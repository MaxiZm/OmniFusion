import litellm
import asyncio
import os
import logging
import random
from typing import Optional
from ..ratelimit.limiter import Slot, rate_limiter
from ..ratelimit.circuit_breaker import CircuitOpenError, circuit_breaker
from ..store.providers import resolve_provider_for_model
from ..providers.validation import validate_base_url
from ..providers.capabilities import get_provider_type_from_model, filter_params
from ..api.errors import OmniFusionError
from ..api.model_names import is_fusion_model_reference

logger = logging.getLogger("omnifusion.llm")


class StreamingResponseWrapper:
    def __init__(self, response, slot: Slot, chunk_timeout: Optional[float] = None):
        self.response = response
        self.slot = slot
        self.released = False
        # Per-chunk deadline: prevents a stalled upstream stream from hanging
        # indefinitely while holding the per-key/provider concurrency slots. The
        # stage timeout only covers stream *setup*, not iteration, so we enforce
        # it again on every chunk.
        self.chunk_timeout = chunk_timeout

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            if self.chunk_timeout:
                return await asyncio.wait_for(
                    self.response.__anext__(), timeout=self.chunk_timeout
                )
            return await self.response.__anext__()
        except StopAsyncIteration:
            self.release()
            raise
        except BaseException:
            # Includes asyncio.TimeoutError on a stalled stream — release the slot
            # and propagate so the caller aborts the response.
            self.release()
            raise

    def release(self):
        if not self.released:
            self.slot.release()
            self.released = True

    def __del__(self):
        self.release()


class LLMClient:
    @staticmethod
    async def acompletion(
        provider_id: str,
        model: str,
        messages: list,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        timeout: Optional[int] = None,
        stream: bool = False,
        **kwargs,
    ):
        """
        Wraps litellm.acompletion with rate limiting, timeouts, param filtering, and backoff retries.
        """
        if is_fusion_model_reference(model):
            raise OmniFusionError(
                f"Recursive fusion model invocation is blocked for model '{model}'.",
                status_code=400,
                type_="invalid_request_error",
                code="recursive_fusion_model",
            )

        # 1. Convert Pydantic ChatMessage objects to dicts if they aren't dicts already
        dict_messages = []
        for m in messages:
            if hasattr(m, "model_dump"):
                dict_messages.append(m.model_dump())
            elif isinstance(m, dict):
                dict_messages.append(m)
            else:
                dict_messages.append(dict(m))

        # 2. Dynamically resolve provider configs from DB
        provider = await resolve_provider_for_model(model)

        provider_type = None
        if provider:
            provider_id = provider["id"]
            provider_type = provider["type"]

            # Decrypt secret keys or resolve env refs
            if provider.get("api_key"):
                api_key = provider["api_key"]
                from ..secrets.redact import redactor

                redactor.add_secret(api_key)
            elif provider.get("api_key_ref"):
                ref = provider["api_key_ref"]
                api_key = os.getenv(ref)
                if api_key:
                    from ..secrets.redact import redactor

                    redactor.add_secret(api_key)

            # Perform SSRF validation on the provider base URL if present
            if provider.get("base_url"):
                api_base = validate_base_url(provider["base_url"], provider_type)

        if not provider_type:
            provider_type = get_provider_type_from_model(model)

        # 2b. Canonicalize the model for custom OpenAI/Anthropic-compatible providers.
        # LiteLLM routes by a provider prefix; a custom provider's bare model name
        # (e.g. "my-model" from the UI) must become "openai/my-model" /
        # "anthropic/my-model" with api_base pointing at the custom endpoint.
        if provider_type == "custom_openai" and not model.startswith("openai/"):
            model = f"openai/{model}"
        elif provider_type == "custom_anthropic" and not model.startswith("anthropic/"):
            model = f"anthropic/{model}"
        elif provider_type == "openrouter" and not model.startswith("openrouter/"):
            model = f"openrouter/{model}"

        # 3. Filter outgoing LLM arguments based on the resolved provider type capabilities
        filtered_kwargs = filter_params(provider_type, kwargs)

        # 4. Construct LiteLLM config
        call_kwargs = {
            "model": model,
            "messages": dict_messages,
            "stream": stream,
            **filtered_kwargs,
        }

        if api_key:
            call_kwargs["api_key"] = api_key
        if api_base:
            call_kwargs["api_base"] = api_base

        # 5. Provider circuit breaker: fail fast once a provider is unhealthy.
        if not circuit_breaker.allow_request(provider_id):
            raise CircuitOpenError(provider_id)

        # 6. Concurrency control and retry loop on 429
        retries = 3
        backoff = 1.0  # start at 1s

        for attempt in range(retries + 1):
            slot = await rate_limiter.acquire(provider_id)
            try:
                if timeout:
                    res = await asyncio.wait_for(
                        litellm.acompletion(**call_kwargs), timeout=timeout
                    )
                else:
                    res = await litellm.acompletion(**call_kwargs)

                if stream:
                    # Carry the stage timeout into stream iteration as a per-chunk
                    # deadline so a hung stream can't hold slots indefinitely.
                    wrapped_res = StreamingResponseWrapper(
                        res, slot, chunk_timeout=timeout
                    )
                    slot = None
                    circuit_breaker.record_success(provider_id)
                    return wrapped_res
                else:
                    circuit_breaker.record_success(provider_id)
                    return res
            except Exception as e:
                # Check if it's a rate limit error (429)
                is_rate_limit = False
                err_str = str(e).lower()
                status_code = getattr(e, "status_code", None)

                if status_code == 429:
                    is_rate_limit = True
                elif (
                    "rate_limit" in err_str
                    or "rate limit" in err_str
                    or "429" in err_str
                ):
                    is_rate_limit = True

                if is_rate_limit and attempt < retries:
                    sleep_time = backoff * (2**attempt) + random.uniform(0, 0.5)
                    logger.warning(
                        f"Upstream rate limit hit on model {model} (provider {provider_id}). "
                        f"Retrying in {sleep_time:.2f}s... (Attempt {attempt + 1}/{retries})"
                    )
                    slot.release()
                    slot = None
                    await asyncio.sleep(sleep_time)
                else:
                    circuit_breaker.record_failure(provider_id)
                    raise e
            finally:
                if slot is not None:
                    slot.release()


llm_client = LLMClient()

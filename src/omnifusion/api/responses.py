from __future__ import annotations

import json
import time
import uuid
from typing import Any, Literal, Optional, Union

from fastapi import APIRouter, Depends, Request, Response
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, JsonValue, field_validator

from .auth import verify_api_key
from .chat import create_chat_completion
from .schemas import ChatCompletionRequest, StreamOptions, ToolChoice, ToolDefinition

router = APIRouter()


class ResponsesRequest(BaseModel):
    model: str
    input: Union[str, list[Any]]
    instructions: Optional[str] = None
    tools: Optional[list[ToolDefinition]] = None
    tool_choice: Optional[ToolChoice] = None
    stream: bool = False
    max_output_tokens: Optional[int] = None
    metadata: Optional[dict[str, JsonValue]] = None
    previous_response_id: Optional[str] = None

    @field_validator("previous_response_id")
    @classmethod
    def reject_stateful_conversation(cls, value):
        if value is not None:
            raise ValueError("previous_response_id is not supported.")
        return value


def input_to_messages(input_value: str | list[Any], instructions: str | None) -> list[dict]:
    messages = []
    if instructions:
        messages.append({"role": "system", "content": instructions})

    if isinstance(input_value, str):
        messages.append({"role": "user", "content": input_value})
        return messages

    for item in input_value:
        if isinstance(item, str):
            messages.append({"role": "user", "content": item})
            continue
        if not isinstance(item, dict):
            raise ValueError("Responses input array items must be strings or objects.")

        item_type = item.get("type")
        if item_type in {"input_text", "text"}:
            messages.append({"role": "user", "content": item.get("text", "")})
        elif item_type == "message" or "role" in item:
            messages.append(
                {
                    "role": item.get("role", "user"),
                    "content": item.get("content", ""),
                }
            )
        else:
            raise ValueError(f"Unsupported Responses input item type: {item_type}")

    return messages


def chat_usage_to_response_usage(usage: dict[str, Any] | None) -> dict[str, int]:
    usage = usage or {}
    return {
        "input_tokens": int(usage.get("prompt_tokens", 0) or 0),
        "output_tokens": int(usage.get("completion_tokens", 0) or 0),
        "total_tokens": int(usage.get("total_tokens", 0) or 0),
    }


def shape_response(chat_result: dict[str, Any], text: str | None = None) -> dict[str, Any]:
    if text is None:
        choices = chat_result.get("choices") or []
        message = choices[0].get("message", {}) if choices else {}
        text = message.get("content") or ""

    return {
        "id": chat_result.get("id") or f"resp-{uuid.uuid4()}",
        "object": "response",
        "created_at": chat_result.get("created") or int(time.time()),
        "model": chat_result.get("model"),
        "status": "completed",
        "output": [
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": text}],
            }
        ],
        "usage": chat_usage_to_response_usage(chat_result.get("usage")),
    }


def response_event(event_type: Literal["response.output_text.delta", "response.completed"], data: dict) -> str:
    return f"event: {event_type}\ndata: {json.dumps(data)}\n\n"


async def response_stream(chat_stream: StreamingResponse, model: str):
    response_id = f"resp-{uuid.uuid4()}"
    collected = ""
    latest_model = model
    collected_usage: dict[str, Any] = {}
    buffer = ""

    async for chunk in chat_stream.body_iterator:
        buffer += chunk.decode("utf-8") if isinstance(chunk, bytes) else str(chunk)
        while "\n\n" in buffer:
            raw_event, buffer = buffer.split("\n\n", 1)
            data_lines = [
                line.removeprefix("data: ")
                for line in raw_event.splitlines()
                if line.startswith("data: ")
            ]
            if not data_lines:
                continue
            raw_data = "\n".join(data_lines)
            if raw_data == "[DONE]":
                continue
            payload = json.loads(raw_data)
            latest_model = payload.get("model", latest_model)
            # A terminal usage chunk carries a usage block (with empty choices);
            # capture it so response.completed can report real usage.
            if payload.get("usage"):
                collected_usage = payload["usage"]
            choices = payload.get("choices") or []
            delta = (choices[0].get("delta") or {}).get("content") if choices else None
            if delta:
                collected += delta
                yield response_event(
                    "response.output_text.delta",
                    {
                        "type": "response.output_text.delta",
                        "response_id": response_id,
                        "delta": delta,
                    },
                )

    completed = shape_response(
        {
            "id": response_id,
            "created": int(time.time()),
            "model": latest_model,
            "usage": collected_usage,
        },
        text=collected,
    )
    yield response_event(
        "response.completed",
        {"type": "response.completed", "response": completed},
    )


@router.post("/responses")
async def create_response(
    request: Request,
    body: ResponsesRequest,
    response: Response,
    key_hash: str = Depends(verify_api_key),
):
    chat_body = ChatCompletionRequest(
        model=body.model,
        messages=input_to_messages(body.input, body.instructions),
        tools=body.tools,
        tool_choice=body.tool_choice,
        stream=body.stream,
        max_tokens=body.max_output_tokens,
        metadata=body.metadata,
        # Ask the underlying chat stream for a terminal usage chunk so the streamed
        # response.completed event can report usage (parity with non-stream).
        stream_options=StreamOptions(include_usage=True) if body.stream else None,
    )
    chat_result = await create_chat_completion(request, chat_body, response, key_hash)
    if isinstance(chat_result, StreamingResponse):
        return StreamingResponse(
            response_stream(chat_result, chat_body.model),
            media_type="text/event-stream",
            headers=dict(chat_result.headers),
        )
    return shape_response(chat_result)

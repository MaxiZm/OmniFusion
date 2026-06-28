import logging
import time
from enum import Enum

from ..api.errors import OmniFusionError

logger = logging.getLogger("omnifusion.circuit_breaker")


class CircuitState(str, Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreaker:
    def __init__(self, failure_threshold: int = 5, cooldown_seconds: float = 30.0):
        self.failure_threshold = max(1, int(failure_threshold))
        self.cooldown_seconds = max(0.0, float(cooldown_seconds))
        self._circuits: dict[str, _ProviderCircuit] = {}

    def configure_from_settings(self, cfg) -> None:
        self.configure(
            getattr(cfg, "omnifusion_circuit_breaker_failure_threshold", self.failure_threshold),
            getattr(cfg, "omnifusion_circuit_breaker_cooldown_seconds", self.cooldown_seconds),
        )

    def configure(self, failure_threshold: int, cooldown_seconds: float) -> None:
        self.failure_threshold = max(1, int(failure_threshold))
        self.cooldown_seconds = max(0.0, float(cooldown_seconds))
        for circuit in self._circuits.values():
            circuit.failure_threshold = self.failure_threshold
            circuit.cooldown_seconds = self.cooldown_seconds

    def _circuit(self, provider_id: str) -> "_ProviderCircuit":
        if provider_id not in self._circuits:
            self._circuits[provider_id] = _ProviderCircuit(
                provider_id,
                self.failure_threshold,
                self.cooldown_seconds,
            )
        return self._circuits[provider_id]

    def allow_request(self, provider_id: str) -> bool:
        return self._circuit(provider_id).allow_request()

    def record_success(self, provider_id: str) -> None:
        self._circuit(provider_id).record_success()

    def record_failure(self, provider_id: str) -> None:
        self._circuit(provider_id).record_failure()

    def get_all_states(self) -> dict[str, dict]:
        return {
            provider_id: {
                "state": circuit.state.value,
                "failures": circuit.consecutive_failures,
            }
            for provider_id, circuit in self._circuits.items()
        }


class _ProviderCircuit:
    def __init__(self, provider_id: str, failure_threshold: int, cooldown_seconds: float):
        self.provider_id = provider_id
        self.failure_threshold = failure_threshold
        self.cooldown_seconds = cooldown_seconds
        self.state = CircuitState.CLOSED
        self.consecutive_failures = 0
        self.last_failure_time = 0.0
        self._probe_in_flight = False

    def allow_request(self) -> bool:
        if self.state == CircuitState.CLOSED:
            return True

        if self.state == CircuitState.OPEN:
            if time.monotonic() - self.last_failure_time < self.cooldown_seconds:
                return False
            self.state = CircuitState.HALF_OPEN
            self._probe_in_flight = True
            logger.info("Circuit for provider '%s' entering half-open state", self.provider_id)
            return True

        if self._probe_in_flight:
            return False
        self._probe_in_flight = True
        return True

    def record_success(self) -> None:
        self.state = CircuitState.CLOSED
        self.consecutive_failures = 0
        self._probe_in_flight = False

    def record_failure(self) -> None:
        self.consecutive_failures += 1
        self.last_failure_time = time.monotonic()
        self._probe_in_flight = False
        if self.state == CircuitState.HALF_OPEN:
            self.state = CircuitState.OPEN
            logger.warning("Circuit for provider '%s' reopened after failed probe", self.provider_id)
        elif self.consecutive_failures >= self.failure_threshold:
            self.state = CircuitState.OPEN
            logger.warning(
                "Circuit for provider '%s' opened after %s failures",
                self.provider_id,
                self.consecutive_failures,
            )


class CircuitOpenError(OmniFusionError):
    def __init__(self, provider_id: str):
        super().__init__(
            f"Provider '{provider_id}' circuit breaker is open; retry after cooldown.",
            status_code=503,
            type_="server_error",
            code="provider_circuit_open",
        )


circuit_breaker = CircuitBreaker()


def configure_from_settings(cfg) -> None:
    circuit_breaker.configure_from_settings(cfg)

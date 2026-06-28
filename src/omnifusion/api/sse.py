"""Request-side streaming helpers.

SSE emission (chunk framing, terminal usage chunk, [DONE]) is owned by the single
canonical StreamingAdapter in fusion.runtime.streaming; this module only inspects
the request for streaming intent.
"""


def wants_usage(body) -> bool:
    """True when the request opted into a streaming usage chunk via stream_options."""
    so = getattr(body, "stream_options", None)
    return bool(so and getattr(so, "include_usage", False))

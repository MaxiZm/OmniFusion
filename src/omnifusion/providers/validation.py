import urllib.parse
import ipaddress
import logging
from ..api.errors import ConfigurationError
from ..settings import settings

logger = logging.getLogger("omnifusion.ssrf")

# Provider types that legitimately point at self-hosted / local endpoints
# (loopback or private LAN). These are configured exclusively by the
# authenticated admin with an explicit base_url, so private egress is expected.
LOCAL_PROVIDER_TYPES = ("ollama", "lmstudio", "custom_openai", "custom_anthropic")


def _check_ip_address(ip: str, provider_type: str) -> None:
    """Raises ConfigurationError if the IP is in a blocked range."""
    clean_ip = ip.split('%')[0]
    ip_obj = ipaddress.ip_address(clean_ip)

    # Block cloud metadata endpoints
    if str(ip_obj) == "169.254.169.254" or str(ip_obj) == "fd00:ec2::254":
        raise ConfigurationError(f"Egress to cloud metadata {ip} is blocked")

    if ip_obj.is_loopback or ip_obj.is_private or ip_obj.is_link_local:
        raise ConfigurationError(
            f"Egress to private/loopback IP {ip} is blocked for provider type {provider_type}"
        )


def validate_base_url(url: str, provider_type: str) -> str:
    """
    SSRF protection. Rejects cloud metadata, loopback, and local network egress
    unless OMNIFUSION_ALLOW_PRIVATE_EGRESS is set, or it's a local provider.

    DNS-rebinding / TOCTOU mitigation:
    - For http:// providers we return the URL with the hostname replaced by the
      validated resolved IP, so litellm connects to the pinned IP rather than
      re-resolving at connect time.
    - For https:// providers we return the original (hostname) URL after validating
      all resolved IPs. Substituting the IP would break TLS — the Host/SNI would
      become the raw IP, causing certificate validation failures for hostname-addressed
      endpoints. The remaining TOCTOU window for https is acceptable given that:
        1. All resolved IPs are checked at save time (attacker must pass check).
        2. A successful DNS rebind still requires the provider's TLS cert to match,
           which is infeasible without compromising the CA.
    - In either case the function fails-closed on gaierror.
    """
    if not url:
        return url

    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise ConfigurationError("base_url scheme must be http or https")

    hostname = parsed.hostname
    if not hostname:
        raise ConfigurationError("base_url must have a hostname")

    # Local / self-hosted providers are allowed to hit loopback/private
    if provider_type in LOCAL_PROVIDER_TYPES:
        return url

    if settings.omnifusion_allow_private_egress:
        return url

    # Resolve hostname synchronously (we're called at save time, not on the hot path)
    import socket
    try:
        addr_info = socket.getaddrinfo(hostname, None)
        resolved_ips = []
        for family, socktype, proto, canonname, sockaddr in addr_info:
            if sockaddr and len(sockaddr) > 0:
                ip = sockaddr[0]
                resolved_ips.append(ip)
    except socket.gaierror as e:
        # Fix (medium): Fail-closed — if we can't resolve, reject rather than allow.
        raise ConfigurationError(
            f"Cannot resolve hostname '{hostname}' for SSRF validation: {e}. "
            f"If this is intentional, set OMNIFUSION_ALLOW_PRIVATE_EGRESS=1."
        )

    if not resolved_ips:
        raise ConfigurationError(
            f"Hostname '{hostname}' resolved to no addresses."
        )

    # Validate all resolved IPs
    for ip in set(resolved_ips):
        _check_ip_address(ip, provider_type)

    # Fix B: Only pin the IP for http:// — for https:// pinning breaks TLS because
    # the Host/SNI becomes a raw IP address, causing certificate validation failures.
    # For https, we return the original hostname URL (all IPs already validated above).
    # The residual TOCTOU window for https is mitigated by TLS cert verification itself.
    if parsed.scheme == "https":
        return url

    # http://: substitute the hostname with the first validated IP to prevent
    # DNS rebinding at litellm's connect time.
    pinned_ip = resolved_ips[0]
    # Handle IPv6: must be wrapped in brackets
    if ":" in pinned_ip:
        pinned_host = f"[{pinned_ip}]"
    else:
        pinned_host = pinned_ip

    # Rebuild the URL with the pinned IP
    port_part = f":{parsed.port}" if parsed.port else ""
    pinned_url = urllib.parse.urlunparse(
        parsed._replace(netloc=f"{pinned_host}{port_part}")
    )

    return pinned_url

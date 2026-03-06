"""IP parsing utilities."""

from __future__ import annotations

import ipaddress
from typing import Optional


def ipv4_to_int(ip: object) -> Optional[int]:
    """Convert IPv4 text to integer, returning None for invalid values."""
    if ip is None:
        return None
    text = str(ip).strip()
    if not text:
        return None
    try:
        return int(ipaddress.IPv4Address(text))
    except (ipaddress.AddressValueError, ValueError):
        return None

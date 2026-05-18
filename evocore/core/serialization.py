"""Private helpers for stable public export payloads."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from importlib import metadata as importlib_metadata


def _stable_sort_key(value: object) -> str:
    return json.dumps(value, sort_keys=True)


def json_safe(value: object) -> object:
    """Return a JSON-safe representation with deterministic container ordering."""
    if value is None or isinstance(value, str | bool | int):
        result = value
    elif isinstance(value, float):
        result = value if math.isfinite(value) else None
    elif hasattr(value, "to_dict") and callable(value.to_dict):
        result = json_safe(value.to_dict())
    elif isinstance(value, Mapping):
        result = {str(key): json_safe(value[key]) for key in sorted(value, key=str)}
    elif isinstance(value, set | frozenset):
        result = sorted(
            (json_safe(item) for item in value),
            key=_stable_sort_key,
        )
    elif isinstance(value, tuple | list) or (
        isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray)
    ):
        result = [json_safe(item) for item in value]
    else:
        result = repr(value)
    return result


def stable_json_dumps(payload: object, *, indent: int | None = None) -> str:
    """Dump a JSON-safe payload with deterministic key ordering."""
    return json.dumps(json_safe(payload), sort_keys=True, indent=indent, allow_nan=False)


def canonical_json_hash(payload: object) -> str:
    """Return a SHA-256 hash over canonical compact JSON."""
    text = json.dumps(
        json_safe(payload),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def package_version() -> str:
    """Return the installed EvoCore version or the local source fallback."""
    try:
        return importlib_metadata.version("evocore")
    except importlib_metadata.PackageNotFoundError:
        return "0.7.0"

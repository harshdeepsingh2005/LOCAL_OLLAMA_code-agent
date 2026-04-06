"""Shared JSON parsing and validation utilities for agents."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


def extract_json_payload(response: str) -> str:
    """Extract a JSON payload from fenced or mixed LLM output without regex-heavy parsing."""
    text = response.strip()

    fence_idx = text.find("```json")
    if fence_idx != -1:
        after = text[fence_idx + len("```json"):]
        end_fence = after.find("```")
        candidate = after[:end_fence] if end_fence != -1 else after
        candidate = candidate.strip()
        if candidate:
            return candidate

    value = _find_first_json_value(text)
    return value if value else text


def parse_json_object(response: str) -> dict[str, Any]:
    """Parse a JSON object from LLM output with self-healing repairs."""
    data = parse_json_value(response)
    if not isinstance(data, dict):
        raise TypeError("JSON root must be an object")
    return data


def parse_json_value(response: str) -> Any:
    """Parse any JSON value from LLM output with self-healing repairs."""
    payload = extract_json_payload(response)
    try:
        data = json.loads(payload)
    except json.JSONDecodeError:
        repaired = _repair_json(payload)
        data = json.loads(repaired)
    return data


def is_safe_relative_path(path: str) -> bool:
    """Validate agent-emitted file paths are relative and do not traverse upward."""
    candidate = (path or "").strip()
    if not candidate:
        return False
    p = Path(candidate)
    if p.is_absolute():
        return False
    return ".." not in p.parts


def _find_first_json_object(text: str) -> str | None:
    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    in_string = False
    escaped = False

    for idx in range(start, len(text)):
        ch = text[idx]

        if in_string:
            if escaped:
                escaped = False
                continue
            if ch == "\\":
                escaped = True
                continue
            if ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
            continue
        if ch == "{":
            depth += 1
            continue
        if ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : idx + 1]

    return None


def _find_first_json_value(text: str) -> str | None:
    """Find the first balanced top-level JSON object or array in text."""
    obj = _find_first_json_object(text)
    arr = _find_first_json_array(text)

    if obj is None:
        return arr
    if arr is None:
        return obj

    obj_idx = text.find(obj)
    arr_idx = text.find(arr)
    return obj if (0 <= obj_idx < arr_idx or arr_idx == -1) else arr


def _find_first_json_array(text: str) -> str | None:
    start = text.find("[")
    if start == -1:
        return None

    depth = 0
    in_string = False
    escaped = False

    for idx in range(start, len(text)):
        ch = text[idx]

        if in_string:
            if escaped:
                escaped = False
                continue
            if ch == "\\":
                escaped = True
                continue
            if ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
            continue
        if ch == "[":
            depth += 1
            continue
        if ch == "]":
            depth -= 1
            if depth == 0:
                return text[start : idx + 1]

    return None


def _repair_json(raw: str) -> str:
    repaired = raw
    repaired = repaired.replace("\u201c", '"').replace("\u201d", '"').replace("\u2019", "'")
    repaired = re.sub(r",\s*([}\]])", r"\1", repaired)
    repaired = _escape_control_chars_in_strings(repaired)
    return repaired


def _escape_control_chars_in_strings(raw_json: str) -> str:
    out: list[str] = []
    in_string = False
    escaped = False

    for ch in raw_json:
        if in_string:
            if escaped:
                out.append(ch)
                escaped = False
                continue

            if ch == "\\":
                out.append(ch)
                escaped = True
                continue

            if ch == '"':
                out.append(ch)
                in_string = False
                continue

            code = ord(ch)
            if code < 0x20:
                if ch == "\n":
                    out.append("\\n")
                elif ch == "\r":
                    out.append("\\r")
                elif ch == "\t":
                    out.append("\\t")
                else:
                    out.append(f"\\u{code:04x}")
                continue

            out.append(ch)
            continue

        out.append(ch)
        if ch == '"':
            in_string = True
            escaped = False

    return "".join(out)

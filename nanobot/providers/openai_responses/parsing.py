"""Parse Responses API SSE streams and SDK response objects."""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from pathlib import Path
import time
from typing import Any, AsyncGenerator

import httpx
import json_repair
from loguru import logger

from nanobot.providers.base import LLMResponse, ToolCallRequest

FINISH_REASON_MAP = {
    "completed": "stop",
    "incomplete": "length",
    "failed": "error",
    "cancelled": "error",
}


def map_finish_reason(status: str | None) -> str:
    """Map a Responses API status string to a Chat-Completions-style finish_reason."""
    return FINISH_REASON_MAP.get(status or "completed", "stop")


async def iter_sse(response: httpx.Response) -> AsyncGenerator[dict[str, Any], None]:
    """Yield parsed JSON events from a Responses API SSE stream."""
    buffer: list[str] = []

    def _flush() -> dict[str, Any] | None:
        data_lines = [l[5:].strip() for l in buffer if l.startswith("data:")]
        buffer.clear()
        if not data_lines:
            return None
        data = "\n".join(data_lines).strip()
        if not data or data == "[DONE]":
            return None
        try:
            return json.loads(data)
        except Exception:
            logger.warning("Failed to parse SSE event JSON: {}", data[:200])
            return None

    async for line in response.aiter_lines():
        if line == "":
            if buffer:
                event = _flush()
                if event is not None:
                    yield event
            continue
        buffer.append(line)

    # Flush any remaining buffer at EOF (#10)
    if buffer:
        event = _flush()
        if event is not None:
            yield event


async def consume_sse(
    response: httpx.Response,
    on_content_delta: Callable[[str], Awaitable[None]] | None = None,
) -> tuple[str, list[ToolCallRequest], str]:
    """Consume a Responses API SSE stream into ``(content, tool_calls, finish_reason)``."""
    content = ""
    tool_calls: list[ToolCallRequest] = []
    tool_call_buffers: dict[str, dict[str, Any]] = {}
    finish_reason = "stop"

    async for event in iter_sse(response):
        event_type = event.get("type")
        if event_type == "response.output_item.added":
            item = event.get("item") or {}
            if item.get("type") == "function_call":
                call_id = item.get("call_id")
                if not call_id:
                    continue
                tool_call_buffers[call_id] = {
                    "id": item.get("id") or "fc_0",
                    "name": item.get("name"),
                    "arguments": item.get("arguments") or "",
                }
        elif event_type == "response.output_text.delta":
            delta_text = event.get("delta") or ""
            content += delta_text
            if on_content_delta and delta_text:
                await on_content_delta(delta_text)
        elif event_type == "response.function_call_arguments.delta":
            call_id = event.get("call_id")
            if call_id and call_id in tool_call_buffers:
                tool_call_buffers[call_id]["arguments"] += event.get("delta") or ""
        elif event_type == "response.function_call_arguments.done":
            call_id = event.get("call_id")
            if call_id and call_id in tool_call_buffers:
                tool_call_buffers[call_id]["arguments"] = event.get("arguments") or ""
        elif event_type == "response.output_item.done":
            item = event.get("item") or {}
            if item.get("type") == "function_call":
                call_id = item.get("call_id")
                if not call_id:
                    continue
                buf = tool_call_buffers.get(call_id) or {}
                args_raw = buf.get("arguments") or item.get("arguments") or "{}"
                try:
                    args = json.loads(args_raw)
                except Exception:
                    logger.warning(
                        "Failed to parse tool call arguments for '{}': {}",
                        buf.get("name") or item.get("name"),
                        args_raw[:200],
                    )
                    args = json_repair.loads(args_raw)
                    if not isinstance(args, dict):
                        args = {"raw": args_raw}
                tool_calls.append(
                    ToolCallRequest(
                        id=f"{call_id}|{buf.get('id') or item.get('id') or 'fc_0'}",
                        name=buf.get("name") or item.get("name") or "",
                        arguments=args,
                    )
                )
        elif event_type == "response.completed":
            status = (event.get("response") or {}).get("status")
            finish_reason = map_finish_reason(status)
        elif event_type in {"error", "response.failed"}:
            detail = event.get("error") or event.get("message") or event
            raise RuntimeError(f"Response failed: {str(detail)[:500]}")

    return content, tool_calls, finish_reason


def parse_response_output(response: Any) -> LLMResponse:
    """Parse an SDK ``Response`` object into an ``LLMResponse``."""
    if not isinstance(response, dict):
        dump = getattr(response, "model_dump", None)
        response = dump() if callable(dump) else vars(response)

    output = response.get("output") or []
    content_parts: list[str] = []
    tool_calls: list[ToolCallRequest] = []
    reasoning_content: str | None = None

    for item in output:
        if not isinstance(item, dict):
            dump = getattr(item, "model_dump", None)
            item = dump() if callable(dump) else vars(item)

        item_type = item.get("type")
        if item_type == "message":
            for block in item.get("content") or []:
                if not isinstance(block, dict):
                    dump = getattr(block, "model_dump", None)
                    block = dump() if callable(dump) else vars(block)
                if block.get("type") == "output_text":
                    content_parts.append(block.get("text") or "")
        elif item_type == "reasoning":
            for s in item.get("summary") or []:
                if not isinstance(s, dict):
                    dump = getattr(s, "model_dump", None)
                    s = dump() if callable(dump) else vars(s)
                if s.get("type") == "summary_text" and s.get("text"):
                    reasoning_content = (reasoning_content or "") + s["text"]
        elif item_type == "function_call":
            call_id = item.get("call_id") or ""
            item_id = item.get("id") or "fc_0"
            args_raw = item.get("arguments") or "{}"
            try:
                args = json.loads(args_raw) if isinstance(args_raw, str) else args_raw
            except Exception:
                logger.warning(
                    "Failed to parse tool call arguments for '{}': {}",
                    item.get("name"),
                    str(args_raw)[:200],
                )
                args = json_repair.loads(args_raw) if isinstance(args_raw, str) else args_raw
                if not isinstance(args, dict):
                    args = {"raw": args_raw}
            tool_calls.append(ToolCallRequest(
                id=f"{call_id}|{item_id}",
                name=item.get("name") or "",
                arguments=args if isinstance(args, dict) else {},
            ))

    usage_raw = response.get("usage") or {}
    if not isinstance(usage_raw, dict):
        dump = getattr(usage_raw, "model_dump", None)
        usage_raw = dump() if callable(dump) else vars(usage_raw)
    usage = {}
    if usage_raw:
        usage = {
            "prompt_tokens": int(usage_raw.get("input_tokens") or 0),
            "completion_tokens": int(usage_raw.get("output_tokens") or 0),
            "total_tokens": int(usage_raw.get("total_tokens") or 0),
        }

    status = response.get("status")
    finish_reason = map_finish_reason(status)

    return LLMResponse(
        content="".join(content_parts) or None,
        tool_calls=tool_calls,
        finish_reason=finish_reason,
        usage=usage,
        reasoning_content=reasoning_content if isinstance(reasoning_content, str) else None,
    )


def _event_get(event: Any, key: str, default: Any = None) -> Any:
    if isinstance(event, dict):
        return event.get(key, default)
    return getattr(event, key, default)


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(key): _to_jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(item) for item in value]
    dump = getattr(value, "model_dump", None)
    if callable(dump):
        return _to_jsonable(dump())
    if hasattr(value, "__dict__"):
        return _to_jsonable(vars(value))
    return str(value)


def _record_stream_event(record_dir: Path | None, event: Any, index: int) -> None:
    if record_dir is None:
        return
    stream_dir = record_dir / "stream_chunks"
    stream_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y-%m-%dT%H-%M-%S", time.gmtime())
    millis = int((time.time() % 1) * 1000)
    chunk_id = f"{index:06d}_{timestamp}.{millis:03d}Z"
    (stream_dir / f"{chunk_id}.json").write_text(
        json.dumps(_to_jsonable(event), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


async def consume_sdk_stream(
    stream: Any,
    on_content_delta: Callable[[str], Awaitable[None]] | None = None,
    record_dir: Path | None = None,
) -> tuple[str, list[ToolCallRequest], str, dict[str, int], str | None]:
    """Consume an SDK async stream from ``client.responses.create(stream=True)``."""
    content = ""
    tool_calls: list[ToolCallRequest] = []
    tool_call_buffers: dict[str, dict[str, Any]] = {}
    finish_reason = "stop"
    usage: dict[str, int] = {}
    reasoning_content: str | None = None
    event_index = 0

    async for event in stream:
        event_index += 1
        _record_stream_event(record_dir, event, event_index)
        event_type = _event_get(event, "type")
        if event_type == "response.output_item.added":
            item = _event_get(event, "item")
            if item and _event_get(item, "type") == "function_call":
                call_id = _event_get(item, "call_id")
                if not call_id:
                    continue
                tool_call_buffers[call_id] = {
                    "id": _event_get(item, "id") or "fc_0",
                    "name": _event_get(item, "name"),
                    "arguments": _event_get(item, "arguments") or "",
                }
        elif event_type == "response.output_text.delta":
            delta_text = _event_get(event, "delta", "") or ""
            content += delta_text
            if on_content_delta and delta_text:
                await on_content_delta(delta_text)
        elif event_type == "response.function_call_arguments.delta":
            call_id = _event_get(event, "call_id")
            if call_id and call_id in tool_call_buffers:
                tool_call_buffers[call_id]["arguments"] += _event_get(event, "delta", "") or ""
        elif event_type == "response.function_call_arguments.done":
            call_id = _event_get(event, "call_id")
            if call_id and call_id in tool_call_buffers:
                tool_call_buffers[call_id]["arguments"] = _event_get(event, "arguments", "") or ""
        elif event_type == "response.output_item.done":
            item = _event_get(event, "item")
            if item and _event_get(item, "type") == "function_call":
                call_id = _event_get(item, "call_id")
                if not call_id:
                    continue
                buf = tool_call_buffers.get(call_id) or {}
                args_raw = buf.get("arguments") or _event_get(item, "arguments") or "{}"
                try:
                    args = json.loads(args_raw)
                except Exception:
                    logger.warning(
                        "Failed to parse tool call arguments for '{}': {}",
                        buf.get("name") or _event_get(item, "name"),
                        str(args_raw)[:200],
                    )
                    args = json_repair.loads(args_raw)
                    if not isinstance(args, dict):
                        args = {"raw": args_raw}
                tool_calls.append(
                    ToolCallRequest(
                        id=f"{call_id}|{buf.get('id') or _event_get(item, 'id') or 'fc_0'}",
                        name=buf.get("name") or _event_get(item, "name") or "",
                        arguments=args,
                    )
                )
        elif event_type == "response.completed":
            resp = _event_get(event, "response")
            status = _event_get(resp, "status") if resp else None
            finish_reason = map_finish_reason(status)
            if resp:
                usage_obj = _event_get(resp, "usage")
                if usage_obj:
                    usage = {
                        "prompt_tokens": int(_event_get(usage_obj, "input_tokens", 0) or 0),
                        "completion_tokens": int(_event_get(usage_obj, "output_tokens", 0) or 0),
                        "total_tokens": int(_event_get(usage_obj, "total_tokens", 0) or 0),
                    }
                for out_item in _event_get(resp, "output") or []:
                    if _event_get(out_item, "type") == "reasoning":
                        for s in _event_get(out_item, "summary") or []:
                            if _event_get(s, "type") == "summary_text":
                                text = _event_get(s, "text")
                                if text:
                                    reasoning_content = (reasoning_content or "") + text
        elif event_type in {"error", "response.failed"}:
            detail = _event_get(event, "error") or _event_get(event, "message") or event
            raise RuntimeError(f"Response failed: {str(detail)[:500]}")

    return content, tool_calls, finish_reason, usage, reasoning_content

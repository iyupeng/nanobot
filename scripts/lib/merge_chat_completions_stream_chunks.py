from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path
from typing import Any


_SKIP_MESSAGE_FIELDS = {"role", "content",
                        "tool_calls", "reasoning_content", "refusal"}
_SKIP_TOOL_CALL_FIELDS = {"index", "id", "type", "function"}
_SKIP_FUNCTION_FIELDS = {"name", "arguments"}


def _load_chunks(stream_chunks_dir: Path) -> list[dict[str, Any]]:
    chunk_paths = sorted(
        path for path in stream_chunks_dir.iterdir() if path.is_file() and path.suffix == ".json"
    )
    if not chunk_paths:
        raise ValueError(f"No JSON chunk files found in {stream_chunks_dir}")

    chunks: list[dict[str, Any]] = []
    for chunk_path in chunk_paths:
        with chunk_path.open("r", encoding="utf-8") as file:
            payload = json.load(file)
        if not isinstance(payload, dict):
            raise ValueError(
                f"Chunk file must contain a JSON object: {chunk_path}")
        chunks.append(payload)
    return chunks


def _append_text(parts: list[str], value: Any) -> None:
    if isinstance(value, str) and value:
        parts.append(value)


def _append_content(buffer: dict[str, Any], value: Any) -> None:
    if value is None:
        return
    if isinstance(value, str):
        buffer["content_text_parts"].append(value)
        return

    if buffer["content_items"] is None:
        existing_text = "".join(buffer["content_text_parts"])
        if existing_text:
            buffer["content_items"] = [{"type": "text", "text": existing_text}]
        else:
            buffer["content_items"] = []
        buffer["content_text_parts"].clear()

    if isinstance(value, list):
        buffer["content_items"].extend(deepcopy(value))
        return

    buffer["content_items"].append(deepcopy(value))


def _tool_call_buffer(choice_buffer: dict[str, Any], tool_call_index: int) -> dict[str, Any]:
    return choice_buffer["tool_calls"].setdefault(tool_call_index, {
        "id": None,
        "type": "function",
        "function_name_parts": [],
        "function_arguments_parts": [],
        "extra": {},
        "function_extra": {},
    })


def _merge_tool_call(choice_buffer: dict[str, Any], tool_call: dict[str, Any], fallback_index: int) -> None:
    tool_call_index = tool_call.get("index")
    if not isinstance(tool_call_index, int):
        tool_call_index = fallback_index

    buffer = _tool_call_buffer(choice_buffer, tool_call_index)
    if tool_call.get("id"):
        buffer["id"] = str(tool_call["id"])
    if tool_call.get("type"):
        buffer["type"] = str(tool_call["type"])

    function_payload = tool_call.get("function")
    if isinstance(function_payload, dict):
        _append_text(buffer["function_name_parts"],
                     function_payload.get("name"))
        _append_text(buffer["function_arguments_parts"],
                     function_payload.get("arguments"))
        for key, value in function_payload.items():
            if key in _SKIP_FUNCTION_FIELDS or value is None:
                continue
            buffer["function_extra"][key] = deepcopy(value)

    for key, value in tool_call.items():
        if key in _SKIP_TOOL_CALL_FIELDS or value is None:
            continue
        buffer["extra"][key] = deepcopy(value)


def _finalize_content(choice_buffer: dict[str, Any]) -> Any:
    if choice_buffer["content_items"] is not None:
        trailing_text = "".join(choice_buffer["content_text_parts"])
        if trailing_text:
            choice_buffer["content_items"].append(
                {"type": "text", "text": trailing_text})
        return choice_buffer["content_items"] or None

    content = "".join(choice_buffer["content_text_parts"])
    return content or None


def _finalize_tool_calls(choice_buffer: dict[str, Any]) -> list[dict[str, Any]] | None:
    if not choice_buffer["tool_calls"]:
        return None

    tool_calls: list[dict[str, Any]] = []
    for _, tool_call in sorted(choice_buffer["tool_calls"].items()):
        function_payload: dict[str, Any] = {
            "name": "".join(tool_call["function_name_parts"]),
            "arguments": "".join(tool_call["function_arguments_parts"]),
        }
        function_payload.update(tool_call["function_extra"])

        merged_tool_call: dict[str, Any] = {
            "id": tool_call["id"],
            "type": tool_call["type"],
            "function": function_payload,
        }
        merged_tool_call.update(tool_call["extra"])
        tool_calls.append(merged_tool_call)

    return tool_calls


def _merge_chat_completion_chunks_impl(chunks: list[dict[str, Any]]) -> dict[str, Any]:
    if not chunks:
        raise ValueError("At least one chunk is required")

    merged: dict[str, Any] = {}
    choice_buffers: dict[int, dict[str, Any]] = {}

    for chunk in chunks:
        for key, value in chunk.items():
            if key == "choices" or value is None:
                continue
            merged[key] = deepcopy(value)

        for fallback_index, choice in enumerate(chunk.get("choices") or []):
            if not isinstance(choice, dict):
                raise ValueError("Each choice chunk must be a JSON object")

            choice_index = choice.get("index")
            if not isinstance(choice_index, int):
                choice_index = fallback_index

            choice_buffer = choice_buffers.setdefault(choice_index, {
                "message_role": None,
                "content_text_parts": [],
                "content_items": None,
                "reasoning_content_parts": [],
                "refusal_parts": [],
                "message_extra": {},
                "tool_calls": {},
                "finish_reason": None,
                "logprobs": None,
            })

            if choice.get("finish_reason") is not None:
                choice_buffer["finish_reason"] = choice["finish_reason"]
            if choice.get("logprobs") is not None:
                choice_buffer["logprobs"] = deepcopy(choice["logprobs"])

            delta = choice.get("delta")
            if not isinstance(delta, dict):
                continue

            if delta.get("role"):
                choice_buffer["message_role"] = delta["role"]
            _append_content(choice_buffer, delta.get("content"))
            _append_text(choice_buffer["reasoning_content_parts"], delta.get(
                "reasoning_content"))
            _append_text(choice_buffer["refusal_parts"], delta.get("refusal"))

            for tool_call_fallback_index, tool_call in enumerate(delta.get("tool_calls") or []):
                if not isinstance(tool_call, dict):
                    raise ValueError(
                        "Each streamed tool call must be a JSON object")
                _merge_tool_call(choice_buffer, tool_call,
                                 tool_call_fallback_index)

            for key, value in delta.items():
                if key in _SKIP_MESSAGE_FIELDS or value is None:
                    continue
                choice_buffer["message_extra"][key] = deepcopy(value)

    merged_choices: list[dict[str, Any]] = []
    for choice_index, choice_buffer in sorted(choice_buffers.items()):
        message: dict[str, Any] = {
            "role": choice_buffer["message_role"] or "assistant",
            "content": _finalize_content(choice_buffer),
        }

        reasoning_content = "".join(choice_buffer["reasoning_content_parts"])
        if reasoning_content:
            message["reasoning_content"] = reasoning_content

        refusal = "".join(choice_buffer["refusal_parts"])
        if refusal:
            message["refusal"] = refusal

        tool_calls = _finalize_tool_calls(choice_buffer)
        if tool_calls is not None:
            message["tool_calls"] = tool_calls

        message.update(choice_buffer["message_extra"])

        merged_choice = {
            "index": choice_index,
            "message": message,
            "finish_reason": choice_buffer["finish_reason"],
        }
        if choice_buffer["logprobs"] is not None:
            merged_choice["logprobs"] = choice_buffer["logprobs"]
        merged_choices.append(merged_choice)

    merged["choices"] = merged_choices

    object_name = merged.get("object")
    if isinstance(object_name, str) and object_name.endswith(".chunk"):
        merged["object"] = object_name[: -len(".chunk")]

    return merged


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Merge streamed OpenAI-compatible chat completion chunks into a single "
            "non-stream response JSON file."
        )
    )
    parser.add_argument("stream_chunks_dir", type=Path,
                        help="Directory containing chunk JSON files")
    parser.add_argument("response_json_path", type=Path,
                        help="Output path for merged response JSON")
    return parser.parse_args()


def merge_chat_completion_chunks(stream_chunks_dir: Path, response_json_path: Path) -> None:
    chunks = _load_chunks(stream_chunks_dir)
    merged_response = _merge_chat_completion_chunks_impl(chunks)

    response_json_path.parent.mkdir(parents=True, exist_ok=True)
    with response_json_path.open("w", encoding="utf-8") as file:
        json.dump(merged_response, file, ensure_ascii=False, indent=2)
        file.write("\n")


def main() -> int:
    args = _parse_args()
    stream_chunks_dir = args.stream_chunks_dir.expanduser().resolve()
    response_json_path = args.response_json_path.expanduser().resolve()

    if not stream_chunks_dir.exists() or not stream_chunks_dir.is_dir():
        raise SystemExit(
            f"stream_chunks_dir is not a directory: {stream_chunks_dir}")

    merge_chat_completion_chunks(stream_chunks_dir, response_json_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

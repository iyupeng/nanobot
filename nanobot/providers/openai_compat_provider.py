"""OpenAI-compatible provider for all non-Anthropic LLM APIs."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import secrets
import string
import uuid
from collections.abc import Awaitable, Callable
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import json_repair
from loguru import logger
from openai import AsyncOpenAI

from nanobot.providers.base import LLMProvider, LLMResponse, ToolCallRequest

if TYPE_CHECKING:
    from nanobot.providers.registry import ProviderSpec

_ALLOWED_MSG_KEYS = frozenset({
    "role", "content", "tool_calls", "tool_call_id", "name",
    "reasoning_content", "extra_content",
})
_ALNUM = string.ascii_letters + string.digits

_STANDARD_TC_KEYS = frozenset({"id", "type", "index", "function"})
_STANDARD_FN_KEYS = frozenset({"name", "arguments"})
_DEFAULT_OPENROUTER_HEADERS = {
    "HTTP-Referer": "https://github.com/HKUDS/nanobot",
    "X-OpenRouter-Title": "nanobot",
    "X-OpenRouter-Categories": "cli-agent,personal-agent",
}
_REPLAY_TAG_RE = re.compile(r"\[replay:([^\]]+)\]")

_current_recorded_session_storage_dir_name: str | None = None
# map session_key and its current iteration index during replay
_current_replayed_iteration_index_mappings = {}
_recorded_sessions: dict[str, list[dict[str, Any]]] = {}


def _short_tool_id() -> str:
    """9-char alphanumeric ID compatible with all providers (incl. Mistral)."""
    return "".join(secrets.choice(_ALNUM) for _ in range(9))


def _get(obj: Any, key: str) -> Any:
    """Get a value from dict or object attribute, returning None if absent."""
    if isinstance(obj, dict):
        return obj.get(key)
    return getattr(obj, key, None)


def _coerce_dict(value: Any) -> dict[str, Any] | None:
    """Try to coerce *value* to a dict; return None if not possible or empty."""
    if value is None:
        return None
    if isinstance(value, dict):
        return value if value else None
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        dumped = model_dump()
        if isinstance(dumped, dict) and dumped:
            return dumped
    return None


def _extract_tc_extras(tc: Any) -> tuple[
    dict[str, Any] | None,
    dict[str, Any] | None,
    dict[str, Any] | None,
]:
    """Extract (extra_content, provider_specific_fields, fn_provider_specific_fields).

    Works for both SDK objects and dicts.  Captures Gemini ``extra_content``
    verbatim and any non-standard keys on the tool-call / function.
    """
    extra_content = _coerce_dict(_get(tc, "extra_content"))

    tc_dict = _coerce_dict(tc)
    prov = None
    fn_prov = None
    if tc_dict is not None:
        leftover = {k: v for k, v in tc_dict.items()
                    if k not in _STANDARD_TC_KEYS and k != "extra_content" and v is not None}
        if leftover:
            prov = leftover
        fn = _coerce_dict(tc_dict.get("function"))
        if fn is not None:
            fn_leftover = {k: v for k, v in fn.items()
                          if k not in _STANDARD_FN_KEYS and v is not None}
            if fn_leftover:
                fn_prov = fn_leftover
    else:
        prov = _coerce_dict(_get(tc, "provider_specific_fields"))
        fn_obj = _get(tc, "function")
        if fn_obj is not None:
            fn_prov = _coerce_dict(_get(fn_obj, "provider_specific_fields"))

    return extra_content, prov, fn_prov


def _uses_openrouter_attribution(spec: "ProviderSpec | None", api_base: str | None) -> bool:
    """Apply Nanobot attribution headers to OpenRouter requests by default."""
    if spec and spec.name == "openrouter":
        return True
    return bool(api_base and "openrouter" in api_base.lower())


def _message_text(message: dict[str, Any] | None) -> str:
    """Extract text content from a chat message payload."""
    if not isinstance(message, dict):
        return ""
    content = message.get("content")
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts: list[str] = []
    for item in content:
        if isinstance(item, str):
            parts.append(item)
            continue
        if not isinstance(item, dict):
            continue
        text = item.get("text")
        if isinstance(text, str):
            parts.append(text)
    return "".join(parts)


def _first_user_message(messages: list[dict[str, Any]]) -> dict[str, Any] | None:
    for message in messages:
        if message.get("role") == "user":
            return message
    return None


def _is_new_conversation(messages: list[dict[str, Any]]) -> bool:
    user_count = sum(1 for message in messages if message.get("role") == "user")
    return bool(messages) and user_count == 1 and messages[-1].get("role") == "user"


def _extract_replay_session_id(messages: list[dict[str, Any]]) -> str | None:
    first_user = _first_user_message(messages)
    if not first_user:
        return None
    first_user_text = _message_text(first_user)
    match = _REPLAY_TAG_RE.search(first_user_text)
    if not match:
        return None
    return match.group(1)


def _timestamp_slug() -> str:
    return datetime.utcnow().isoformat(timespec="milliseconds").replace(":", "-") + "Z"


def _recording_storage_root() -> Path | None:
    storage_directory = os.getenv("RECORD_LLM_INPUT_AND_OUTPUT_STORAGE_DIRECTORY")
    if not storage_directory:
        return None
    return Path(storage_directory).expanduser() / "sessions"


def _recording_enabled() -> bool:
    return os.getenv("RECORD_LLM_INPUT_AND_OUTPUT") == "1"

def _replay_llm_output_first_token_delay_in_millis() -> int:
    default_delay = 3000
    delay_str = os.getenv("REPLAY_LLM_OUTPUT_FIRST_TOKEN_DELAY_IN_MILLIS", str(default_delay))
    try:
        delay = int(delay_str)
        if delay < 0:
            raise ValueError("Delay cannot be negative.")
        return delay
    except ValueError:
        logger.warning(
            "Invalid value for REPLAY_LLM_OUTPUT_FIRST_TOKEN_DELAY_IN_MILLIS: '{}'. Using default of {} ms.",
            delay_str,
            default_delay,
        )
        return default_delay

def _to_jsonable(value: Any) -> Any:
    mapping = _coerce_dict(value)
    if mapping is not None:
        return {str(key): _to_jsonable(val) for key, val in mapping.items()}
    if isinstance(value, list):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_to_jsonable(payload), indent=2), encoding="utf-8")


def _start_recording_iteration(messages: list[dict[str, Any]], kwargs: dict[str, Any]) -> Path | None:
    global _current_recorded_session_storage_dir_name

    if not _recording_enabled():
        logger.warning("Recording LLM input and output is disabled.")
        return None
    sessions_directory = _recording_storage_root()
    if sessions_directory is None:
        logger.error(
            "Storage directory for recording LLM input and output is not set. "
            "Please set RECORD_LLM_INPUT_AND_OUTPUT_STORAGE_DIRECTORY."
        )
        return None

    logger.warning(
        "Recording LLM input and output is enabled. Storage directory: {}",
        sessions_directory.parent,
    )

    if _current_recorded_session_storage_dir_name is None and not _is_new_conversation(messages):
        logger.warning("Recording a session from the middle.")

    if _current_recorded_session_storage_dir_name is None or _is_new_conversation(messages):
        _current_recorded_session_storage_dir_name = f"{_timestamp_slug()}_{uuid.uuid4()}"
        logger.warning(
            "Recording a new session to directory: {}",
            sessions_directory / _current_recorded_session_storage_dir_name,
        )

    iteration_directory = (
        sessions_directory
        / _current_recorded_session_storage_dir_name
        / "iterations"
        / _timestamp_slug()
    )
    logger.warning(
        "Recording LLM input for current iteration in directory: {}",
        iteration_directory,
    )
    _write_json(iteration_directory / "params.json", kwargs)
    return iteration_directory


def _parse_chunk_timestamp(path: Path) -> float | None:
    match = re.match(r"^\d+_(\d{4}-\d{2}-\d{2})T(\d{2}-\d{2}-\d{2}\.\d{3}Z)\.json$", path.name)
    if not match:
        return None
    timestamp_str = f"{match.group(1)}T{match.group(2).replace('-', ':')}"
    try:
        return datetime.strptime(timestamp_str, "%Y-%m-%dT%H:%M:%S.%fZ").timestamp()
    except ValueError:
        return None


def _load_replay_stream_chunks(iteration_directory: Path) -> tuple[list[dict[str, Any]], list[float | None]]:
    stream_chunks_directory = iteration_directory / "stream_chunks"
    if not stream_chunks_directory.exists():
        # logger.error("No iterations found for replay session: {}", iteration_directory.parent.parent.name)
        return [], []

    chunks: list[dict[str, Any]] = []
    timestamps: list[float | None] = []
    for chunk_path in sorted(stream_chunks_directory.iterdir()):
        if not chunk_path.is_file() or chunk_path.suffix != ".json":
            continue
        try:
            chunks.append(json.loads(chunk_path.read_text(encoding="utf-8")))
            timestamps.append(_parse_chunk_timestamp(chunk_path))
        except json.JSONDecodeError:
            logger.exception("Failed to parse chunk file: {}", chunk_path)
            return [], []
    if not chunks:
        logger.warning("No valid chunk files found for iteration directory: {}", iteration_directory)
        return [], []
    if len(chunks) != len(timestamps):
        logger.warning(
            "Mismatch between number of chunk files and timestamps for iteration directory: {}",
            iteration_directory,
        )
        return [], []
    return chunks, timestamps


def _record_stream_chunk(iteration_directory: Path, chunk_index: int, chunk: Any) -> None:
    chunk_id = f"{chunk_index:06d}_{_timestamp_slug()}"
    if chunk_index == 1:
        logger.warning(
            "Recording streaming chunks for current iteration in directory: {}",
            iteration_directory / "stream_chunks",
        )
    _write_json(iteration_directory / "stream_chunks" / f"{chunk_id}.json", chunk)


def _load_replay_response(iteration_directory: Path) -> dict[str, Any] | None:
    response_path = iteration_directory / "response.json"
    if not response_path.exists():
        return None
    try:
        return json.loads(response_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def _load_available_replay_sessions() -> dict[str, list[dict[str, Any]]]:
    sessions_directory = _recording_storage_root()
    if sessions_directory is None:
        logger.error(
            "Replay requested but RECORD_LLM_INPUT_AND_OUTPUT_STORAGE_DIRECTORY is not set."
        )
        return {}
    if not sessions_directory.exists():
        return {}

    available_sessions: dict[str, list[dict[str, Any]]] = {}
    for session_directory in sorted(path for path in sessions_directory.iterdir() if path.is_dir()):
        iterations_directory = session_directory / "iterations"
        if not iterations_directory.exists():
            continue

        session_iterations: list[dict[str, Any]] = []
        for iteration_directory in sorted(path for path in iterations_directory.iterdir() if path.is_dir()):
            replay_response = _load_replay_response(iteration_directory)
            replay_chunks, replay_timestamps = _load_replay_stream_chunks(iteration_directory)
            session_iterations.append({
                "iteration_directory": iteration_directory,
                "response": replay_response,
                "chunks": replay_chunks,
                "timestamps": replay_timestamps,
            })

        if session_iterations:
            available_sessions[session_directory.name] = session_iterations

    return available_sessions


_recorded_sessions = _load_available_replay_sessions()
logger.warning(f"Loaded {len(_recorded_sessions)} recorded sessions ready for replay.")


def _load_replay_iteration_data(messages: list[dict[str, Any]], session_key: str | None) -> dict[str, Any] | None:
    replay_session_id = _extract_replay_session_id(messages)
    if not replay_session_id:
        logger.warning("No replay session ID found in the first user message. Streaming from provider.")
        return None

    if not session_key:
        logger.warning("session_key is not provided, skipping checking replay")
        return None

    logger.warning(f"Current agent session: {session_key}. Target replay session: {replay_session_id}")

    if _is_new_conversation(messages):
        _current_replayed_iteration_index_mappings[session_key] = -1

    session_iterations = _recorded_sessions.get(replay_session_id)
    if not session_iterations:
        logger.warning("Current agent session: {}. No replay iterations found for session: {}", session_key, replay_session_id)
        return None

    iteration_directory = session_iterations[0]["iteration_directory"].parent.parent
    logger.warning("Current agent session: {}. Replaying session from directory: {}", session_key, iteration_directory)

    next_index = _current_replayed_iteration_index_mappings[session_key] + 1
    if next_index >= len(session_iterations):
        logger.warning("Current agent session: {}. No more iterations available to replay for session: {}", session_key, replay_session_id)
        return None
    _current_replayed_iteration_index_mappings[session_key] = next_index
    logger.warning(
        "Current agent session: {}. Replaying iteration ({}/{}) for session.",
        session_key,
        next_index + 1,
        len(session_iterations),
    )
    return session_iterations[next_index]


class OpenAICompatProvider(LLMProvider):
    """Unified provider for all OpenAI-compatible APIs.

    Receives a resolved ``ProviderSpec`` from the caller — no internal
    registry lookups needed.
    """

    def __init__(
        self,
        api_key: str | None = None,
        api_base: str | None = None,
        default_model: str = "gpt-4o",
        extra_headers: dict[str, str] | None = None,
        spec: ProviderSpec | None = None,
    ):
        super().__init__(api_key, api_base)
        self.default_model = default_model
        self.extra_headers = extra_headers or {}
        self._spec = spec

        if api_key and spec and spec.env_key:
            self._setup_env(api_key, api_base)

        effective_base = api_base or (spec.default_api_base if spec else None) or None
        default_headers = {"x-session-affinity": uuid.uuid4().hex}
        if _uses_openrouter_attribution(spec, effective_base):
            default_headers.update(_DEFAULT_OPENROUTER_HEADERS)
        if extra_headers:
            default_headers.update(extra_headers)

        self._client = AsyncOpenAI(
            api_key=api_key or "no-key",
            base_url=effective_base,
            default_headers=default_headers,
        )

    def _setup_env(self, api_key: str, api_base: str | None) -> None:
        """Set environment variables based on provider spec."""
        spec = self._spec
        if not spec or not spec.env_key:
            return
        if spec.is_gateway:
            os.environ[spec.env_key] = api_key
        else:
            os.environ.setdefault(spec.env_key, api_key)
        effective_base = api_base or spec.default_api_base
        for env_name, env_val in spec.env_extras:
            resolved = env_val.replace("{api_key}", api_key).replace("{api_base}", effective_base)
            os.environ.setdefault(env_name, resolved)

    @staticmethod
    def _apply_cache_control(
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]] | None]:
        """Inject cache_control markers for prompt caching."""
        cache_marker = {"type": "ephemeral"}
        new_messages = list(messages)

        def _mark(msg: dict[str, Any]) -> dict[str, Any]:
            content = msg.get("content")
            if isinstance(content, str):
                return {**msg, "content": [
                    {"type": "text", "text": content, "cache_control": cache_marker},
                ]}
            if isinstance(content, list) and content:
                nc = list(content)
                nc[-1] = {**nc[-1], "cache_control": cache_marker}
                return {**msg, "content": nc}
            return msg

        if new_messages and new_messages[0].get("role") == "system":
            new_messages[0] = _mark(new_messages[0])
        if len(new_messages) >= 3:
            new_messages[-2] = _mark(new_messages[-2])

        new_tools = tools
        if tools:
            new_tools = list(tools)
            new_tools[-1] = {**new_tools[-1], "cache_control": cache_marker}
        return new_messages, new_tools

    @staticmethod
    def _normalize_tool_call_id(tool_call_id: Any) -> Any:
        """Normalize to a provider-safe 9-char alphanumeric form."""
        if not isinstance(tool_call_id, str):
            return tool_call_id
        if len(tool_call_id) == 9 and tool_call_id.isalnum():
            return tool_call_id
        return hashlib.sha1(tool_call_id.encode()).hexdigest()[:9]

    def _sanitize_messages(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Strip non-standard keys, normalize tool_call IDs."""
        sanitized = LLMProvider._sanitize_request_messages(messages, _ALLOWED_MSG_KEYS)
        id_map: dict[str, str] = {}

        def map_id(value: Any) -> Any:
            if not isinstance(value, str):
                return value
            return id_map.setdefault(value, self._normalize_tool_call_id(value))

        for clean in sanitized:
            if isinstance(clean.get("tool_calls"), list):
                normalized = []
                for tc in clean["tool_calls"]:
                    if not isinstance(tc, dict):
                        normalized.append(tc)
                        continue
                    tc_clean = dict(tc)
                    tc_clean["id"] = map_id(tc_clean.get("id"))
                    normalized.append(tc_clean)
                clean["tool_calls"] = normalized
            if "tool_call_id" in clean and clean["tool_call_id"]:
                clean["tool_call_id"] = map_id(clean["tool_call_id"])
        return sanitized

    # ------------------------------------------------------------------
    # Build kwargs
    # ------------------------------------------------------------------

    def _build_kwargs(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        model: str | None,
        max_tokens: int,
        temperature: float,
        reasoning_effort: str | None,
        tool_choice: str | dict[str, Any] | None,
    ) -> dict[str, Any]:
        model_name = model or self.default_model
        spec = self._spec

        if spec and spec.supports_prompt_caching:
            messages, tools = self._apply_cache_control(messages, tools)

        if spec and spec.strip_model_prefix:
            model_name = model_name.split("/")[-1]

        kwargs: dict[str, Any] = {
            "model": model_name,
            "messages": self._sanitize_messages(self._sanitize_empty_content(messages)),
            "temperature": temperature,
        }

        if spec and getattr(spec, "supports_max_completion_tokens", False):
            kwargs["max_completion_tokens"] = max(1, max_tokens)
        else:
            kwargs["max_tokens"] = max(1, max_tokens)

        if spec:
            model_lower = model_name.lower()
            for pattern, overrides in spec.model_overrides:
                if pattern in model_lower:
                    kwargs.update(overrides)
                    break

        if reasoning_effort:
            kwargs["reasoning_effort"] = reasoning_effort

        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = tool_choice or "auto"

        return kwargs

    # ------------------------------------------------------------------
    # Response parsing
    # ------------------------------------------------------------------

    @staticmethod
    def _maybe_mapping(value: Any) -> dict[str, Any] | None:
        if isinstance(value, dict):
            return value
        model_dump = getattr(value, "model_dump", None)
        if callable(model_dump):
            dumped = model_dump()
            if isinstance(dumped, dict):
                return dumped
        return None

    @classmethod
    def _extract_text_content(cls, value: Any) -> str | None:
        if value is None:
            return None
        if isinstance(value, str):
            return value
        if isinstance(value, list):
            parts: list[str] = []
            for item in value:
                item_map = cls._maybe_mapping(item)
                if item_map:
                    text = item_map.get("text")
                    if isinstance(text, str):
                        parts.append(text)
                        continue
                text = getattr(item, "text", None)
                if isinstance(text, str):
                    parts.append(text)
                    continue
                if isinstance(item, str):
                    parts.append(item)
            return "".join(parts) or None
        return str(value)

    @classmethod
    def _extract_usage(cls, response: Any) -> dict[str, int]:
        usage_obj = None
        response_map = cls._maybe_mapping(response)
        if response_map is not None:
            usage_obj = response_map.get("usage")
        elif hasattr(response, "usage") and response.usage:
            usage_obj = response.usage

        usage_map = cls._maybe_mapping(usage_obj)
        if usage_map is not None:
            return {
                "prompt_tokens": int(usage_map.get("prompt_tokens") or 0),
                "completion_tokens": int(usage_map.get("completion_tokens") or 0),
                "total_tokens": int(usage_map.get("total_tokens") or 0),
            }

        if usage_obj:
            return {
                "prompt_tokens": getattr(usage_obj, "prompt_tokens", 0) or 0,
                "completion_tokens": getattr(usage_obj, "completion_tokens", 0) or 0,
                "total_tokens": getattr(usage_obj, "total_tokens", 0) or 0,
            }
        return {}

    def _parse(self, response: Any) -> LLMResponse:
        if isinstance(response, str):
            return LLMResponse(content=response, finish_reason="stop")

        response_map = self._maybe_mapping(response)
        if response_map is not None:
            choices = response_map.get("choices") or []
            if not choices:
                content = self._extract_text_content(
                    response_map.get("content") or response_map.get("output_text")
                )
                if content is not None:
                    return LLMResponse(
                        content=content,
                        finish_reason=str(response_map.get("finish_reason") or "stop"),
                        usage=self._extract_usage(response_map),
                    )
                return LLMResponse(content="Error: API returned empty choices.", finish_reason="error")

            choice0 = self._maybe_mapping(choices[0]) or {}
            msg0 = self._maybe_mapping(choice0.get("message")) or {}
            content = self._extract_text_content(msg0.get("content"))
            finish_reason = str(choice0.get("finish_reason") or "stop")

            raw_tool_calls: list[Any] = []
            reasoning_content = msg0.get("reasoning_content")
            for ch in choices:
                ch_map = self._maybe_mapping(ch) or {}
                m = self._maybe_mapping(ch_map.get("message")) or {}
                tool_calls = m.get("tool_calls")
                if isinstance(tool_calls, list) and tool_calls:
                    raw_tool_calls.extend(tool_calls)
                    if ch_map.get("finish_reason") in ("tool_calls", "stop"):
                        finish_reason = str(ch_map["finish_reason"])
                if not content:
                    content = self._extract_text_content(m.get("content"))
                if not reasoning_content:
                    reasoning_content = m.get("reasoning_content")

            parsed_tool_calls = []
            for tc in raw_tool_calls:
                tc_map = self._maybe_mapping(tc) or {}
                fn = self._maybe_mapping(tc_map.get("function")) or {}
                args = fn.get("arguments", {})
                if isinstance(args, str):
                    args = json_repair.loads(args)
                ec, prov, fn_prov = _extract_tc_extras(tc)
                parsed_tool_calls.append(ToolCallRequest(
                    id=_short_tool_id(),
                    name=str(fn.get("name") or ""),
                    arguments=args if isinstance(args, dict) else {},
                    extra_content=ec,
                    provider_specific_fields=prov,
                    function_provider_specific_fields=fn_prov,
                ))

            return LLMResponse(
                content=content,
                tool_calls=parsed_tool_calls,
                finish_reason=finish_reason,
                usage=self._extract_usage(response_map),
                reasoning_content=reasoning_content if isinstance(reasoning_content, str) else None,
            )

        if not response.choices:
            return LLMResponse(content="Error: API returned empty choices.", finish_reason="error")

        choice = response.choices[0]
        msg = choice.message
        content = msg.content
        finish_reason = choice.finish_reason

        raw_tool_calls: list[Any] = []
        for ch in response.choices:
            m = ch.message
            if hasattr(m, "tool_calls") and m.tool_calls:
                raw_tool_calls.extend(m.tool_calls)
                if ch.finish_reason in ("tool_calls", "stop"):
                    finish_reason = ch.finish_reason
            if not content and m.content:
                content = m.content

        tool_calls = []
        for tc in raw_tool_calls:
            args = tc.function.arguments
            if isinstance(args, str):
                args = json_repair.loads(args)
            ec, prov, fn_prov = _extract_tc_extras(tc)
            tool_calls.append(ToolCallRequest(
                id=_short_tool_id(),
                name=tc.function.name,
                arguments=args,
                extra_content=ec,
                provider_specific_fields=prov,
                function_provider_specific_fields=fn_prov,
            ))

        return LLMResponse(
            content=content,
            tool_calls=tool_calls,
            finish_reason=finish_reason or "stop",
            usage=self._extract_usage(response),
            reasoning_content=getattr(msg, "reasoning_content", None) or None,
        )

    @classmethod
    def _parse_chunks(cls, chunks: list[Any]) -> LLMResponse:
        content_parts: list[str] = []
        tc_bufs: dict[int, dict[str, Any]] = {}
        finish_reason = "stop"
        usage: dict[str, int] = {}

        def _accum_tc(tc: Any, idx_hint: int) -> None:
            """Accumulate one streaming tool-call delta into *tc_bufs*."""
            tc_index: int = _get(tc, "index") if _get(tc, "index") is not None else idx_hint
            buf = tc_bufs.setdefault(tc_index, {
                "id": "", "name": "", "arguments": "",
                "extra_content": None, "prov": None, "fn_prov": None,
            })
            tc_id = _get(tc, "id")
            if tc_id:
                buf["id"] = str(tc_id)
            fn = _get(tc, "function")
            if fn is not None:
                fn_name = _get(fn, "name")
                if fn_name:
                    buf["name"] = str(fn_name)
                fn_args = _get(fn, "arguments")
                if fn_args:
                    buf["arguments"] += str(fn_args)
            ec, prov, fn_prov = _extract_tc_extras(tc)
            if ec:
                buf["extra_content"] = ec
            if prov:
                buf["prov"] = prov
            if fn_prov:
                buf["fn_prov"] = fn_prov

        for chunk in chunks:
            if isinstance(chunk, str):
                content_parts.append(chunk)
                continue

            chunk_map = cls._maybe_mapping(chunk)
            if chunk_map is not None:
                choices = chunk_map.get("choices") or []
                if not choices:
                    usage = cls._extract_usage(chunk_map) or usage
                    text = cls._extract_text_content(
                        chunk_map.get("content") or chunk_map.get("output_text")
                    )
                    if text:
                        content_parts.append(text)
                    continue
                choice = cls._maybe_mapping(choices[0]) or {}
                if choice.get("finish_reason"):
                    finish_reason = str(choice["finish_reason"])
                delta = cls._maybe_mapping(choice.get("delta")) or {}
                text = cls._extract_text_content(delta.get("content"))
                if text:
                    content_parts.append(text)
                for idx, tc in enumerate(delta.get("tool_calls") or []):
                    _accum_tc(tc, idx)
                usage = cls._extract_usage(chunk_map) or usage
                continue

            if not chunk.choices:
                usage = cls._extract_usage(chunk) or usage
                continue
            choice = chunk.choices[0]
            if choice.finish_reason:
                finish_reason = choice.finish_reason
            delta = choice.delta
            if delta and delta.content:
                content_parts.append(delta.content)
            for tc in (delta.tool_calls or []) if delta else []:
                _accum_tc(tc, getattr(tc, "index", 0))

        return LLMResponse(
            content="".join(content_parts) or None,
            tool_calls=[
                ToolCallRequest(
                    id=b["id"] or _short_tool_id(),
                    name=b["name"],
                    arguments=json_repair.loads(b["arguments"]) if b["arguments"] else {},
                    extra_content=b.get("extra_content"),
                    provider_specific_fields=b.get("prov"),
                    function_provider_specific_fields=b.get("fn_prov"),
                )
                for b in tc_bufs.values()
            ],
            finish_reason=finish_reason,
            usage=usage,
        )

    @staticmethod
    def _handle_error(e: Exception) -> LLMResponse:
        body = getattr(e, "doc", None) or getattr(getattr(e, "response", None), "text", None)
        msg = f"Error: {body.strip()[:500]}" if body and body.strip() else f"Error calling LLM: {e}"
        return LLMResponse(content=msg, finish_reason="error")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        reasoning_effort: str | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        session_key: str | None = None,
    ) -> LLMResponse:
        kwargs = self._build_kwargs(
            messages, tools, model, max_tokens, temperature,
            reasoning_effort, tool_choice,
        )

        replay_iteration_data = _load_replay_iteration_data(messages, session_key)
        if replay_iteration_data is not None:
            replay_iteration_directory = replay_iteration_data["iteration_directory"]
            replay_response = replay_iteration_data.get("response")
            if replay_response is not None:
                logger.warning("Replaying non-stream response from directory: {}", replay_iteration_directory)
                # delay before first token to simulate LLM thinking time, configurable via env var
                first_token_delay_ms = _replay_llm_output_first_token_delay_in_millis()
                if first_token_delay_ms > 0:
                    logger.warning("Delaying first token by {} ms to simulate LLM thinking time.", first_token_delay_ms)
                    await asyncio.sleep(first_token_delay_ms / 1000.0)
                return self._parse(replay_response)
            replay_chunks = replay_iteration_data.get("chunks") or []
            if replay_chunks:
                logger.warning(
                    "Replaying {} streamed chunks as a non-stream response.",
                    len(replay_chunks),
                )
                # delay before first token to simulate LLM thinking time, configurable via env var
                first_token_delay_ms = _replay_llm_output_first_token_delay_in_millis()
                if first_token_delay_ms > 0:
                    logger.warning("Delaying first token by {} ms to simulate LLM thinking time.", first_token_delay_ms)
                    await asyncio.sleep(first_token_delay_ms / 1000.0)
                return self._parse_chunks(replay_chunks)

        iteration_directory = _start_recording_iteration(messages, kwargs)
        try:
            logger.warning("Requesting LLM provider...(non-stream)")
            response = await self._client.chat.completions.create(**kwargs)
            if iteration_directory is not None:
                _write_json(iteration_directory / "response.json", response)
            return self._parse(response)
        except Exception as e:
            return self._handle_error(e)

    async def chat_stream(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        reasoning_effort: str | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        on_content_delta: Callable[[str], Awaitable[None]] | None = None,
        session_key: str | None = None,
    ) -> LLMResponse:
        kwargs = self._build_kwargs(
            messages, tools, model, max_tokens, temperature,
            reasoning_effort, tool_choice,
        )
        kwargs["stream"] = True
        kwargs["stream_options"] = {"include_usage": True}

        replay_iteration_data = _load_replay_iteration_data(messages, session_key)
        if replay_iteration_data is not None:
            replay_iteration_directory = replay_iteration_data["iteration_directory"]
            replay_chunks = replay_iteration_data.get("chunks") or []
            replay_timestamps = replay_iteration_data.get("timestamps") or []
            if replay_chunks:
                logger.warning("Streaming {} chunks from replay.", len(replay_chunks))
                # delay before first token to simulate LLM thinking time, configurable via env var
                first_token_delay_ms = _replay_llm_output_first_token_delay_in_millis()
                if first_token_delay_ms > 0:
                    logger.warning("Delaying first token by {} ms to simulate LLM thinking time.", first_token_delay_ms)
                    await asyncio.sleep(first_token_delay_ms / 1000.0)
                previous_timestamp: float | None = None
                for chunk, timestamp in zip(replay_chunks, replay_timestamps, strict=False):
                    if previous_timestamp is not None and timestamp is not None:
                        await asyncio.sleep(max(0.0, timestamp - previous_timestamp))
                    previous_timestamp = timestamp if timestamp is not None else previous_timestamp
                    if on_content_delta:
                        delta = self._extract_text_content(
                            ((chunk.get("choices") or [{}])[0].get("delta") or {}).get("content")
                        )
                        if delta:
                            await on_content_delta(delta)
                return self._parse_chunks(replay_chunks)

            replay_response = replay_iteration_data.get("response")
            if replay_response is not None:
                logger.warning(
                    "No replay stream chunks found; replaying stored non-stream response from {}",
                    replay_iteration_directory,
                )
                # delay before first token to simulate LLM thinking time, configurable via env var
                first_token_delay_ms = _replay_llm_output_first_token_delay_in_millis()
                if first_token_delay_ms > 0:
                    logger.warning("Delaying first token by {} ms to simulate LLM thinking time.", first_token_delay_ms)
                    await asyncio.sleep(first_token_delay_ms / 1000.0)
                replayed = self._parse(replay_response)
                if on_content_delta and replayed.content:
                    await on_content_delta(replayed.content)
                return replayed

        iteration_directory = _start_recording_iteration(messages, kwargs)
        try:
            logger.warning("Requesting LLM provider...(stream)")
            stream = await self._client.chat.completions.create(**kwargs)
            chunks: list[Any] = []
            chunk_index = 0
            async for chunk in stream:
                chunks.append(chunk)
                chunk_index += 1
                if iteration_directory is not None:
                    _record_stream_chunk(iteration_directory, chunk_index, chunk)
                if on_content_delta and chunk.choices:
                    text = getattr(chunk.choices[0].delta, "content", None)
                    if text:
                        await on_content_delta(text)
            return self._parse_chunks(chunks)
        except Exception as e:
            return self._handle_error(e)

    def get_default_model(self) -> str:
        return self.default_model

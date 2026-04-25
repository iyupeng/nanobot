"""Agent loop: the core processing engine."""

from __future__ import annotations

import asyncio
import json
import os
import time
from contextlib import AsyncExitStack, nullcontext
from pathlib import Path
from typing import TYPE_CHECKING, Any, Awaitable, Callable

from loguru import logger
from nanobot.utils.profiler import ProfilerTrace
from nanobot.agent.context import ContextBuilder
from nanobot.agent.hook import AgentHook, AgentHookContext, CompositeHook
from nanobot.agent.memory import MemoryConsolidator
from nanobot.agent.runner import AgentRunSpec, AgentRunner
from nanobot.agent.subagent import SubagentManager
from nanobot.agent.tools.cron import CronTool
from nanobot.agent.skills import BUILTIN_SKILLS_DIR
from nanobot.agent.tools.filesystem import EditFileTool, ListDirTool, ReadFileTool, WriteFileTool
from nanobot.agent.tools.message import MessageTool
from nanobot.agent.tools.registry import ToolRegistry
from nanobot.agent.tools.shell import ExecTool
from nanobot.agent.tools.spawn import SpawnTool
from nanobot.agent.tools.web import WebFetchTool, WebSearchTool
from nanobot.bus.events import InboundMessage, OutboundMessage
from nanobot.command import CommandContext, CommandRouter, register_builtin_commands
from nanobot.bus.queue import MessageBus
from nanobot.providers.base import LLMProvider
from nanobot.session.manager import Session, SessionManager

if TYPE_CHECKING:
    from nanobot.config.schema import ChannelsConfig, ExecToolConfig, WebSearchConfig
    from nanobot.cron.service import CronService


class _LoopHook(AgentHook):
    def __init__(self, agent_loop: "AgentLoop", on_progress: Callable[..., Awaitable[None]] | None = None, on_stream: Callable[[str], Awaitable[None]] | None = None, on_stream_end: Callable[..., Awaitable[None]] | None = None, *, channel: str = "cli", chat_id: str = "direct", message_id: str | None = None) -> None:
        self._loop = agent_loop
        self._on_progress = on_progress
        self._on_stream = on_stream
        self._on_stream_end = on_stream_end
        self._channel = channel
        self._chat_id = chat_id
        self._message_id = message_id
        self._stream_buf = ""

    def wants_streaming(self) -> bool:
        return self._on_stream is not None

    async def on_stream(self, context: AgentHookContext, delta: str) -> None:
        from nanobot.utils.helpers import strip_think
        prev_clean = strip_think(self._stream_buf)
        self._stream_buf += delta
        new_clean = strip_think(self._stream_buf)
        incremental = new_clean[len(prev_clean):]
        if incremental and self._on_stream:
            await self._on_stream(incremental)

    async def on_stream_end(self, context: AgentHookContext, *, resuming: bool) -> None:
        if self._on_stream_end:
            await self._on_stream_end(resuming=resuming)
        self._stream_buf = ""

    async def before_execute_tools(self, context: AgentHookContext) -> None:
        if self._on_progress:
            if not self._on_stream:
                thought = self._loop._strip_think(context.response.content if context.response else None)
                if thought:
                    await self._on_progress(thought)
            tool_hint = self._loop._strip_think(self._loop._tool_hint(context.tool_calls))
            await self._on_progress(tool_hint, tool_hint=True)
        for tc in context.tool_calls:
            args_str = json.dumps(tc.arguments, ensure_ascii=False)
            logger.info("Tool call: {}({})", tc.name, args_str[:200])
        self._loop._set_tool_context(self._channel, self._chat_id, self._message_id)

    def finalize_content(self, context: AgentHookContext, content: str | None) -> str | None:
        return self._loop._strip_think(content)


class _LoopHookChain(AgentHook):
    __slots__ = ("_primary", "_extras")

    def __init__(self, primary: AgentHook, extra_hooks: list[AgentHook]) -> None:
        self._primary = primary
        self._extras = CompositeHook(extra_hooks)

    def wants_streaming(self) -> bool:
        return self._primary.wants_streaming() or self._extras.wants_streaming()

    async def before_iteration(self, context: AgentHookContext) -> None:
        await self._primary.before_iteration(context)
        await self._extras.before_iteration(context)

    async def on_stream(self, context: AgentHookContext, delta: str) -> None:
        await self._primary.on_stream(context, delta)
        await self._extras.on_stream(context, delta)

    async def on_stream_end(self, context: AgentHookContext, *, resuming: bool) -> None:
        await self._primary.on_stream_end(context, resuming=resuming)
        await self._extras.on_stream_end(context, resuming=resuming)

    async def before_execute_tools(self, context: AgentHookContext) -> None:
        await self._primary.before_execute_tools(context)
        await self._extras.before_execute_tools(context)

    async def after_iteration(self, context: AgentHookContext) -> None:
        await self._primary.after_iteration(context)
        await self._extras.after_iteration(context)

    def finalize_content(self, context: AgentHookContext, content: str | None) -> str | None:
        content = self._primary.finalize_content(context, content)
        return self._extras.finalize_content(context, content)


class AgentLoop:
    _TOOL_RESULT_MAX_CHARS = 16_000

    def __init__(self, bus: MessageBus, provider: LLMProvider, workspace: Path, model: str | None = None, max_iterations: int = 40, context_window_tokens: int = 65_536, web_search_config: WebSearchConfig | None = None, web_proxy: str | None = None, exec_config: ExecToolConfig | None = None, cron_service: CronService | None = None, restrict_to_workspace: bool = False, session_manager: SessionManager | None = None, mcp_servers: dict | None = None, channels_config: ChannelsConfig | None = None, timezone: str | None = None, hooks: list[AgentHook] | None = None):
        from nanobot.config.schema import ExecToolConfig, WebSearchConfig
        self.bus = bus
        self.channels_config = channels_config
        self.provider = provider
        self.workspace = workspace
        self.model = model or provider.get_default_model()
        self.max_iterations = max_iterations
        self.context_window_tokens = context_window_tokens
        self.web_search_config = web_search_config or WebSearchConfig()
        self.web_proxy = web_proxy
        self.exec_config = exec_config or ExecToolConfig()
        self.cron_service = cron_service
        self.restrict_to_workspace = restrict_to_workspace
        self._start_time = time.time()
        self._last_usage: dict[str, int] = {}
        self._extra_hooks: list[AgentHook] = hooks or []
        self._profiler = ProfilerTrace()
        self.context = ContextBuilder(workspace, timezone=timezone)
        self.sessions = session_manager or SessionManager(workspace)
        self.tools = ToolRegistry()
        self.runner = AgentRunner(provider)
        self.subagents = SubagentManager(provider=provider, workspace=workspace, bus=bus, model=self.model, web_search_config=self.web_search_config, web_proxy=web_proxy, exec_config=self.exec_config, restrict_to_workspace=restrict_to_workspace)
        self._running = False
        self._mcp_servers = mcp_servers or {}
        self._mcp_stack: AsyncExitStack | None = None
        self._mcp_connected = False
        self._mcp_connecting = False
        self._active_tasks: dict[str, list[asyncio.Task]] = {}
        self._background_tasks: list[asyncio.Task] = []
        self._session_locks: dict[str, asyncio.Lock] = {}
        _max = int(os.environ.get("NANOBOT_MAX_CONCURRENT_REQUESTS", "3"))
        self._concurrency_gate: asyncio.Semaphore | None = (asyncio.Semaphore(_max) if _max > 0 else None)
        self.memory_consolidator = MemoryConsolidator(workspace=workspace, provider=provider, model=self.model, sessions=self.sessions, context_window_tokens=context_window_tokens, build_messages=self.context.build_messages, get_tool_definitions=self.tools.get_definitions, max_completion_tokens=provider.generation.max_tokens)
        self._register_default_tools()
        self.commands = CommandRouter()
        register_builtin_commands(self.commands)

    def _register_default_tools(self) -> None:
        allowed_dir = self.workspace if self.restrict_to_workspace else None
        extra_read = [BUILTIN_SKILLS_DIR] if allowed_dir else None
        self.tools.register(ReadFileTool(workspace=self.workspace, allowed_dir=allowed_dir, extra_allowed_dirs=extra_read))
        for cls in (WriteFileTool, EditFileTool, ListDirTool):
            self.tools.register(cls(workspace=self.workspace, allowed_dir=allowed_dir))
        if self.exec_config.enable:
            self.tools.register(ExecTool(working_dir=str(self.workspace), timeout=self.exec_config.timeout, restrict_to_workspace=self.restrict_to_workspace, path_append=self.exec_config.path_append))
        self.tools.register(WebSearchTool(config=self.web_search_config, proxy=self.web_proxy))
        self.tools.register(WebFetchTool(proxy=self.web_proxy))
        self.tools.register(MessageTool(send_callback=self.bus.publish_outbound))
        self.tools.register(SpawnTool(manager=self.subagents))
        if self.cron_service:
            self.tools.register(CronTool(self.cron_service, default_timezone=self.context.timezone or "UTC"))

    async def _connect_mcp(self) -> None:
        if self._mcp_connected or self._mcp_connecting or not self._mcp_servers:
            return
        self._mcp_connecting = True
        from nanobot.agent.tools.mcp import connect_mcp_servers
        try:
            self._mcp_stack = AsyncExitStack()
            await self._mcp_stack.__aenter__()
            await connect_mcp_servers(self._mcp_servers, self.tools, self._mcp_stack)
            self._mcp_connected = True
        except BaseException as e:
            logger.error("Failed to connect MCP servers (will retry next message): {}", e)
            if self._mcp_stack:
                try:
                    await self._mcp_stack.aclose()
                except Exception:
                    pass
                self._mcp_stack = None
        finally:
            self._mcp_connecting = False

    def _set_tool_context(self, channel: str, chat_id: str, message_id: str | None = None) -> None:
        for name in ("message", "spawn", "cron"):
            if tool := self.tools.get(name):
                if hasattr(tool, "set_context"):
                    tool.set_context(channel, chat_id, *([message_id] if name == "message" else []))

    @staticmethod
    def _strip_think(text: str | None) -> str | None:
        if not text:
            return None
        from nanobot.utils.helpers import strip_think
        return strip_think(text) or None

    @staticmethod
    def _tool_hint(tool_calls: list) -> str:
        def _fmt(tc):
            args = (tc.arguments[0] if isinstance(tc.arguments, list) else tc.arguments) or {}
            val = next(iter(args.values()), None) if isinstance(args, dict) else None
            if not isinstance(val, str):
                return tc.name
            return f'{tc.name}("{val[:40]}…")' if len(val) > 40 else f'{tc.name}("{val}")'
        return ", ".join(_fmt(tc) for tc in tool_calls)

    async def _run_agent_loop(self, initial_messages: list[dict], on_progress: Callable[..., Awaitable[None]] | None = None, on_stream: Callable[[str], Awaitable[None]] | None = None, on_stream_end: Callable[..., Awaitable[None]] | None = None, *, channel: str = "cli", chat_id: str = "direct", message_id: str | None = None, session_key: str | None = None) -> tuple[str | None, list[str], list[dict]]:
        loop_hook = _LoopHook(self, on_progress=on_progress, on_stream=on_stream, on_stream_end=on_stream_end, channel=channel, chat_id=chat_id, message_id=message_id)
        hook: AgentHook = _LoopHookChain(loop_hook, self._extra_hooks) if self._extra_hooks else loop_hook
        result = await self.runner.run(AgentRunSpec(initial_messages=initial_messages, tools=self.tools, model=self.model, max_iterations=self.max_iterations, hook=hook, error_message="Sorry, I encountered an error calling the AI model.", concurrent_tools=True, profiler=self._profiler), session_key=session_key)
        self._last_usage = result.usage
        if result.stop_reason == "max_iterations":
            logger.warning("Max iterations ({}) reached", self.max_iterations)
        elif result.stop_reason == "error":
            logger.error("LLM returned error: {}", (result.final_content or "")[:200])
        return result.final_content, result.tools_used, result.messages

    async def run(self) -> None:
        self._running = True
        await self._connect_mcp()
        logger.info("Agent loop started")
        while self._running:
            try:
                msg = await asyncio.wait_for(self.bus.consume_inbound(), timeout=1.0)
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                if not self._running or asyncio.current_task().cancelling():
                    raise
                continue
            except Exception as e:
                logger.warning("Error consuming inbound message: {}, continuing...", e)
                continue
            raw = msg.content.strip()
            if self.commands.is_priority(raw):
                ctx = CommandContext(msg=msg, session=None, key=msg.session_key, raw=raw, loop=self)
                result = await self.commands.dispatch_priority(ctx)
                if result:
                    await self.bus.publish_outbound(result)
                continue
            task = asyncio.create_task(self._dispatch(msg))
            self._active_tasks.setdefault(msg.session_key, []).append(task)
            task.add_done_callback(lambda t, k=msg.session_key: self._active_tasks.get(k, []) and self._active_tasks[k].remove(t) if t in self._active_tasks.get(k, []) else None)

    async def _dispatch(self, msg: InboundMessage) -> None:
        lock = self._session_locks.setdefault(msg.session_key, asyncio.Lock())
        gate = self._concurrency_gate or nullcontext()
        async with lock, gate:
            try:
                on_stream = on_stream_end = None
                if msg.metadata.get("_wants_stream"):
                    stream_base_id = f"{msg.session_key}:{time.time_ns()}"
                    stream_segment = 0
                    def _current_stream_id() -> str:
                        return f"{stream_base_id}:{stream_segment}"
                    async def on_stream(delta: str) -> None:
                        meta = dict(msg.metadata or {})
                        meta["_stream_delta"] = True
                        meta["_stream_id"] = _current_stream_id()
                        await self.bus.publish_outbound(OutboundMessage(channel=msg.channel, chat_id=msg.chat_id, content=delta, metadata=meta))
                    async def on_stream_end(*, resuming: bool = False) -> None:
                        nonlocal stream_segment
                        meta = dict(msg.metadata or {})
                        meta["_stream_end"] = True
                        meta["_stream_id"] = _current_stream_id()
                        meta["_stream_resuming"] = resuming
                        await self.bus.publish_outbound(OutboundMessage(channel=msg.channel, chat_id=msg.chat_id, content="", metadata=meta))
                        if resuming:
                            stream_segment += 1
                response = await self._process_message(msg, on_stream=on_stream, on_stream_end=on_stream_end)
                if response:
                    await self.bus.publish_outbound(response)
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.exception("Error processing message")
                await self.bus.publish_outbound(OutboundMessage(channel=msg.channel, chat_id=msg.chat_id, content=f"Sorry, I encountered an error: {e}"))

    def _schedule_background(self, coro) -> None:
        task = asyncio.create_task(coro)
        self._background_tasks.append(task)
        task.add_done_callback(self._background_tasks.remove)

    async def _process_message(self, msg: InboundMessage, session_key: str | None = None, on_progress: Callable[[str], Awaitable[None]] | None = None, on_stream: Callable[[str], Awaitable[None]] | None = None, on_stream_end: Callable[..., Awaitable[None]] | None = None) -> OutboundMessage | None:
        prof = self._profiler
        if msg.channel == "system":
            prof.push("prepare_before_agent_loop")
            channel, chat_id = (msg.chat_id.split(":", 1) if ":" in msg.chat_id else ("cli", msg.chat_id))
            logger.info("Processing system message from {}", msg.sender_id)
            key = f"{channel}:{chat_id}"
            session = self.sessions.get_or_create(key)
            await self.memory_consolidator.maybe_consolidate_by_tokens(session)
            self._set_tool_context(channel, chat_id, msg.metadata.get("message_id"))
            history = session.get_history(max_messages=0)
            current_role = "assistant" if msg.sender_id == "subagent" else "user"
            messages = self.context.build_messages(history=history, current_message=msg.content, channel=channel, chat_id=chat_id, current_role=current_role)
            prof.pop()
            prof.push("agent_loop")
            final_content, _, all_msgs = await self._run_agent_loop(messages, channel=channel, chat_id=chat_id, message_id=msg.metadata.get("message_id"))
            prof.pop()
            prof.push("session_end")
            self._save_turn(session, all_msgs, 1 + len(history))
            self.sessions.save(session)
            self._schedule_background(self.memory_consolidator.maybe_consolidate_by_tokens(session))
            outbound = OutboundMessage(channel=channel, chat_id=chat_id, content=final_content or "Background task completed.")
            prof.pop()
            return outbound

        prof.push("prepare_before_agent_loop")
        preview = msg.content[:80] + "..." if len(msg.content) > 80 else msg.content
        logger.info("Processing message from {}:{}: {}", msg.channel, msg.sender_id, preview)
        key = session_key or msg.session_key
        session = self.sessions.get_or_create(key)
        raw = msg.content.strip()
        ctx = CommandContext(msg=msg, session=session, key=key, raw=raw, loop=self)
        result = await self.commands.dispatch(ctx)
        if result:
            return result
        await self.memory_consolidator.maybe_consolidate_by_tokens(session)
        self._set_tool_context(msg.channel, msg.chat_id, msg.metadata.get("message_id"))
        if message_tool := self.tools.get("message"):
            if isinstance(message_tool, MessageTool):
                message_tool.start_turn()
        history = session.get_history(max_messages=0)
        initial_messages = self.context.build_messages(history=history, current_message=msg.content, media=msg.media if msg.media else None, channel=msg.channel, chat_id=msg.chat_id)
        prof.pop()

        async def _bus_progress(content: str, *, tool_hint: bool = False) -> None:
            meta = dict(msg.metadata or {})
            meta["_progress"] = True
            meta["_tool_hint"] = tool_hint
            await self.bus.publish_outbound(OutboundMessage(channel=msg.channel, chat_id=msg.chat_id, content=content, metadata=meta))

        prof.push("agent_loop")
        final_content, _, all_msgs = await self._run_agent_loop(initial_messages, on_progress=on_progress or _bus_progress, on_stream=on_stream, on_stream_end=on_stream_end, channel=msg.channel, chat_id=msg.chat_id, message_id=msg.metadata.get("message_id"), session_key=key)
        prof.pop()

        prof.push("session_end")
        if final_content is None:
            final_content = "I've completed processing but have no response to give."
        self._save_turn(session, all_msgs, 1 + len(history))
        self.sessions.save(session)
        self._schedule_background(self.memory_consolidator.maybe_consolidate_by_tokens(session))
        if (mt := self.tools.get("message")) and isinstance(mt, MessageTool) and mt._sent_in_turn:
            prof.pop()
            return None
        preview = final_content[:120] + "..." if len(final_content) > 120 else final_content
        logger.info("Response to {}:{}: {}", msg.channel, msg.sender_id, preview)
        meta = dict(msg.metadata or {})
        if on_stream is not None:
            meta["_streamed"] = True
        outbound = OutboundMessage(channel=msg.channel, chat_id=msg.chat_id, content=final_content, metadata=meta)
        prof.pop()
        return outbound

    @staticmethod
    def _image_placeholder(block: dict[str, Any]) -> dict[str, str]:
        path = (block.get("_meta") or {}).get("path", "")
        return {"type": "text", "text": f"[image: {path}]" if path else "[image]"}

    def _sanitize_persisted_blocks(self, content: list[dict[str, Any]], *, truncate_text: bool = False, drop_runtime: bool = False) -> list[dict[str, Any]]:
        filtered: list[dict[str, Any]] = []
        for block in content:
            if not isinstance(block, dict):
                filtered.append(block)
                continue
            if drop_runtime and block.get("type") == "text" and isinstance(block.get("text"), str) and block["text"].startswith(ContextBuilder._RUNTIME_CONTEXT_TAG):
                continue
            if block.get("type") == "image_url" and block.get("image_url", {}).get("url", "").startswith("data:image/"):
                filtered.append(self._image_placeholder(block))
                continue
            if block.get("type") == "input_image":
                filtered.append(self._image_placeholder(block))
                continue
            if truncate_text and block.get("type") == "text":
                text = block.get("text")
                if isinstance(text, str) and len(text) > self._TOOL_RESULT_MAX_CHARS:
                    trimmed = text[:self._TOOL_RESULT_MAX_CHARS]
                    filtered.append({**block, "text": trimmed + "\n...[truncated]"})
                    continue
            filtered.append(block)
        return filtered

    def _save_turn(self, session: Session, messages: list[dict], start_index: int) -> None:
        from datetime import datetime
        for msg in messages[start_index:]:
            entry = dict(msg)
            if isinstance(entry.get("content"), list):
                filtered = self._sanitize_persisted_blocks(entry["content"], truncate_text=entry.get("role") == "tool", drop_runtime=entry.get("role") == "system")
                if not filtered:
                    continue
                entry["content"] = filtered
            entry.setdefault("timestamp", datetime.now().isoformat())
            session.messages.append(entry)
        session.updated_at = datetime.now()

    async def process_direct(self, content: str, session_key: str = "cli:direct", channel: str = "cli", chat_id: str = "direct", on_progress: Callable[[str], Awaitable[None]] | None = None, on_stream: Callable[[str], Awaitable[None]] | None = None, on_stream_end: Callable[..., Awaitable[None]] | None = None) -> OutboundMessage | None:
        prof = self._profiler
        prof.push("run", category="mainloop")
        prof.push("connect_mcp")
        await self._connect_mcp()
        prof.pop()
        msg = InboundMessage(channel=channel, sender_id="user", chat_id=chat_id, content=content)
        response = await self._process_message(msg, session_key=session_key, on_progress=on_progress, on_stream=on_stream, on_stream_end=on_stream_end)
        prof.pop()
        return response

    async def close_mcp(self) -> None:
        pass

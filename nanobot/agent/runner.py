"""Shared execution loop for tool-using agents."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

from nanobot.utils.profiler import ProfilerTrace
from nanobot.agent.hook import AgentHook, AgentHookContext
from nanobot.agent.tools.registry import ToolRegistry
from nanobot.providers.base import LLMProvider, ToolCallRequest
from nanobot.utils.helpers import build_assistant_message

_DEFAULT_MAX_ITERATIONS_MESSAGE = (
    "I reached the maximum number of tool call iterations ({max_iterations}) "
    "without completing the task. You can try breaking the task into smaller steps."
)
_DEFAULT_ERROR_MESSAGE = "Sorry, I encountered an error calling the AI model."


@dataclass(slots=True)
class AgentRunSpec:
    initial_messages: list[dict[str, Any]]
    tools: ToolRegistry
    model: str
    max_iterations: int
    temperature: float | None = None
    max_tokens: float | None = None
    reasoning_effort: str | None = None
    hook: AgentHook | None = None
    error_message: str | None = _DEFAULT_ERROR_MESSAGE
    max_iterations_message: str | None = None
    concurrent_tools: bool = False
    fail_on_tool_error: bool = False
    profiler: ProfilerTrace = field(default_factory=ProfilerTrace)


@dataclass(slots=True)
class AgentRunResult:
    final_content: str | None
    messages: list[dict[str, Any]]
    tools_used: list[str] = field(default_factory=list)
    usage: dict[str, int] = field(default_factory=dict)
    stop_reason: str = "completed"
    error: str | None = None
    tool_events: list[dict[str, str]] = field(default_factory=list)
    profiler: ProfilerTrace | None = None


class AgentRunner:
    def __init__(self, provider: LLMProvider):
        self.provider = provider

    async def run(self, spec: AgentRunSpec, session_key: str | None = None) -> AgentRunResult:
        hook = spec.hook or AgentHook()
        prof = spec.profiler
        messages = list(spec.initial_messages)
        final_content: str | None = None
        tools_used: list[str] = []
        usage = {"prompt_tokens": 0, "completion_tokens": 0}
        error: str | None = None
        stop_reason = "completed"
        tool_events: list[dict[str, str]] = []

        for iteration in range(spec.max_iterations):
            prof.push("iteration_start", category="mainloop")
            context = AgentHookContext(iteration=iteration, messages=messages, profiler=prof)
            await hook.before_iteration(context)

            kwargs: dict[str, Any] = {"messages": messages, "tools": spec.tools.get_definitions(), "model": spec.model, "session_key": session_key}
            if spec.temperature is not None:
                kwargs["temperature"] = spec.temperature
            if spec.max_tokens is not None:
                kwargs["max_tokens"] = spec.max_tokens
            if spec.reasoning_effort is not None:
                kwargs["reasoning_effort"] = spec.reasoning_effort

            prof.push("llm_call", category="model")
            if hook.wants_streaming():
                async def _stream(delta: str) -> None:
                    await hook.on_stream(context, delta)
                response = await self.provider.chat_stream_with_retry(**kwargs, on_content_delta=_stream)
            else:
                response = await self.provider.chat_with_retry(**kwargs)
            prof.pop(args={
                "model": spec.model,
                "messages": messages,
                "response": response,
            })

            raw_usage = response.usage or {}
            usage = {"prompt_tokens": int(raw_usage.get("prompt_tokens", 0) or 0), "completion_tokens": int(raw_usage.get("completion_tokens", 0) or 0)}
            context.response = response
            context.usage = usage
            context.tool_calls = list(response.tool_calls)

            if response.has_tool_calls:
                prof.push("tool_calls")
                if hook.wants_streaming():
                    await hook.on_stream_end(context, resuming=True)
                messages.append(build_assistant_message(response.content or "", tool_calls=[tc.to_openai_tool_call() for tc in response.tool_calls], reasoning_content=response.reasoning_content, thinking_blocks=response.thinking_blocks))
                tools_used.extend(tc.name for tc in response.tool_calls)
                await hook.before_execute_tools(context)
                results, new_events, fatal_error = await self._execute_tools(spec, iteration, response.tool_calls)
                prof.pop(args={"tool_calls": response.tool_calls})
                tool_events.extend(new_events)
                context.tool_results = list(results)
                context.tool_events = list(new_events)
                if fatal_error is not None:
                    error = f"Error: {type(fatal_error).__name__}: {fatal_error}"
                    stop_reason = "tool_error"
                    context.error = error
                    context.stop_reason = stop_reason
                    await hook.after_iteration(context)
                    prof.pop(args={"finish_reason": response.finish_reason, "stop_reason": stop_reason})
                    break
                for tool_call, result in zip(response.tool_calls, results):
                    messages.append({"role": "tool", "tool_call_id": tool_call.id, "name": tool_call.name, "content": result})
                await hook.after_iteration(context)
                prof.pop(args={"finish_reason": response.finish_reason, "stop_reason": "tool_calls"})
                continue

            if hook.wants_streaming():
                await hook.on_stream_end(context, resuming=False)
            clean = hook.finalize_content(context, response.content)
            if response.finish_reason == "error":
                final_content = clean or spec.error_message or _DEFAULT_ERROR_MESSAGE
                stop_reason = "error"
                error = final_content
                context.final_content = final_content
                context.error = error
                context.stop_reason = stop_reason
                await hook.after_iteration(context)
                prof.pop(args={"finish_reason": response.finish_reason, "stop_reason": stop_reason})
                break

            messages.append(build_assistant_message(clean, reasoning_content=response.reasoning_content, thinking_blocks=response.thinking_blocks))
            final_content = clean
            context.final_content = final_content
            context.stop_reason = stop_reason
            await hook.after_iteration(context)
            prof.pop(args={"finish_reason": response.finish_reason, "stop_reason": stop_reason})
            break
        else:
            stop_reason = "max_iterations"
            final_content = (spec.max_iterations_message or _DEFAULT_MAX_ITERATIONS_MESSAGE).format(max_iterations=spec.max_iterations)

        return AgentRunResult(final_content=final_content, messages=messages, tools_used=tools_used, usage=usage, stop_reason=stop_reason, error=error, tool_events=tool_events, profiler=prof)

    async def _execute_tools(self, spec: AgentRunSpec, iteration: int, tool_calls: list[ToolCallRequest]) -> tuple[list[Any], list[dict[str, str]], BaseException | None]:
        prof = spec.profiler
        prof.push("execute_tools", category="tool")
        if spec.concurrent_tools:
            tool_results = await asyncio.gather(*(self._run_tool(spec, iteration, tool_call) for tool_call in tool_calls))
        else:
            tool_results = [await self._run_tool(spec, iteration, tool_call) for tool_call in tool_calls]
        results: list[Any] = []
        events: list[dict[str, str]] = []
        fatal_error: BaseException | None = None
        for result, event, error in tool_results:
            results.append(result)
            events.append(event)
            if error is not None and fatal_error is None:
                fatal_error = error
        prof.pop(args={"tool_count": len(tool_calls), "concurrent": spec.concurrent_tools})
        return results, events, fatal_error

    async def _run_tool(self, spec: AgentRunSpec, iteration: int, tool_call: ToolCallRequest) -> tuple[Any, dict[str, str], BaseException | None]:
        prof = spec.profiler
        prof.push(tool_call.name, category="tool")
        try:
            result = await spec.tools.execute(tool_call.name, tool_call.arguments)
        except asyncio.CancelledError:
            raise
        except BaseException as exc:
            event = {"name": tool_call.name, "status": "error", "detail": str(exc)}
            prof.pop(status="error", args={"tool_call": tool_call, "error": exc})
            if spec.fail_on_tool_error:
                return f"Error: {type(exc).__name__}: {exc}", event, exc
            return f"Error: {type(exc).__name__}: {exc}", event, None
        detail = "" if result is None else str(result)
        detail = detail.replace("\n", " ").strip()
        if not detail:
            detail = "(empty)"
        elif len(detail) > 120:
            detail = detail[:120] + "..."
        status = "error" if isinstance(result, str) and result.startswith("Error") else "ok"
        prof.pop(status=status, args={"tool_call": tool_call, "result": result})
        return result, {"name": tool_call.name, "status": status, "detail": detail}, None


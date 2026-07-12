"""Task() / Tasks() delegation to isolated sub-agents.

Sub-agents always run in 'auto' mode: they're isolated, self-contained
delegated work, so per-tool-call prompts would just block the whole
session waiting on a decision the user can't fully see context for.
Genuinely dangerous shell commands still always prompt regardless of mode.
"""

import json
from concurrent.futures import ThreadPoolExecutor, as_completed

from rich.panel import Panel
from rich.markup import escape

from bardgent import config
from bardgent.config import console, console_lock, log_event
from bardgent.state import AgentState
from bardgent.session import trim_history
from bardgent.utils import truncate_output
from bardgent.model import stream_agent_response, call_model


def run_subagent(task_prompt, max_iters=15, render=True, label=None):
    """Run one isolated sub-agent to completion and return its final text.

    render=True  -> normal single-Task behaviour: live-streamed output.
    render=False -> used when multiple sub-agents run concurrently (Tasks);
                     uses plain blocking model calls and lock-protected
                     status lines so parallel threads don't fight over the
                     terminal.
    """
    from bardgent.tool_schemas import dispatch_tool, SUBAGENT_TOOLS

    tag = f"[{label}] " if label else ''
    sub_system_prompt = (
        "You are a focused sub-agent spawned to complete one delegated task.\n"
        "Use the available tools as needed, then reply with ONLY the final "
        "result, no meta-commentary about being a sub-agent.\n\n" + config.SYSTEM_INFO
    )
    sub_state = AgentState(sub_system_prompt, name='sub', track_session=False, mode='auto')
    sub_state.messages.append({'role': 'user', 'content': task_prompt})

    if render:
        console.print(Panel(task_prompt, title='[bold magenta]SUB-AGENT started', border_style='magenta'))
    else:
        with console_lock:
            console.print(f"[bold magenta]{tag}SUB-AGENT started:[/bold magenta] {task_prompt[:100]}")
    log_event(f"SUBAGENT {tag}START: {task_prompt[:200]!r}")

    for i in range(max_iters):
        if render:
            final_text, tool_calls, _ = stream_agent_response(sub_state.messages, SUBAGENT_TOOLS)
        else:
            final_text, tool_calls, _ = call_model(sub_state.messages, SUBAGENT_TOOLS)
            if tool_calls:
                names = ', '.join(tc['name'] for tc in tool_calls)
                with console_lock:
                    console.print(f"[dim magenta]{tag}iteration {i + 1}: running {names}[/dim magenta]")

        if not tool_calls:
            result = final_text.strip()
            if render:
                console.print(Panel(result or '(empty result)', title='[bold magenta]SUB-AGENT finished', border_style='magenta'))
            else:
                with console_lock:
                    console.print(Panel(result or '(empty result)', title=f'[bold magenta]{tag}SUB-AGENT finished', border_style='magenta'))
            log_event(f"SUBAGENT {tag}DONE")
            return result

        sub_state.messages.append({
            "role": "assistant",
            "content": final_text or None,
            "tool_calls": [
                {"id": tc['id'], "type": "function",
                 "function": {"name": tc['name'], "arguments": tc['arguments']}}
                for tc in tool_calls
            ]
        })
        for tc in tool_calls:
            try:
                args = json.loads(tc['arguments'] or '{}')
            except json.JSONDecodeError as e:
                result = f"Error: could not parse arguments for '{tc['name']}': {e}"
            else:
                if render:
                    result = dispatch_tool(tc['name'], args, sub_state)
                else:
                    with console_lock:
                        result = dispatch_tool(tc['name'], args, sub_state)
            sub_state.messages.append({
                'role': 'tool', 'tool_call_id': tc['id'],
                'content': truncate_output(str(result)),
            })
        trim_history(sub_state)

    log_event(f"SUBAGENT {tag}HIT MAX ITERATIONS")
    return "(sub-agent hit max iterations without finishing)"


def run_subagents_parallel(prompts, max_iters=15, max_workers=5):
    """Run several sub-agents concurrently (Tasks tool). Returns a single
    string combining every sub-agent's labeled result, in original order."""
    n = len(prompts)
    results = [None] * n

    def worker(i, prompt):
        label = f"Task {i + 1}/{n}"
        return i, run_subagent(prompt, max_iters=max_iters, render=False, label=label)

    with console_lock:
        console.print(Panel(
            "\n".join(f"{i + 1}. {p[:100]}" for i, p in enumerate(prompts)),
            title=f"[bold magenta]Running {n} sub-agents concurrently[/bold magenta]",
            border_style='magenta',
        ))

    with ThreadPoolExecutor(max_workers=min(n, max_workers)) as ex:
        futures = [ex.submit(worker, i, p) for i, p in enumerate(prompts)]
        for fut in as_completed(futures):
            i, result = fut.result()
            results[i] = result

    combined = "\n\n".join(f"[Sub-agent {i + 1} result]:\n{r}" for i, r in enumerate(results))
    log_event(f"PARALLEL SUBAGENTS DONE ({n} tasks)")
    return combined
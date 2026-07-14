"""Task() / Tasks() delegation to isolated sub-agents.

Sub-agents always run in 'auto' mode: they're isolated, self-contained
delegated work, so per-tool-call prompts would just block the whole
session waiting on a decision the user can't fully see context for.
Genuinely dangerous shell commands still always prompt regardless of mode
- UNLESS the sub-agent is unattended (e.g. a scheduled task with nobody at
the keyboard), in which case dangerous actions are auto-denied instead of
blocking on input() forever (see state.ask_approval).
"""

import json
from concurrent.futures import ThreadPoolExecutor, as_completed

from rich.panel import Panel
from rich.markup import escape

from bardgent import config
from bardgent.config import console, console_lock, log_event
from bardgent.state import AgentState
from bardgent.utils import truncate_output
from bardgent.model import stream_agent_response, call_model
from bardgent.system_prompt import build_skills_and_rules_block
from bardgent.project_instructions import format_project_instructions_section
from bardgent.memory import memory_context_block

# Returned verbatim by run_subagent() when it hits max_iters without a final
# answer. Exposed as a constant (rather than making callers match a literal
# string) so scheduler.py can reliably tell an incomplete run apart from a
# genuinely finished one.
INCOMPLETE_RESULT = "(sub-agent hit max iterations without finishing)"

# Default tool-loop budget for delegated work. Main agent uses 100; sub-agents
# used to cap at 15 and frequently returned INCOMPLETE_RESULT on multi-file jobs.
DEFAULT_SUBAGENT_MAX_ITERS = 40


def _sub_system_prompt(unattended=False):
    # Same skills catalogue + tool-usage rules as the main agent, so
    # sub-agents follow identical conventions (Read paging, Edit fuzzy-match
    # behaviour, background Bash jobs, when to reach for a skill, etc.)
    # instead of a stripped-down prompt that drifts out of sync over time.
    # Also inject project instructions and long-term memory so delegated work
    # sees the same AGENTS.md / user facts as the main agent.
    intro = (
        "You are a focused sub-agent spawned to complete one delegated task.\n"
        "Use the available tools as needed, then reply with ONLY the final "
        "result, no meta-commentary about being a sub-agent."
    )
    if unattended:
        intro += (
            "\n\nThis run is a SCHEDULED TASK, executing on its own with nobody watching. "
            "Whatever plain text you give as your final answer is automatically delivered to "
            "the user over Telegram right after you finish - you do not send it yourself, and "
            "you have no way to call the Telegram Bot API directly (no bot token, no chat id, "
            "no HTTP access to Telegram). Never write curl commands, Python scripts, or "
            "instructions for setting up a bot or chat id, and never ask the user for a bot "
            "token or chat id - that delivery step is already handled outside of you. Just "
            "produce the finished result itself (e.g. the actual news summary, the actual "
            "report) as your final answer, written in normal markdown (**bold**, bullet lists, "
            "etc.) - it gets converted to Telegram's formatting automatically. Work "
            "efficiently: you have a limited number of tool calls, so don't repeat searches "
            "that already gave you enough to work with."
        )
    project_instructions = format_project_instructions_section()
    memory_block = memory_context_block()
    return (
        intro
        + "\n\n" + config.get_system_info()
        + "\n\n" + project_instructions
        + "\n\nKNOWN USER MEMORY (same as main agent):\n" + memory_block
        + "\n\n" + build_skills_and_rules_block()
    )


def run_subagent(task_prompt, max_iters=DEFAULT_SUBAGENT_MAX_ITERS, render=True, label=None, unattended=False):
    """Run one isolated sub-agent to completion and return its final text.

    render=True  -> normal single-Task behaviour: live-streamed output.
    render=False -> used when multiple sub-agents run concurrently (Tasks),
                     or when running unattended (scheduled tasks); uses plain
                     blocking model calls and lock-protected status lines so
                     parallel/background threads don't fight over the
                     terminal.
    unattended=True -> nobody is present to answer an approval prompt (e.g.
                     a scheduled task firing on its own clock). Dangerous
                     shell commands are auto-denied rather than blocking on
                     input() forever; everything else behaves like normal
                     'auto' mode. The sub-agent is also told explicitly that
                     its answer gets auto-delivered over Telegram, so it
                     doesn't try to "help" by inventing a bot-token setup
                     flow of its own.
    """
    from bardgent.tool_schemas import dispatch_tool, SUBAGENT_TOOLS

    tag = f"[{label}] " if label else ''
    sub_state = AgentState(
        _sub_system_prompt(unattended=unattended), name='sub', track_session=False, mode='auto',
        unattended=unattended,
    )
    sub_state.messages.append({'role': 'user', 'content': task_prompt})

    if render:
        console.print(Panel(task_prompt, title='[bold magenta]SUB-AGENT started', border_style='magenta'))
    else:
        with console_lock:
            console.print(f"[bold magenta]{tag}SUB-AGENT started:[/bold magenta] {task_prompt[:100]}")
    log_event(f"SUBAGENT {tag}START: {task_prompt[:200]!r}")

    made_any_tool_call = False
    nudged_no_tool_first_turn = False

    for i in range(max_iters):
        if render:
            final_text, tool_calls, _, _ = stream_agent_response(sub_state.messages, SUBAGENT_TOOLS)
        else:
            final_text, tool_calls, _, _ = call_model(sub_state.messages, SUBAGENT_TOOLS)
            if tool_calls:
                names = ', '.join(tc['name'] for tc in tool_calls)
                with console_lock:
                    console.print(f"[dim magenta]{tag}iteration {i + 1}: running {names}[/dim magenta]")

        if not tool_calls:
            result = final_text.strip()

            # Guard against a run declaring itself "done" after only
            # announcing intent ("I'll search for current news...") without
            # ever actually calling a tool - this happens occasionally with
            # unattended runs and, left unchecked, delivers an empty
            # non-answer as if it were the finished result. Nudge it once to
            # actually do the work instead of accepting that as final.
            if (unattended and i == 0 and not made_any_tool_call
                    and not nudged_no_tool_first_turn and len(result) < 200):
                nudged_no_tool_first_turn = True
                log_event(
                    f"SUBAGENT {tag}first-turn no-tool short reply looked like an intent "
                    f"statement, not a result; nudging to actually do the work: {result[:100]!r}"
                )
                sub_state.messages.append({'role': 'assistant', 'content': result})
                sub_state.messages.append({
                    'role': 'user',
                    'content': (
                        "That was a statement of intent, not the finished result. Nobody is "
                        "watching this run - actually use your tools now (WebSearch, Fetch, etc.) "
                        "and then give the real, complete final answer with the actual content, "
                        "not a description of what you're about to do."
                    ),
                })
                continue

            if render:
                console.print(Panel(result or '(empty result)', title='[bold magenta]SUB-AGENT finished', border_style='magenta'))
            else:
                with console_lock:
                    console.print(Panel(result or '(empty result)', title=f'[bold magenta]{tag}SUB-AGENT finished', border_style='magenta'))
            log_event(f"SUBAGENT {tag}DONE")
            return result

        made_any_tool_call = True

        for n, tc in enumerate(tool_calls):
            if not tc.get('id'):
                tc['id'] = f'call_{n}_{abs(hash(tc.get("name") or "")) % 10**8:08d}'

        sub_state.messages.append({
            "role": "assistant",
            "content": final_text or '',
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
                'content': truncate_output(str(result)) or '(no output)',
            })
        # Avoid mid-loop trim (same Gemini empty-contents failure mode as main).

    log_event(f"SUBAGENT {tag}HIT MAX ITERATIONS")
    return INCOMPLETE_RESULT


def run_subagents_parallel(prompts, max_iters=DEFAULT_SUBAGENT_MAX_ITERS, max_workers=5):
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
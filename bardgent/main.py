#!/usr/bin/env python3
"""BardGent entry point. Run with: python main.py"""

import os
import sys
import json
import atexit

from prompt_toolkit import PromptSession
from prompt_toolkit.completion import WordCompleter
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.application import run_in_terminal
from rich.text import Text
from rich.panel import Panel

from bardgent import config
from bardgent.config import console, log_event
from bardgent.ui import print_welcome
from bardgent.state import AgentState
from bardgent.system_prompt import SYSTEM_PROMPT, refresh_system_message
from bardgent.commands import COMMANDS, handle_command, switch_mode, MODE_CYCLE
from bardgent.session import (
    save_session, trim_history, total_history_tokens,
    do_summary_and_compact, sanitize_history,
)
from bardgent.model import stream_agent_response, print_usage
from bardgent.tool_schemas import TOOLS, dispatch_tool
from bardgent.utils import truncate_output
from bardgent.status_bar import (
    enable_status_bar, disable_status_bar, install_resize_handler, draw_status_bar,
    make_bottom_toolbar, suspend_status_bar, resume_status_bar,
)
from bardgent.exec_tools import cleanup_jobs
from bardgent import scheduler


def _prompt_user(prompt_session, state, message):
    suspend_status_bar()
    try:
        return prompt_session.prompt(message, multiline=False).strip()
    finally:
        resume_status_bar(state)

def main():
    # Detached scheduler process (keeps firing tasks after the terminal closes).
    # Invoked by ensure_daemon_running(); not meant for interactive use.
    if '--scheduler-daemon' in sys.argv:
        scheduler.run_daemon_forever()
        return

    if not os.environ.get('GEMINI_API_KEY'):
        console.print("[bold red]Error: GEMINI_API_KEY is not set.[/bold red]")
        console.print("Please set it in your environment or add it to ~/.bardgent/.env:")
        console.print("[yellow]GEMINI_API_KEY=your_key_here[/yellow]")
        sys.exit(1)

    print_welcome()
    log_event("=== Bardgent session start ===")

    state = AgentState(SYSTEM_PROMPT, name='main')
    mode_keys = KeyBindings()

    @mode_keys.add('s-tab')
    def _cycle_mode(event):
        def _do_switch():
            idx = MODE_CYCLE.index(state.mode) if state.mode in MODE_CYCLE else 0
            switch_mode(state, MODE_CYCLE[(idx + 1) % len(MODE_CYCLE)])
        run_in_terminal(_do_switch)

    prompt_session = PromptSession(
            completer=WordCompleter(list(COMMANDS.keys()), sentence=True),
            key_bindings=mode_keys,
            # Pinned footer while waiting for USER: input (works in Warp + classic TTYs).
            bottom_toolbar=make_bottom_toolbar(state),
            # Rows reserved for the completion popup (e.g. slash-command menu).
            # Too small a value here means the popup has nowhere to render and
            # silently doesn't show up — this does NOT affect the toolbar.
            reserve_space_for_menu=8,
            erase_when_done=True,
        )
    
    enable_status_bar()
    atexit.register(disable_status_bar)
    atexit.register(cleanup_jobs)
    install_resize_handler(state)

    # Recurring scheduled tasks run in a *detached* daemon process so they
    # keep firing after you quit Bardgent / close the terminal. On-demand
    # `/schedule run` still executes inside this process. See scheduler.py.
    ok, daemon_msg = scheduler.ensure_daemon_running()
    if ok:
        console.print(f'[dim]Scheduler daemon: {daemon_msg}[/dim]')
    else:
        console.print(f'[yellow]Scheduler daemon: {daemon_msg}[/yellow]')
        console.print('[yellow]Scheduled tasks will not run until the daemon is up '
                      '(/schedule daemon start).[/yellow]')

    auto_continue = False
    while True:
        try:
            if auto_continue:
                user_input = "Please proceed with executing the proposed plan."
                console.print(Text("USER: ", style="bold green") + Text(user_input))
                state.messages.append({'role': 'user', 'content': user_input})
                auto_continue = False
            else:
                try:
                    user_input = _prompt_user(
                        prompt_session, state,
                        HTML('<ansigreen><b>USER: </b></ansigreen>'),
                    )
                except KeyboardInterrupt:
                    resume_status_bar(state)
                    continue
                except EOFError:
                    console.print('Goodbye!')
                    break
                console.print(Text("USER: ", style="bold green") + Text(user_input))
        except KeyboardInterrupt:
            resume_status_bar(state)
            continue
        except EOFError:
            console.print('Goodbye!')
            break

        if user_input.lower() in ['exit', 'quit']:
            console.print('Goodbye!')
            break

        if user_input.startswith('/'):
            result = handle_command(user_input, state)
            if result == 'handled':
                continue
            elif result is None:
                console.print(f'[bold red]Unknown command: {user_input}[/bold red]')
                continue
        else:
            state.messages.append({'role': 'user', 'content': user_input})

        refresh_system_message(state)

        if total_history_tokens(state) > config.AUTO_SUMMARY_TOKEN_THRESHOLD:
            console.print('[dim]Context getting large, auto-summarizing...[/dim]')
            log_event("AUTO-SUMMARY triggered")
            do_summary_and_compact(state)
        else:
            trim_history(state)

        # Always re-validate before the model call (resume/trim/failed retries
        # can leave Gemini-invalid turn order in the history).
        sanitize_history(state)

        try:
            plan_completed_successfully = False
            for _ in range(config.MAX_ITERATIONS):
                final_text, tool_calls, finish_reason, usage = stream_agent_response(state.messages, TOOLS)
                if usage:
                    state.last_prompt_tokens = usage.prompt_tokens + usage.completion_tokens
                if tool_calls:
                    # Guarantee ids — some Gemini streams omit them on deltas.
                    for n, tc in enumerate(tool_calls):
                        if not tc.get('id'):
                            tc['id'] = f'call_{n}_{abs(hash(tc.get("name") or "")) % 10**8:08d}'

                    state.messages.append({
                        "role": "assistant",
                        "content": final_text or '',
                        "tool_calls": [
                            {
                                "id": tc['id'],
                                "type": "function",
                                "function": {
                                    "name": tc['name'],
                                    "arguments": tc['arguments']
                                }
                            }
                            for tc in tool_calls
                        ]
                    })

                    for tool_call in tool_calls:
                        name = tool_call['name']
                        try:
                            args = json.loads(tool_call['arguments'] or '{}')
                        except json.JSONDecodeError as e:
                            result = f"Error: could not parse arguments for '{name}': {e}. Re-issue the call with valid JSON."
                        else:
                            result = dispatch_tool(name, args, state)

                        state.messages.append({
                            'role': 'tool',
                            'tool_call_id': tool_call['id'],
                            'content': truncate_output(str(result)) or '(no output)',
                        })
                    # Do NOT trim mid tool-loop: aggressive trims were wiping the
                    # user turn and leaving system-only history (Gemini 400:
                    # "contents is not specified"). Trimming runs at turn start.
                    draw_status_bar(state)
                    continue

                final_text = config.remove_thoughts(final_text.strip())
                state.messages.append({'role': 'assistant', 'content': final_text})
                print_usage(usage)
                if state.telegram_enabled and state.telegram_chat_id and final_text:
                    from bardgent.telegram import send_telegram_message
                    if not send_telegram_message(final_text, state.telegram_chat_id):
                        console.print('[dim red]Could not deliver message to Telegram (see bardgent.log).[/dim red]')
                plan_completed_successfully = True
                break
            else:
                console.print(f'[bold red]Hit max iterations ({config.MAX_ITERATIONS}) without a final answer.[/bold red]')

            if plan_completed_successfully and state.mode == 'plan':
                console.print()
                button_strip = (
                    "Select a mode to execute the proposed plan:\n\n"
                    "  [bold white on green]  1  [/bold white on green] [bold green]NORMAL MODE[/bold green] (Approve each action)\n"
                    "  [bold white on red]  2  [/bold white on red] [bold red]AUTO MODE[/bold red]   (Auto-approve non-dangerous actions)\n"
                    "  [bold white on grey37]  3  [/bold white on grey37] [dim]KEEP PLAN[/dim]   (Stay in read-only mode)\n"
                )
                console.print(Panel(button_strip, title="[bold cyan]Next Action[/bold cyan]", border_style="cyan", expand=False))

                try:
                    ans = _prompt_user(
                        prompt_session, state,
                        HTML('<ansiyellow><b>Select option [1/2/3] (3): </b></ansiyellow>'),
                    )
                except (KeyboardInterrupt, EOFError):
                    resume_status_bar(state)
                    ans = '3'

                if ans == '1':
                    switch_mode(state, 'normal')
                    auto_continue = True
                elif ans == '2':
                    switch_mode(state, 'auto')
                    auto_continue = True
                else:
                    console.print("[dim]Remaining in PLAN mode.[/dim]")

        except KeyboardInterrupt:
            # Mid-turn interrupt can leave assistant tool_calls without results.
            sanitize_history(state)
            console.print('\n[yellow]Interrupted! back to prompt.[/yellow]')
        except Exception as e:
            sanitize_history(state)
            console.print(f'\n[bold red]Error during turn: {type(e).__name__}: {e}[/bold red]')
            log_event(f"TURN ERROR: {type(e).__name__}: {e}")

        if state.messages[1:]:
            save_session(state)
        trim_history(state)
        draw_status_bar(state, force=True)


if __name__ == '__main__':
    main()
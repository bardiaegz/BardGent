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
from bardgent.system_prompt import SYSTEM_PROMPT
from bardgent.commands import COMMANDS, handle_command, switch_mode, MODE_CYCLE
from bardgent.session import (
    save_session, trim_history, total_history_tokens,
    do_summary_and_compact, sanitize_history,
)
from bardgent.model import stream_agent_response
from bardgent.tool_schemas import TOOLS, dispatch_tool
from bardgent.utils import truncate_output
from bardgent.status_bar import enable_status_bar, disable_status_bar, install_resize_handler


def main():
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
    )

    enable_status_bar()
    atexit.register(disable_status_bar)
    install_resize_handler(state)

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
                    user_input = prompt_session.prompt(HTML('<ansigreen><b>USER: </b></ansigreen>'), multiline=False).strip()
                except KeyboardInterrupt:
                    continue
                except EOFError:
                    console.print('Goodbye!')
                    break
        except KeyboardInterrupt:
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
                final_text, tool_calls, finish_reason = stream_agent_response(state.messages, TOOLS)

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
                    continue

                final_text = config.remove_thoughts(final_text.strip())
                state.messages.append({'role': 'assistant', 'content': final_text})
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
                    ans = prompt_session.prompt(
                        HTML('<ansiyellow><b>Select option [1/2/3] (3): </b></ansiyellow>'),
                        multiline=False
                    ).strip()
                except (KeyboardInterrupt, EOFError):
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


if __name__ == '__main__':
    main()

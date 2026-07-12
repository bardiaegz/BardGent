"""Slash commands (/model, /clear, /resume, /plan, ...) and mode switching."""

from rich.panel import Panel
from rich.markup import escape

from bardgent import config, skills
from bardgent.config import console, log_event
from bardgent.state import ask_approval
from bardgent.session import (
    session_file_name, do_summary_and_compact, resume_session, list_sessions,
)
from bardgent.checkpoints import list_checkpoints, restore_checkpoint
from bardgent.telegram import discover_telegram_chat_id, send_telegram_message, _save_telegram_chat_id

COMMANDS = {
    '/summary': 'Summarize the current conversation',
    '/model': 'Show the current model, or switch: /model <name>',
    '/clear': 'Clear history and start a new session',
    '/resume': 'Pick a past session and resume it',
    '/telegram': 'Toggle sending the agent\'s final answers to Telegram',
    '/plan': 'Switch to PLAN mode (read-only exploration, agent proposes a plan)',
    '/normal': 'Switch to NORMAL mode (approve each action, default)',
    '/auto': 'Switch to AUTO mode (auto-approve everything except dangerous commands)',
    '/mode': 'Show the current mode',
    '/skills': 'List installed skills (auto-detected capability packs)',
    '/checkpoints': 'List recent git checkpoints (auto-created on Write/Edit)',
    '/restore': 'Restore the working tree to a checkpoint: /restore <n>',
    '/exit': 'Quit Bardgent',
}

MODE_CYCLE = ['normal', 'auto', 'plan']
_MODE_COLOR = {'plan': 'cyan', 'normal': 'white', 'auto': 'bold red'}


def switch_mode(state, new_mode, announce=True):
    """Single source of truth for changing state.mode, used by /plan,
    /normal, /auto, and the shift+tab shortcut alike."""
    old_mode = state.mode
    if new_mode == old_mode:
        if announce:
            console.print(f'[dim]Already in {new_mode} mode.[/dim]')
        return
    state.mode = new_mode
    if announce:
        color = _MODE_COLOR.get(new_mode, 'white')
        console.print(f'[{color}]Mode switched: {old_mode} -> {new_mode}[/{color}]')
        if new_mode == 'plan':
            console.print('[dim]Only read-only tools are allowed until you switch back with /normal or /auto.[/dim]')
        elif new_mode == 'auto':
            console.print('[dim]Non-dangerous actions will be auto-approved without asking. '
                          'Dangerous commands (rm, sudo, chmod, kill, ...) still always ask.[/dim]')
    log_event(f"[{state.name}] MODE {old_mode} -> {new_mode}")


def handle_command(user_input, state):
    cmd = user_input.strip().lower()

    if cmd == '/model' or cmd.startswith('/model '):
        parts = user_input.strip().split(maxsplit=1)
        if len(parts) == 1:
            console.print(f'Current model: [bold cyan]{config.MODEL}[/bold cyan]')
        else:
            new_model = parts[1].strip()
            try:
                available = [m.id for m in config.client.models.list().data]
            except Exception as e:
                available = None
                console.print(f'[dim]Could not reach server to verify model list: {e}[/dim]')
            if available and new_model not in available:
                console.print(f"[yellow]Warning: '{new_model}' was not found on the server.[/yellow]")
                console.print(f"[dim]Available: {', '.join(available)}[/dim]")
                if not ask_approval(state, 'model_switch_unverified', "Switch to it anyway?"):
                    return 'handled'
            config.MODEL = new_model
            console.print(f'[bold green]Model switched to {config.MODEL}.[/bold green]')
        return 'handled'

    if cmd == '/clear':
        from bardgent.ui import print_welcome  # local import: avoids a circular import with main.py
        del state.messages[1:]
        state.session_file = config.SESSION_DIR / session_file_name()
        console.clear()
        print_welcome()
        console.print("[bold green]New session started.[/bold green]")
        return 'handled'

    if cmd == '/resume':
        resume_session(state)
        return 'handled'

    if cmd in ('/plan', '/normal', '/auto'):
        switch_mode(state, cmd[1:])
        return 'handled'

    if cmd == '/mode':
        console.print(f'Current mode: [bold cyan]{state.mode}[/bold cyan]')
        return 'handled'

    if cmd == '/skills':
        skills.refresh_skills()
        console.print(Panel(escape(skills.list_skills_text()), title='[bold cyan]Installed skills', border_style='cyan'))
        return 'handled'

    if cmd == '/checkpoints':
        console.print(Panel(escape(list_checkpoints()), title='[bold cyan]Git checkpoints', border_style='cyan'))
        return 'handled'

    if cmd == '/restore' or cmd.startswith('/restore '):
        parts = user_input.strip().split(maxsplit=1)
        if len(parts) == 1:
            console.print('[yellow]Usage: /restore <n>  (see /checkpoints for indices)[/yellow]')
            return 'handled'
        if not ask_approval(state, 'restore_checkpoint',
                            f"This overwrites files in the working tree to match checkpoint #{parts[1].strip()}. Continue?"):
            console.print('[yellow]Restore cancelled.[/yellow]')
            return 'handled'
        console.print(restore_checkpoint(parts[1].strip()))
        return 'handled'

    if cmd == '/exit':
        console.print('Goodbye!')
        import sys
        sys.exit(0)

    if cmd == '/summary':
        do_summary_and_compact(state)
        return 'handled'

    if cmd == '/telegram':
        if not config.TELEGRAM_BOT_TOKEN:
            console.print('[bold red]No TELEGRAM_BOT_TOKEN found (check your .env file). Cannot enable Telegram.[/bold red]')
            return 'handled'

        if state.telegram_enabled:
            state.telegram_enabled = False
            console.print('[yellow]Telegram messaging turned off.[/yellow]')
            log_event("TELEGRAM disabled")
            return 'handled'

        if not state.telegram_chat_id:
            chat_id = discover_telegram_chat_id()
            if not chat_id:
                console.print('[bold red]No message received from the bot in time. '
                              'Message your bot on Telegram, then run /telegram again.[/bold red]')
                return 'handled'
            state.telegram_chat_id = chat_id
            _save_telegram_chat_id(chat_id)
            console.print(f'[bold green]Telegram linked (chat_id={chat_id}).[/bold green]')
            log_event(f"TELEGRAM linked chat_id={chat_id}")

        state.telegram_enabled = True
        console.print('[bold green]Telegram messaging turned on, final answers will be sent there too.[/bold green]')
        log_event("TELEGRAM enabled")
        return 'handled'

    return None
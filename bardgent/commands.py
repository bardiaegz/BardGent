"""Slash commands (/model, /clear, /resume, /plan, ...) and mode switching."""

from rich.panel import Panel
from rich.markup import escape

from bardgent import config, skills, scheduler
from bardgent.config import console, log_event
from bardgent.state import ask_approval
from bardgent.session import (
    session_file_name, do_summary_and_compact, resume_session, list_sessions,
)
from bardgent.checkpoints import list_checkpoints, restore_checkpoint
from bardgent.telegram import discover_telegram_chat_id, send_telegram_message, _save_telegram_chat_id
from bardgent.skills import install_skill_from_github
from bardgent.system_prompt import refresh_system_message

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
    '/skill install': 'Install a skill from GitHub: /skill install <github_url>',
    '/checkpoints': 'List recent git checkpoints (auto-created on Write/Edit)',
    '/restore': 'Restore the working tree to a checkpoint: /restore <n>',
    '/schedule': 'Create a scheduled task: /schedule <spec> :: <prompt>. Manage: /schedule pause|resume|delete|run <id>',
    '/schedules': 'List scheduled tasks (next/last run, status)',
    '/exit': 'Quit Bardgent',
}

MODE_CYCLE = ['normal', 'auto', 'plan']
_MODE_COLOR = {'plan': 'cyan', 'normal': 'white', 'auto': 'bold red'}

SCHEDULE_HELP = (
    "Usage: /schedule <spec> :: <prompt>\n"
    "  Specs:\n"
    "    every 30m | every 2h | every 1d   (recurring interval, min 60s)\n"
    "    daily 09:00 | daily 6pm            (once a day)\n"
    "    weekly mon 09:00                   (once a week)\n"
    "    once 2026-07-20 09:00              (a single one-off run)\n"
    "    cron */15 * * * *                  (standard 5-field cron)\n"
    "  Example:\n"
    "    /schedule daily 09:00 :: Summarize my unread Slack messages from the last 24 hours\n"
    "  Manage existing tasks:\n"
    "    /schedules                (list all)\n"
    "    /schedule pause <id>\n"
    "    /schedule resume <id>\n"
    "    /schedule delete <id>\n"
    "    /schedule run <id>        (run it right now, on demand)"
)


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
        refresh_system_message(state)
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

    if cmd == '/schedules':
        console.print(Panel(escape(scheduler.list_tasks_text()), title='[bold cyan]Scheduled tasks', border_style='cyan'))
        return 'handled'

    if cmd == '/schedule' or cmd.startswith('/schedule '):
        rest = user_input.strip()[len('/schedule'):].strip()
        if not rest:
            console.print(Panel(SCHEDULE_HELP, title='[bold cyan]/schedule help', border_style='cyan'))
            return 'handled'

        first_word = rest.split(maxsplit=1)[0].lower()
        if first_word in ('pause', 'resume', 'delete', 'remove', 'cancel', 'run'):
            parts = rest.split(maxsplit=1)
            if len(parts) < 2:
                console.print(f'[yellow]Usage: /schedule {first_word} <id>  (see /schedules for ids)[/yellow]')
                return 'handled'
            task_id = parts[1].strip()

            if first_word == 'pause':
                ok = scheduler.set_enabled(task_id, False)
                console.print(f'[green]Paused {task_id}.[/green]' if ok else f'[red]No scheduled task with id {task_id}.[/red]')
            elif first_word == 'resume':
                ok = scheduler.set_enabled(task_id, True)
                console.print(f'[green]Resumed {task_id}.[/green]' if ok else f'[red]No scheduled task with id {task_id}.[/red]')
            elif first_word in ('delete', 'remove', 'cancel'):
                ok = scheduler.remove_task(task_id)
                console.print(f'[green]Deleted {task_id}.[/green]' if ok else f'[red]No scheduled task with id {task_id}.[/red]')
            elif first_word == 'run':
                task = scheduler.get_task(task_id)
                if not task:
                    console.print(f'[red]No scheduled task with id {task_id}.[/red]')
                else:
                    console.print(f'[cyan]Running {task_id} ("{task["name"]}") now in the background...[/cyan]')
                    scheduler.run_task_in_background(task_id)
            return 'handled'

        if '::' not in rest:
            console.print(Panel(SCHEDULE_HELP, title='[bold cyan]/schedule help', border_style='cyan'))
            return 'handled'

        spec_part, prompt_part = rest.split('::', 1)
        task, err = scheduler.add_task(prompt_part.strip(), spec_part.strip())
        if err:
            console.print(f'[bold red]Could not create scheduled task:[/bold red] {err}')
        else:
            nr_text = scheduler.format_dt(task.get('next_run'))
            console.print(f'[bold green]Scheduled task created:[/bold green] {task["id"]}  (next run: {nr_text})')
            if not config.TELEGRAM_BOT_TOKEN:
                console.print('[dim]Note: no TELEGRAM_BOT_TOKEN configured, results will only be visible via /schedules, not delivered to Telegram.[/dim]')
            elif not state.telegram_chat_id:
                console.print('[dim]Note: Telegram isn\'t linked yet - run /telegram once to link it so results get delivered there too.[/dim]')
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

    if cmd == '/skill install' or cmd.startswith('/skill install '):
        parts = user_input.strip().split(maxsplit=2)
        if len(parts) < 3:
            console.print('[yellow]Usage: /skill install <github_url>[/yellow]')
            console.print('[dim]Example: /skill install https://github.com/alirezarezvani/claude-skills[/dim]')
            return 'handled'
        
        github_url = parts[2].strip()
        console.print(f'[cyan]Installing skill from {github_url}...[/cyan]')
        result = install_skill_from_github(github_url)
        console.print(result)
        skills.refresh_skills()
        return 'handled'

    return None
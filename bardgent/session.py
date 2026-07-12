"""Session save/resume/replay, history trimming, and LLM-based compaction.

Deliberately does not import bardgent.state, to keep the import graph
one-directional (state.py imports this module, not the reverse).
"""

import glob
import json
import time
from pathlib import Path

from bardgent import config
from bardgent.config import console, log_event
from bardgent.utils import count_tokens


def session_file_name():
    return f"{config.SESSION_PREFIX}{time.strftime('%Y-%m-%d_%H-%M-%S')}.json"


def session_title(msgs):
    for m in msgs:
        if m.get('role') == 'user' and m.get('content'):
            first_line = m['content'].strip().splitlines()[0]
            return first_line[:60] + ('…' if len(first_line) > 60 else '')
    for m in msgs:
        if m.get('content'):
            first_line = str(m['content']).removeprefix(config.SUMMARY_PREFIX).strip().splitlines()[0]
            return first_line[:60] + ('…' if len(first_line) > 60 else '')
    return '(empty session)'


def save_session(state):
    if not state.messages[1:] or state.session_file is None:
        return
    data = {
        'updated': time.strftime('%Y-%m-%d %H:%M:%S'),
        'title': session_title(state.messages[1:]),
        'messages': state.messages[1:],
    }
    try:
        with open(state.session_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    except OSError as e:
        console.print(f"[dim red]Failed to auto-save session: {e}[/dim red]")
        log_event(f"SESSION SAVE FAILED: {e}")


def list_sessions():
    sessions = []
    for path in glob.glob(str(config.SESSION_DIR / f'{config.SESSION_PREFIX}*.json')):
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if data.get('messages'):
                sessions.append((path, data))
        except (OSError, json.JSONDecodeError):
            continue
    sessions.sort(key=lambda s: s[1].get('updated', ''), reverse=True)
    return sessions


def replay_transcript(state):
    from rich.text import Text
    from rich.panel import Panel
    from rich.markdown import Markdown
    from rich.console import Group

    for m in state.messages[1:]:
        role = m.get('role')
        content = m.get('content') or ''
        if role == 'user':
            console.print(Text('USER: ', style='bold green') + Text(content))
        elif role == 'assistant':
            clean = content.removeprefix(config.SUMMARY_PREFIX).strip()
            if content.startswith(config.SUMMARY_PREFIX):
                console.print(Panel(Text(clean), title='[bold cyan]SUMMARY', border_style='cyan'))
            elif clean:
                console.print(Group(Text('AGENT:', style='bold cyan'), Markdown(clean)))


def resume_session(state):
    from rich.text import Text
    from rich.panel import Panel

    sessions = [s for s in list_sessions() if Path(s[0]) != state.session_file]
    if not sessions:
        console.print('[yellow]No saved sessions found.[/yellow]')
        return
    sessions = sessions[:10]

    listing = Text()
    for i, (path, data) in enumerate(sessions, 1):
        listing.append(f'{i}. ', style='bold cyan')
        listing.append(data.get('title') or '(untitled)')
        listing.append(f"   {data.get('updated', '?')} · {len(data['messages'])} messages\n", style='dim')

    console.print(Panel(listing, title='[bold cyan]Resume a session', border_style='cyan'))
    choice = input(f"Resume which session? [1-{len(sessions)}, q to cancel] (1): ").strip().lower()
    if choice in ('q', 'quit', 'n', 'no'):
        return
    try:
        path, data = sessions[(int(choice) if choice else 1) - 1]
    except (ValueError, IndexError):
        console.print('[bold red]Invalid choice.[/bold red]')
        return

    del state.messages[1:]
    state.messages.extend(data['messages'])
    state.session_file = Path(path)
    console.print()
    replay_transcript(state)
    console.print(Text(f"\nResumed \"{data.get('title', '')}\" ({len(state.messages) - 1} messages).", style='bold green'))


def _message_tokens(m):
    body = count_tokens(str(m.get('content') or ''))
    if m.get('tool_calls'):
        body += count_tokens(json.dumps(m['tool_calls']))
    return body


def total_history_tokens(state):
    return sum(_message_tokens(m) for m in state.messages[1:])


def trim_history(state):
    msgs = state.messages
    total = total_history_tokens(state)
    if total <= config.MAX_HISTORY_TOKENS and len(msgs) <= config.MAX_HISTORY_MESSAGES + 1:
        return
    cut = 1
    running = total
    while cut < len(msgs) and (running > config.MAX_HISTORY_TOKENS or len(msgs) - cut > config.MAX_HISTORY_MESSAGES):
        running -= _message_tokens(msgs[cut])
        cut += 1
    while cut < len(msgs) and msgs[cut].get('role') == 'tool':
        cut += 1
    del msgs[1:cut]


def do_summary_and_compact(state):
    from rich.panel import Panel
    from bardgent.model import print_usage  # local import: model.py doesn't import session at module load

    if len(state.messages) <= 1:
        console.print('[yellow]Nothing to summarize yet.[/yellow]')
        return

    temp_messages = state.messages + [{
        'role': 'user',
        'content': 'Summarize our conversation so far, concisely, keeping key facts/decisions.'
    }]

    response = config.client.chat.completions.create(
        model=config.MODEL,
        messages=temp_messages,
        temperature=config.TEMPERATURE,
        max_tokens=config.RESPONSE_TOKEN_RESERVE,
    )
    summary_text = response.choices[0].message.content or ''
    print_usage(getattr(response, 'usage', None))

    del state.messages[1:]
    state.messages.append({
        'role': 'assistant',
        'content': config.SUMMARY_PREFIX + summary_text
    })

    save_session(state)
    console.print(Panel(summary_text, title='[bold cyan]SUMMARY (history compacted)', border_style='cyan'))

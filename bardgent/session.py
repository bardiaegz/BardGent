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
    # Persist a Gemini-safe transcript so /resume never reloads a broken chain.
    sanitize_history(state)
    if not state.messages[1:]:
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
    sanitize_history(state)
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


def _is_safe_history_start(m):
    """Gemini requires function-call turns to follow a user or tool turn.

    A trimmed/resumed history must not begin with orphan tool results or an
    assistant message that only contains tool_calls (those need a prior user).
    Plain assistant text (e.g. a summary) is allowed as a start.
    """
    if m.get('role') == 'user':
        return True
    if m.get('role') == 'assistant' and not m.get('tool_calls'):
        return True
    return False


def _tool_call_ids(m):
    return [tc.get('id') for tc in (m.get('tool_calls') or []) if tc.get('id')]


def _ensure_tool_call_ids(assistant_msg, tool_msgs):
    """Fill missing tool_call ids and re-bind following tool rows by position.

    Some Gemini/Gemma streams omit ids on tool-call deltas. Without ids our
    sanitizer used to treat the group as incomplete and drop it.
    """
    tcs = assistant_msg.get('tool_calls') or []
    if not tcs:
        return
    for n, tc in enumerate(tcs):
        if not tc.get('id'):
            tc['id'] = f'call_{n}_{abs(hash((tc.get("function") or {}).get("name") or tc.get("name") or n)) % 10**8:08d}'
    # Pair tool results that lack tool_call_id (or have a stale one) by order.
    for n, tool_msg in enumerate(tool_msgs):
        if n < len(tcs):
            expected = tcs[n].get('id')
            if not tool_msg.get('tool_call_id'):
                tool_msg['tool_call_id'] = expected


def _consume_tool_group(body, start):
    """If body[start] is assistant+tool_calls, return (end_index, needed, found).

    end_index is the first index after the contiguous matching tool responses
    (or start+1 when there are none). found maps tool_call_id -> tool message.

    When ids are missing, assigns them and pairs tool rows by position.
    """
    m = body[start]
    tcs = list(m.get('tool_calls') or [])
    # Peek contiguous tool rows so we can repair ids first.
    j = start + 1
    tool_rows = []
    while j < len(body) and body[j].get('role') == 'tool':
        tool_rows.append(body[j])
        j += 1

    _ensure_tool_call_ids(m, tool_rows[:len(tcs)])

    needed = set(_tool_call_ids(m))
    found = {}
    # Prefer id matches; fall back to positional for extras already repaired.
    for tool_msg in tool_rows:
        tcid = tool_msg.get('tool_call_id')
        if tcid in needed and tcid not in found:
            found[tcid] = tool_msg
        elif tcid in needed:
            continue  # duplicate
        else:
            # Unrelated tool row (different parent) — stop; leave for later.
            # But if we already have all needed, ignore trailing extras.
            if set(found) == needed and needed:
                break
            # If ids were never on the tool rows, positional repair should have
            # set them; anything left unmatched means a different group.
            if not needed:
                break
            # tool row doesn't match this group — stop consuming
            break

    # Recompute end as start+1 + number of consumed tool rows that we took.
    end = start + 1
    while end < len(body) and body[end].get('role') == 'tool':
        tcid = body[end].get('tool_call_id')
        if tcid in found and found[tcid] is body[end]:
            end += 1
            continue
        if tcid in needed:
            # duplicate for this group
            end += 1
            continue
        break
    return end, needed, found


def sanitize_history(state):
    """Repair message history so the Gemini OpenAI-compat API accepts it.

    Fixes common corruptions from trim/resume/interrupt:
      - history starting mid tool-call chain
      - orphan `tool` messages
      - assistant tool_calls missing some/all tool responses
      - consecutive user messages (failed turns re-appended)
      - null content on assistant tool-call messages
      - empty conversation (system-only) which Gemini rejects as
        "contents is not specified"
    """
    msgs = state.messages
    if len(msgs) <= 1:
        return

    system = msgs[0] if msgs[0].get('role') == 'system' else None
    body = list(msgs[1:] if system is not None else msgs)
    if not body:
        return

    # Drop leading orphan tool rows (parent assistant call already trimmed away).
    while body and body[0].get('role') == 'tool':
        body.pop(0)

    # If history starts mid tool-loop with a *complete* call group, keep it by
    # inserting a synthetic user turn (Gemini requires user|tool before calls).
    if body and body[0].get('role') == 'assistant' and body[0].get('tool_calls'):
        _end, _needed, _found = _consume_tool_group(body, 0)
        if _needed and set(_found) == _needed:
            body.insert(0, {
                'role': 'user',
                'content': (
                    '[Earlier conversation was trimmed; '
                    'continuing from recent tool activity.]'
                ),
            })

    cleaned = []
    i = 0
    dropped = 0

    while i < len(body):
        m = body[i]
        role = m.get('role')

        # Orphan tool results — never valid without a parent call in `cleaned`.
        if role == 'tool':
            dropped += 1
            i += 1
            continue

        if role == 'user':
            content = m.get('content') or ''
            while i + 1 < len(body) and body[i + 1].get('role') == 'user':
                i += 1
                nxt = body[i].get('content') or ''
                if nxt and nxt != content:
                    content = f'{content}\n\n{nxt}' if content else nxt
            cleaned.append({'role': 'user', 'content': content or '(empty)'})
            i += 1
            continue

        if role == 'assistant' and m.get('tool_calls'):
            end, needed, found = _consume_tool_group(body, i)
            # Gemini: a function-call turn must follow a user or tool response.
            prev = cleaned[-1] if cleaned else None
            prev_ok = (
                prev is not None
                and prev.get('role') in ('user', 'tool')
            )
            complete = bool(needed) and set(found) == needed
            if not prev_ok or not complete:
                log_event(
                    f"SANITIZE: dropping tool-call group "
                    f"(prev_ok={prev_ok}, needed={sorted(needed)}, found={sorted(found)})"
                )
                dropped += 1 + (end - i - 1)
                i = end
                continue

            cleaned.append({
                'role': 'assistant',
                # Gemini is happier with "" than null alongside tool_calls.
                'content': m.get('content') if m.get('content') is not None else '',
                'tool_calls': m['tool_calls'],
            })
            for tc in m['tool_calls']:
                tcid = tc.get('id')
                if tcid in found:
                    tool_msg = found[tcid]
                    # Empty tool content can make Gemini drop the whole turn.
                    if not tool_msg.get('content'):
                        tool_msg = dict(tool_msg)
                        tool_msg['content'] = '(no output)'
                    cleaned.append(tool_msg)
            i = end
            continue

        if role == 'assistant':
            content = m.get('content') if m.get('content') is not None else ''
            cleaned.append({
                'role': 'assistant',
                'content': content if content != '' else '(empty)',
            })
            i += 1
            continue

        # Unknown role — drop rather than poison the request.
        dropped += 1
        i += 1

    if dropped:
        log_event(f"SANITIZE: dropped {dropped} message(s) while repairing history")

    # Never leave system-only history: Gemini returns
    # "GenerateContentRequest.contents: contents is not specified".
    if not cleaned:
        log_event("SANITIZE: history empty after repair; inserting recovery user turn")
        cleaned = [{
            'role': 'user',
            'content': (
                '[Conversation history was reset because it became invalid. '
                'Please continue from here.]'
            ),
        }]
    elif not any(m.get('role') == 'user' for m in cleaned):
        cleaned.insert(0, {
            'role': 'user',
            'content': (
                '[Earlier conversation was trimmed; continuing from recent activity.]'
            ),
        })

    if system is not None:
        state.messages[:] = [system] + cleaned
    else:
        state.messages[:] = cleaned


def trim_history(state):
    """Drop old messages when over token/count limits.

    Never deletes the entire conversation (that yields Gemini
    "contents is not specified"). Always keeps a minimum tail; sanitize
    will re-insert a synthetic user turn if the tail no longer starts
    with one.
    """
    msgs = state.messages
    total = total_history_tokens(state)
    if total <= config.MAX_HISTORY_TOKENS and len(msgs) <= config.MAX_HISTORY_MESSAGES + 1:
        sanitize_history(state)
        return

    min_keep = max(getattr(config, 'MIN_HISTORY_MESSAGES', 6), 2)
    # Highest cut index we may use: always leave at least min_keep body msgs.
    max_cut = max(1, len(msgs) - min_keep)

    cut = 1
    running = total
    while cut < max_cut and (
        running > config.MAX_HISTORY_TOKENS
        or len(msgs) - cut > config.MAX_HISTORY_MESSAGES
    ):
        running -= _message_tokens(msgs[cut])
        cut += 1

    def _can_advance():
        return cut < max_cut and cut < len(msgs)

    # Never leave the window starting mid tool-call group.
    while _can_advance() and msgs[cut].get('role') == 'tool':
        cut += 1

    # Prefer a user (or plain assistant) as the first kept message.
    while _can_advance() and not _is_safe_history_start(msgs[cut]):
        m = msgs[cut]
        cut += 1
        if m.get('role') == 'assistant' and m.get('tool_calls'):
            needed = set(_tool_call_ids(m))
            n_tools = len(m.get('tool_calls') or []) if not needed else None
            seen = 0
            while _can_advance() and msgs[cut].get('role') == 'tool':
                tcid = msgs[cut].get('tool_call_id')
                if needed:
                    if tcid in needed:
                        needed.discard(tcid)
                    cut += 1
                    if not needed:
                        break
                else:
                    cut += 1
                    seen += 1
                    if n_tools is not None and seen >= n_tools:
                        break

    # Final safety: never delete everything after system.
    if cut >= len(msgs):
        cut = max_cut
    cut = min(cut, max_cut)

    if cut > 1:
        del msgs[1:cut]
        log_event(f"TRIM: removed {cut - 1} leading message(s); kept {len(msgs) - 1}")

    sanitize_history(state)


def do_summary_and_compact(state):
    from rich.panel import Panel
    from bardgent.model import print_usage  # local import: model.py doesn't import session at module load

    if len(state.messages) <= 1:
        console.print('[yellow]Nothing to summarize yet.[/yellow]')
        return

    sanitize_history(state)
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

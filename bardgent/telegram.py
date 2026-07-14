"""Optional Telegram delivery of the agent's final answers."""

import re
import json
import time
import requests

from bardgent import config
from bardgent.config import console, log_event
from bardgent.utils import with_retries


def _load_telegram_chat_id():
    if config.TELEGRAM_CHATID_FILE.exists():
        try:
            return json.loads(config.TELEGRAM_CHATID_FILE.read_text(encoding='utf-8')).get('chat_id')
        except (OSError, json.JSONDecodeError):
            return None
    return None


def _save_telegram_chat_id(chat_id):
    try:
        config.TELEGRAM_CHATID_FILE.write_text(json.dumps({'chat_id': chat_id}), encoding='utf-8')
    except OSError as e:
        console.print(f'[dim red]Could not save Telegram chat id: {e}[/dim red]')


def discover_telegram_chat_id(timeout=30):
    """Poll getUpdates until the user messages the bot, then return their chat id."""
    from rich.panel import Panel
    console.print(Panel(
        "Open Telegram, find your bot, and send it any message (e.g. /start).\n"
        f"Waiting up to {timeout}s...",
        title='[bold cyan]Telegram setup', border_style='cyan'))
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            resp = with_retries(requests.get, f'{config.TELEGRAM_API_BASE}/getUpdates', timeout=10, retries=2)
            resp.raise_for_status()
            data = resp.json()
        except (requests.RequestException, ValueError) as e:
            console.print(f'[dim red]Telegram poll failed: {e}[/dim red]')
            time.sleep(2)
            continue
        results = data.get('result', [])
        if results:
            chat = results[-1].get('message', {}).get('chat', {})
            chat_id = chat.get('id')
            if chat_id:
                return chat_id
        time.sleep(2)
    return None


# ---------------------------------------------------------------------------
# Markdown -> Telegram formatting
# ---------------------------------------------------------------------------
# The agent writes normal markdown (**bold**, `code`, "- " bullets, "# "
# headers, [text](url) links). Telegram's legacy Markdown parse_mode doesn't
# understand GitHub-style double-asterisk bold, and MarkdownV2 requires
# escaping a long list of punctuation that shows up in ordinary prose
# constantly (. ! - ( ) etc), which is fragile and easy to get subtly wrong.
# HTML parse_mode is the most forgiving option: escape the raw text first,
# then translate the handful of markdown patterns we actually see into the
# small HTML subset Telegram supports (<b> <i> <code> <pre> <a>).

def _escape_html(text):
    return text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')


def markdown_to_telegram_html(text):
    """Best-effort markdown -> Telegram-HTML conversion. Never raises -
    if a pattern doesn't match anything it's simply left alone."""
    text = _escape_html(text)

    # Fenced code blocks first, so their contents aren't touched by the
    # bold/italic/link passes below.
    text = re.sub(r'```(?:\w*\n)?(.*?)```', lambda m: f'<pre>{m.group(1)}</pre>', text, flags=re.DOTALL)
    text = re.sub(r'`([^`\n]+)`', r'<code>\1</code>', text)

    # Bold (**x** / __x__) before italic, so a leading "* " bullet marker
    # doesn't get mistaken for a stray italic delimiter.
    text = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', text)
    text = re.sub(r'__(.+?)__', r'<b>\1</b>', text)

    # Italic (*x* / _x_) - single delimiter, matched within one line only.
    text = re.sub(r'(?<!\*)\*([^*\n]+?)\*(?!\*)', r'<i>\1</i>', text)
    text = re.sub(r'(?<!_)_([^_\n]+?)_(?!_)', r'<i>\1</i>', text)

    # Links [text](url)
    text = re.sub(r'\[([^\]]+)\]\((https?://[^\s)]+)\)', r'<a href="\2">\1</a>', text)

    # Headers "# Text" / "## Text" -> a bold line (Telegram has no <h*> tags).
    text = re.sub(r'^#{1,6}\s*(.+)$', r'<b>\1</b>', text, flags=re.MULTILINE)

    # Bullets "- item" / "* item" -> a plain bullet character.
    text = re.sub(r'^[*\-]\s+', '• ', text, flags=re.MULTILINE)

    return text


def _chunk_text(text, max_len):
    """Split on blank lines, then bullet-item boundaries, then plain
    newlines, then spaces, in that preference order - so a markdown
    emphasis span (and the HTML tag it becomes), or a single bullet point,
    doesn't get sliced in half across two separate Telegram messages any
    more often than it has to."""
    if len(text) <= max_len:
        return [text]
    chunks = []
    remaining = text
    while len(remaining) > max_len:
        window = remaining[:max_len]
        split_at = window.rfind('\n\n')
        if split_at == -1:
            bullet_matches = list(re.finditer(r'\n(?=[•\-\*]\s)', window))
            if bullet_matches:
                split_at = bullet_matches[-1].start()
        if split_at == -1:
            split_at = window.rfind('\n')
        if split_at == -1:
            split_at = window.rfind(' ')
        if split_at <= 0:
            split_at = max_len
        chunks.append(remaining[:split_at])
        remaining = remaining[split_at:].lstrip('\n')
    if remaining:
        chunks.append(remaining)
    return chunks


def _send_one(text, chat_id, parse_mode=None):
    payload = {'chat_id': chat_id, 'text': text}
    if parse_mode:
        payload['parse_mode'] = parse_mode
    try:
        resp = with_retries(
            requests.post, f'{config.TELEGRAM_API_BASE}/sendMessage',
            json=payload, timeout=10, retries=2,
        )
    except requests.RequestException as e:
        log_event(f"TELEGRAM SEND FAILED (network): {e}")
        return False
    if resp.status_code != 200:
        # Most commonly a 400 from a formatting edge case (e.g. an unclosed
        # tag from an unusual markdown pattern) - not worth retrying as-is,
        # the caller falls back to plain text instead.
        log_event(f"TELEGRAM SEND FAILED ({resp.status_code}): {resp.text[:300]}")
        return False
    return True


def send_telegram_message(text, chat_id, header=None):
    """Send `text` (written as markdown, the way the agent naturally writes)
    to `chat_id`, rendered with Telegram's HTML formatting. If a chunk's
    formatting ever fails to parse, that chunk is resent as plain text
    instead of being silently dropped.

    `header` (optional) is prepended only to the first chunk - e.g. a
    "Scheduled task X finished (ok)" line. When the message needs more than
    one chunk, each one gets a "part i/n" footer so it's clear they belong
    together (and, just as usefully, that two "part 1/1" deliveries close
    together are two separate runs, not one message rendering oddly)."""
    if not config.TELEGRAM_BOT_TOKEN or not chat_id or not text:
        return False

    reserve = (len(header) + 4) if header else 0
    chunks = _chunk_text(text, max(config.TELEGRAM_MAX_LEN - reserve - 24, 500))
    n = len(chunks)

    ok = True
    for i, chunk in enumerate(chunks, 1):
        piece = chunk
        if header and i == 1:
            piece = f"{header}\n\n{piece}"
        if n > 1:
            piece = f"{piece}\n\n— part {i}/{n} —"
        if not _send_one(markdown_to_telegram_html(piece), chat_id, parse_mode='HTML'):
            log_event("TELEGRAM: formatted send failed, retrying this chunk as plain text")
            if not _send_one(piece, chat_id, parse_mode=None):
                ok = False
    return ok
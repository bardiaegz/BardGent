"""Optional Telegram delivery of the agent's final answers."""

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


def send_telegram_message(text, chat_id):
    if not config.TELEGRAM_BOT_TOKEN or not chat_id or not text:
        return False
    ok = True
    for i in range(0, len(text), config.TELEGRAM_MAX_LEN):
        chunk = text[i:i + config.TELEGRAM_MAX_LEN]
        try:
            resp = with_retries(
                requests.post, f'{config.TELEGRAM_API_BASE}/sendMessage',
                json={'chat_id': chat_id, 'text': chunk}, timeout=10, retries=2,
            )
            resp.raise_for_status()
        except requests.RequestException as e:
            log_event(f"TELEGRAM SEND FAILED: {e}")
            ok = False
    return ok

"""Per-agent conversation state, and the y/N/always approval prompt."""

import threading

from bardgent import config
from bardgent.config import console, log_event
from bardgent.session import session_file_name
from bardgent.telegram import _load_telegram_chat_id


class AgentState:
    def __init__(self, system_prompt, name='main', track_session=True, mode='normal'):
        self.name = name
        self.messages = [{'role': 'system', 'content': system_prompt}]
        self.shell_cwd = __import__('os').getcwd()
        self.approved_for_session = set()
        self.approval_lock = threading.RLock()
        self.session_file = (config.SESSION_DIR / session_file_name()) if track_session else None
        self.telegram_enabled = False
        self.telegram_chat_id = _load_telegram_chat_id() if name == 'main' else None
        self.mode = mode if mode in config.VALID_MODES else 'normal'


def ask_approval(state, key, question, dangerous=False):
    """Ask the user to approve an action. 'a' remembers the approval for this session.

    Mode behaviour:
      - auto:   non-dangerous actions are auto-approved with no prompt at all.
                Dangerous actions ALWAYS still prompt, even in auto mode.
      - plan/normal: unchanged, per-action prompts, with 'a' to remember.
    """
    with state.approval_lock:
        if state.mode == 'auto' and not dangerous:
            console.print(f"[dim]auto-approved ({key}) [auto mode][/dim]")
            log_event(f"[{state.name}] approval '{key}' -> auto-mode auto-approved")
            return True
        if dangerous:
            answer = input(f"{question} [y/N]: ").strip().lower()
            log_event(f"[{state.name}] approval(dangerous) '{key}' -> {answer!r}")
            return answer in ('y', 'yes')
        if key in state.approved_for_session:
            console.print(f"[dim]auto-approved ({key})[/dim]")
            return True
        answer = input(f"{question} [Y/n/a=always]: ").strip().lower()
        if answer in ('a', 'always'):
            state.approved_for_session.add(key)
            log_event(f"[{state.name}] approval '{key}' -> always")
            return True
        approved = answer in ('', 'y', 'yes')
        log_event(f"[{state.name}] approval '{key}' -> {approved}")
        return approved

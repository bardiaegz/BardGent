import os
import sys
import json
import time
import glob
import shlex
import logging
import platform
import threading
from pathlib import Path
from urllib.parse import urlparse, parse_qs
import requests
from bs4 import BeautifulSoup
from openai import (
    OpenAI,
    APIError,
    APIConnectionError,
    APITimeoutError,
    RateLimitError,
    InternalServerError,
)
from rich.console import Console, Group
from rich.panel import Panel
from rich.text import Text
from rich.markdown import Markdown
from rich.syntax import Syntax
from rich.live import Live
from rich.spinner import Spinner
from prompt_toolkit import PromptSession
from prompt_toolkit.completion import WordCompleter
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.application import run_in_terminal
import fnmatch
import re
import difflib
import subprocess
import datetime
import shutil
import atexit
import signal
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv

load_dotenv()
load_dotenv(Path.expanduser(Path("~/.bardgent/.env")))

console = Console()

python_path = sys.executable
operating_system = platform.platform()
working_directory = os.getcwd()
home_directory = os.path.expanduser('~')

client = OpenAI(
    base_url='https://generativelanguage.googleapis.com/v1beta/openai/',
    api_key=os.environ.get('GEMINI_API_KEY', '')
)

MODEL = 'gemma-4-31b-it'
TEMPERATURE = 0.2
MAX_ITERATIONS = 30
MAX_HISTORY_MESSAGES = 30
MAX_TOOL_OUTPUT = 24_000
BASH_TIMEOUT_SECONDS = 60
RESPONSE_TOKEN_RESERVE = 8_192        # tokens guaranteed free for the model's own reply
MAX_HISTORY_TOKENS = 180_000           # rough budget for kept history
AUTO_SUMMARY_TOKEN_THRESHOLD = 140_000  # auto-compact via LLM summary above this

# Retry schedule for transient API failures (500s, connection drops, rate
# limits, timeouts). Delays escalate then cap, giving the backend room to
# recover without hammering it or waiting forever.
MODEL_MAX_RETRIES = 10
MODEL_RETRY_DELAYS = [3, 5, 8, 13, 21, 30, 45, 60, 60, 60]
RETRYABLE_ERRORS = (APIError, APIConnectionError, APITimeoutError, RateLimitError, InternalServerError)

# ---------------------------------------------------------------------------
# Modes: plan / normal / auto (Claude Code style)
#   plan   -> read-only tools only; agent must propose a plan and wait
#   normal -> existing per-action approval behaviour (unchanged default)
#   auto   -> everything auto-approved EXCEPT dangerous commands, which still
#             always require an explicit y/N from the user
# ---------------------------------------------------------------------------
VALID_MODES = ('plan', 'normal', 'auto')
READONLY_TOOLS = {'Read', 'Glob', 'Grep', 'WebSearch', 'Fetch', 'read_memory', 'list_memory'}

# ---------------------------------------------------------------------------
# Custom tool permissions file: lets a user pre-approve certain bash command
# prefixes (and, in future, tools) per-project without re-approving every
# session. Lives at .bardgent/permissions.json in the current directory.
# ---------------------------------------------------------------------------
PERMISSIONS_DIR = Path.cwd() / '.bardgent'
PERMISSIONS_FILE = PERMISSIONS_DIR / 'permissions.json'
DEFAULT_PERMISSIONS = {
    "auto_approve_bash_prefixes": [],
    "auto_approve_tools": [],
    "extra_dangerous_patterns": []
}


def load_permissions():
    if PERMISSIONS_FILE.exists():
        try:
            data = json.loads(PERMISSIONS_FILE.read_text(encoding='utf-8'))
        except (OSError, json.JSONDecodeError) as e:
            console.print(f"[dim red]Could not read {PERMISSIONS_FILE}: {e}. Using defaults.[/dim red]")
            return dict(DEFAULT_PERMISSIONS)
        merged = dict(DEFAULT_PERMISSIONS)
        for k in DEFAULT_PERMISSIONS:
            if k in data and isinstance(data[k], list):
                merged[k] = data[k]
        return merged
    try:
        PERMISSIONS_DIR.mkdir(exist_ok=True)
        PERMISSIONS_FILE.write_text(json.dumps(DEFAULT_PERMISSIONS, indent=2), encoding='utf-8')
    except OSError:
        pass
    return dict(DEFAULT_PERMISSIONS)


PERMISSIONS = load_permissions()


def is_permitted_bash_prefix(command):
    cmd = command.strip()
    for prefix in PERMISSIONS.get('auto_approve_bash_prefixes', []):
        prefix = prefix.strip()
        if not prefix:
            continue
        if cmd == prefix or cmd.startswith(prefix + ' '):
            return True
    return False


def is_tool_permitted(name):
    return name in PERMISSIONS.get('auto_approve_tools', [])

CHECKPOINT_REF = 'refs/bardgent/checkpoints'
CHECKPOINT_LOG = PERMISSIONS_DIR / 'checkpoints.json'
CHECKPOINT_INDEX_FILE = PERMISSIONS_DIR / 'checkpoint.index'
GLOBAL_DIR = Path.home() / '.bardgent'
GLOBAL_DIR.mkdir(exist_ok=True)


def _git_root(path):
    try:
        result = subprocess.run(
            ['git', 'rev-parse', '--show-toplevel'],
            cwd=os.path.dirname(os.path.abspath(path)) or '.',
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return None


def _load_checkpoint_log():
    if CHECKPOINT_LOG.exists():
        try:
            return json.loads(CHECKPOINT_LOG.read_text(encoding='utf-8'))
        except (OSError, json.JSONDecodeError):
            return []
    return []


def _save_checkpoint_log(entries):
    try:
        PERMISSIONS_DIR.mkdir(exist_ok=True)
        CHECKPOINT_LOG.write_text(json.dumps(entries, indent=2), encoding='utf-8')
    except OSError as e:
        log_event(f"CHECKPOINT LOG SAVE FAILED: {e}")


def make_git_checkpoint(path, message):
    root = _git_root(path)
    if not root:
        return None
    try:
        env = os.environ.copy()
        env['GIT_INDEX_FILE'] = str(CHECKPOINT_INDEX_FILE)
        subprocess.run(['git', 'add', '-A'], cwd=root, env=env, capture_output=True, text=True, timeout=20)
        tree = subprocess.run(['git', 'write-tree'], cwd=root, env=env, capture_output=True, text=True, timeout=20)
        if tree.returncode != 0:
            log_event(f"CHECKPOINT write-tree failed: {tree.stderr.strip()}")
            return None
        tree_hash = tree.stdout.strip()

        parent_args = []
        parent = subprocess.run(['git', 'rev-parse', CHECKPOINT_REF], cwd=root, capture_output=True, text=True, timeout=5)
        if parent.returncode == 0:
            parent_args = ['-p', parent.stdout.strip()]
        else:
            head = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=root, capture_output=True, text=True, timeout=5)
            if head.returncode == 0:
                parent_args = ['-p', head.stdout.strip()]

        commit = subprocess.run(
            ['git', 'commit-tree', tree_hash, *parent_args, '-m', message],
            cwd=root, env=env, capture_output=True, text=True, timeout=20,
        )
        if commit.returncode != 0:
            log_event(f"CHECKPOINT commit-tree failed: {commit.stderr.strip()}")
            return None
        commit_hash = commit.stdout.strip()

        upd = subprocess.run(['git', 'update-ref', CHECKPOINT_REF, commit_hash], cwd=root, capture_output=True, text=True, timeout=10)
        if upd.returncode != 0:
            log_event(f"CHECKPOINT update-ref failed: {upd.stderr.strip()}")
            return None

        entries = _load_checkpoint_log()
        entries.append({
            'time': time.strftime('%Y-%m-%d %H:%M:%S'),
            'message': message,
            'commit': commit_hash,
            'root': root,
        })
        entries = entries[-100:]  # keep the log from growing forever
        _save_checkpoint_log(entries)
        log_event(f"CHECKPOINT {commit_hash[:10]} ({message})")
        return commit_hash
    except (OSError, subprocess.SubprocessError) as e:
        log_event(f"CHECKPOINT failed: {type(e).__name__}: {e}")
        return None


def list_checkpoints():
    entries = _load_checkpoint_log()
    if not entries:
        return '(no checkpoints yet. checkpoints are created automatically on Write/Edit inside a git repo)'
    lines = []
    for i, e in enumerate(entries, 1):
        lines.append(f"{i}. {e['time']}  {e['commit'][:10]}  {e['message']}")
    return '\n'.join(lines)


def restore_checkpoint(index):
    entries = _load_checkpoint_log()
    try:
        idx = int(index)
    except (TypeError, ValueError):
        return f"Invalid checkpoint index: {index!r}"
    if idx < 1 or idx > len(entries):
        return f"No checkpoint at index {idx}. Use /checkpoints to see valid indices."
    entry = entries[idx - 1]
    root, commit = entry['root'], entry['commit']
    try:
        result = subprocess.run(
            ['git', 'checkout', commit, '--', '.'],
            cwd=root, capture_output=True, text=True, timeout=30,
        )
        if result.returncode != 0:
            return f"Restore failed: {result.stderr.strip()}"
    except (OSError, subprocess.SubprocessError) as e:
        return f"Restore failed: {type(e).__name__}: {e}"
    log_event(f"RESTORED checkpoint #{idx} ({commit[:10]})")
    return f"Working tree in {root} restored to checkpoint #{idx} ({entry['time']}, {commit[:10]}). Your git branch/HEAD/index are untouched. only file contents were overwritten."


# ---------------------------------------------------------------------------
# File-watch / hot context: warn (rather than silently proceed) when a file
# the model already saw via Read() has since changed on disk, so it doesn't
# build an Edit() on stale content.
# ---------------------------------------------------------------------------
_known_mtimes = {}


def _record_mtime(path):
    try:
        _known_mtimes[path] = os.path.getmtime(path)
    except OSError:
        pass


def _stale_warning(path):
    prev = _known_mtimes.get(path)
    if prev is None:
        return ''
    try:
        current = os.path.getmtime(path)
    except OSError:
        return ''
    if current != prev:
        return (f"\n[NOTE: {path} changed on disk since it was last read in this session "
                 f"(external edit, or another process wrote to it). The content above/used "
                 f"here is the CURRENT version. refresh your understanding before making "
                 f"further assumptions about its old contents.]")
    return ''

MEMORY_FILE = GLOBAL_DIR / 'Bardgent.md'
SESSION_DIR = GLOBAL_DIR / "sessions"
SESSION_DIR.mkdir(exist_ok=True)
SESSION_PREFIX = "session_"
SUMMARY_PREFIX = '[Conversation summary so far]: '

BACKUP_DIR = GLOBAL_DIR / "backups"
BACKUP_DIR.mkdir(exist_ok=True)
last_backup = {}

TELEGRAM_BOT_TOKEN = os.environ.get('TELEGRAM_BOT_TOKEN', '')
TELEGRAM_API_BASE = f'https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}'
TELEGRAM_CHATID_FILE = GLOBAL_DIR / 'telegram_chat_id.json'
TELEGRAM_MAX_LEN = 4000  # stay under Telegram's 4096-char hard limit with margin

THOUGHT_TAG_RE = re.compile(r"<(?:thought|think)>.*?</(?:thought|think)>", re.DOTALL | re.IGNORECASE)
THOUGHT_OPEN_RE = re.compile(r"<(?:thought|think)>", re.IGNORECASE)


def remove_thoughts(text):
    if not text:
        return text
    text = THOUGHT_TAG_RE.sub("", text)
    m = THOUGHT_OPEN_RE.search(text)
    if m:
        text = text[:m.start()]
    return text.strip()

def _load_telegram_chat_id():
    if TELEGRAM_CHATID_FILE.exists():
        try:
            return json.loads(TELEGRAM_CHATID_FILE.read_text(encoding='utf-8')).get('chat_id')
        except (OSError, json.JSONDecodeError):
            return None
    return None


def _save_telegram_chat_id(chat_id):
    try:
        TELEGRAM_CHATID_FILE.write_text(json.dumps({'chat_id': chat_id}), encoding='utf-8')
    except OSError as e:
        console.print(f'[dim red]Could not save Telegram chat id: {e}[/dim red]')


def discover_telegram_chat_id(timeout=30):
    """Poll getUpdates until the user messages the bot, then return their chat id."""
    console.print(Panel(
        "Open Telegram, find your bot, and send it any message (e.g. /start).\n"
        f"Waiting up to {timeout}s...",
        title='[bold cyan]Telegram setup', border_style='cyan'))
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            resp = _with_retries(requests.get, f'{TELEGRAM_API_BASE}/getUpdates', timeout=10, retries=2)
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
    if not TELEGRAM_BOT_TOKEN or not chat_id or not text:
        return False
    ok = True
    for i in range(0, len(text), TELEGRAM_MAX_LEN):
        chunk = text[i:i + TELEGRAM_MAX_LEN]
        try:
            resp = _with_retries(
                requests.post, f'{TELEGRAM_API_BASE}/sendMessage',
                json={'chat_id': chat_id, 'text': chunk}, timeout=10, retries=2,
            )
            resp.raise_for_status()
        except requests.RequestException as e:
            log_event(f"TELEGRAM SEND FAILED: {e}")
            ok = False
    return ok


LOG_FILE = GLOBAL_DIR / 'bardgent.log'
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
)

def log_event(msg):
    logging.info(msg)

def Read(file_path):
    path = os.path.abspath(os.path.expanduser(file_path))
    warning = _stale_warning(path)
    with open(path, 'r') as f:
        content = f.read()
    _record_mtime(path)
    return content + warning

class AgentState:
    def __init__(self, system_prompt, name='main', track_session=True, mode='normal'):
        self.name = name
        self.messages = [{'role': 'system', 'content': system_prompt}]
        self.shell_cwd = os.getcwd()
        self.approved_for_session = set()
        self.approval_lock = threading.RLock()
        self.session_file = (SESSION_DIR / session_file_name()) if track_session else None
        self.telegram_enabled = False
        self.telegram_chat_id = _load_telegram_chat_id() if name == 'main' else None
        self.mode = mode if mode in VALID_MODES else 'normal'


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


ADD_STYLE = 'white on dark_green'
DEL_STYLE = 'white on dark_red'


def confirm_diff(old, new, path, tool_name, state):
    """Show a Claude Code style diff (full-width green/red line backgrounds,
    line numbers) of the proposed change and ask for approval."""
    diff = list(difflib.unified_diff(old.splitlines(), new.splitlines(), lineterm=''))
    body = Text()
    bar_width = max(console.width - 6, 40)
    old_no = new_no = 1
    first_hunk = True

    for line in diff:
        if line.startswith(('+++', '---')):
            continue
        if line.startswith('@@'):
            m = re.match(r'@@ -(\d+)(?:,\d+)? \+(\d+)(?:,\d+)? @@', line)
            if m:
                old_no, new_no = int(m.group(1)), int(m.group(2))
            if not first_hunk:
                body.append('   ⋮\n', style='dim')
            first_hunk = False
            continue
        if line.startswith('+'):
            lexer = Path(path).suffix.lstrip('.') or 'python'
            try:
                syntax = Syntax(line[1:], lexer, theme='monokai', line_numbers=False)
                rendered = console.render(syntax)
                for segment in rendered.spans:
                    segment.style = f"{segment.style} on dark_green" if segment.style else "white on dark_green"
                body.append(rendered)
            except Exception:
                body.append(f"{new_no:>4} + {line[1:]}".ljust(bar_width), style=ADD_STYLE)
            body.append('\n')
            new_no += 1
        elif line.startswith('-'):
            lexer = Path(path).suffix.lstrip('.') or 'python'
            try:
                syntax = Syntax(line[1:], lexer, theme='monokai', line_numbers=False)
                rendered = console.render(syntax)
                for segment in rendered.spans:
                    segment.style = f"{segment.style} on dark_red" if segment.style else "white on dark_red"
                body.append(rendered)
            except Exception:
                body.append(f"{new_no:>4} + {line[1:]}".ljust(bar_width), style=DEL_STYLE)
            body.append('\n')
            new_no += 1
        else:
            body.append(f"{new_no:>4}   {line[1:]}\n", style='dim')
            old_no += 1
            new_no += 1

    if not body:
        body = Text('(no changes)', style='dim')
    with state.approval_lock:
        console.print(Panel(body, title=f"[bold yellow]{tool_name}: {path}", border_style='yellow'))
        return ask_approval(state, tool_name, "Apply this change?")


def _make_backup(path, old_content):
    """Save the pre-edit content of a file so it can be restored with Undo()."""
    ts = time.strftime('%Y%m%d_%H%M%S')
    backup_path = BACKUP_DIR / f"{Path(path).name}.{ts}.bak"
    backup_path.write_text(old_content, encoding='utf-8')
    last_backup[path] = backup_path
    return backup_path


def Write(file_path, content, state):
    path = os.path.abspath(os.path.expanduser(file_path))
    old = ''
    existed = os.path.exists(path)
    if existed:
        with open(path, 'r') as f:
            old = f.read()
    if not confirm_diff(old, content, path, 'Write', state):
        return f"Write to {path} rejected by user. Do NOT retry it or a variation of it, continue with what you already have, or ask the user in your final answer."
    if existed:
        _make_backup(path, old)
    with open(path, 'w') as f:
        f.write(content)
    _record_mtime(path)
    log_event(f"[{state.name}] WRITE {path}")
    checkpoint = make_git_checkpoint(path, f"Write: {os.path.basename(path)}")
    checkpoint_note = f' [checkpoint {checkpoint[:10]}]' if checkpoint else ''
    suffix = ' (previous version backed up, use Undo to revert)' if existed else ''
    return f'Wrote to {path}{suffix}{checkpoint_note}'


def find_fuzzy_match(content, old_str, threshold=0.6):
    """Fallback for when Edit's old_str doesn't match verbatim (e.g. minor
    whitespace/indentation drift, or content shifted slightly since the model
    last saw it). Slides a same-line-length window across the file and
    returns the closest-matching block if it clears `threshold`, else None."""
    old_lines = old_str.splitlines()
    content_lines = content.splitlines()
    n = len(old_lines)
    if n == 0 or len(content_lines) < n or len(content_lines) > 20000:
        return None, 0.0
    best_ratio = 0.0
    best_block = None
    for start in range(0, len(content_lines) - n + 1):
        block = '\n'.join(content_lines[start:start + n])
        ratio = difflib.SequenceMatcher(None, block, old_str).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            best_block = block
    if best_ratio >= threshold:
        return best_block, best_ratio
    return None, best_ratio


def Edit(file_path, old_str, new_str, state):
    path = os.path.abspath(os.path.expanduser(file_path))
    with open(path, 'r') as f:
        content = f.read()
    stale = _stale_warning(path)
    count = content.count(old_str)

    if count == 0:
        match, ratio = find_fuzzy_match(content, old_str)
        if match is None:
            hint = f" (closest block was only {ratio:.0%} similar)" if ratio else ''
            return f"Error: old_str not found in {path}{hint}. Re-Read the file and re-check the exact text.{stale}"
        console.print(Panel(
            Text(match), title=f"[bold yellow]Fuzzy match ({ratio:.0%} similar), old_str wasn't found verbatim",
            border_style='yellow'
        ))
        if not ask_approval(state, 'Edit_fuzzy', f"Use this {ratio:.0%}-similar block as the edit target instead?"):
            return f"Edit to {path} rejected, old_str not found exactly and the fuzzy match was declined. Re-Read the file and use the exact text."
        old_str = match
        count = content.count(old_str)
        if count != 1:
            return f"Error: the fuzzy-matched block occurs {count} times in {path}; make old_str more specific."
    elif count > 1:
        return f"Error: old_str matches {count} times in {path}, must be unique"

    new_content = content.replace(old_str, new_str)
    if not confirm_diff(content, new_content, path, 'Edit', state):
        return f"Edit to {path} rejected by user. Do NOT retry it or a variation of it, continue with what you already have, or ask the user in your final answer."
    _make_backup(path, content)
    with open(path, 'w') as f:
        f.write(new_content)
    _record_mtime(path)
    log_event(f"[{state.name}] EDIT {path}")
    checkpoint = make_git_checkpoint(path, f"Edit: {os.path.basename(path)}")
    checkpoint_note = f' [checkpoint {checkpoint[:10]}]' if checkpoint else ''
    return f'Edited {path} (previous version backed up, use Undo to revert){checkpoint_note}{stale}'


def Undo(file_path):
    path = os.path.abspath(os.path.expanduser(file_path))
    backup_path = last_backup.get(path)
    if not backup_path or not backup_path.exists():
        return f"No backup available for {path} in this session. If it's tracked in git, /checkpoints + /restore <n> can roll back the whole project instead."
    with open(path, 'w', encoding='utf-8') as f:
        f.write(backup_path.read_text(encoding='utf-8'))
    _record_mtime(path)
    log_event(f"UNDO {path} <- {backup_path.name}")
    del last_backup[path]
    return f"Restored {path} from backup {backup_path.name}."


def Glob(pattern):
    matches = glob.glob(os.path.expanduser(pattern), recursive=True)
    return '\n'.join(matches) if matches else '(no matches)'


MAX_GREP_MATCHES = 200
SKIP_DIRS = {'__pycache__', 'node_modules', 'venv', '.venv', 'dist', 'build'}


def Grep(pattern, path='.', include=None):
    try:
        regex = re.compile(pattern)
    except re.error as e:
        return f"Error: invalid regex: {e}"
    root = os.path.abspath(os.path.expanduser(path))
    matches = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if not d.startswith('.') and d not in SKIP_DIRS]
        for filename in sorted(filenames):
            if include and not fnmatch.fnmatch(filename, include):
                continue
            file_path = os.path.join(dirpath, filename)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    for lineno, line in enumerate(f, 1):
                        if regex.search(line):
                            rel = os.path.relpath(file_path, root)
                            matches.append(f"{rel}:{lineno}: {line.rstrip()}")
                            if len(matches) >= MAX_GREP_MATCHES:
                                return '\n'.join(matches) + f"\n(stopped at {MAX_GREP_MATCHES} matches)"
            except (UnicodeDecodeError, OSError):
                continue
    return '\n'.join(matches) if matches else '(no matches)'

DANGEROUS_PATTERNS = [
    r'\brm\b', r'\brmdir\b', r'\bmv\b', r'\bdd\b',
    r'\bsudo\b', r'\bchmod\b', r'\bchown\b',
    r'\bkill\b', r'\bpkill\b', r'\bkillall\b',
    r'>\s*/', r'\btruncate\b', r'\bmkfs\b',
    r'--force\b', r'--hard\b',
    r'\bshutdown\b', r'\breboot\b',
]

DANGEROUS_BINARIES = {
    'rm', 'rmdir', 'mv', 'dd', 'sudo', 'chmod', 'chown',
    'kill', 'pkill', 'killall', 'truncate', 'mkfs',
    'shutdown', 'reboot', 'mkswap', 'fdisk', 'parted',
}
CODE_INTERPRETERS = {'python', 'python3', 'perl', 'ruby', 'node', 'php'}
INLINE_EXEC_FLAGS = {'-c', '-e'}
SPLIT_OPERATORS = {';', '&&', '||', '|'}


def command_segments(command):
    """Split a shell command string into its sub-commands on ; && || |."""
    try:
        tokens = shlex.split(command)
    except ValueError:
        return [command]
    segments, current = [], []
    for tok in tokens:
        if tok in SPLIT_OPERATORS:
            if current:
                segments.append(current)
            current = []
        else:
            current.append(tok)
    if current:
        segments.append(current)
    return [' '.join(seg) for seg in segments] or [command]


def is_dangerous(command):
    extra_patterns = PERMISSIONS.get('extra_dangerous_patterns', [])
    for seg in command_segments(command):
        try:
            words = shlex.split(seg)
        except ValueError:
            words = seg.split()
        if not words:
            continue
        first = os.path.basename(words[0])
        if first in DANGEROUS_BINARIES:
            return True
        if first in CODE_INTERPRETERS and any(f in words for f in INLINE_EXEC_FLAGS):
            return True
        if any(re.search(p, seg) for p in DANGEROUS_PATTERNS):
            return True
        if any(re.search(p, seg) for p in extra_patterns):
            return True
    return False


CWD_MARKER = '__BARDGENT_CWD__'
console_lock = threading.RLock()


def Bash(command, state, timeout=BASH_TIMEOUT_SECONDS):
    danger = is_dangerous(command)
    permitted = (not danger) and is_permitted_bash_prefix(command)
    first_word = (command.strip().split() or [''])[0]
    key = f"Bash:{first_word}"
    with state.approval_lock:
        if permitted:
            console.print(f"[dim]auto-approved ({key}) [permissions.json][/dim]")
            log_event(f"[{state.name}] approval '{key}' -> permitted via permissions.json")
        else:
            if danger or key not in state.approved_for_session:
                color = "red" if danger else "yellow"
                title = "Bash wants to run (DANGEROUS)" if danger else "Bash wants to run"
                console.print(Panel(command, title=f"[bold {color}]{title}",
                                    subtitle=f"[dim]in {state.shell_cwd}", border_style=color))
            if not ask_approval(state, key, "Run this command?", dangerous=danger):
                return "Command rejected by user. Do NOT retry it or a variation of it, continue with what you already have, or ask the user in your final answer."
    # append a marker echoing $PWD so `cd` persists to the next Bash call
    wrapped = command + f'\nprintf "\\n{CWD_MARKER}%s" "$PWD"'
    try:
        result = subprocess.run(
            wrapped, shell=True, capture_output=True, text=True,
            cwd=state.shell_cwd, timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        log_event(f"[{state.name}] BASH TIMEOUT after {timeout}s: {command!r}")
        return (f"Command timed out after {timeout}s and was killed. "
                f"If this command is expected to run long, break it into smaller steps "
                f"or run it in the background with `&` and poll for completion.")
    stdout, sep, after = result.stdout.rpartition(CWD_MARKER)
    if sep:
        new_dir = after.strip()
        if new_dir and os.path.isdir(new_dir):
            state.shell_cwd = new_dir
        stdout = stdout[:-1] if stdout.endswith('\n') else stdout
    else:
        stdout = result.stdout
    log_event(f"[{state.name}] BASH: {command!r} (exit={result.returncode})")
    return stdout + result.stderr


SYSTEM_INFO = f"""[CRITICAL SYSTEM INFO]:
- Python Executable Path: {python_path}
- Operating System: {operating_system}
- Current Working Directory: {working_directory}
- User Home Directory: {home_directory}"""


def read_memory():
    if MEMORY_FILE.exists():
        # console.print(f'\n[bold green]⚙ TOOL:[/bold green] READING MEMORY FROM Bardgent.md\n')
        return MEMORY_FILE.read_text(encoding='utf-8')
    return ''


def save_memory(text: str):
    text = text.strip()
    memories = set()
    if MEMORY_FILE.exists():
        for line in MEMORY_FILE.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line.startswith("-"):
                memories.add(line[1:].strip().lower())

    if text.lower() in memories:
        return "Memory already exists."

    with open(MEMORY_FILE, "a", encoding="utf-8") as f:
        f.write(f"\n- {text}\n")

    # console.print(f'\n[bold green]⚙ TOOL:[/bold green] SAVING MEMORY TO Bardgent.md\n')
    log_event(f"MEMORY SAVE: {text}")
    return "Memory saved."


def list_memory():
    """Return saved memories as a numbered list so the user/model can reference an index to delete."""
    if not MEMORY_FILE.exists():
        return '(no memories saved)'
    mem_lines = [l.strip() for l in MEMORY_FILE.read_text(encoding='utf-8').splitlines() if l.strip().startswith('-')]
    if not mem_lines:
        return '(no memories saved)'
    return '\n'.join(f'{i}. {l[1:].strip()}' for i, l in enumerate(mem_lines, 1))


def delete_memory(index):
    """Delete a memory by its 1-based index as shown by list_memory()."""
    if not MEMORY_FILE.exists():
        return 'No memory file exists.'
    try:
        idx = int(index)
    except (TypeError, ValueError):
        return f"Error: index must be an integer, got {index!r}. Call list_memory() first."
    lines = MEMORY_FILE.read_text(encoding='utf-8').splitlines()
    mem_line_positions = [i for i, l in enumerate(lines) if l.strip().startswith('-')]
    if idx < 1 or idx > len(mem_line_positions):
        return f"Error: no memory at index {idx}. Use list_memory() to see valid indices."
    removed = lines[mem_line_positions[idx - 1]]
    del lines[mem_line_positions[idx - 1]]
    MEMORY_FILE.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    log_event(f"MEMORY DELETE #{idx}: {removed}")
    return f'Deleted memory #{idx} ({removed.strip("- ")}).'
def _with_retries(func, *args, retries=3, backoff=1.5, **kwargs):
    last_exc = None
    for attempt in range(retries):
        try:
            return func(*args, **kwargs)
        except requests.RequestException as e:
            last_exc = e
            if attempt < retries - 1:
                time.sleep(backoff ** attempt)
    raise last_exc


def WebSearch(query):
    console.print(f'Web Search: [bold green]{query}[/bold green]\n')
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
    try:
        resp = _with_retries(
            requests.post, 'https://html.duckduckgo.com/html/',
            data={'q': query}, headers=headers, timeout=10,
        )
        resp.raise_for_status()
    except requests.RequestException as e:
        return f"Web search failed after retries: {type(e).__name__}: {e}"
    soup = BeautifulSoup(resp.text, 'html.parser')
    results = []
    for r in soup.select('.result')[:8]:
        a = r.select_one('a.result__a')
        if not a:
            continue
        url = a.get('href', '')
        uddg = parse_qs(urlparse(url).query).get('uddg')
        if uddg:
            url = uddg[0]
        snippet = r.select_one('.result__snippet')
        entry = f"{a.get_text(strip=True)}\n{url}"
        if snippet:
            entry += f"\n{snippet.get_text(strip=True)}"
        results.append(entry)
    return '\n\n'.join(results) if results else '(no results)'


def Fetch(link, state):
    console.print(Panel(link, title='[bold yellow]Fetch wants to run', border_style='yellow'))

    if not is_tool_permitted('Fetch') and not ask_approval(state, 'Fetch', 'Fetch this page?'):
        return 'Fetch rejected by user.'

    # console.print(f'\n[bold green]⚙ TOOL:[/bold green] Fetch\n')

    headers = {
        'User-Agent': (
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) '
            'AppleWebKit/537.36 Chrome/120 Safari/537.36'
        )
    }

    try:
        resp = _with_retries(requests.get, link, headers=headers, timeout=10)
        if resp.status_code == 403:
            return f"Could not fetch page (403 Forbidden): {link}"
        resp.raise_for_status()
    except requests.RequestException as e:
        return f"Fetch failed after retries: {type(e).__name__}: {e}"

    soup = BeautifulSoup(resp.text, 'html.parser')
    for tag in soup(['script', 'style', 'noscript']):
        tag.decompose()
    return soup.get_text(separator='\n', strip=True)


def print_welcome():
    console.print(f"[bold italic magenta]Welcome to Bardgent[/bold italic magenta]!")
    console.print("Type 'exit' or 'quit' to leave.")
    console.print("[dim]Shift+Tab cycles mode (normal -> auto -> plan), or use /normal, /auto, /plan.[/dim]")
    # if IS_WARP:
    #     console.print(
    #         "[dim]Warp terminal detected: it doesn't support a pinned bottom status bar, "
    #         "so context/mode will be shown as a line instead.[/dim]"
    #     )
    console.print()


DATETIME = datetime.datetime.now().astimezone()

SYSTEM_PROMPT = f"""
You are a helpful coding agent.
Your name is Bardgent made by Bardia.
Don't use emoji.

DATETIME: {DATETIME.strftime('%Y-%B-%d %I:%M %p %Z')}

{SYSTEM_INFO}

You have access to these tools:

File tools:
- Read(file_path): Read the content of a file.
- Write(file_path, content): Write or overwrite a file. Always show the user a diff and ask for approval before writing. Automatically backed up, the user or you can call Undo(file_path) to revert.
- Edit(file_path, old_str, new_str): Replace an exact unique string inside a file. Prefer Edit over Write for small changes. Automatically backed up.
- Undo(file_path): Restore a file to how it was before the most recent Write/Edit in this session.
- Glob(pattern): Find files by name using glob patterns.
- Grep(pattern, path, include): Search inside files using regex.

Execution tools:
- Bash(command): Execute shell commands. The shell keeps its working directory between calls, so `cd` persists. Commands are killed after {BASH_TIMEOUT_SECONDS}s if they hang.

Web tools:
- WebSearch(query): Search the web and return results.
- Fetch(link): Fetch and extract text from a web page.

Memory tools:
- read_memory(): Read long-term memory.
- save_memory(memory): Save useful user facts or preferences.
- list_memory(): List saved memories with their index numbers.
- delete_memory(index): Delete a memory by the index shown in list_memory().

Delegation:
- Task(prompt): Delegate a single self-contained, multi-step subtask (e.g. a broad
  codebase search, a multi-file investigation, or a repetitive bulk operation) to an
  isolated sub-agent. The sub-agent has its own context and its own copy of the
  file/exec/web tools (but cannot itself call Task/Tasks). It returns only its final
  result to you, which keeps your own context small. Use it when a subtask would
  otherwise take many tool calls whose intermediate output you don't need to see.
- Tasks(prompts): Like Task, but delegates MULTIPLE independent sub-tasks that run
  CONCURRENTLY. Use this instead of several Task calls when the sub-tasks don't
  depend on each other's results (e.g. investigate 3 unrelated modules at once).

Modes (the user controls this with /plan, /normal, /auto):
- plan: you may only use read-only tools (Read, Glob, Grep, WebSearch, Fetch,
  read_memory, list_memory). Any mutating tool call is blocked with an explanation.
  Investigate, then present a concrete step-by-step plan in your final answer and
  stop, wait for the user to review it and switch modes before you execute anything.
- normal: default behaviour. Every Write/Edit/Bash/etc. asks the user for approval
  (they can approve once, always for that action this session, or reject).
- auto: everything is auto-approved WITHOUT prompting, except genuinely dangerous
  shell commands (rm, sudo, chmod, kill, etc.), which always still require an
  explicit yes from the user no matter the mode. Use plain, direct action in auto
  mode, you won't be interrupted for routine approvals.

Checkpoints:
- Every applied Write/Edit is backed up automatically (Undo(file_path) reverts the
  single most recent change to that file).
- If the file lives inside a git repository, a full project-wide checkpoint is also
  silently snapshotted (the user can list them with /checkpoints and roll the whole
  working tree back to one with /restore <n>, this never touches their git branch,
  HEAD, or staged changes).

Rules:

- Always use this exact Python executable path when executing Python files:
  {python_path}

- When the user gives a relative path (for example Desktop/foo/app.py),
  first try it relative to the current working directory and home directory before searching.

- Before modifying files:
  - Prefer Edit for small targeted changes.
  - Use Write only when replacing the entire file or creating a new file.
  - Always review the diff shown by the tool and respect the user's approval.
  - If Edit reports old_str wasn't found, re-Read the file before retrying blindly,
    a fuzzy-match fallback may offer the closest block, but exact text is preferred.

- For exploring a codebase:
  - Use Glob to discover files instead of guessing filenames.
  - Use Grep to search for functions, classes, variables, or keywords.

- For Bash:
  - Think before executing commands.
  - Avoid destructive commands unless explicitly requested.
  - The Bash working directory persists between calls.

- After every tool call:
  - Read and understand the result.
  - Decide whether another tool call is needed.
  - Only provide the final answer when the task is complete.

- For questions that may depend on previous conversations:
  call read_memory() before answering.

- Only call save_memory() when the user explicitly tells you a new fact about themselves.
- Never save information inferred by you.
- Never save information retrieved from read_memory().

You are a coding agent. Prefer taking action with tools over only explaining what could be done.
"""


def session_file_name():
    return f"{SESSION_PREFIX}{time.strftime('%Y-%m-%d_%H-%M-%S')}.json"


def session_title(msgs):
    for m in msgs:
        if m.get('role') == 'user' and m.get('content'):
            first_line = m['content'].strip().splitlines()[0]
            return first_line[:60] + ('…' if len(first_line) > 60 else '')
    for m in msgs:
        if m.get('content'):
            first_line = str(m['content']).removeprefix(SUMMARY_PREFIX).strip().splitlines()[0]
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
    for path in glob.glob(str(SESSION_DIR / f'{SESSION_PREFIX}*.json')):
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if data.get('messages'):
                sessions.append((path, data))
        except (OSError, json.JSONDecodeError):
            continue
    sessions.sort(key=lambda s: s[1].get('updated', ''), reverse=True)
    return sessions


COMMANDS = {
    '/summary': 'Summarize the current conversation',
    '/model': 'Show the current model, or switch: /model <n>',
    '/clear': 'Clear history and start a new session',
    '/resume': 'Pick a past session and resume it',
    '/telegram': 'Toggle sending the agent\'s final answers to Telegram',
    '/plan': 'Switch to PLAN mode (read-only exploration, agent proposes a plan)',
    '/normal': 'Switch to NORMAL mode (approve each action, default)',
    '/auto': 'Switch to AUTO mode (auto-approve everything except dangerous commands)',
    '/mode': 'Show the current mode',
    '/checkpoints': 'List recent git checkpoints (auto-created on Write/Edit)',
    '/restore': 'Restore the working tree to a checkpoint: /restore <n>',
    '/exit': 'Quit Bardgent',
}


def replay_transcript(state):
    for m in state.messages[1:]:
        role = m.get('role')
        content = m.get('content') or ''
        if role == 'user':
            console.print(Text('USER: ', style='bold green') + Text(content))
        elif role == 'assistant':
            clean = content.removeprefix(SUMMARY_PREFIX).strip()
            if content.startswith(SUMMARY_PREFIX):
                console.print(Panel(Text(clean), title='[bold cyan]SUMMARY', border_style='cyan'))
            elif clean:
                console.print(Group(Text('AGENT:', style='bold cyan'), Markdown(clean)))


def resume_session(state):
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


def do_summary_and_compact(state):
    if len(state.messages) <= 1:
        console.print('[yellow]Nothing to summarize yet.[/yellow]')
        return

    temp_messages = state.messages + [{
        'role': 'user',
        'content': 'Summarize our conversation so far, concisely, keeping key facts/decisions.'
    }]

    response = client.chat.completions.create(
        model=MODEL,
        messages=temp_messages,
        temperature=TEMPERATURE,
        max_tokens=RESPONSE_TOKEN_RESERVE,
    )
    summary_text = response.choices[0].message.content or ''
    print_usage(getattr(response, 'usage', None))

    del state.messages[1:]
    state.messages.append({
        'role': 'assistant',
        'content': SUMMARY_PREFIX + summary_text
    })

    save_session(state)
    console.print(Panel(summary_text, title='[bold cyan]SUMMARY (history compacted)', border_style='cyan'))
    # draw_status_bar(state, force=True)


# Order that shift+tab cycles through.
MODE_CYCLE = ['normal', 'auto', 'plan']
_MODE_COLOR = {'plan': 'cyan', 'normal': 'white', 'auto': 'bold red'}


def switch_mode(state, new_mode, announce=True):
    """Single source of truth for changing state.mode, used by /plan,
    /normal, /auto, and the shift+tab shortcut alike, so all three stay in
    sync with the same messaging, logging, and status-bar redraw."""
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
    # draw_status_bar(state, force=True)


def handle_command(user_input, state):
    global MODEL
    cmd = user_input.strip().lower()

    if cmd == '/model' or cmd.startswith('/model '):
        parts = user_input.strip().split(maxsplit=1)
        if len(parts) == 1:
            console.print(f'Current model: [bold cyan]{MODEL}[/bold cyan]')
        else:
            new_model = parts[1].strip()
            try:
                available = [m.id for m in client.models.list().data]
            except Exception as e:
                available = None
                console.print(f'[dim]Could not reach server to verify model list: {e}[/dim]')
            if available and new_model not in available:
                console.print(f"[yellow]Warning: '{new_model}' was not found on the server.[/yellow]")
                console.print(f"[dim]Available: {', '.join(available)}[/dim]")
                if not ask_approval(state, 'model_switch_unverified', "Switch to it anyway?"):
                    return 'handled'
            MODEL = new_model
            console.print(f'[bold green]Model switched to {MODEL}.[/bold green]')
        return 'handled'

    if cmd == '/clear':
        del state.messages[1:]
        state.session_file = SESSION_DIR / session_file_name()
        console.clear()
        print_welcome()
        console.print('[bold green]New session started.[/bold green]')
        # draw_status_bar(state, force=True)
        return 'handled'

    if cmd == '/resume':
        resume_session(state)
        # draw_status_bar(state, force=True)
        return 'handled'

    if cmd in ('/plan', '/normal', '/auto'):
        switch_mode(state, cmd[1:])
        return 'handled'

    if cmd == '/mode':
        console.print(f'Current mode: [bold cyan]{state.mode}[/bold cyan]')
        return 'handled'

    if cmd == '/checkpoints':
        console.print(Panel(list_checkpoints(), title='[bold cyan]Git checkpoints', border_style='cyan'))
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
        sys.exit(0)

    if cmd == '/summary':
        do_summary_and_compact(state)
        return 'handled'

    if cmd == '/telegram':
        if not TELEGRAM_BOT_TOKEN:
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

TOOLS = [
    {'type': 'function', 'function': {
        'name': 'read_memory', 'description': 'Read long-term memory.',
        'parameters': {'type': 'object', 'properties': {}},
    }},
    {'type': 'function', 'function': {
        'name': 'save_memory', 'description': 'Save useful user facts or preferences.',
        'parameters': {'type': 'object', 'properties': {
            'memory': {'type': 'string'}
        }, 'required': ['memory']},
    }},
    {'type': 'function', 'function': {
        'name': 'list_memory', 'description': 'List saved memories with their index numbers.',
        'parameters': {'type': 'object', 'properties': {}},
    }},
    {'type': 'function', 'function': {
        'name': 'delete_memory', 'description': 'Delete a saved memory by the index shown in list_memory().',
        'parameters': {'type': 'object', 'properties': {
            'index': {'type': 'integer', 'description': '1-based index from list_memory()'}
        }, 'required': ['index']},
    }},
    {'type': 'function', 'function': {
        'name': 'Fetch', 'description': 'Fetch the content of a web page',
        'parameters': {'type': 'object', 'properties': {
            'link': {'type': 'string', 'description': 'the link of the web page to fetch'}
        }, 'required': ['link']},
    }},
    {'type': 'function', 'function': {
        'name': 'WebSearch',
        'description': 'Search the web (DuckDuckGo), returns titles, URLs and snippets. Use Fetch afterwards to read a promising result.',
        'parameters': {'type': 'object', 'properties': {
            'query': {'type': 'string', 'description': 'the search query'}
        }, 'required': ['query']},
    }},
    {'type': 'function', 'function': {
        'name': 'Read', 'description': 'Read a file from disk',
        'parameters': {'type': 'object', 'properties': {'file_path': {'type': 'string'}}, 'required': ['file_path']}
    }},
    {'type': 'function', 'function': {
        'name': 'Write', 'description': 'Write (overwrite) full content to a file. Backs up any existing file first.',
        'parameters': {'type': 'object', 'properties': {
            'file_path': {'type': 'string', 'description': 'the path of the file to write to'},
            'content': {'type': 'string', 'description': 'the content to write to the file'}
        }, 'required': ['file_path', 'content']}
    }},
    {'type': 'function', 'function': {
        'name': 'Edit', 'description': 'Replace an exact string match inside a file (must match exactly once). Backs up the file first.',
        'parameters': {'type': 'object', 'properties': {
            'file_path': {'type': 'string'},
            'old_str': {'type': 'string', 'description': 'exact text to find'},
            'new_str': {'type': 'string', 'description': 'text to replace it with'}
        }, 'required': ['file_path', 'old_str', 'new_str']}
    }},
    {'type': 'function', 'function': {
        'name': 'Undo', 'description': 'Restore a file to its state before the most recent Write/Edit in this session.',
        'parameters': {'type': 'object', 'properties': {
            'file_path': {'type': 'string'}
        }, 'required': ['file_path']}
    }},
    {'type': 'function', 'function': {
        'name': 'Glob', 'description': 'List/search files matching a glob pattern, e.g. "**/*.py"',
        'parameters': {'type': 'object', 'properties': {'pattern': {'type': 'string'}}, 'required': ['pattern']}
    }},
    {'type': 'function', 'function': {
        'name': 'Grep', 'description': 'Search file contents for a regex pattern, returns matches as path:line_number: line',
        'parameters': {'type': 'object', 'properties': {
            'pattern': {'type': 'string', 'description': 'regex pattern to search for'},
            'path': {'type': 'string', 'description': 'directory to search in (default: current directory)'},
            'include': {'type': 'string', 'description': 'only search files matching this glob, e.g. "*.py"'}
        }, 'required': ['pattern']}
    }},
    {'type': 'function', 'function': {
        'name': 'Bash', 'description': f'Execute a shell command (killed after {BASH_TIMEOUT_SECONDS}s if it hangs)',
        'parameters': {'type': 'object', 'properties': {'command': {'type': 'string', 'description': 'the command to execute'}}, 'required': ['command']}
    }},
    {'type': 'function', 'function': {
        'name': 'Task',
        'description': 'Delegate a single self-contained sub-task (e.g. large codebase search, multi-step investigation) to an isolated sub-agent. Returns only its final result.',
        'parameters': {'type': 'object', 'properties': {
            'prompt': {'type': 'string', 'description': 'the full task for the sub-agent to complete'}
        }, 'required': ['prompt']}
    }},
    {'type': 'function', 'function': {
        'name': 'Tasks',
        'description': 'Delegate MULTIPLE independent sub-tasks to isolated sub-agents that run CONCURRENTLY (in parallel), not one after another. Use this instead of several Task calls when the sub-tasks do not depend on each other (e.g. investigate 3 different modules, run 2 independent checks). Returns each sub-agent\'s final result, labeled by task number, in the original order.',
        'parameters': {'type': 'object', 'properties': {
            'prompts': {
                'type': 'array', 'items': {'type': 'string'},
                'description': 'list of independent, self-contained task descriptions, one per sub-agent'
            }
        }, 'required': ['prompts']}
    }},
]

# Sub-agents get every tool except Task/Tasks themselves, to prevent recursive spawning.
SUBAGENT_TOOLS = [t for t in TOOLS if t['function']['name'] not in ('Task', 'Tasks')]

# Required-argument schema, used to validate model-supplied tool args before
# dispatch instead of letting a missing key raise KeyError mid-turn.
REQUIRED_ARGS = {t['function']['name']: t['function']['parameters'].get('required', []) for t in TOOLS}


def validate_args(name, args):
    missing = [k for k in REQUIRED_ARGS.get(name, []) if k not in args]
    if missing:
        return f"Error: missing required argument(s) {missing} for tool '{name}'. Re-check the tool schema and try again."
    return None


def dispatch_tool(name, args, state):
    """Single place that both the main loop and sub-agents call to run a tool.
    Validates arguments and isolates exceptions per-tool-call so one bad call
    can't take down the rest of the turn."""
    if state.mode == 'plan' and name not in READONLY_TOOLS:
        msg = (
            f"'{name}' is not available in PLAN MODE. You may only explore using "
            f"{', '.join(sorted(READONLY_TOOLS))}. Investigate as needed, then present "
            f"your plan in your final answer and wait, the user will switch you to "
            f"normal or auto mode (/normal or /auto) to let you execute it."
        )
        log_event(f"[{state.name}] PLAN MODE BLOCKED '{name}'")
        return msg
    err = validate_args(name, args)
    if err:
        log_event(f"[{state.name}] VALIDATION FAILED for '{name}': {err}")
        return err
    try:
        if name == 'Task':
            return run_subagent(args['prompt'])
        elif name == 'Tasks':
            prompts = args.get('prompts') or []
            if not isinstance(prompts, list) or not prompts:
                return "Error: 'prompts' must be a non-empty list of task strings."
            return run_subagents_parallel(prompts)
        elif name == 'read_memory':
            return read_memory()
        elif name == 'save_memory':
            return save_memory(args['memory'])
        elif name == 'list_memory':
            return list_memory()
        elif name == 'delete_memory':
            return delete_memory(args['index'])
        elif name == 'WebSearch':
            return WebSearch(args['query'])
        elif name == 'Fetch':
            return Fetch(args['link'], state)
        elif name == 'Read':
            return Read(args['file_path'])
        elif name == 'Write':
            return Write(args['file_path'], args['content'], state)
        elif name == 'Edit':
            return Edit(args['file_path'], args['old_str'], args['new_str'], state)
        elif name == 'Undo':
            return Undo(args['file_path'])
        elif name == 'Glob':
            return Glob(args['pattern'])
        elif name == 'Grep':
            return Grep(args['pattern'], args.get('path', '.'), args.get('include'))
        elif name == 'Bash':
            return Bash(args['command'], state)
        else:
            return 'Unknown tool'
    except Exception as e:
        log_event(f"[{state.name}] TOOL '{name}' RAISED: {type(e).__name__}: {e}")
        return f"Error running tool '{name}': {type(e).__name__}: {e}. Do not blindly retry, adjust the arguments or approach."


def render_agent(text):
    return Group(Text('AGENT:', style='bold cyan'), Markdown(text))


def stream_agent_response(messages, tools):
    """Retry wrapper around _stream_agent_response_once(). Retries on
    transient API errors (500s, timeouts, connection drops, rate limits)
    up to MODEL_MAX_RETRIES times, waiting according to MODEL_RETRY_DELAYS
    between attempts. Anything else (bad args, programming errors) is not
    retried and propagates immediately."""
    for attempt in range(1, MODEL_MAX_RETRIES + 1):
        try:
            return _stream_agent_response_once(messages, tools)
        except RETRYABLE_ERRORS as e:
            log_event(f"MODEL CALL FAILED (attempt {attempt}/{MODEL_MAX_RETRIES}): {type(e).__name__}: {e}")
            if attempt == MODEL_MAX_RETRIES:
                console.print(f"[bold red]Giving up after {MODEL_MAX_RETRIES} attempts: {type(e).__name__}: {e}[/bold red]")
                raise
            delay = MODEL_RETRY_DELAYS[min(attempt - 1, len(MODEL_RETRY_DELAYS) - 1)]
            console.print(
                f"[bold red]API error (attempt {attempt}/{MODEL_MAX_RETRIES})[/bold red]\n "
                # f"{type(e).__name__}: {e}"
                f"[yellow]Retrying in {delay}s...[/yellow]"
            )
            time.sleep(delay)


def _stream_agent_response_once(messages, tools):
    """
    Calls the model with stream=True and renders the reply live:
      - Shows a spinner ("Thinking...") until the first chunk arrives.
      - As text tokens stream in, live-updates the rendered markdown
        (word-by-word streaming).
      - Tool call arguments arrive split across many chunks. Each chunk
        only carries a fragment of the JSON string (e.g. '{"file' then
        '_path": "a.py' then '"}'), plus an `index` saying which tool
        call it belongs to (a response can stream several tool calls
        interleaved). We ONLY concatenate fragments by index here and
        never json.loads() until the stream is fully consumed, parsing
        a partial fragment is exactly what caused the "incomplete JSON /
        bad chunking" error.

    Returns: (final_text, tool_calls, finish_reason)
      final_text    -> str, the assistant's plain text content (may be '')
      tool_calls    -> list of {'id', 'name', 'arguments'} dicts, in order
      finish_reason -> str or None
    """
    stream = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        tools=tools,
        temperature=TEMPERATURE,
        max_tokens=RESPONSE_TOKEN_RESERVE,
        stream=True,
        stream_options={'include_usage': True},
    )

    content_parts = []
    tool_calls = {}          # index -> {'id':..., 'name':..., 'arguments': ''}
    finish_reason = None
    usage = None
    has_output = False

    spinner = Spinner('dots', text=Text(' Thinking...', style='cyan'))

    status_state = _current_state_for_resize[0]

    # auto_refresh=False is deliberate: Live's default mode repaints itself
    # from a background thread on a timer, which was racing with the manual
    # sys.stdout writes in # draw_status_bar() below (both threads writing to
    # stdout at once). That race is what produced the raw escape-code text
    # ("...[45;1H[2K[32m Context: ...") showing up as garbage, repeated many
    # times, in the middle of the chat. With auto_refresh off, Live only
    # repaints when we explicitly call live.refresh(), from this same
    # (main) thread, right after # draw_status_bar() -- so the two writers are
    # fully serialized and can never interleave.
    with Live(spinner, console=console, refresh_per_second=12, transient=False, auto_refresh=False) as live:
        # draw_status_bar(status_state, force=True)
        live.refresh()
        for chunk in stream:
            if getattr(chunk, 'usage', None):
                usage = chunk.usage
            if not chunk.choices:
                continue
            choice = chunk.choices[0]
            delta = choice.delta

            if choice.finish_reason:
                finish_reason = choice.finish_reason

            if delta and delta.content:
                content_parts.append(delta.content)
                display_text = remove_thoughts(''.join(content_parts))
                if display_text:
                    # first real (non-thought) text has arrived, switch from
                    # the loading spinner to the actual rendered answer
                    has_output = True
                    live.update(render_agent(display_text))
                    live.refresh()
                else:
                    # still inside a <thought>/<think> block, keep showing
                    # the loading spinner instead of raw thinking text, and
                    # do NOT mark has_output yet (so a TOOL call that follows
                    # a thought still gets its "⚙ TOOL:" announcement below)
                    live.update(spinner)
                    live.refresh()

            if delta and delta.tool_calls:
                for tc_delta in delta.tool_calls:
                    idx = tc_delta.index
                    entry = tool_calls.setdefault(idx, {'id': None, 'name': None, 'arguments': ''})
                    if tc_delta.id:
                        entry['id'] = tc_delta.id
                    if tc_delta.function:
                        if tc_delta.function.name:
                            entry['name'] = tc_delta.function.name
                        if tc_delta.function.arguments:
                            entry['arguments'] += tc_delta.function.arguments
                if not has_output:
                    names = ', '.join(t['name'] for t in tool_calls.values() if t['name'])
                    live.update(Text(f'⚙ TOOL: {names}', style='dim cyan'))
                    live.refresh()
                    has_output = True

            # Keep the pinned bottom bar repainted while tokens stream in.
            # This is throttled (see draw_status_bar) so a fast stream of
            # small chunks can't flood stdout with dozens of near-identical
            # redraws per second.
            # draw_status_bar(status_state)

        if not has_output:
            live.update(Text(''))
            live.refresh()
        # draw_status_bar(status_state, force=True)

    # print_usage(usage)

    # Drop any malformed/nameless tool-call fragments (occasionally seen from
    # the Gemini API) so we never trigger a wasted extra round-trip on them,
    # this was the cause of the duplicated "AGENT:" block: a phantom
    # tool_calls delta with no name would still count as "has tool calls",
    # so the loop would run the model again for a second, real answer while
    # the first (empty) answer stayed printed on screen from the first pass.
    ordered_calls = [tool_calls[i] for i in sorted(tool_calls.keys()) if tool_calls[i].get('name')]
    final_text = ''.join(content_parts)
    final_text = remove_thoughts(final_text)
    return final_text, ordered_calls, finish_reason


def _call_model_once(messages, tools):
    """Blocking, non-streaming model call. Used by concurrently-run sub-agents
    (see Tasks/run_subagents_parallel) where we deliberately avoid rich's Live
    rendering, since multiple Live instances writing to the same terminal at
    once will corrupt the display."""
    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        tools=tools,
        temperature=TEMPERATURE,
        max_tokens=RESPONSE_TOKEN_RESERVE,
    )
    choice = response.choices[0]
    msg = choice.message
    text = remove_thoughts(msg.content or '')
    calls = [
        {'id': tc.id, 'name': tc.function.name, 'arguments': tc.function.arguments}
        for tc in (msg.tool_calls or [])
    ]
    return text, calls, choice.finish_reason


def call_model(messages, tools):
    """Same retry policy as stream_agent_response, for the non-streaming path."""
    for attempt in range(1, MODEL_MAX_RETRIES + 1):
        try:
            return _call_model_once(messages, tools)
        except RETRYABLE_ERRORS as e:
            log_event(f"MODEL CALL (non-stream) FAILED (attempt {attempt}/{MODEL_MAX_RETRIES}): {type(e).__name__}: {e}")
            if attempt == MODEL_MAX_RETRIES:
                raise
            delay = MODEL_RETRY_DELAYS[min(attempt - 1, len(MODEL_RETRY_DELAYS) - 1)]
            time.sleep(delay)


def print_usage(usage):
    if not usage:
        return
    in_tok, out_tok = usage.prompt_tokens, usage.completion_tokens
    console.print(f'[dim]tokens: {in_tok} in / {out_tok} out[/dim]')

def count_tokens(text):
    if not text:
        return 0
    return max(1, len(text) // 4)


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
    if total <= MAX_HISTORY_TOKENS and len(msgs) <= MAX_HISTORY_MESSAGES + 1:
        return
    cut = 1
    running = total
    while cut < len(msgs) and (running > MAX_HISTORY_TOKENS or len(msgs) - cut > MAX_HISTORY_MESSAGES):
        running -= _message_tokens(msgs[cut])
        cut += 1
    while cut < len(msgs) and msgs[cut].get('role') == 'tool':
        cut += 1
    del msgs[1:cut]


def truncate_output(text):
    if len(text) <= MAX_TOOL_OUTPUT:
        return text
    return text[:MAX_TOOL_OUTPUT] + f"\n... [output truncated, {len(text) - MAX_TOOL_OUTPUT} more chars not shown]"

def run_subagent(task_prompt, max_iters=15, render=True, label=None):
    """Run one isolated sub-agent to completion and return its final text.

    render=True  -> normal single-Task behaviour: live-streamed output via
                     rich Live, printed straight to the terminal (unchanged
                     from before).
    render=False -> used when multiple sub-agents run concurrently (Tasks).
                     Uses plain blocking model calls (no Live) and only
                     prints brief, lock-protected status lines, so parallel
                     threads never fight over the terminal.

    Sub-agents always run in 'auto' mode: they're isolated, self-contained
    delegated work, so per-tool-call prompts would just block the whole
    session waiting on a decision the user can't fully see context for.
    Genuinely dangerous shell commands still always prompt regardless of
    mode (see is_dangerous / ask_approval).
    """
    tag = f"[{label}] " if label else ''
    sub_system_prompt = (
        "You are a focused sub-agent spawned to complete one delegated task.\n"
        "Use the available tools as needed, then reply with ONLY the final "
        "result, no meta-commentary about being a sub-agent.\n\n" + SYSTEM_INFO
    )
    sub_state = AgentState(sub_system_prompt, name='sub', track_session=False, mode='auto')
    sub_state.messages.append({'role': 'user', 'content': task_prompt})

    if render:
        console.print(Panel(task_prompt, title='[bold magenta]SUB-AGENT started', border_style='magenta'))
    else:
        with console_lock:
            console.print(f"[bold magenta]{tag}SUB-AGENT started:[/bold magenta] {task_prompt[:100]}")
    log_event(f"SUBAGENT {tag}START: {task_prompt[:200]!r}")

    for i in range(max_iters):
        if render:
            final_text, tool_calls, _ = stream_agent_response(sub_state.messages, SUBAGENT_TOOLS)
        else:
            final_text, tool_calls, _ = call_model(sub_state.messages, SUBAGENT_TOOLS)
            if tool_calls:
                names = ', '.join(tc['name'] for tc in tool_calls)
                with console_lock:
                    console.print(f"[dim magenta]{tag}iteration {i + 1}: running {names}[/dim magenta]")

        if not tool_calls:
            result = final_text.strip()
            if render:
                console.print(Panel(result or '(empty result)', title='[bold magenta]SUB-AGENT finished', border_style='magenta'))
            else:
                with console_lock:
                    console.print(Panel(result or '(empty result)', title=f'[bold magenta]{tag}SUB-AGENT finished', border_style='magenta'))
            log_event(f"SUBAGENT {tag}DONE")
            return result

        sub_state.messages.append({
            "role": "assistant",
            "content": final_text or None,
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
                'content': truncate_output(str(result)),
            })
        trim_history(sub_state)

    log_event(f"SUBAGENT {tag}HIT MAX ITERATIONS")
    return "(sub-agent hit max iterations without finishing)"


def run_subagents_parallel(prompts, max_iters=15, max_workers=5):
    """Run several sub-agents concurrently (Tasks tool). Each gets its own
    isolated AgentState and runs in the non-Live 'render=False' mode so their
    output doesn't corrupt each other's terminal rendering. Returns a single
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


# ---------------------------------------------------------------------------
# Bottom-of-terminal context usage bar (Claude Code style)
#
# Works by shrinking the terminal's scroll region to leave the very last row
# free, then repeatedly repainting that last row in place (save cursor ->
# jump to last row -> clear it -> draw the bar -> restore cursor) every time
# something meaningful happens (a reply streams in, a tool runs, history is
# trimmed/summarized, etc). Everything else in the app keeps behaving
# exactly as before and simply scrolls above that reserved row.
# ---------------------------------------------------------------------------

CONTEXT_WINDOW_TOKENS = 256_000
_status_bar_enabled = False

# Warp (the macOS/Linux/Windows terminal app) renders output through its own
# custom block-based UI rather than fully emulating a VT100 grid. It doesn't
# support the "shrink the scroll region, then save/jump/clear/restore the
# cursor" trick this pinned bar depends on: the lone ESC in `ESC 7` / `ESC 8`
# (save/restore cursor) gets swallowed while the trailing digit prints
# literally, and the scroll-margin sequence isn't honored either, which is
# exactly the garbled "7[45;1H[2K[32m Context: ...8" text showing up in the
# chat. There's no reliable escape sequence workaround for this, so on Warp
# we skip the pinned-row approach entirely and fall back to printing the
# same info as a normal, scrolling line at a few meaningful checkpoints
# (see print_status_line / the IS_WARP branch in draw_status_bar below)
# instead of trying to redraw a fixed row.
IS_WARP = os.environ.get('TERM_PROGRAM') == 'WarpTerminal'


def _term_size():
    size = shutil.get_terminal_size(fallback=(80, 24))
    return size.columns, size.lines


def enable_status_bar():
    """Reserve the last terminal row for the status bar."""
    global _status_bar_enabled
    if not sys.stdout.isatty() or IS_WARP:
        return
    cols, rows = _term_size()
    if rows < 3:
        return
    sys.stdout.write(f"\x1b[1;{rows - 1}r")   # scroll region = rows 1..rows-1
    sys.stdout.write(f"\x1b[{rows - 1};1H")   # park cursor at bottom of region
    sys.stdout.flush()
    _status_bar_enabled = True


def disable_status_bar():
    """Restore full-screen scrolling and clear the reserved row. Safe to call
    multiple times (e.g. once at exit, once on Ctrl-C)."""
    global _status_bar_enabled
    if not _status_bar_enabled or not sys.stdout.isatty():
        return
    cols, rows = _term_size()
    sys.stdout.write("\x1b[r")                # reset scroll region to full screen
    sys.stdout.write(f"\x1b[{rows};1H")
    sys.stdout.write("\x1b[2K")
    sys.stdout.flush()
    _status_bar_enabled = False


def context_usage_tokens(state):
    """Best-effort estimate of tokens currently occupying the model's context
    window: system prompt + full running history."""
    used = count_tokens(state.messages[0].get('content', '')) if state.messages else 0
    used += total_history_tokens(state)
    return used


def _bar_color(pct):
    if pct < 0.5:
        return '32'   # green
    if pct < 0.8:
        return '33'   # yellow
    return '31'       # red


MODE_COLORS = {'plan': '36', 'normal': '37', 'auto': '31'}
MODE_LABELS = {'plan': 'PLAN', 'normal': 'NORMAL', 'auto': 'AUTO'}


def format_status_bar(state, width):
    used = context_usage_tokens(state)
    pct = min(used / CONTEXT_WINDOW_TOKENS, 1.0)
    color = _bar_color(pct)

    mode_color = MODE_COLORS.get(state.mode, '37')
    mode_tag = f" \x1b[1;{mode_color}m[{MODE_LABELS.get(state.mode, state.mode.upper())}]\x1b[0m"

    label = f" Context: "
    stats = f" {used:,}/{CONTEXT_WINDOW_TOKENS:,} tokens ({pct * 100:.1f}%) "
    model_tag = f" model:{MODEL} "

    bar_width = max(10, min(30, width - len(label) - len(stats) - len(model_tag) - len(MODE_LABELS.get(state.mode, state.mode.upper())) - 8))
    filled = int(bar_width * pct)
    bar = '█' * filled + '░' * (bar_width - filled)

    line = f"\x1b[{color}m{label}[{bar}]{stats}\x1b[2m|{model_tag}\x1b[0m{mode_tag}"
    # Strip to visible width so we never wrap onto the next (reserved) row.
    visible_len = (len(label) + 1 + bar_width + 1 + len(stats) + 1 + len(model_tag)
                   + len(MODE_LABELS.get(state.mode, state.mode.upper())) + 3)
    if visible_len > width:
        # Fall back to a plain, guaranteed-short line if the terminal is tiny.
        line = (f"\x1b[{color}m Context: {used:,}/{CONTEXT_WINDOW_TOKENS:,} ({pct * 100:.0f}%) \x1b[0m"
                f"\x1b[1;{mode_color}m[{MODE_LABELS.get(state.mode, state.mode.upper())}]\x1b[0m")
    return line


_last_bar_draw_at = [0.0]
MIN_BAR_REDRAW_INTERVAL = 0.08  # ~12/sec ceiling, matches the Live refresh rate
_last_plain_status_line = [None]  # dedupe so Warp doesn't print identical lines back to back


def print_status_line(state):
    """Fallback for terminals that can't do the pinned-row trick (Warp):
    print the same information as one normal, scrolling line using Rich
    markup, which Rich translates safely for whatever terminal it's on,
    no raw cursor-repositioning escape codes involved at all."""
    if not sys.stdout.isatty():
        return
    used = context_usage_tokens(state)
    pct = min(used / CONTEXT_WINDOW_TOKENS, 1.0)
    color_name = {'32': 'green', '33': 'yellow', '31': 'red'}[_bar_color(pct)]
    mode = state.mode
    mode_color_name = {'36': 'cyan', '37': 'white', '31': 'red'}[MODE_COLORS.get(mode, '37')]
    mode_label = MODE_LABELS.get(mode, mode.upper())

    bar_width = 24
    filled = int(bar_width * pct)
    bar = '█' * filled + '░' * (bar_width - filled)

    key = (used, mode)
    if key == _last_plain_status_line[0]:
        return
    _last_plain_status_line[0] = key

    console.print(
        f"[{color_name}]Context: [{bar}] {used:,}/{CONTEXT_WINDOW_TOKENS:,} tokens "
        f"({pct * 100:.1f}%)[/{color_name}] [dim]| model:{MODEL}[/dim] "
        f"[bold {mode_color_name}][{mode_label}][/bold {mode_color_name}]"
    )


def  draw_status_bar(state, force=False):
    """Repaint the reserved bottom row without disturbing the cursor position
    the rest of the app is using.

    Throttled by default: during token-by-token streaming this can otherwise
    be called dozens of times a second, which is both wasteful and (when
    interleaved with any other stdout writer) the source of the corrupted,
    repeated status-bar text some users saw ("...[45;1H[2K..." showing up as
    literal text in the chat). Pass force=True for the handful of call sites
    where we redraw once after something meaningful changed (mode switch,
    turn finished, prompt about to be shown, etc.) and always want it to
    happen immediately.

    On Warp (see IS_WARP above) the pinned row itself isn't usable, so
    instead of attempting it we only ever emit the plain-line fallback, and
    only at those same "force" checkpoints, never on every streamed token,
    which would otherwise spam a line into the chat for each chunk.
    """
    if IS_WARP:
        if force:
            print_status_line(state)
        return
    if not _status_bar_enabled or not sys.stdout.isatty():
        return
    now = time.monotonic()
    if not force and (now - _last_bar_draw_at[0]) < MIN_BAR_REDRAW_INTERVAL:
        return
    _last_bar_draw_at[0] = now
    cols, rows = _term_size()
    if rows < 3:
        return
    bar_text = format_status_bar(state, cols)
    sys.stdout.write("\x1b7")                 # save cursor
    sys.stdout.write(f"\x1b[{rows};1H")       # jump to last row
    sys.stdout.write("\x1b[2K")               # clear it
    sys.stdout.write(bar_text)
    sys.stdout.write("\x1b8")                 # restore cursor
    sys.stdout.flush()


def _handle_resize(signum, frame):
    """On SIGWINCH: re-establish the scroll region for the new size and redraw."""
    if not _status_bar_enabled:
        return
    cols, rows = _term_size()
    if rows < 3:
        return
    sys.stdout.write(f"\x1b[1;{rows - 1}r")
    sys.stdout.flush()
    # draw_status_bar(_current_state_for_resize[0], force=True)


_current_state_for_resize = [None]  # small mutable box so the signal handler can see current state


def install_resize_handler(state):
    _current_state_for_resize[0] = state
    if hasattr(signal, 'SIGWINCH'):
        try:
            signal.signal(signal.SIGWINCH, _handle_resize)
        except (ValueError, OSError):
            pass  # not the main thread, or platform doesn't support it


def main():
    global state
    
    # Check for API Key before starting
    if not os.environ.get('GEMINI_API_KEY'):
        console.print("[bold red]Error: GEMINI_API_KEY is not set.[/bold red]")
        console.print("Please set it in your environment or add it to ~/.bardgent/.env:")
        console.print("[yellow]GEMINI_API_KEY=your_key_here[/yellow]")
        sys.exit(1)

    print_welcome()
    log_event("=== Bardgent session start ===")

    state = AgentState(SYSTEM_PROMPT, name='main')
    _mode_keys = KeyBindings()

    # key-bindings callback
    @_mode_keys.add('s-tab')
    def _cycle_mode(event):
        def _do_switch():
            idx = MODE_CYCLE.index(state.mode) if state.mode in MODE_CYCLE else 0
            switch_mode(state, MODE_CYCLE[(idx + 1) % len(MODE_CYCLE)])
        run_in_terminal(_do_switch)

    global prompt_session
    prompt_session = PromptSession(
        completer=WordCompleter(list(COMMANDS.keys()), sentence=True),
        key_bindings=_mode_keys,
    )

    enable_status_bar()
    atexit.register(disable_status_bar)
    install_resize_handler(state)
    # draw_status_bar(state, force=True)

    while True:
        try:
            user_input = prompt_session.prompt(HTML('<ansigreen><b>USER: </b></ansigreen>'), multiline=False).strip()
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

        if total_history_tokens(state) > AUTO_SUMMARY_TOKEN_THRESHOLD:
            console.print('[dim]Context getting large, auto-summarizing...[/dim]')
            log_event("AUTO-SUMMARY triggered")
            do_summary_and_compact(state)
        else:
            trim_history(state)

        try:
            for _ in range(MAX_ITERATIONS):
                final_text, tool_calls, finish_reason = stream_agent_response(state.messages, TOOLS)

                if tool_calls:
                    state.messages.append({
                        "role": "assistant",
                        "content": final_text or None,
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
                            'content': truncate_output(str(result))
                        })
                    trim_history(state)
                    continue

                final_text = final_text.strip()
                final_text = remove_thoughts(final_text)
                state.messages.append({'role': 'assistant', 'content': final_text})
                if state.telegram_enabled and state.telegram_chat_id and final_text:
                    if not send_telegram_message(final_text, state.telegram_chat_id):
                        console.print('[dim red]Could not deliver message to Telegram (see bardgent.log).[/dim red]')
                break
            else:
                console.print(f'[bold red]Hit max iterations ({MAX_ITERATIONS}) without a final answer.[/bold red]')

        except KeyboardInterrupt:
            console.print('\n[yellow]Interrupted! back to prompt.[/yellow]')
        except Exception as e:
            console.print(f'\n[bold red]Error during turn: {type(e).__name__}: {e}[/bold red]')
            log_event(f"TURN ERROR: {type(e).__name__}: {e}")

        if state.messages[1:]:
            save_session(state)
        trim_history(state)

if __name__ == '__main__':
    main()
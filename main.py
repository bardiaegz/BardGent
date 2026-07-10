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
from openai import OpenAI
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
import fnmatch
import re
import difflib
import subprocess
import datetime

console = Console()

python_path = sys.executable
operating_system = platform.platform()
working_directory = os.getcwd()
home_directory = os.path.expanduser('~')

client = OpenAI(
    base_url='http://localhost:8080',
    api_key='sk-no-key-required'
)

MODEL = 'unsloth/gemma-4-26B-A4B-it-GGUF:UD-IQ3_S'
TEMPERATURE = 0.2
MAX_ITERATIONS = 30
MAX_HISTORY_MESSAGES = 30
MAX_TOOL_OUTPUT = 8_000
BASH_TIMEOUT_SECONDS = 60
MAX_HISTORY_TOKENS = 6_000          # rough budget for kept history
AUTO_SUMMARY_TOKEN_THRESHOLD = 5_000  # auto-compact via LLM summary above this

MEMORY_FILE = Path('Bardgent.md')
SESSION_DIR = Path.cwd() / ".bardgent_sessions"
SESSION_DIR.mkdir(exist_ok=True)
SESSION_PREFIX = ".bardgent_session_"
SUMMARY_PREFIX = '[Conversation summary so far]: '

BACKUP_DIR = Path.cwd() / ".bardgent_backups"
BACKUP_DIR.mkdir(exist_ok=True)
last_backup = {}

LOG_FILE = Path.cwd() / 'bardgent.log'
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
)

def log_event(msg):
    logging.info(msg)

def Read(file_path):
    path = os.path.abspath(os.path.expanduser(file_path))
    with open(path, 'r') as f:
        return f.read()

class AgentState:
    def __init__(self, system_prompt, name='main', track_session=True):
        self.name = name
        self.messages = [{'role': 'system', 'content': system_prompt}]
        self.shell_cwd = os.getcwd()
        self.approved_for_session = set()
        self.approval_lock = threading.RLock()
        self.session_file = (SESSION_DIR / session_file_name()) if track_session else None


def ask_approval(state, key, question, dangerous=False):
    """Ask the user to approve an action. 'a' remembers the approval for this session."""
    with state.approval_lock:
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
        return f"Write to {path} rejected by user. Do NOT retry it or a variation of it — continue with what you already have, or ask the user in your final answer."
    if existed:
        _make_backup(path, old)
    with open(path, 'w') as f:
        f.write(content)
    log_event(f"[{state.name}] WRITE {path}")
    suffix = ' (previous version backed up — use Undo to revert)' if existed else ''
    return f'Wrote to {path}{suffix}'


def Edit(file_path, old_str, new_str, state):
    path = os.path.abspath(os.path.expanduser(file_path))
    with open(path, 'r') as f:
        content = f.read()
    count = content.count(old_str)
    if count == 0:
        return f"Error: old_str not found in {path}"
    if count > 1:
        return f"Error: old_str matches {count} times in {path}, must be unique"
    new_content = content.replace(old_str, new_str)
    if not confirm_diff(content, new_content, path, 'Edit', state):
        return f"Edit to {path} rejected by user. Do NOT retry it or a variation of it — continue with what you already have, or ask the user in your final answer."
    _make_backup(path, content)
    with open(path, 'w') as f:
        f.write(new_content)
    log_event(f"[{state.name}] EDIT {path}")
    return f'Edited {path} (previous version backed up — use Undo to revert)'


def Undo(file_path):
    path = os.path.abspath(os.path.expanduser(file_path))
    backup_path = last_backup.get(path)
    if not backup_path or not backup_path.exists():
        return f"No backup available for {path} in this session."
    with open(path, 'w', encoding='utf-8') as f:
        f.write(backup_path.read_text(encoding='utf-8'))
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
    return False


CWD_MARKER = '__BARDGENT_CWD__'


def Bash(command, state, timeout=BASH_TIMEOUT_SECONDS):
    danger = is_dangerous(command)
    first_word = (command.strip().split() or [''])[0]
    key = f"Bash:{first_word}"
    with state.approval_lock:
        if danger or key not in state.approved_for_session:
            color = "red" if danger else "yellow"
            title = "Bash wants to run (DANGEROUS)" if danger else "Bash wants to run"
            console.print(Panel(command, title=f"[bold {color}]{title}",
                                subtitle=f"[dim]in {state.shell_cwd}", border_style=color))
        if not ask_approval(state, key, "Run this command?", dangerous=danger):
            return "Command rejected by user. Do NOT retry it or a variation of it — continue with what you already have, or ask the user in your final answer."
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

    if not ask_approval(state, 'Fetch', 'Fetch this page?'):
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
    console.print("Type 'exit' or 'quit' to leave.\n")


DATETIME = datetime.datetime.now().astimezone()

SYSTEM_PROMPT = f"""
You are a helpful coding agent.
Your name is Bardgent made by Bardia.

DATETIME: {DATETIME.strftime('%Y-%B-%d %I:%M %p %Z')}

{SYSTEM_INFO}

You have access to these tools:

File tools:
- Read(file_path): Read the content of a file.
- Write(file_path, content): Write or overwrite a file. Always show the user a diff and ask for approval before writing. Automatically backed up — the user or you can call Undo(file_path) to revert.
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
- Task(prompt): Delegate a self-contained, multi-step subtask (e.g. a broad codebase
  search, a multi-file investigation, or a repetitive bulk operation) to an isolated
  sub-agent. The sub-agent has its own context and its own copy of the file/exec/web
  tools (but cannot itself call Task). It returns only its final result to you, which
  keeps your own context small. Use it when a subtask would otherwise take many tool
  calls whose intermediate output you don't need to see yourself.

Rules:

- Always use this exact Python executable path when executing Python files:
  {python_path}

- When the user gives a relative path (for example Desktop/foo/app.py),
  first try it relative to the current working directory and home directory before searching.

- Before modifying files:
  - Prefer Edit for small targeted changes.
  - Use Write only when replacing the entire file or creating a new file.
  - Always review the diff shown by the tool and respect the user's approval.

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
        temperature=TEMPERATURE
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
        return 'handled'

    if cmd == '/resume':
        resume_session(state)
        return 'handled'

    if cmd == '/exit':
        console.print('Goodbye!')
        sys.exit(0)

    if cmd == '/summary':
        do_summary_and_compact(state)
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
        'description': 'Delegate a self-contained sub-task (e.g. large codebase search, multi-step investigation) to an isolated sub-agent. Returns only its final result.',
        'parameters': {'type': 'object', 'properties': {
            'prompt': {'type': 'string', 'description': 'the full task for the sub-agent to complete'}
        }, 'required': ['prompt']}
    }},
]

# Sub-agents get every tool except Task itself, to prevent recursive spawning.
SUBAGENT_TOOLS = [t for t in TOOLS if t['function']['name'] != 'Task']

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
    err = validate_args(name, args)
    if err:
        log_event(f"[{state.name}] VALIDATION FAILED for '{name}': {err}")
        return err
    try:
        if name == 'Task':
            return run_subagent(args['prompt'])
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
        return f"Error running tool '{name}': {type(e).__name__}: {e}. Do not blindly retry — adjust the arguments or approach."


def render_agent(text):
    return Group(Text('AGENT:', style='bold cyan'), Markdown(text))


def stream_agent_response(messages, tools):
    stream = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        tools=tools,
        temperature=TEMPERATURE,
        stream=True,
        stream_options={'include_usage': True},
    )

    content_parts = []
    tool_calls = {}          # index -> {'id':..., 'name':..., 'arguments': ''}
    finish_reason = None
    usage = None
    has_output = False

    spinner = Spinner('dots', text=Text(' Thinking...', style='cyan'))

    with Live(spinner, console=console, refresh_per_second=12, transient=False) as live:
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
                has_output = True
                live.update(render_agent(''.join(content_parts)))

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
                    has_output = True

        if not has_output:
            live.update(Text(''))

    # print_usage(usage)

    ordered_calls = [tool_calls[i] for i in sorted(tool_calls.keys())]
    final_text = ''.join(content_parts)
    return final_text, ordered_calls, finish_reason


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

def run_subagent(task_prompt, max_iters=15):
    sub_system_prompt = (
        "You are a focused sub-agent spawned to complete one delegated task.\n"
        "Use the available tools as needed, then reply with ONLY the final "
        "result — no meta-commentary about being a sub-agent.\n\n" + SYSTEM_INFO
    )
    sub_state = AgentState(sub_system_prompt, name='sub', track_session=False)
    sub_state.messages.append({'role': 'user', 'content': task_prompt})

    console.print(Panel(task_prompt, title='[bold magenta]SUB-AGENT started', border_style='magenta'))
    log_event(f"SUBAGENT START: {task_prompt[:200]!r}")

    for _ in range(max_iters):
        final_text, tool_calls, _ = stream_agent_response(sub_state.messages, SUBAGENT_TOOLS)

        if not tool_calls:
            result = final_text.strip()
            console.print(Panel(result or '(empty result)', title='[bold magenta]SUB-AGENT finished', border_style='magenta'))
            log_event("SUBAGENT DONE")
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
                result = dispatch_tool(tc['name'], args, sub_state)
            sub_state.messages.append({
                'role': 'tool', 'tool_call_id': tc['id'],
                'content': truncate_output(str(result)),
            })
        trim_history(sub_state)

    log_event("SUBAGENT HIT MAX ITERATIONS")
    return "(sub-agent hit max iterations without finishing)"

print_welcome()
log_event("=== Bardgent session start ===")

prompt_session = PromptSession(completer=WordCompleter(list(COMMANDS.keys()), sentence=True))
state = AgentState(SYSTEM_PROMPT, name='main')

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

    # Prefer an LLM-generated summary over a blind trim once history gets
    # large — it preserves the gist instead of just dropping old messages.
    if total_history_tokens(state) > AUTO_SUMMARY_TOKEN_THRESHOLD:
        console.print('[dim]Context getting large — auto-summarizing...[/dim]')
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
            state.messages.append({'role': 'assistant', 'content': final_text})
            # Not re-printed here: Live already rendered it on screen as it
            # streamed in (stream_agent_response uses transient=False).
            break
        else:
            console.print(f'[bold red]Hit max iterations ({MAX_ITERATIONS}) without a final answer.[/bold red]')

    except KeyboardInterrupt:
        console.print('\n[yellow]Interrupted — back to prompt.[/yellow]')
    except Exception as e:
        console.print(f'\n[bold red]Error during turn: {type(e).__name__}: {e}[/bold red]')
        log_event(f"TURN ERROR: {type(e).__name__}: {e}")

    if state.messages[1:]:
        save_session(state)
    trim_history(state)
import os
import sys
import json
import time
import glob
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

MEMORY_FILE = Path('Bardgent.md')
SESSION_DIR = Path.cwd() / ".bardgent_sessions"
SESSION_DIR.mkdir(exist_ok=True)
SESSION_PREFIX = ".bardgent_session_"
SUMMARY_PREFIX = '[Conversation summary so far]: '

def Read(file_path):
    path = os.path.abspath(os.path.expanduser(file_path))
    with open(path, 'r') as f:
        return f.read()

approved_for_session = set()
approval_lock = threading.RLock()


def ask_approval(key, question, dangerous=False):
    """Ask the user to approve an action. 'a' remembers the approval for this session."""
    with approval_lock:
        # TODO: will add command and dangerous in the next commit.
        if dangerous:
            answer = input(f"{question} [y/N]: ").strip().lower()
            return answer in ('y', 'yes')
        if key in approved_for_session:
            console.print(f"[dim]auto-approved ({key})[/dim]")
            return True
        answer = input(f"{question} [Y/n/a=always]: ").strip().lower()
        if answer in ('a', 'always'):
            approved_for_session.add(key)
            return True
        return answer in ('', 'y', 'yes')

ADD_STYLE = 'white on dark_green'
DEL_STYLE = 'white on dark_red'

def confirm_diff(old, new, path, tool_name):
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
            # Guess lexer based on file extension, default to python
            lexer = Path(path).suffix.lstrip('.') or 'python'
            try:
                syntax = Syntax(line[1:], lexer, theme='monokai', line_numbers=False)
                # Render syntax to a Text object, then apply background style
                rendered = console.render(syntax)
                for segment in rendered.spans:
                    segment.style = f"{segment.style} on dark_green" if segment.style else "white on dark_green"
                body.append(rendered)
            except:
                body.append(f"{new_no:>4} + {line[1:]}".ljust(bar_width), style=ADD_STYLE)
            body.append('\n')
            new_no += 1
        # (Repeat similar logic for '-' lines with dark_red)
        elif line.startswith('-'):
            # Guess lexer based on file extension, default to python
            lexer = Path(path).suffix.lstrip('.') or 'python'
            try:
                syntax = Syntax(line[1:], lexer, theme='monokai', line_numbers=False)
                # Render syntax to a Text object, then apply background style
                rendered = console.render(syntax)
                for segment in rendered.spans:
                    segment.style = f"{segment.style} on dark_red" if segment.style else "white on dark_red"
                body.append(rendered)
            except:
                body.append(f"{new_no:>4} + {line[1:]}".ljust(bar_width), style=DEL_STYLE)
            body.append('\n')
            new_no += 1
        # (Repeat similar logic for '-' lines with dark_red)
        else:
            body.append(f"{new_no:>4}   {line[1:]}\n", style='dim')
            old_no += 1
            new_no += 1

    if not body:
        body = Text('(no changes)', style='dim')
    with approval_lock:
        console.print(Panel(body, title=f"[bold yellow]{tool_name}: {path}", border_style='yellow'))
        return ask_approval(tool_name, "Apply this change?")

def Write(file_path, content):
    path = os.path.abspath(os.path.expanduser(file_path))
    old = ''
    if os.path.exists(path):
        with open(path, 'r') as f:
            old = f.read()
    if not confirm_diff(old, content, path, 'Write'):
        return f"Write to {path} rejected by user. Do NOT retry it or a variation of it. continue with what you already have, or ask the user in your final answer."
    with open(path, 'w') as f:
        f.write(content)
    return f'Wrote to {path}'

def Edit(file_path, old_str, new_str):
    path = os.path.abspath(os.path.expanduser(file_path))
    with open(path, 'r') as f:
        content = f.read()
    count = content.count(old_str)
    if count == 0:
        return f"Error: old_str not found in {path}"
    if count > 1:
        return f"Error: old_str matches {count} times in {path}, must be unique"
    new_content = content.replace(old_str, new_str)
    if not confirm_diff(content, new_content, path, 'Edit'):
        return f"Edit to {path} rejected by user. Do NOT retry it or a variation of it. continue with what you already have, or ask the user in your final answer."
    with open(path, 'w') as f:
        f.write(new_content)
    return f'Edited {path}'

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
    r'\|\s*(sh|bash)\b', r'--force\b', r'--hard\b',
]

def is_dangerous(command):
    return any(re.search(p, command) for p in DANGEROUS_PATTERNS)

shell_cwd = os.getcwd()
CWD_MARKER = '__BARDGENT_CWD__'

def Bash(command):
    global shell_cwd
    danger = is_dangerous(command)
    first_word = (command.strip().split() or [''])[0]
    key = f"Bash:{first_word}"
    with approval_lock:
        if danger or key not in approved_for_session:
            color = "red" if danger else "yellow"
            title = "Bash wants to run (DANGEROUS)" if danger else "Bash wants to run"
            console.print(Panel(command, title=f"[bold {color}]{title}",
                                subtitle=f"[dim]in {shell_cwd}", border_style=color))
        if not ask_approval(key, "Run this command?", dangerous=danger):
            return "Command rejected by user. Do NOT retry it or a variation of it. continue with what you already have, or ask the user in your final answer."
    # append a marker echoing $PWD so `cd` persists to the next Bash call
    wrapped = command + f'\nprintf "\\n{CWD_MARKER}%s" "$PWD"'
    result = subprocess.run(wrapped, shell=True, capture_output=True, text=True, cwd=shell_cwd)
    stdout, sep, after = result.stdout.rpartition(CWD_MARKER)
    if sep:
        new_dir = after.strip()
        if new_dir and os.path.isdir(new_dir):
            shell_cwd = new_dir
        stdout = stdout[:-1] if stdout.endswith('\n') else stdout
    else:
        stdout = result.stdout
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
    return "Memory saved."


def WebSearch(query):
    # console.print(f'\n[bold green]⚙ TOOL:[/bold green] Web Search: {query}\n')
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
    resp = requests.post('https://html.duckduckgo.com/html/', data={'q': query}, headers=headers, timeout=10)
    console.status("Searching the web...")
    resp.raise_for_status()
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

def Fetch(link):
    console.print(Panel(link, title='[bold yellow]Fetch wants to run', border_style='yellow'))

    if not ask_approval('Fetch', 'Fetch this page?'):
        return 'Fetch rejected by user.'

    # console.print(f'\n[bold green]⚙ TOOL:[/bold green] Fetch\n')

    headers = {
        'User-Agent': (
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) '
            'AppleWebKit/537.36 Chrome/120 Safari/537.36'
        )
    }

    try:
        resp = requests.get(link, headers=headers, timeout=10)

        if resp.status_code == 403:
            return f"Could not fetch page (403 Forbidden): {link}"

        resp.raise_for_status()

    except requests.RequestException as e:
        return f"Fetch failed: {type(e).__name__}: {e}"

    soup = BeautifulSoup(resp.text, 'html.parser')

    for tag in soup(['script', 'style', 'noscript']):
        tag.decompose()

    return soup.get_text(separator='\n', strip=True)

def print_welcome():
    console.print(f"[bold italic magenta]Welcome to Bardgent[/bold italic magenta]!")
    console.print("Type 'exit' or 'quit' to leave.\n")

DATETIME = datetime.datetime.now().astimezone()

messages = [
    {
        "role": "system",
        "content": f"""
You are a helpful coding agent.
Your name is Bardgent made by Bardia.

DATETIME: {DATETIME.strftime('%Y-%B-%d %I:%M %p %Z')}

{SYSTEM_INFO}

You have access to these tools:

File tools:
- Read(file_path): Read the content of a file.
- Write(file_path, content): Write or overwrite a file. Always show the user a diff and ask for approval before writing.
- Edit(file_path, old_str, new_str): Replace an exact unique string inside a file. Prefer Edit over Write for small changes.
- Glob(pattern): Find files by name using glob patterns.
- Grep(pattern, path, include): Search inside files using regex.

Execution tools:
- Bash(command): Execute shell commands. The shell keeps its working directory between calls, so `cd` persists.

Web tools:
- WebSearch(query): Search the web and return results.
- Fetch(link): Fetch and extract text from a web page.

Memory tools:
- read_memory(): Read long-term memory.
- save_memory(memory): Save useful user facts or preferences.

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
    }
]

def session_file_name():
    return f"{SESSION_PREFIX}{time.strftime('%Y-%m-%d_%H-%M-%S')}.json"

current_session_file = SESSION_DIR / session_file_name()

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


def save_session():
    if not messages[1:]:
        return
    data = {
        'updated': time.strftime('%Y-%m-%d %H:%M:%S'),
        'title': session_title(messages[1:]),
        'messages': messages[1:],
    }
    try:
        with open(current_session_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    except OSError as e:
        console.print(f"[dim red]Failed to auto-save session: {e}[/dim red]")

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
    '/model': 'Show the current model, or switch: /model <name>',
    '/clear': 'Clear history and start a new session',
    '/resume': 'Pick a past session and resume it',
    '/exit': 'Quit Bardgent',
}

def replay_transcript():
    for m in messages[1:]:
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

def resume_session():
    global current_session_file
    sessions = [s for s in list_sessions() if Path(s[0]) != current_session_file]
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

    del messages[1:]
    messages.extend(data['messages'])
    current_session_file = Path(path)
    console.print()
    replay_transcript()
    console.print(Text(f"\nResumed \"{data.get('title', '')}\" ({len(messages) - 1} messages).", style='bold green'))

def do_summary_and_compact():
    if len(messages) <= 1:
        console.print('[yellow]Nothing to summarize yet.[/yellow]')
        return

    temp_messages = messages + [{
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

    del messages[1:]
    messages.append({
        'role': 'assistant',
        'content': SUMMARY_PREFIX + summary_text
    })

    save_session()
    console.print(Panel(summary_text, title='[bold cyan]SUMMARY (history compacted)', border_style='cyan'))

def handle_command(user_input):
    global current_session_file, MODEL
    cmd = user_input.strip().lower()

    if cmd == '/model' or cmd.startswith('/model '):
        parts = user_input.strip().split(maxsplit=1)
        if len(parts) == 1:
            console.print(f'Current model: [bold cyan]{MODEL}[/bold cyan]')
        else:
            MODEL = parts[1].strip()
            console.print(f'[bold green]Model switched to {MODEL}.[/bold green]')
        return 'handled'

    if cmd == '/clear':
        del messages[1:]
        current_session_file = SESSION_DIR / session_file_name()
        console.clear()
        print_welcome()
        console.print('[bold green]New session started.[/bold green]')
        return 'handled'

    if cmd == '/resume':
        resume_session()
        return 'handled'

    if cmd == '/exit':
        console.print('Goodbye!')
        sys.exit(0)

    if cmd == '/summary':
        do_summary_and_compact()
        return 'handled'

    return None

tools = [
    {
        'type': 'function',
        'function': {
            'name': 'read_memory',
            'description': 'Read long-term memory.',
            'parameters': {'type': 'object', 'properties': {}},
        }
    },
    {
        'type': 'function',
        'function': {
            'name': 'save_memory',
            'description': 'Save useful user facts or preferences.',
            'parameters': {
                'type': 'object',
                'properties': {
                    'memory': {'type': 'string'}
                },
                'required': ['memory']
            },
        }
    },
    {
        'type': 'function',
        'function': {
            'name': 'Fetch',
            'description': 'Fetch the content of a web page',
            'parameters': {
                'type': 'object',
                'properties': {
                    'link': {
                        'type': 'string',
                        'description': 'the link of the web page to fetch'
                    }
                },
                'required': ['link']
            },
        }
    },
    {
        'type': 'function',
        'function': {
            'name': 'WebSearch',
            'description': 'Search the web (DuckDuckGo), returns titles, URLs and snippets. Use Fetch afterwards to read a promising result.',
            'parameters': {
                'type': 'object',
                'properties': {
                    'query': {
                        'type': 'string',
                        'description': 'the search query'
                    }
                },
                'required': ['query']
            },
        }
    },
        {'type': 'function', 'function': {
        'name': 'Read', 'description': 'Read a file from disk',
        'parameters': {'type': 'object', 'properties': {'file_path': {'type': 'string'}}, 'required': ['file_path']}
    }},
    {'type': 'function', 'function': {
        'name': 'Write', 'description': 'Write (overwrite) full content to a file',
        'parameters': {'type': 'object', 'properties': {
            'file_path': {'type': 'string', 'description': 'the path of the file to write to'},
            'content': {'type': 'string', 'description': 'the content to write to the file'}
        }, 'required': ['file_path', 'content']}
    }},
    {'type': 'function', 'function': {
        'name': 'Edit', 'description': 'Replace an exact string match inside a file (must match exactly once)',
        'parameters': {'type': 'object', 'properties': {
            'file_path': {'type': 'string'},
            'old_str': {'type': 'string', 'description': 'exact text to find'},
            'new_str': {'type': 'string', 'description': 'text to replace it with'}
        }, 'required': ['file_path', 'old_str', 'new_str']}
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
        'name': 'Bash', 'description': 'Execute a shell command',
        'parameters': {'type': 'object', 'properties': {'command': {'type': 'string', 'description': 'the command to execute'}}, 'required': ['command']}
    }},

]

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
                            # Concatenate only. Never parse here.
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


def trim_history():
    if len(messages) <= MAX_HISTORY_MESSAGES + 1:
        return
    cut = len(messages) - MAX_HISTORY_MESSAGES
    while cut < len(messages) and messages[cut].get('role') == 'tool':
        cut += 1
    del messages[1:cut]


def truncate_output(text):
    if len(text) <= MAX_TOOL_OUTPUT:
        return text
    return text[:MAX_TOOL_OUTPUT] + f"\n... [output truncated, {len(text) - MAX_TOOL_OUTPUT} more chars not shown]"

print_welcome()

prompt_session = PromptSession(completer=WordCompleter(list(COMMANDS.keys()), sentence=True))

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
        result = handle_command(user_input)
        if result == 'handled':
            continue
        elif result is None:
            console.print(f'[bold red]Unknown command: {user_input}[/bold red]')
            continue
    else:
        messages.append({'role': 'user', 'content': user_input})

    trim_history()

    try:
        for _ in range(MAX_ITERATIONS):
            final_text, tool_calls, finish_reason = stream_agent_response(messages, tools)

            if tool_calls:
                messages.append({
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
                    # arguments is the FULLY reassembled string at this point
                    # (stream_agent_response already joined every chunk by
                    # index), so this is the only place it gets parsed.
                    args = json.loads(tool_call['arguments'] or '{}')

                    if name == 'read_memory':
                        result = read_memory()
                    elif name == 'save_memory':
                        result = save_memory(args['memory'])
                    elif name == 'WebSearch':
                        result = WebSearch(args['query'])
                    elif name == 'Fetch':
                        result = Fetch(args['link'])
                    elif name == 'Read':
                        result = Read(args['file_path'])
                    elif name == 'Write':
                        result = Write(args['file_path'], args['content'])
                    elif name == 'Edit':
                        result = Edit(args['file_path'], args['old_str'], args['new_str'])
                    elif name == 'Glob':
                        result = Glob(args['pattern'])
                    elif name == 'Grep':
                        result = Grep(args['pattern'], args.get('path', '.'), args.get('include'))
                    elif name == 'Bash':
                        result = Bash(args['command'])
                    else:
                        result = 'Unknown tool'

                    messages.append({
                        'role': 'tool',
                        'tool_call_id': tool_call['id'],
                        'content': truncate_output(str(result))
                    })
                continue

            final_text = final_text.strip()
            messages.append({'role': 'assistant', 'content': final_text})
            # Not re-printed here: Live already rendered it on screen as it
            # streamed in (stream_agent_response uses transient=False).
            break
        else:
            console.print(f'[bold red]Hit max iterations ({MAX_ITERATIONS}) without a final answer.[/bold red]')

    except KeyboardInterrupt:
        console.print('\n[yellow]Interrupted. back to prompt.[/yellow]')
    except Exception as e:
        console.print(f'\n[bold red]Error during turn: {type(e).__name__}: {e}[/bold red]')

    if messages[1:]:
        save_session()
    trim_history()
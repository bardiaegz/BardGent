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
from prompt_toolkit import PromptSession
from prompt_toolkit.completion import WordCompleter
from prompt_toolkit.formatted_text import HTML

console = Console()

python_path = sys.executable
operating_system = platform.platform()
working_directory = os.getcwd()
home_directory = os.path.expanduser('~')

client = OpenAI(
    base_url='http://localhost:8080',
    api_key='sk-no-key-required'
)

MODEL = 'yuxinlu1/gemma-4-12B-agentic-fable5-composer2.5-v2-3.5x-tau2-GGUF:Q4_K_M'
TEMPERATURE = 0.2
MAX_ITERATIONS = 10
MAX_HISTORY_MESSAGES = 30
MAX_TOOL_OUTPUT = 8_000

MEMORY_FILE = Path('Bardgent.md')
SESSION_DIR = Path.cwd() / ".bardgent_sessions"
SESSION_DIR.mkdir(exist_ok=True)
SESSION_PREFIX = ".bardgent_session_"
SUMMARY_PREFIX = '[Conversation summary so far]: '

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


SYSTEM_INFO = f"""[CRITICAL SYSTEM INFO]:
- Python Executable Path: {python_path}
- Operating System: {operating_system}
- Current Working Directory: {working_directory}
- User Home Directory: {home_directory}"""

def read_memory():
    if MEMORY_FILE.exists():
        console.print(f'\n[bold green]⚙ TOOL:[/bold green] READING MEMORY FROM Bardgent.md\n')
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

    console.print(f'\n[bold green]⚙ TOOL:[/bold green] SAVING MEMORY TO Bardgent.md\n')
    return "Memory saved."


def WebSearch(query):
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
    resp = requests.post('https://html.duckduckgo.com/html/', data={'q': query}, headers=headers, timeout=10)
    resp.raise_for_status()
    console.print(f'\n[bold green]⚙ TOOL:[/bold green] Web Search: {query}\n')
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

    console.print(f'\n[bold green]⚙ TOOL:[/bold green] Fetch\n')

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


messages = [
    {
        'role': 'system',
        'content': f"""
You are helpful agent and your name is Bardgent made by Bardia.

{SYSTEM_INFO}

You have access to these tools:
- read_memory(): read long-term memory
- save_memory(memory): save useful facts
- WebSearch: Websearch the web
- Fetch: Fetch web pages

Only save information that will be useful in future conversations.
Before answering questions that may depend on past context, call read_memory.
Only call save_memory when the user explicitly tells you a new fact about themselves in their most recent message.

Never save information that came from read_memory.
Never save information that you inferred.
Never save information that already exists in memory.
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
]

def render_agent(text):
    return Group(Text('AGENT:', style='bold cyan'), Markdown(text))


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
            response = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                tools=tools,
                temperature=TEMPERATURE,
            )

            assistant_message = response.choices[0].message
            # print_usage(getattr(response, 'usage', None))

            if assistant_message.tool_calls:
                messages.append({
                    "role": "assistant",
                    "content": assistant_message.content,
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments
                            }
                        }
                        for tc in assistant_message.tool_calls
                    ]
                })

                for tool_call in assistant_message.tool_calls:
                    name = tool_call.function.name
                    args = json.loads(tool_call.function.arguments or '{}')

                    if name == 'read_memory':
                        result = read_memory()
                    elif name == 'save_memory':
                        result = save_memory(args['memory'])
                    elif name == 'WebSearch':
                        result = WebSearch(args['query'])
                    elif name == 'Fetch':
                        result = Fetch(args['link'])
                    else:
                        result = 'Unknown tool'

                    messages.append({
                        'role': 'tool',
                        'tool_call_id': tool_call.id,
                        'content': truncate_output(str(result))
                    })
                continue

            final_text = (assistant_message.content or '').strip()
            messages.append({'role': 'assistant', 'content': final_text})
            console.print(render_agent(final_text))
            break
        else:
            console.print(f'[bold red]Hit max iterations ({MAX_ITERATIONS}) without a final answer.[/bold red]')

    except KeyboardInterrupt:
        console.print('\n[yellow]Interrupted — back to prompt.[/yellow]')
    except Exception as e:
        console.print(f'\n[bold red]Error during turn: {type(e).__name__}: {e}[/bold red]')

    if messages[1:]:
        save_session()
    trim_history()
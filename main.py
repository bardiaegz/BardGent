import dotenv
import os
import sys
from openai import OpenAI
import requests
from bs4 import BeautifulSoup
from rich.console import Console
from pathlib import Path
import json
import platform
from urllib.parse import urlparse, parse_qs
import threading
from rich.panel import Panel

approved_for_session = set()
approval_lock = threading.RLock()
tool_iterations = 0

def ask_approval(key, question, dangerous=False):
    """Ask the user to approve an action. 'a' remembers the approval for this
    session (keyed per tool / command). Dangerous actions always ask, default No."""
    with approval_lock:
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


python_path = sys.executable
operating_system = platform.platform()
working_directory = os.getcwd()
home_directory = os.path.expanduser('~')

console = Console()

client = OpenAI(
    base_url='http://localhost:8080',
    api_key='sk-no-key-required'
)

MEMORY_FILE = Path("Bardgent.md")

def read_memory():
    if MEMORY_FILE.exists():
        console.print(f'\n[bold green]⚙ TOOL:[/bold green] READING MEMORY FROM Bardgent.md\n')
        return MEMORY_FILE.read_text(encoding="utf-8")
    return ""


def save_memory(text: str):
    with open(MEMORY_FILE, "a", encoding="utf-8") as f:
        f.write(f"\n- {text}\n")
    console.print(f'\n[bold green]⚙ TOOL:[/bold green] SAVING MEMORY TO Bardgent.md\n')
    return "Memory saved."

def WebSearch(query):
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
    resp = requests.post('https://html.duckduckgo.com/html/', data={'q': query},
                         headers=headers, timeout=10)
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
    print(results)
    return '\n\n'.join(results) if results else '(no results)'

def Fetch(link):
    console.print(Panel(link, title="[bold yellow]Fetch wants to run", border_style='yellow'))

    if not ask_approval('Fetch', "Fetch this page?"):
        return "Fetch rejected by user."

    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
    resp = requests.get(link, headers=headers, timeout=10)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    return soup.get_text(separator="\n", strip=True)


SYSTEM_INFO = f"""[CRITICAL SYSTEM INFO]:
- Python Executable Path: {python_path}
- Operating System: {operating_system}
- Current Working Directory: {working_directory}
- User Home Directory: {home_directory}"""

messages = [
    {
        "role": "system",
        "content": f"""
You are helpful agent and your name is Bardgent made by Bardia.

{SYSTEM_INFO}

You have access to these tools:


- read_memory(): read long-term memory
- save_memory(memory): save useful facts
- WebSearch: Websearch the web
- Fetch: Fetch web pages

Only save information that will be useful in future conversations.
Before answering questions that may depend on past context, call read_memory.
Only remember facts that the user states directly in the current message.
"""
    }
]

tools = [
    {
        "type": "function",
        "function": {
            "name": "read_memory",
            "description": "Read long-term memory.",
            "parameters": {
                "type": "object",
                "properties": {}
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "save_memory",
            "description": "Save useful user facts or preferences.",
            "parameters": {
                "type": "object",
                "properties": {
                    "memory": {
                        "type": "string"
                    }
                },
                "required": ["memory"]
            }
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
                }
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
                }
            }
    },

]


console.print(f'Welcome to [bold italic magenta]Bardgent[/bold italic magenta]')

while True:
    user_input = console.input(f'[bold green]>>>[/bold green] ')
    if user_input in ['exit', 'quit']:
        console.print('[bold red]Goodbye![/bold red]')
        break
    else:
        messages.append({
            'role': 'user',
            'content': user_input
        })
        response = client.chat.completions.create(
            model='yuxinlu1/gemma-4-12B-agentic-fable5-composer2.5-v2-3.5x-tau2-GGUF:Q4_K_M',
            messages=messages,
            temperature=0.2,
            tools=tools
        )

        assistant_message = response.choices[0].message

        while assistant_message.tool_calls and tool_iterations < 5:
            tool_iterations += 1
            messages.append(assistant_message)

            for tool_call in assistant_message.tool_calls:
                name = tool_call.function.name
                args = json.loads(tool_call.function.arguments or "{}")
                if name == "read_memory":
                    result = read_memory()
                elif name == "save_memory":
                    result = save_memory(args["memory"])
                elif name == "WebSearch":
                    result = WebSearch(args["query"])
                elif name == "Fetch":
                    result = Fetch(args["link"])
                else:
                    result = "Unknown tool"

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result
                })

            response = client.chat.completions.create(
                model="yuxinlu1/gemma-4-12B-agentic-fable5-composer2.5-v2-3.5x-tau2-GGUF:Q4_K_M",
                messages=messages,
                tools=tools,
                temperature=0.2
            )

            assistant_message = response.choices[0].message

        messages.append({
            "role": "assistant",
            "content": assistant_message.content
        })

        print(assistant_message.content)
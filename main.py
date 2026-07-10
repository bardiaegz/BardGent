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

You have access to two tools:


- read_memory(): read long-term memory
- save_memory(memory): save useful facts

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
    }
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

        while assistant_message.tool_calls:
            messages.append(assistant_message)

            for tool_call in assistant_message.tool_calls:
                name = tool_call.function.name
                args = json.loads(tool_call.function.arguments or "{}")
                if name == "read_memory":
                    result = read_memory()
                elif name == "save_memory":
                    result = save_memory(args["memory"])
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


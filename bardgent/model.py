"""Calls to the underlying chat-completions model: streaming (for the main,
single-Live-rendered loop) and blocking (for concurrently-run sub-agents,
where multiple Live renders would corrupt each other's terminal output)."""

import time

from rich.text import Text
from rich.console import Group
from rich.markdown import Markdown
from rich.spinner import Spinner
from rich.live import Live
import traceback

from bardgent import config
from bardgent.config import console, log_event, remove_thoughts


def render_agent(text):
    return Group(Text('AGENT:', style='bold cyan'), Markdown(text))


def print_usage(usage):
    if not usage:
        return
    in_tok, out_tok = usage.prompt_tokens, usage.completion_tokens
    console.print(f'[dim]tokens: {in_tok} in / {out_tok} out[/dim]')


def stream_agent_response(messages, tools):
    """Retry wrapper around _stream_agent_response_once()."""
    for attempt in range(1, config.MODEL_MAX_RETRIES + 1):
        try:
            return _stream_agent_response_once(messages, tools)
        except config.RETRYABLE_ERRORS as e:
            log_event(f"MODEL CALL FAILED (attempt {attempt}/{config.MODEL_MAX_RETRIES}): {type(e).__name__}: {e}")
            if attempt == config.MODEL_MAX_RETRIES:
                console.print(f"[bold red]Giving up after {config.MODEL_MAX_RETRIES} attempts: {type(e).__name__}: {e}[/bold red]")
                raise
            delay = config.MODEL_RETRY_DELAYS[min(attempt - 1, len(config.MODEL_RETRY_DELAYS) - 1)]
            console.print(
                f"[bold red]API error (attempt {attempt}/{config.MODEL_MAX_RETRIES})[/bold red]\n "
                f"[yellow]Retrying in {delay}s...[/yellow]"
            )
            time.sleep(delay)


def _stream_agent_response_once(messages, tools):
    """
    Calls the model with stream=True and renders the reply live:
      - Shows a spinner ("Thinking...") until the first chunk arrives.
      - As text tokens stream in, live-updates the rendered markdown.
      - Tool call arguments arrive split across many chunks; we only
        concatenate fragments by index and never json.loads() until the
        stream is fully consumed.

    Returns: (final_text, tool_calls, finish_reason)
    """
    stream = config.client.chat.completions.create(
        model=config.MODEL,
        messages=messages,
        tools=tools,
        temperature=config.TEMPERATURE,
        max_tokens=config.RESPONSE_TOKEN_RESERVE,
        stream=True,
        stream_options={'include_usage': True},
    )

    content_parts = []
    tool_calls = {}
    finish_reason = None
    usage = None
    has_output = False

    spinner = Spinner('dots', text=Text(' Thinking...', style='cyan'))

    with Live(spinner, console=console, refresh_per_second=12, transient=False, auto_refresh=False) as live:
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
                    has_output = True
                    live.update(render_agent(display_text))
                    live.refresh()
                else:
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

        if not has_output:
            live.update(Text(''))
            live.refresh()

    ordered_calls = [tool_calls[i] for i in sorted(tool_calls.keys()) if tool_calls[i].get('name')]
    final_text = ''.join(content_parts)
    final_text = remove_thoughts(final_text)
    return final_text, ordered_calls, finish_reason


def _call_model_once(messages, tools):
    """Blocking, non-streaming model call. Used by concurrently-run sub-agents."""
    response = config.client.chat.completions.create(
        model=config.MODEL,
        messages=messages,
        tools=tools,
        temperature=config.TEMPERATURE,
        max_tokens=config.RESPONSE_TOKEN_RESERVE,
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
    for attempt in range(1, config.MODEL_MAX_RETRIES + 1):
        try:
            return _call_model_once(messages, tools)
        except config.RETRYABLE_ERRORS as e:
            log_event(f"MODEL CALL (non-stream) FAILED (attempt {attempt}/{config.MODEL_MAX_RETRIES}): {type(e).__name__}: {e}")
            if attempt == config.MODEL_MAX_RETRIES:
                raise
            delay = config.MODEL_RETRY_DELAYS[min(attempt - 1, len(config.MODEL_RETRY_DELAYS) - 1)]
            time.sleep(delay)

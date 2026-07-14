"""
Bottom-of-terminal context usage bar (Claude Code style).

Pins a single status line to the last terminal row by:
  1. Setting the VT100 scroll region to all rows except the last
     (so chat output never scrolls the bar away).
  2. Painting the last row in place (save cursor -> move to last row ->
     clear -> write bar -> restore cursor).

While the user is at the USER: prompt, prompt_toolkit owns the screen; use
`make_bottom_toolbar(state)` as PromptSession(bottom_toolbar=...) and call
`suspend_status_bar()` / `resume_status_bar(state)` around prompt().

Works on standard terminals and Warp (no scrolling-into-chat fallback).
"""

import sys
import time
import shutil
import signal
import json
from bardgent.tool_schemas import TOOLS

from prompt_toolkit.formatted_text import HTML

from bardgent import config

_tools_token_cache = None
_status_bar_enabled = False
_last_bar_draw_at = [0.0]
MIN_BAR_REDRAW_INTERVAL = 0.08
_current_state_for_resize = [None]


def _term_size():
    size = shutil.get_terminal_size(fallback=(80, 24))
    return size.columns, size.lines


def _can_pin():
    return sys.stdout.isatty()


def _set_scroll_region(rows):
    """Reserve the last row for the status bar; content scrolls above it."""
    if rows < 3:
        return
    # DECSTBM (set scroll region) homes the cursor as a side effect on most
    # real terminals. Save/restore around it so setting/reasserting the
    # region never moves the visible cursor - otherwise you get either a
    # jump to the bottom row (old bug: explicit CUP after DECSTBM) or a jump
    # to the top-left that overwrites already-printed lines (what removing
    # that CUP naively causes instead).
    sys.stdout.write("\x1b7")                    # DECSC: save cursor
    sys.stdout.write(f"\x1b[1;{rows - 1}r")       # CSI r: scroll region 1..rows-1
    sys.stdout.write("\x1b8")                    # DECRC: restore cursor

def _reset_scroll_region():
    cols, rows = _term_size()
    sys.stdout.write("\x1b7")            # DECSC: save cursor
    sys.stdout.write("\x1b[r")           # full-screen scroll region
    if rows >= 1:
        sys.stdout.write(f"\x1b[{rows};1H")
        sys.stdout.write("\x1b[2K")      # clear the old status row
    sys.stdout.write("\x1b8")            # DECRC: restore cursor
    sys.stdout.flush()

def enable_status_bar():
    """Reserve the bottom row and mark the bar active."""
    global _status_bar_enabled
    if not _can_pin():
        return
    cols, rows = _term_size()
    if rows < 3:
        return
    _set_scroll_region(rows)
    sys.stdout.flush()
    _status_bar_enabled = True


def disable_status_bar():
    """Release the bottom row (e.g. on exit)."""
    global _status_bar_enabled
    if not _status_bar_enabled or not _can_pin():
        _status_bar_enabled = False
        return
    _reset_scroll_region()
    _status_bar_enabled = False


def suspend_status_bar():
    """Temporarily free the full screen for prompt_toolkit (USER: prompt)."""
    global _status_bar_enabled
    if not _status_bar_enabled or not _can_pin():
        return
    _reset_scroll_region()
    # Keep logical enabled flag False so we don't double-reset; resume will re-enable
    _status_bar_enabled = False


def resume_status_bar(state):
    """Re-pin the bar after prompt_toolkit returns control."""
    enable_status_bar()
    if state is not None:
        draw_status_bar(state, force=True)


# def _tools_token_estimate():
#     """One-time estimate of the TOOLS schema's token cost, sent with every
#     API call but invisible to the naive char-count estimate below."""
#     global _tools_token_cache
#     if _tools_token_cache is None:
#         _tools_token_cache = count_tokens(json.dumps(TOOLS))
#     return _tools_token_cache


def context_usage_tokens(state):
    """Real prompt-token usage as reported by the API on the last request.
    0 before the first exchange (e.g. right after /clear) — we don't
    estimate or guess, only display what the model actually told us."""
    return getattr(state, 'last_prompt_tokens', None) or 0

def _bar_color(pct):
    if pct < 0.5:
        return '32'  # green
    if pct < 0.8:
        return '33'  # yellow
    return '31'  # red


MODE_COLORS = {'plan': '36', 'normal': '37', 'auto': '31'}
MODE_LABELS = {'plan': 'PLAN', 'normal': 'NORMAL', 'auto': 'AUTO'}


def _status_fields(state):
    used = context_usage_tokens(state)
    pct = min(used / config.CONTEXT_WINDOW_TOKENS, 1.0)
    mode = state.mode if state else 'normal'
    return used, pct, mode


def format_status_bar(state, width):
    """ANSI string for the pinned bottom row (raw stdout)."""
    used, pct, mode = _status_fields(state)
    color = _bar_color(pct)
    mode_color = MODE_COLORS.get(mode, '37')
    mode_label = MODE_LABELS.get(mode, mode.upper())

    label = " Context: "
    stats = f" {used:,}/{config.CONTEXT_WINDOW_TOKENS:,} tokens ({pct * 100:.1f}%) "
    model_tag = f" model:{config.MODEL} "
    mode_tag = f" [{mode_label}]"

    overhead = len(label) + len(stats) + len(model_tag) + len(mode_tag) + 4
    bar_width = max(8, min(28, width - overhead))
    filled = int(bar_width * pct)
    bar = '█' * filled + '░' * (bar_width - filled)

    # Visible length without ANSI
    visible = label + f"[{bar}]" + stats + "|" + model_tag + mode_tag
    if len(visible) > width:
        short = f" Context: {used:,}/{config.CONTEXT_WINDOW_TOKENS:,} ({pct * 100:.0f}%) [{mode_label}] "
        if len(short) > width:
            short = short[: max(0, width - 1)]
        return (
            f"\x1b[{color}m{short}\x1b[0m"
            f"\x1b[K"  # clear to end of line
        )

    line = (
        f"\x1b[{color}m{label}[{bar}]{stats}\x1b[0m"
        f"\x1b[2m|{model_tag}\x1b[0m"
        f"\x1b[1;{mode_color}m{mode_tag}\x1b[0m"
        f"\x1b[K"
    )
    return line


def make_bottom_toolbar(state):
    """prompt_toolkit bottom_toolbar callable — pinned while at USER: prompt."""

    def toolbar():
        used, pct, mode = _status_fields(state)
        mode_label = MODE_LABELS.get(mode, mode.upper())
        bar_width = 20
        filled = int(bar_width * pct)
        bar = '█' * filled + '░' * (bar_width - filled)
        # Map pct to a simple color name prompt_toolkit understands
        if pct < 0.5:
            color = 'ansigreen'
        elif pct < 0.8:
            color = 'ansiyellow'
        else:
            color = 'ansired'
        mode_style = {
            'plan': 'ansicyan',
            'normal': 'ansiwhite',
            'auto': 'ansired',
        }.get(mode, 'ansiwhite')
        return HTML(
            f'<style fg="{color}">'
            f' Context: [{bar}] {used:,}/{config.CONTEXT_WINDOW_TOKENS:,} '
            f'({pct * 100:.1f}%) '
            f'</style>'
            f'<style fg="ansibrightblack">|</style>'
            f'<style fg="ansigray"> model:{config.MODEL} </style>'
            f'<style fg="{mode_style}"><b>[{mode_label}]</b></style>'
        )

    return toolbar


def draw_status_bar(state, force=False):
    """Repaint the reserved bottom row. Throttled unless force=True."""
    global _status_bar_enabled
    if not _can_pin() or state is None:
        return
    now = time.monotonic()
    if not force and (now - _last_bar_draw_at[0]) < MIN_BAR_REDRAW_INTERVAL:
        return
    _last_bar_draw_at[0] = now

    cols, rows = _term_size()
    if rows < 3:
        return

    # Re-assert scroll region every paint — Rich Live / other writers may reset it.
    _set_scroll_region(rows)
    _status_bar_enabled = True

    bar_text = format_status_bar(state, cols)
    # DECSC / CUP last row / EL / write / DECRC
    sys.stdout.write("\x1b7")
    sys.stdout.write(f"\x1b[{rows};1H")
    sys.stdout.write("\x1b[2K")
    sys.stdout.write(bar_text)
    sys.stdout.write("\x1b8")
    sys.stdout.flush()


def _handle_resize(signum, frame):
    if not _can_pin():
        return
    cols, rows = _term_size()
    if rows < 3:
        return
    if _status_bar_enabled:
        _set_scroll_region(rows)
        sys.stdout.flush()
    state = _current_state_for_resize[0]
    if state is not None:
        draw_status_bar(state, force=True)


def install_resize_handler(state):
    _current_state_for_resize[0] = state
    if hasattr(signal, 'SIGWINCH'):
        try:
            signal.signal(signal.SIGWINCH, _handle_resize)
        except (ValueError, OSError):
            pass
    draw_status_bar(state, force=True)

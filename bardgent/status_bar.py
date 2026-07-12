"""
Bottom-of-terminal context usage bar (Claude Code style).

Works by shrinking the terminal's scroll region to leave the very last row
free, then repeatedly repainting that last row in place (save cursor ->
jump to last row -> clear it -> draw the bar -> restore cursor).

Warp doesn't support this trick (it renders through its own block-based UI,
not a full VT100 grid), so on Warp we fall back to printing a plain,
scrolling status line at a few meaningful checkpoints instead.
"""

import sys
import time
import shutil
import signal

from bardgent import config
from bardgent.config import console
from bardgent.session import total_history_tokens
from bardgent.utils import count_tokens

_status_bar_enabled = False
_last_bar_draw_at = [0.0]
MIN_BAR_REDRAW_INTERVAL = 0.08
_last_plain_status_line = [None]
_current_state_for_resize = [None]


def _term_size():
    size = shutil.get_terminal_size(fallback=(80, 24))
    return size.columns, size.lines


def enable_status_bar():
    global _status_bar_enabled
    if not sys.stdout.isatty() or config.IS_WARP:
        return
    cols, rows = _term_size()
    if rows < 3:
        return
    sys.stdout.write(f"\x1b[1;{rows - 1}r")
    sys.stdout.write(f"\x1b[{rows - 1};1H")
    sys.stdout.flush()
    _status_bar_enabled = True


def disable_status_bar():
    global _status_bar_enabled
    if not _status_bar_enabled or not sys.stdout.isatty():
        return
    cols, rows = _term_size()
    sys.stdout.write("\x1b[r")
    sys.stdout.write(f"\x1b[{rows};1H")
    sys.stdout.write("\x1b[2K")
    sys.stdout.flush()
    _status_bar_enabled = False


def context_usage_tokens(state):
    used = count_tokens(state.messages[0].get('content', '')) if state.messages else 0
    used += total_history_tokens(state)
    return used


def _bar_color(pct):
    if pct < 0.5:
        return '32'
    if pct < 0.8:
        return '33'
    return '31'


MODE_COLORS = {'plan': '36', 'normal': '37', 'auto': '31'}
MODE_LABELS = {'plan': 'PLAN', 'normal': 'NORMAL', 'auto': 'AUTO'}


def format_status_bar(state, width):
    used = context_usage_tokens(state)
    pct = min(used / config.CONTEXT_WINDOW_TOKENS, 1.0)
    color = _bar_color(pct)

    mode_color = MODE_COLORS.get(state.mode, '37')
    mode_tag = f" \x1b[1;{mode_color}m[{MODE_LABELS.get(state.mode, state.mode.upper())}]\x1b[0m"

    label = f" Context: "
    stats = f" {used:,}/{config.CONTEXT_WINDOW_TOKENS:,} tokens ({pct * 100:.1f}%) "
    model_tag = f" model:{config.MODEL} "

    bar_width = max(10, min(30, width - len(label) - len(stats) - len(model_tag) - len(MODE_LABELS.get(state.mode, state.mode.upper())) - 8))
    filled = int(bar_width * pct)
    bar = '█' * filled + '░' * (bar_width - filled)

    line = f"\x1b[{color}m{label}[{bar}]{stats}\x1b[2m|{model_tag}\x1b[0m{mode_tag}"
    visible_len = (len(label) + 1 + bar_width + 1 + len(stats) + 1 + len(model_tag)
                   + len(MODE_LABELS.get(state.mode, state.mode.upper())) + 3)
    if visible_len > width:
        line = (f"\x1b[{color}m Context: {used:,}/{config.CONTEXT_WINDOW_TOKENS:,} ({pct * 100:.0f}%) \x1b[0m"
                f"\x1b[1;{mode_color}m[{MODE_LABELS.get(state.mode, state.mode.upper())}]\x1b[0m")
    return line


def print_status_line(state):
    """Fallback for terminals that can't do the pinned-row trick (Warp)."""
    if not sys.stdout.isatty():
        return
    used = context_usage_tokens(state)
    pct = min(used / config.CONTEXT_WINDOW_TOKENS, 1.0)
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
        f"[{color_name}]Context: [{bar}] {used:,}/{config.CONTEXT_WINDOW_TOKENS:,} tokens "
        f"({pct * 100:.1f}%)[/{color_name}] [dim]| model:{config.MODEL}[/dim] "
        f"[bold {mode_color_name}][{mode_label}][/bold {mode_color_name}]"
    )


def draw_status_bar(state, force=False):
    """Repaint the reserved bottom row. Throttled unless force=True."""
    if config.IS_WARP:
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
    sys.stdout.write("\x1b7")
    sys.stdout.write(f"\x1b[{rows};1H")
    sys.stdout.write("\x1b[2K")
    sys.stdout.write(bar_text)
    sys.stdout.write("\x1b8")
    sys.stdout.flush()


def _handle_resize(signum, frame):
    if not _status_bar_enabled:
        return
    cols, rows = _term_size()
    if rows < 3:
        return
    sys.stdout.write(f"\x1b[1;{rows - 1}r")
    sys.stdout.flush()


def install_resize_handler(state):
    _current_state_for_resize[0] = state
    if hasattr(signal, 'SIGWINCH'):
        try:
            signal.signal(signal.SIGWINCH, _handle_resize)
        except (ValueError, OSError):
            pass

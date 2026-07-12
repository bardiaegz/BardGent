"""Shell execution, with a persistent cwd across calls and a dangerous-
command allowlist/denylist that always requires an explicit y/N."""

import os
import re
import shlex
import subprocess

from rich.panel import Panel
from rich.markup import escape

from bardgent import config
from bardgent.config import console, log_event
from bardgent.permissions import PERMISSIONS, is_permitted_bash_prefix
from bardgent.state import ask_approval

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


def Bash(command, state, timeout=None):
    timeout = timeout or config.BASH_TIMEOUT_SECONDS
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
                console.print(Panel(escape(command), title=f"[bold {color}]{title}",
                                    subtitle=f"[dim]in {escape(state.shell_cwd)}", border_style=color))
            if not ask_approval(state, key, "Run this command?", dangerous=danger):
                return "Command rejected by user. Do NOT retry it or a variation of it, continue with what you already have, or ask the user in your final answer."
    # append a marker echoing $PWD so `cd` persists to the next Bash call
    wrapped = command + f'\nprintf "\\n{config.CWD_MARKER}%s" "$PWD"'
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
    stdout, sep, after = result.stdout.rpartition(config.CWD_MARKER)
    if sep:
        new_dir = after.strip()
        if new_dir and os.path.isdir(new_dir):
            state.shell_cwd = new_dir
        stdout = stdout[:-1] if stdout.endswith('\n') else stdout
    else:
        stdout = result.stdout
    log_event(f"[{state.name}] BASH: {command!r} (exit={result.returncode})")
    return stdout + result.stderr
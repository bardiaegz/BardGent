"""Shell execution, with a persistent cwd across calls, a dangerous-command
allowlist/denylist that always requires an explicit y/N, and background job
support for long-running commands (servers, watchers, long builds)."""

import os
import re
import time
import uuid
import shlex
import threading
import subprocess

from rich.panel import Panel
from rich.markup import escape

from bardgent import config
from bardgent.config import console, log_event
from bardgent.permissions import PERMISSIONS, is_permitted_bash_prefix
from bardgent.state import ask_approval
from bardgent.utils import truncate_output

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


# ---------------------------------------------------------------------------
# Background jobs (Bash run_in_background=true / ListJobs / Await)
# ---------------------------------------------------------------------------
# In-memory registry, cleared on process restart. Log files persist on disk
# under config.JOBS_DIR in case something needs to inspect them after the
# fact, but the registry itself (needed to poll/await a process) is only
# meaningful for the life of this bardgent process.
_jobs_lock = threading.RLock()
_jobs = {}  # job_id -> dict(process, command, log_path, log_file, start, cwd)


def _new_job_id():
    return uuid.uuid4().hex[:8]


def _launch_background(command, cwd):
    job_id = _new_job_id()
    log_path = config.JOBS_DIR / f'{job_id}.log'
    log_file = open(log_path, 'w', encoding='utf-8')
    process = subprocess.Popen(
        command, shell=True, cwd=cwd,
        stdout=log_file, stderr=subprocess.STDOUT, text=True,
    )
    with _jobs_lock:
        _jobs[job_id] = {
            'process': process,
            'command': command,
            'log_path': log_path,
            'log_file': log_file,
            'start': time.time(),
            'cwd': cwd,
        }
    return job_id


def _read_job_log(entry):
    try:
        entry['log_file'].flush()
    except Exception:
        pass
    try:
        return entry['log_path'].read_text(encoding='utf-8', errors='replace')
    except OSError:
        return '(could not read job log)'


def ListJobs():
    """List background jobs started via Bash(run_in_background=true)."""
    with _jobs_lock:
        if not _jobs:
            return '(no background jobs)'
        lines = []
        for job_id, entry in _jobs.items():
            ret = entry['process'].poll()
            status = 'running' if ret is None else f'exited({ret})'
            elapsed = time.time() - entry['start']
            lines.append(f"{job_id}  [{status}]  {elapsed:.0f}s elapsed  in {entry['cwd']}  $ {entry['command']}")
        return '\n'.join(lines)


def Await(job_id, timeout=None):
    """Wait for a background job to produce output or finish.

    Blocks up to `timeout` seconds (default config.BASH_AWAIT_DEFAULT_SECONDS,
    hard-capped at config.BASH_AWAIT_MAX_SECONDS) for the process to exit,
    then returns whatever it has written to its log so far. If the process is
    still running when the wait expires, this can be called again to keep
    checking - it does not kill the job.
    """
    with _jobs_lock:
        entry = _jobs.get(job_id)
    if entry is None:
        return f"Error: no job with id '{job_id}'. Use ListJobs() to see active jobs."

    timeout = min(float(timeout) if timeout else config.BASH_AWAIT_DEFAULT_SECONDS,
                  config.BASH_AWAIT_MAX_SECONDS)
    process = entry['process']
    try:
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        pass

    ret = process.poll()
    output = truncate_output(_read_job_log(entry))

    if ret is None:
        return (f"Job {job_id} still running after {timeout:.0f}s "
                f"(command: {entry['command']!r}).\n--- output so far ---\n{output or '(no output yet)'}")

    try:
        entry['log_file'].close()
    except Exception:
        pass
    log_event(f"BASH(bg) {job_id} EXITED code={ret}")
    return f"Job {job_id} finished (exit code {ret}).\n--- output ---\n{output or '(no output)'}"


def cleanup_jobs():
    """Best-effort termination of any still-running background jobs, for a
    clean process exit. Registered via atexit in main.py."""
    with _jobs_lock:
        for job_id, entry in _jobs.items():
            process = entry['process']
            if process.poll() is None:
                try:
                    process.terminate()
                except Exception:
                    pass
            try:
                entry['log_file'].close()
            except Exception:
                pass


def Bash(command, state, timeout=None, run_in_background=False):
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
                bg_note = " (background)" if run_in_background else ""
                title = ("Bash wants to run (DANGEROUS)" if danger else "Bash wants to run") + bg_note
                console.print(Panel(escape(command), title=f"[bold {color}]{title}",
                                    subtitle=f"[dim]in {escape(state.shell_cwd)}", border_style=color))
            if not ask_approval(state, key, "Run this command?", dangerous=danger):
                return "Command rejected by user. Do NOT retry it or a variation of it, continue with what you already have, or ask the user in your final answer."

    if run_in_background:
        job_id = _launch_background(command, state.shell_cwd)
        log_event(f"[{state.name}] BASH(bg) START {job_id}: {command!r}")
        return (
            f"Started background job {job_id} in {state.shell_cwd}.\n"
            f"Use Await(job_id=\"{job_id}\") to wait for it and collect output "
            f"(safe to call repeatedly while it's still running), or ListJobs() "
            f"to see all active jobs."
        )

    timeout = timeout or config.BASH_TIMEOUT_SECONDS
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
                f"If this command is expected to run long, re-run it with "
                f"run_in_background=true and poll it with Await()/ListJobs().")
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
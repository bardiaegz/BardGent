"""File-editing tools, with diff-confirmation UI, backups, and fuzzy-match
fallback for Edit(), plus a stale-file warning when disk content has
changed since it was last Read() in this session."""

import os
import re
import glob
import time
import shutil
import difflib
import fnmatch
import subprocess
from pathlib import Path

from rich.text import Text
from rich.panel import Panel
from rich.syntax import Syntax

from bardgent import config
from bardgent.config import console, log_event
from bardgent.checkpoints import make_git_checkpoint
from bardgent.permissions import is_tool_permitted
from bardgent.state import ask_approval

_known_mtimes = {}


def _record_mtime(path):
    try:
        _known_mtimes[path] = os.path.getmtime(path)
    except OSError:
        pass


def _stale_warning(path):
    prev = _known_mtimes.get(path)
    if prev is None:
        return ''
    try:
        current = os.path.getmtime(path)
    except OSError:
        return ''
    if current != prev:
        return (f"\n[NOTE: {path} changed on disk since it was last read in this session "
                 f"(external edit, or another process wrote to it). The content above/used "
                 f"here is the CURRENT version. refresh your understanding before making "
                 f"further assumptions about its old contents.]")
    return ''


def Read(file_path, offset=None, limit=None):
    """Read a file, optionally windowed by 1-based line `offset`/`limit`.

    - No offset/limit: returns the whole file, each line prefixed with its
      1-based line number (like `cat -n`), capped at config.READ_MAX_LIMIT
      lines total so a huge file can't blow the context window by itself.
    - offset only: starts at that line, using config.READ_DEFAULT_LIMIT lines.
    - offset + limit: reads exactly that window (limit still capped at
      config.READ_MAX_LIMIT).

    The "N\t" line-number prefix is display-only — never copy it into an
    Edit() old_str/new_str.
    """
    path = os.path.abspath(os.path.expanduser(file_path))
    warning = _stale_warning(path)
    try:
        with open(path, 'r', encoding='utf-8', errors='replace') as f:
            lines = f.readlines()
    except OSError as e:
        return f"Error reading {path}: {e}"

    total = len(lines)
    start = max((offset - 1), 0) if offset else 0

    if limit is not None:
        window = limit
    elif offset:
        window = config.READ_DEFAULT_LIMIT
    else:
        window = total

    window = min(window, config.READ_MAX_LIMIT)
    hard_end = min(start + window, total)
    selected = lines[start:hard_end]

    _record_mtime(path)

    # main.py separately truncates every tool's raw output to
    # config.MAX_TOOL_OUTPUT characters. If we let that be the thing that cuts
    # us off, the model only sees a generic char-count truncation note with no
    # line info, so it can't ask for the next window precisely. Instead, stop
    # ourselves first and emit the same "pass offset=N" hint we'd use for a
    # line-count cutoff, so behavior is identical either way.
    budget = max(config.MAX_TOOL_OUTPUT - 200, 500)
    out_lines = []
    used = 0
    end = start
    for i, line in enumerate(selected, start + 1):
        entry = f"{i:>6}\t{line}"
        if not entry.endswith('\n'):
            entry += '\n'
        if used + len(entry) > budget and out_lines:
            break
        out_lines.append(entry)
        used += len(entry)
        end = i

    numbered = ''.join(out_lines)

    note = ''
    if end < total:
        note = f"[showing lines {start + 1}-{end} of {total} total; pass offset={end + 1} to continue]\n"
    elif start > 0:
        note = f"[showing lines {start + 1}-{end} of {total} total]\n"

    return note + numbered + warning


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
                body.append(f"{old_no:>4} - {line[1:]}".ljust(bar_width), style=DEL_STYLE)
            body.append('\n')
            old_no += 1
        else:
            body.append(f"{new_no:>4}   {line[1:]}\n", style='dim')
            old_no += 1
            new_no += 1

    if not body:
        body = Text('(no changes)', style='dim')
    with state.approval_lock:
        console.print(Panel(body, title=f"[bold yellow]{tool_name}: {path}", border_style='yellow'))
        if is_tool_permitted(tool_name):
            console.print(f"[dim]auto-approved ({tool_name}) [permissions.json][/dim]")
            log_event(f"[{state.name}] approval '{tool_name}' -> permitted via permissions.json")
            return True
        return ask_approval(state, tool_name, "Apply this change?")


def _make_backup(path, old_content):
    """Save the pre-edit content of a file so it can be restored with Undo()."""
    ts = time.strftime('%Y%m%d_%H%M%S')
    backup_path = config.BACKUP_DIR / f"{Path(path).name}.{ts}.bak"
    backup_path.write_text(old_content, encoding='utf-8')
    config.last_backup[path] = backup_path
    return backup_path


def Write(file_path, content, state):
    path = os.path.abspath(os.path.expanduser(file_path))
    parent = os.path.dirname(path)
    old = ''
    existed = os.path.exists(path)
    if existed:
        with open(path, 'r') as f:
            old = f.read()
    if not confirm_diff(old, content, path, 'Write', state):
        return f"Write to {path} rejected by user. Do NOT retry it or a variation of it, continue with what you already have, or ask the user in your final answer."

    created_dirs = False
    if parent and not os.path.isdir(parent):
        os.makedirs(parent, exist_ok=True)
        created_dirs = True

    if existed:
        _make_backup(path, old)
    with open(path, 'w') as f:
        f.write(content)
    _record_mtime(path)
    log_event(f"[{state.name}] WRITE {path}" + (" (created parent dirs)" if created_dirs else ""))
    checkpoint = make_git_checkpoint(path, f"Write: {os.path.basename(path)}")
    checkpoint_note = f' [checkpoint {checkpoint[:10]}]' if checkpoint else ''
    suffix = ' (previous version backed up, use Undo to revert)' if existed else ''
    dirs_note = f' (created directory {parent})' if created_dirs else ''
    return f'Wrote to {path}{suffix}{dirs_note}{checkpoint_note}'


def find_fuzzy_match(content, old_str, threshold=0.6):
    """Fallback for when Edit's old_str doesn't match verbatim (e.g. minor
    whitespace/indentation drift, or content shifted slightly since the model
    last saw it). Slides a same-line-length window across the file and
    returns the closest-matching block if it clears `threshold`, else None."""
    old_lines = old_str.splitlines()
    content_lines = content.splitlines()
    n = len(old_lines)
    if n == 0 or len(content_lines) < n or len(content_lines) > 20000:
        return None, 0.0
    best_ratio = 0.0
    best_block = None
    for start in range(0, len(content_lines) - n + 1):
        block = '\n'.join(content_lines[start:start + n])
        ratio = difflib.SequenceMatcher(None, block, old_str).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            best_block = block
    if best_ratio >= threshold:
        return best_block, best_ratio
    return None, best_ratio


def Edit(file_path, old_str, new_str, state):
    path = os.path.abspath(os.path.expanduser(file_path))
    with open(path, 'r') as f:
        content = f.read()
    stale = _stale_warning(path)
    count = content.count(old_str)

    if count == 0:
        match, ratio = find_fuzzy_match(content, old_str)
        if match is None:
            hint = f" (closest block was only {ratio:.0%} similar)" if ratio else ''
            return f"Error: old_str not found in {path}{hint}. Re-Read the file and re-check the exact text.{stale}"
        console.print(Panel(
            Text(match), title=f"[bold yellow]Fuzzy match ({ratio:.0%} similar), old_str wasn't found verbatim",
            border_style='yellow'
        ))
        if not ask_approval(state, 'Edit_fuzzy', f"Use this {ratio:.0%}-similar block as the edit target instead?"):
            return f"Edit to {path} rejected, old_str not found exactly and the fuzzy match was declined. Re-Read the file and use the exact text."
        old_str = match
        count = content.count(old_str)
        if count != 1:
            return f"Error: the fuzzy-matched block occurs {count} times in {path}; make old_str more specific."
    elif count > 1:
        return f"Error: old_str matches {count} times in {path}, must be unique"

    new_content = content.replace(old_str, new_str)
    if not confirm_diff(content, new_content, path, 'Edit', state):
        return f"Edit to {path} rejected by user. Do NOT retry it or a variation of it, continue with what you already have, or ask the user in your final answer."
    _make_backup(path, content)
    with open(path, 'w') as f:
        f.write(new_content)
    _record_mtime(path)
    log_event(f"[{state.name}] EDIT {path}")
    checkpoint = make_git_checkpoint(path, f"Edit: {os.path.basename(path)}")
    checkpoint_note = f' [checkpoint {checkpoint[:10]}]' if checkpoint else ''
    return f'Edited {path} (previous version backed up, use Undo to revert){checkpoint_note}{stale}'


def Undo(file_path):
    path = os.path.abspath(os.path.expanduser(file_path))
    backup_path = config.last_backup.get(path)
    if not backup_path or not backup_path.exists():
        return f"No backup available for {path} in this session. If it's tracked in git, /checkpoints + /restore <n> can roll back the whole project instead."
    with open(path, 'w', encoding='utf-8') as f:
        f.write(backup_path.read_text(encoding='utf-8'))
    _record_mtime(path)
    log_event(f"UNDO {path} <- {backup_path.name}")
    del config.last_backup[path]
    return f"Restored {path} from backup {backup_path.name}."


def Glob(pattern):
    matches = glob.glob(os.path.expanduser(pattern), recursive=True)
    return '\n'.join(matches) if matches else '(no matches)'


MAX_GREP_MATCHES = 200
SKIP_DIRS = {'__pycache__', 'node_modules', 'venv', '.venv', 'dist', 'build'}

_RIPGREP_PATH = shutil.which('rg')


def _ripgrep_grep(pattern, root, include):
    cmd = [_RIPGREP_PATH, '--line-number', '--no-heading', '--color=never', '-e', pattern]
    if include:
        cmd += ['-g', include]
    cmd.append(root)
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError) as e:
        return None, f"Error running ripgrep: {type(e).__name__}: {e}"

    # rg exit codes: 0 = matches found, 1 = no matches (not an error), 2+ = real error.
    if result.returncode not in (0, 1):
        return None, f"Error running ripgrep: {result.stderr.strip() or 'unknown error'}"

    out = result.stdout.strip('\n')
    if not out:
        return '(no matches)', None

    lines = out.split('\n')
    truncated = len(lines) > MAX_GREP_MATCHES
    lines = lines[:MAX_GREP_MATCHES]
    rel_lines = []
    for line in lines:
        head, sep, rest = line.partition(':')
        if sep:
            try:
                rel = os.path.relpath(head, root)
            except ValueError:
                rel = head
            rel_lines.append(f"{rel}:{rest}")
        else:
            rel_lines.append(line)
    text = '\n'.join(rel_lines)
    if truncated:
        text += f"\n(stopped at {MAX_GREP_MATCHES} matches)"
    return text, None


def _python_grep(pattern, root, include):
    """Pure-Python fallback used when ripgrep isn't installed on the system."""
    try:
        regex = re.compile(pattern)
    except re.error as e:
        return f"Error: invalid regex: {e}"
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


def Grep(pattern, path='.', include=None):
    """Search file contents for a regex pattern.

    Uses the system `rg` (ripgrep) binary when available, since it's far
    faster on large trees and already understands .gitignore; falls back to
    a pure-Python walk (identical output format) when rg isn't installed.
    """
    root = os.path.abspath(os.path.expanduser(path))
    if _RIPGREP_PATH:
        text, err = _ripgrep_grep(pattern, root, include)
        if err is None:
            return text
        # ripgrep itself failed (bad regex, missing path, etc.) - surface it
        # rather than silently falling back, since the Python path may behave
        # differently on regex edge cases.
        return err
    return _python_grep(pattern, root, include)
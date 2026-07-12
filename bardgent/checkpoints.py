"""
Silent, git-backed project checkpoints. Every applied Write/Edit inside a
git repo gets snapshotted onto a side ref (refs/bardgent/checkpoints) via a
throwaway index file, so it never touches the user's real HEAD, branch, or
staged changes. /checkpoints lists them, /restore <n> rolls the working
tree back to one.
"""

import os
import json
import time
import subprocess

from bardgent import config
from bardgent.config import log_event


def _git_root(path):
    try:
        result = subprocess.run(
            ['git', 'rev-parse', '--show-toplevel'],
            cwd=os.path.dirname(os.path.abspath(path)) or '.',
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return None


def _load_checkpoint_log():
    if config.CHECKPOINT_LOG.exists():
        try:
            return json.loads(config.CHECKPOINT_LOG.read_text(encoding='utf-8'))
        except (OSError, json.JSONDecodeError):
            return []
    return []


def _save_checkpoint_log(entries):
    try:
        config.PERMISSIONS_DIR.mkdir(exist_ok=True)
        config.CHECKPOINT_LOG.write_text(json.dumps(entries, indent=2), encoding='utf-8')
    except OSError as e:
        log_event(f"CHECKPOINT LOG SAVE FAILED: {e}")


def make_git_checkpoint(path, message):
    root = _git_root(path)
    if not root:
        return None
    try:
        env = os.environ.copy()
        env['GIT_INDEX_FILE'] = str(config.CHECKPOINT_INDEX_FILE)
        subprocess.run(['git', 'add', '-A'], cwd=root, env=env, capture_output=True, text=True, timeout=20)
        tree = subprocess.run(['git', 'write-tree'], cwd=root, env=env, capture_output=True, text=True, timeout=20)
        if tree.returncode != 0:
            log_event(f"CHECKPOINT write-tree failed: {tree.stderr.strip()}")
            return None
        tree_hash = tree.stdout.strip()

        parent_args = []
        parent = subprocess.run(['git', 'rev-parse', config.CHECKPOINT_REF], cwd=root, capture_output=True, text=True, timeout=5)
        if parent.returncode == 0:
            parent_args = ['-p', parent.stdout.strip()]
        else:
            head = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=root, capture_output=True, text=True, timeout=5)
            if head.returncode == 0:
                parent_args = ['-p', head.stdout.strip()]

        commit = subprocess.run(
            ['git', 'commit-tree', tree_hash, *parent_args, '-m', message],
            cwd=root, env=env, capture_output=True, text=True, timeout=20,
        )
        if commit.returncode != 0:
            log_event(f"CHECKPOINT commit-tree failed: {commit.stderr.strip()}")
            return None
        commit_hash = commit.stdout.strip()

        upd = subprocess.run(['git', 'update-ref', config.CHECKPOINT_REF, commit_hash], cwd=root, capture_output=True, text=True, timeout=10)
        if upd.returncode != 0:
            log_event(f"CHECKPOINT update-ref failed: {upd.stderr.strip()}")
            return None

        entries = _load_checkpoint_log()
        entries.append({
            'time': time.strftime('%Y-%m-%d %H:%M:%S'),
            'message': message,
            'commit': commit_hash,
            'root': root,
        })
        entries = entries[-100:]
        _save_checkpoint_log(entries)
        log_event(f"CHECKPOINT {commit_hash[:10]} ({message})")
        return commit_hash
    except (OSError, subprocess.SubprocessError) as e:
        log_event(f"CHECKPOINT failed: {type(e).__name__}: {e}")
        return None


def list_checkpoints():
    entries = _load_checkpoint_log()
    if not entries:
        return '(no checkpoints yet. checkpoints are created automatically on Write/Edit inside a git repo)'
    lines = []
    for i, e in enumerate(entries, 1):
        lines.append(f"{i}. {e['time']}  {e['commit'][:10]}  {e['message']}")
    return '\n'.join(lines)


def restore_checkpoint(index):
    entries = _load_checkpoint_log()
    try:
        idx = int(index)
    except (TypeError, ValueError):
        return f"Invalid checkpoint index: {index!r}"
    if idx < 1 or idx > len(entries):
        return f"No checkpoint at index {idx}. Use /checkpoints to see valid indices."
    entry = entries[idx - 1]
    root, commit = entry['root'], entry['commit']
    try:
        result = subprocess.run(
            ['git', 'checkout', commit, '--', '.'],
            cwd=root, capture_output=True, text=True, timeout=30,
        )
        if result.returncode != 0:
            return f"Restore failed: {result.stderr.strip()}"
    except (OSError, subprocess.SubprocessError) as e:
        return f"Restore failed: {type(e).__name__}: {e}"
    log_event(f"RESTORED checkpoint #{idx} ({commit[:10]})")
    return f"Working tree in {root} restored to checkpoint #{idx} ({entry['time']}, {commit[:10]}). Your git branch/HEAD/index are untouched. only file contents were overwritten."

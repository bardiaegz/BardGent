"""
Custom tool permissions file: lets a user pre-approve certain bash command
prefixes (and specific tools) per-project without re-approving every
session. Lives at .bardgent/permissions.json in the current directory.
"""

import json

from bardgent import config
from bardgent.config import console

DEFAULT_PERMISSIONS = {
    "auto_approve_bash_prefixes": [],
    "auto_approve_tools": [],
    "extra_dangerous_patterns": []
}


def load_permissions():
    permissions_file = config.PERMISSIONS_DIR / 'permissions.json'
    if permissions_file.exists():
        try:
            data = json.loads(permissions_file.read_text(encoding='utf-8'))
        except (OSError, json.JSONDecodeError) as e:
            console.print(f"[dim red]Could not read {permissions_file}: {e}. Using defaults.[/dim red]")
            return dict(DEFAULT_PERMISSIONS)
        merged = dict(DEFAULT_PERMISSIONS)
        for k in DEFAULT_PERMISSIONS:
            if k in data and isinstance(data[k], list):
                merged[k] = data[k]
        return merged
    try:
        config.PERMISSIONS_DIR.mkdir(exist_ok=True)
        permissions_file.write_text(json.dumps(DEFAULT_PERMISSIONS, indent=2), encoding='utf-8')
    except OSError:
        pass
    return dict(DEFAULT_PERMISSIONS)


PERMISSIONS = load_permissions()


def is_permitted_bash_prefix(command):
    cmd = command.strip()
    for prefix in PERMISSIONS.get('auto_approve_bash_prefixes', []):
        prefix = prefix.strip()
        if not prefix:
            continue
        if cmd == prefix or cmd.startswith(prefix + ' '):
            return True
    return False


def is_tool_permitted(name):
    return name in PERMISSIONS.get('auto_approve_tools', [])

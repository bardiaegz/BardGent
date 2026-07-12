"""
Skills: Claude-Code style, auto-detected capability packs.

A "skill" is a folder containing a SKILL.md file:

    skills/
      docx/
        SKILL.md
        reference.py        <- any bundled scripts/templates the skill needs
      git-commit/
        SKILL.md

SKILL.md starts with a small frontmatter block:

    ---
    name: docx
    description: Use this skill whenever the user wants to create or edit Word documents...
    ---
    (full body: step by step instructions, conventions, gotchas, examples)

How auto-detection works (mirrors Claude Code):
  1. At startup we scan every skill directory and read ONLY the frontmatter
     of each SKILL.md (cheap - a few lines each). This gives a catalogue of
     "name: description" pairs.
  2. That catalogue (not the full bodies) is embedded in the system prompt,
     so the model always knows what capabilities exist without spending
     tokens on content it may never need.
  3. Before starting a task whose description matches, the model is
     instructed to call the `Skill(name)` tool, which loads and returns the
     FULL body of that one SKILL.md on demand. The model decides which
     skill(s) apply, and can load more than one per task if several match.
  4. Any extra files that live in that skill's folder (scripts, templates,
     reference docs) are just plain files on disk - the model can find and
     read them with Glob/Grep/Read/Bash once it knows the folder path
     (returned alongside the skill body).

Skills are looked up in three places, in this priority order, so a project
can override a user override a bundled default of the same name:
  1. ./.bardgent/skills/<name>/SKILL.md   (project-local)
  2. ~/.bardgent/skills/<name>/SKILL.md   (user-global, installed once)
  3. <package>/../skills/<name>/SKILL.md  (bundled with bardgent itself)
"""

from pathlib import Path
from urllib.parse import urlparse

from bardgent import config

FRONTMATTER_DELIM = "---"

_PACKAGE_ROOT = Path(__file__).resolve().parent.parent

SKILL_DIRS = [
    Path.cwd() / ".bardgent" / "skills",     # project-local (highest priority)
    config.GLOBAL_DIR / "skills",            # user-global
    _PACKAGE_ROOT / "skills",                # bundled with bardgent
]


def _parse_frontmatter(text):
    """Tiny `key: value` frontmatter parser - no external YAML dependency."""
    lines = text.splitlines()
    if not lines or lines[0].strip() != FRONTMATTER_DELIM:
        return {}, text
    meta = {}
    i = 1
    while i < len(lines) and lines[i].strip() != FRONTMATTER_DELIM:
        line = lines[i]
        if ':' in line:
            key, _, value = line.partition(':')
            meta[key.strip()] = value.strip().strip('"').strip("'")
        i += 1
    body = '\n'.join(lines[i + 1:]).strip()
    return meta, body


def discover_skills():
    """Scan all skill directories, return {name: {'description', 'path', 'dir'}}.

    Directories earlier in SKILL_DIRS win on name collisions (project-local
    overrides user-global overrides bundled), since we only set a name the
    first time we see it.
    """
    registry = {}
    for base in SKILL_DIRS:
        if not base.is_dir():
            continue
        for entry in sorted(base.iterdir()):
            skill_md = entry / "SKILL.md"
            if not (entry.is_dir() and skill_md.exists()):
                continue
            try:
                text = skill_md.read_text(encoding='utf-8')
            except OSError:
                continue
            meta, _ = _parse_frontmatter(text)
            name = meta.get('name') or entry.name
            if name in registry:
                continue  # higher-priority dir already claimed this name
            registry[name] = {
                'description': meta.get('description') or '(no description provided)',
                'path': str(skill_md),
                'dir': str(entry),
            }
    return registry


# Computed once at import time; refresh_skills() can rescan mid-session
# (e.g. after the user drops a new skill folder in without restarting).
SKILL_REGISTRY = discover_skills()


def refresh_skills():
    global SKILL_REGISTRY
    SKILL_REGISTRY = discover_skills()
    return SKILL_REGISTRY


def format_skills_catalogue(registry=None):
    registry = registry if registry is not None else SKILL_REGISTRY
    if not registry:
        return '(no skills installed)'
    lines = []
    for name, info in sorted(registry.items()):
        lines.append(f"- {name}: {info['description']}")
    return '\n'.join(lines)


def load_skill(name, registry=None):
    """Return the full SKILL.md body (instructions) for `name`, on demand."""
    registry = registry if registry is not None else SKILL_REGISTRY
    info = registry.get(name)
    if not info:
        available = ', '.join(sorted(registry.keys())) or '(none)'
        return f"Error: no skill named '{name}'. Available skills: {available}. Use list_skills() to double check."
    try:
        text = Path(info['path']).read_text(encoding='utf-8')
    except OSError as e:
        return f"Error reading skill '{name}': {e}"
    _, body = _parse_frontmatter(text)
    return (
        f"[Skill loaded: {name}]\n"
        f"(folder: {info['dir']} -- if this skill mentions bundled scripts, templates, "
        f"or reference files, they live in that folder; use Read/Glob/Bash to access them)\n\n"
        f"{body}"
    )


def list_skills_text(registry=None):
    registry = registry if registry is not None else SKILL_REGISTRY
    if not registry:
        return ('(no skills installed)\n'
                f'Add one by creating {config.GLOBAL_DIR / "skills"}/<name>/SKILL.md '
                '(or ./.bardgent/skills/<name>/SKILL.md for a project-local skill).')
    lines = ['Installed skills:']
    for name, info in sorted(registry.items()):
        lines.append(f"  {name} - {info['description']}  (folder: {info['dir']})")
    return '\n'.join(lines)


def install_skill_from_github(github_url: str) -> str:
    """Install a skill from a GitHub repository.
    
    Args:
        github_url: GitHub repository URL (e.g., https://github.com/alirezarezvani/claude-skills)
    
    Returns:
        Status message string
    """
    import subprocess
    import tempfile
    import shutil
    from urllib.parse import urlparse
    
    # Parse the GitHub URL
    parsed = urlparse(github_url)
    if parsed.netloc != 'github.com':
        return f"[bold red]Error:[/bold red] Only GitHub URLs are supported (got {parsed.netloc})"
    
    path_parts = parsed.path.strip('/').split('/')
    if len(path_parts) < 2:
        return "[bold red]Error:[/bold red] Invalid GitHub URL format. Expected: https://github.com/owner/repo"
    
    owner, repo = path_parts[0], path_parts[1]
    
    # Determine target directory (user-global skills directory)
    target_dir = config.GLOBAL_DIR / "skills"
    target_dir.mkdir(parents=True, exist_ok=True)
    
    # Clone to a temporary directory
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_path = Path(tmpdir) / repo
        try:
            # Clone the repository
            result = subprocess.run(
                ['git', 'clone', '--depth', '1', github_url, str(repo_path)],
                capture_output=True,
                text=True,
                timeout=60
            )
            if result.returncode != 0:
                return f"[bold red]Error cloning repository:[/bold red] {result.stderr.strip()}"
            
            # Find all skill directories (directories containing SKILL.md) - recursively
            skill_dirs = []
            for skill_md in repo_path.rglob("SKILL.md"):
                skill_dirs.append(skill_md.parent)
            
            if not skill_dirs:
                return "[bold yellow]No skills found in repository (no directories with SKILL.md found).[/bold yellow]"
            
            installed = []
            skipped = []
            
            for skill_dir in skill_dirs:
                skill_name = skill_dir.name
                target_skill_dir = target_dir / skill_name
                
                if target_skill_dir.exists():
                    skipped.append(skill_name)
                    continue
                
                # Copy the skill directory
                shutil.copytree(skill_dir, target_skill_dir)
                installed.append(skill_name)
            
            # Refresh the skills registry
            refresh_skills()
            
            # Build result message
            lines = []
            if installed:
                lines.append(f"[bold green]Installed {len(installed)} skill(s):[/bold green]")
                for name in installed:
                    lines.append(f"  [green]✓[/green] {name}")
            if skipped:
                lines.append(f"[yellow]Skipped {len(skipped)} already installed:[/yellow]")
                for name in skipped:
                    lines.append(f"  [yellow]⊘[/yellow] {name} (already exists)")
            
            return '\n'.join(lines)
            
        except subprocess.TimeoutExpired:
            return "[bold red]Error:[/bold red] Git clone timed out (60s timeout)"
        except Exception as e:
            return f"[bold red]Error:[/bold red] {str(e)}"
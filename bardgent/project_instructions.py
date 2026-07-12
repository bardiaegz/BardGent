"""Load project-level agent instructions for injection into system prompts.

Checked (in order). Every existing file is included, labelled by path, so a
project can layer general rules (AGENTS.md) with bardgent-specific notes
(.bardgent/instructions.md) without either being discarded.

  1. ./AGENTS.md
  2. ./CLAUDE.md
  3. ./.bardgent/AGENTS.md
  4. ./.bardgent/instructions.md
  5. ./.bardgent/RULES.md
"""

from pathlib import Path

from bardgent import config

# Relative to the process cwd at load time (same as the rest of bardgent).
INSTRUCTION_CANDIDATES = (
    'AGENTS.md',
    'CLAUDE.md',
    '.bardgent/AGENTS.md',
    '.bardgent/instructions.md',
    '.bardgent/RULES.md',
)

# Cap so a huge AGENTS.md cannot blow the context window alone.
MAX_TOTAL_CHARS = 40_000
MAX_PER_FILE_CHARS = 20_000


def _read_instruction_file(path: Path) -> str | None:
    try:
        if not path.is_file():
            return None
        text = path.read_text(encoding='utf-8', errors='replace').strip()
    except OSError:
        return None
    if not text:
        return None
    if len(text) > MAX_PER_FILE_CHARS:
        text = (
            text[:MAX_PER_FILE_CHARS]
            + f"\n\n... [truncated, file was {len(text)} chars; "
            f"showing first {MAX_PER_FILE_CHARS}]"
        )
    return text


def load_project_instructions(cwd=None):
    """Return a single markdown block of project instructions, or '' if none."""
    root = Path(cwd) if cwd is not None else Path.cwd()
    sections = []
    total = 0
    for rel in INSTRUCTION_CANDIDATES:
        path = root / rel
        text = _read_instruction_file(path)
        if text is None:
            continue
        header = f"### From {rel}\n\n"
        block = header + text
        if total + len(block) > MAX_TOTAL_CHARS:
            remaining = MAX_TOTAL_CHARS - total
            if remaining > 200:
                sections.append(
                    block[:remaining]
                    + "\n\n... [further project instructions truncated]"
                )
            break
        sections.append(block)
        total += len(block)

    if not sections:
        return ''
    return (
        "[PROJECT INSTRUCTIONS — follow these for this repository; "
        "they override general defaults when they conflict]:\n\n"
        + "\n\n".join(sections)
    )


def format_project_instructions_section(cwd=None):
    """Same as load_project_instructions, or a short note when nothing is found."""
    body = load_project_instructions(cwd=cwd)
    if body:
        return body
    return (
        "[PROJECT INSTRUCTIONS]: (none found. Optional files: AGENTS.md, "
        "CLAUDE.md, .bardgent/AGENTS.md, .bardgent/instructions.md, "
        ".bardgent/RULES.md)"
    )

"""Long-term memory: a flat Bardgent.md bullet list in the global config dir."""

from bardgent import config
from bardgent.config import log_event


def read_memory():
    if config.MEMORY_FILE.exists():
        return config.MEMORY_FILE.read_text(encoding='utf-8')
    return ''


def save_memory(text: str):
    text = text.strip()
    memories = set()
    if config.MEMORY_FILE.exists():
        for line in config.MEMORY_FILE.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line.startswith("-"):
                memories.add(line[1:].strip().lower())

    if text.lower() in memories:
        return "Memory already exists."

    with open(config.MEMORY_FILE, "a", encoding="utf-8") as f:
        f.write(f"\n- {text}\n")

    log_event(f"MEMORY SAVE: {text}")
    return "Memory saved."


def memory_context_block():
    """Formatted saved memories, ready to embed directly in a system prompt.

    This is what makes memory actually reliable: instead of depending on the
    model remembering to call read_memory() before every question that might
    need it (it often won't), the current contents are baked straight into
    the system message and refreshed every turn (see system_prompt.py /
    main.py). The read_memory/save_memory/list_memory/delete_memory tools
    stay available for the model to manage memory explicitly, but answering
    from known facts no longer requires a tool call at all.
    """
    mem = read_memory().strip()
    if not mem:
        return '(none saved yet)'
    return mem


def list_memory():
    """Return saved memories as a numbered list so the user/model can reference an index to delete."""
    if not config.MEMORY_FILE.exists():
        return '(no memories saved)'
    mem_lines = [l.strip() for l in config.MEMORY_FILE.read_text(encoding='utf-8').splitlines() if l.strip().startswith('-')]
    if not mem_lines:
        return '(no memories saved)'
    return '\n'.join(f'{i}. {l[1:].strip()}' for i, l in enumerate(mem_lines, 1))


def delete_memory(index):
    """Delete a memory by its 1-based index as shown by list_memory()."""
    if not config.MEMORY_FILE.exists():
        return 'No memory file exists.'
    try:
        idx = int(index)
    except (TypeError, ValueError):
        return f"Error: index must be an integer, got {index!r}. Call list_memory() first."
    lines = config.MEMORY_FILE.read_text(encoding='utf-8').splitlines()
    mem_line_positions = [i for i, l in enumerate(lines) if l.strip().startswith('-')]
    if idx < 1 or idx > len(mem_line_positions):
        return f"Error: no memory at index {idx}. Use list_memory() to see valid indices."
    removed = lines[mem_line_positions[idx - 1]]
    del lines[mem_line_positions[idx - 1]]
    config.MEMORY_FILE.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    log_event(f"MEMORY DELETE #{idx}: {removed}")
    return f'Deleted memory #{idx} ({removed.strip("- ")}).'
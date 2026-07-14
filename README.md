# Bardgent

A terminal coding agent with tools for files, shell, web, memory, skills, sub-agents, and scheduled tasks.

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -e .
```

Set an API key (env name is historical; any OpenAI-compatible key works with the configured base URL):

```bash
# ~/.bardgent/.env  or export in your shell
GEMINI_API_KEY=your_key_here
```

Then run:

```bash
bardgent
```

## Defaults

| Setting | Value |
|---------|--------|
| API base | `https://integrate.api.nvidia.com/v1` |
| Model | `nvidia/nemotron-3-ultra-550b-a55b` |
| Config dir | `~/.bardgent/` |

## Modes

| Mode | Behavior |
|------|----------|
| **normal** | Approve Write/Edit/Bash (and similar) per action |
| **auto** | Auto-approve non-dangerous actions; dangerous shell still prompts |
| **plan** | Read-only tools only; agent proposes a plan, then you choose how to execute |

Cycle with **Shift+Tab** or `/normal`, `/auto`, `/plan`.

## Tools (high level)

- **Files:** Read, Write, Edit, Undo, Glob, Grep  
- **Shell:** Bash (cwd persists), background jobs via `ListJobs` / `Await`  
- **Web:** WebSearch (DuckDuckGo), Fetch  
- **Memory:** save / list / delete long-term notes (`~/.bardgent/Bardgent.md`)  
- **Skills:** Claude Code–style `SKILL.md` packs; `/skills`, `/skill install <github_url>`  
- **Delegation:** `Task` / `Tasks` (parallel sub-agents)  
- **Schedule:** recurring or one-off tasks; `/schedule`, `/schedules`

## Project instructions

Place any of these in the project root (all that exist are loaded):

- `AGENTS.md`
- `.bardgent/AGENTS.md`, `.bardgent/instructions.md`, `.bardgent/RULES.md`

## Skills locations

1. `./.bardgent/skills/<name>/SKILL.md` (project)
2. `~/.bardgent/skills/<name>/SKILL.md` (user)
3. Package `skills/` (optional bundled defaults)

## Permissions

Per-project file: `.bardgent/permissions.json`

```json
{
  "auto_approve_bash_prefixes": ["git status", "pytest"],
  "auto_approve_tools": ["Fetch", "Write", "Edit"],
  "extra_dangerous_patterns": []
}
```

## Useful commands

Type `/help` in the REPL for the full list. Highlights: `/resume`, `/checkpoints`, `/restore`, `/telegram`, `/summary`, `/clear`.

## License

Use and modify as you like for personal / project use.

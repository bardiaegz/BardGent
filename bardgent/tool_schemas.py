"""OpenAI-style tool schemas, argument validation, and the single dispatch
point every tool call (main loop or sub-agent) goes through."""

from bardgent import config, memory, skills, scheduler
from bardgent.config import log_event
from bardgent.web_tools import WebSearch, Fetch
from bardgent.fs_tools import Read, Write, Edit, Undo, Glob, Grep
from bardgent.exec_tools import Bash, ListJobs, Await

TOOLS = [
    {'type': 'function', 'function': {
        'name': 'read_memory', 'description': 'Read long-term memory.',
        'parameters': {'type': 'object', 'properties': {}},
    }},
    {'type': 'function', 'function': {
        'name': 'save_memory', 'description': 'Save useful user facts or preferences.',
        'parameters': {'type': 'object', 'properties': {
            'memory': {'type': 'string'}
        }, 'required': ['memory']},
    }},
    {'type': 'function', 'function': {
        'name': 'list_memory', 'description': 'List saved memories with their index numbers.',
        'parameters': {'type': 'object', 'properties': {}},
    }},
    {'type': 'function', 'function': {
        'name': 'delete_memory', 'description': 'Delete a saved memory by the index shown in list_memory().',
        'parameters': {'type': 'object', 'properties': {
            'index': {'type': 'integer', 'description': '1-based index from list_memory()'}
        }, 'required': ['index']},
    }},
    {'type': 'function', 'function': {
        'name': 'list_skills',
        'description': 'List every installed skill with its name, description, and folder. Use this if you are unsure what skills are available, or to double-check a name before calling Skill().',
        'parameters': {'type': 'object', 'properties': {}},
    }},
    {'type': 'function', 'function': {
        'name': 'Skill',
        'description': (
            'Load the full instructions for one installed skill by name. A catalogue of '
            '"name: description" pairs for every installed skill is included in your system '
            'prompt - whenever the current task matches a skill\'s description, call Skill(name) '
            'BEFORE starting that task (the same way Claude Code loads a SKILL.md before acting), '
            'then follow its instructions. You may load more than one skill per task if several '
            'match. The skill body may reference bundled scripts/templates/resources living in the '
            'same folder; use Read/Glob/Bash to reach those once you know the folder path.'
        ),
        'parameters': {'type': 'object', 'properties': {
            'name': {'type': 'string', 'description': 'the skill name, exactly as shown in the catalogue'}
        }, 'required': ['name']},
    }},
    {'type': 'function', 'function': {
        'name': 'Fetch', 'description': 'Fetch the content of a web page',
        'parameters': {'type': 'object', 'properties': {
            'link': {'type': 'string', 'description': 'the link of the web page to fetch'}
        }, 'required': ['link']},
    }},
    {'type': 'function', 'function': {
        'name': 'WebSearch',
        'description': 'Search the web (DuckDuckGo), returns titles, URLs and snippets. Use Fetch afterwards to read a promising result.',
        'parameters': {'type': 'object', 'properties': {
            'query': {'type': 'string', 'description': 'the search query'}
        }, 'required': ['query']},
    }},
    {'type': 'function', 'function': {
        'name': 'Read',
        'description': (
            'Read a file from disk. Each returned line is prefixed with its 1-based line '
            'number (e.g. "12\\t..."), which is display-only - never include it when building '
            'an Edit() old_str/new_str. By default returns the whole file, capped at '
            f'{config.READ_MAX_LIMIT} lines; use offset (and optionally limit) to page through '
            'files larger than that.'
        ),
        'parameters': {'type': 'object', 'properties': {
            'file_path': {'type': 'string'},
            'offset': {'type': 'integer', 'description': f'1-based line number to start reading from. Defaults to line 1; when set without limit, reads {config.READ_DEFAULT_LIMIT} lines from there.'},
            'limit': {'type': 'integer', 'description': f'max number of lines to return (capped at {config.READ_MAX_LIMIT})'},
        }, 'required': ['file_path']}
    }},
    {'type': 'function', 'function': {
        'name': 'Write', 'description': 'Write (overwrite) full content to a file. Creates missing parent directories automatically. Backs up any existing file first.',
        'parameters': {'type': 'object', 'properties': {
            'file_path': {'type': 'string', 'description': 'the path of the file to write to'},
            'content': {'type': 'string', 'description': 'the content to write to the file'}
        }, 'required': ['file_path', 'content']}
    }},
    {'type': 'function', 'function': {
        'name': 'Edit', 'description': 'Replace an exact string match inside a file (must match exactly once). Backs up the file first.',
        'parameters': {'type': 'object', 'properties': {
            'file_path': {'type': 'string'},
            'old_str': {'type': 'string', 'description': 'exact text to find'},
            'new_str': {'type': 'string', 'description': 'text to replace it with'}
        }, 'required': ['file_path', 'old_str', 'new_str']}
    }},
    {'type': 'function', 'function': {
        'name': 'Undo', 'description': 'Restore a file to its state before the most recent Write/Edit in this session.',
        'parameters': {'type': 'object', 'properties': {
            'file_path': {'type': 'string'}
        }, 'required': ['file_path']}
    }},
    {'type': 'function', 'function': {
        'name': 'Glob', 'description': 'List/search files matching a glob pattern, e.g. "**/*.py"',
        'parameters': {'type': 'object', 'properties': {'pattern': {'type': 'string'}}, 'required': ['pattern']}
    }},
    {'type': 'function', 'function': {
        'name': 'Grep',
        'description': (
            'Search file contents for a regex pattern, returns matches as path:line_number: line. '
            'Uses ripgrep under the hood when it is installed on the system (much faster, respects '
            '.gitignore), and falls back to an equivalent pure-Python search otherwise.'
        ),
        'parameters': {'type': 'object', 'properties': {
            'pattern': {'type': 'string', 'description': 'regex pattern to search for'},
            'path': {'type': 'string', 'description': 'directory to search in (default: current directory)'},
            'include': {'type': 'string', 'description': 'only search files matching this glob, e.g. "*.py"'}
        }, 'required': ['pattern']}
    }},
    {'type': 'function', 'function': {
        'name': 'Bash',
        'description': (
            f'Execute a shell command (killed after {config.BASH_TIMEOUT_SECONDS}s if it hangs, '
            'unless run in the background). For long-running or blocking commands (dev servers, '
            'watchers, long builds/tests), set run_in_background=true: it returns a job_id '
            'immediately instead of blocking, which you then check with Await(job_id) or '
            'ListJobs().'
        ),
        'parameters': {'type': 'object', 'properties': {
            'command': {'type': 'string', 'description': 'the command to execute'},
            'run_in_background': {'type': 'boolean', 'description': 'if true, run asynchronously and return a job_id immediately instead of blocking'},
            'timeout': {'type': 'integer', 'description': 'foreground-only: override the default timeout in seconds before the command is killed'},
        }, 'required': ['command']}
    }},
    {'type': 'function', 'function': {
        'name': 'ListJobs',
        'description': 'List background jobs started via Bash(run_in_background=true), with their status (running/exited) and elapsed time.',
        'parameters': {'type': 'object', 'properties': {}},
    }},
    {'type': 'function', 'function': {
        'name': 'Await',
        'description': (
            'Wait for a background job (started via Bash run_in_background=true) to finish or '
            'produce output, and return what it has written so far. Safe to call repeatedly on '
            'a still-running job - it does not kill it.'
        ),
        'parameters': {'type': 'object', 'properties': {
            'job_id': {'type': 'string', 'description': 'the job_id returned by Bash(run_in_background=true)'},
            'timeout': {'type': 'integer', 'description': f'max seconds to wait for this call (default {config.BASH_AWAIT_DEFAULT_SECONDS}, hard-capped at {config.BASH_AWAIT_MAX_SECONDS})'},
        }, 'required': ['job_id']}
    }},
    {'type': 'function', 'function': {
        'name': 'Task',
        'description': 'Delegate a single self-contained sub-task (e.g. large codebase search, multi-step investigation) to an isolated sub-agent. Returns only its final result.',
        'parameters': {'type': 'object', 'properties': {
            'prompt': {'type': 'string', 'description': 'the full task for the sub-agent to complete'}
        }, 'required': ['prompt']}
    }},
    {'type': 'function', 'function': {
        'name': 'Tasks',
        'description': 'Delegate MULTIPLE independent sub-tasks to isolated sub-agents that run CONCURRENTLY (in parallel), not one after another. Use this instead of several Task calls when the sub-tasks do not depend on each other (e.g. investigate 3 different modules at once). Returns each sub-agent\'s final result, labeled by task number, in the original order.',
        'parameters': {'type': 'object', 'properties': {
            'prompts': {
                'type': 'array', 'items': {'type': 'string'},
                'description': 'list of independent, self-contained task descriptions, one per sub-agent'
            }
        }, 'required': ['prompts']}
    }},
    {'type': 'function', 'function': {
        'name': 'ScheduleTask',
        'description': (
            'Create a recurring (or one-off) scheduled task, the same way Claude Cowork lets you '
            'describe recurring work once and have it run automatically. The prompt is saved and run '
            'later as its own isolated sub-agent session (same tools available, unattended - dangerous '
            'shell commands are auto-declined rather than prompted for, since nobody is watching). '
            'The finished result is delivered to the user over Telegram (if linked via /telegram), and '
            'always recorded for review via ListScheduledTasks. Use this whenever the user asks for '
            'something to happen "every day/week", "on a schedule", "recurring", or similar - do not '
            'just say you will remember to do it, actually call this tool.'
        ),
        'parameters': {'type': 'object', 'properties': {
            'prompt': {'type': 'string', 'description': 'the full task instructions to run each time, exactly as a sub-agent should receive them'},
            'schedule': {
                'type': 'string',
                'description': (
                    "when to run it. Formats: 'every 30m' / 'every 2h' / 'every 1d' (recurring interval, "
                    "min 60s), 'daily 09:00' or 'daily 6pm', 'weekly mon 09:00', "
                    "'once 2026-07-20 09:00' (single run), or 'cron */15 * * * *' (standard 5-field cron)."
                ),
            },
            'name': {'type': 'string', 'description': 'short human-readable label for the task (optional, derived from the prompt if omitted)'},
        }, 'required': ['prompt', 'schedule']}
    }},
    {'type': 'function', 'function': {
        'name': 'ListScheduledTasks',
        'description': 'List every scheduled task (id, name, schedule, next/last run, enabled state, run count).',
        'parameters': {'type': 'object', 'properties': {}},
    }},
    {'type': 'function', 'function': {
        'name': 'ToggleScheduledTask',
        'description': 'Pause or resume a scheduled task by id (from ListScheduledTasks). Paused tasks never fire until resumed.',
        'parameters': {'type': 'object', 'properties': {
            'task_id': {'type': 'string', 'description': "the task's id, e.g. 'sched_ab12cd34'"},
            'enabled': {'type': 'boolean', 'description': 'true to resume/enable, false to pause'},
        }, 'required': ['task_id', 'enabled']}
    }},
    {'type': 'function', 'function': {
        'name': 'CancelScheduledTask',
        'description': 'Permanently delete a scheduled task by id (from ListScheduledTasks). This cannot be undone.',
        'parameters': {'type': 'object', 'properties': {
            'task_id': {'type': 'string', 'description': "the task's id, e.g. 'sched_ab12cd34'"}
        }, 'required': ['task_id']}
    }},
    {'type': 'function', 'function': {
        'name': 'RunScheduledTaskNow',
        'description': (
            'Trigger a scheduled task on demand, right now, instead of waiting for its next '
            'scheduled time. Runs in the background and returns immediately - the result is '
            'delivered over Telegram (if linked) and recorded, exactly like a normal scheduled run.'
        ),
        'parameters': {'type': 'object', 'properties': {
            'task_id': {'type': 'string', 'description': "the task's id, e.g. 'sched_ab12cd34'"}
        }, 'required': ['task_id']}
    }},
]

# Sub-agents get every tool except Task/Tasks (no recursive spawning) and
# schedule mutators (only the main agent / user CLI should create or cancel
# recurring work). ListScheduledTasks stays so they can observe schedules.
_SUBAGENT_EXCLUDED = {
    'Task', 'Tasks',
    'ScheduleTask', 'ToggleScheduledTask', 'CancelScheduledTask', 'RunScheduledTaskNow',
}
SUBAGENT_TOOLS = [t for t in TOOLS if t['function']['name'] not in _SUBAGENT_EXCLUDED]

REQUIRED_ARGS = {t['function']['name']: t['function']['parameters'].get('required', []) for t in TOOLS}


def validate_args(name, args):
    missing = [k for k in REQUIRED_ARGS.get(name, []) if k not in args]
    if missing:
        return f"Error: missing required argument(s) {missing} for tool '{name}'. Re-check the tool schema and try again."
    return None


def dispatch_tool(name, args, state):
    """Single place that both the main loop and sub-agents call to run a tool.
    Validates arguments and isolates exceptions per-tool-call so one bad call
    can't take down the rest of the turn."""
    if state.mode == 'plan' and name not in config.READONLY_TOOLS:
        msg = (
            f"'{name}' is not available in PLAN MODE. You may only explore using "
            f"{', '.join(sorted(config.READONLY_TOOLS))}. Investigate as needed, then present "
            f"your plan in your final answer and wait, the user will switch you to "
            f"normal or auto mode (/normal or /auto) to let you execute it."
        )
        log_event(f"[{state.name}] PLAN MODE BLOCKED '{name}'")
        return msg
    err = validate_args(name, args)
    if err:
        log_event(f"[{state.name}] VALIDATION FAILED for '{name}': {err}")
        return err
    try:
        if name == 'Task':
            from bardgent.subagents import run_subagent
            return run_subagent(args['prompt'])
        elif name == 'Tasks':
            prompts = args.get('prompts') or []
            if not isinstance(prompts, list) or not prompts:
                return "Error: 'prompts' must be a non-empty list of task strings."
            from bardgent.subagents import run_subagents_parallel
            return run_subagents_parallel(prompts)
        elif name == 'read_memory':
            return memory.read_memory()
        elif name == 'save_memory':
            return memory.save_memory(args['memory'])
        elif name == 'list_memory':
            return memory.list_memory()
        elif name == 'delete_memory':
            return memory.delete_memory(args['index'])
        elif name == 'list_skills':
            return skills.list_skills_text()
        elif name == 'Skill':
            return skills.load_skill(args['name'])
        elif name == 'WebSearch':
            return WebSearch(args['query'])
        elif name == 'Fetch':
            return Fetch(args['link'], state)
        elif name == 'Read':
            return Read(args['file_path'], args.get('offset'), args.get('limit'))
        elif name == 'Write':
            return Write(args['file_path'], args['content'], state)
        elif name == 'Edit':
            return Edit(args['file_path'], args['old_str'], args['new_str'], state)
        elif name == 'Undo':
            return Undo(args['file_path'])
        elif name == 'Glob':
            return Glob(args['pattern'])
        elif name == 'Grep':
            return Grep(args['pattern'], args.get('path', '.'), args.get('include'))
        elif name == 'Bash':
            return Bash(
                args['command'], state,
                timeout=args.get('timeout'),
                run_in_background=bool(args.get('run_in_background', False)),
            )
        elif name == 'ListJobs':
            return ListJobs()
        elif name == 'Await':
            return Await(args['job_id'], args.get('timeout'))
        elif name == 'ScheduleTask':
            task, sched_err = scheduler.add_task(args['prompt'], args['schedule'], args.get('name'))
            if sched_err:
                return f"Error: could not create scheduled task: {sched_err}"
            nr = scheduler.format_dt(task.get('next_run'))
            return (
                f"Scheduled task created: {task['id']} (\"{task['name']}\"). "
                f"Schedule: {task['schedule_spec']}. Next run: {nr}. "
                f"Results will be sent over Telegram if linked (/telegram), and are always "
                f"visible via ListScheduledTasks / /schedules."
            )
        elif name == 'ListScheduledTasks':
            return scheduler.list_tasks_text()
        elif name == 'ToggleScheduledTask':
            ok = scheduler.set_enabled(args['task_id'], bool(args['enabled']))
            if not ok:
                return f"Error: no scheduled task with id '{args['task_id']}'. Use ListScheduledTasks() to see valid ids."
            return f"Task {args['task_id']} {'resumed' if args['enabled'] else 'paused'}."
        elif name == 'CancelScheduledTask':
            ok = scheduler.remove_task(args['task_id'])
            if not ok:
                return f"Error: no scheduled task with id '{args['task_id']}'. Use ListScheduledTasks() to see valid ids."
            return f"Task {args['task_id']} deleted."
        elif name == 'RunScheduledTaskNow':
            task = scheduler.get_task(args['task_id'])
            if not task:
                return f"Error: no scheduled task with id '{args['task_id']}'. Use ListScheduledTasks() to see valid ids."
            if scheduler.is_task_running(args['task_id']):
                return f"Task {args['task_id']} (\"{task['name']}\") is already running - not starting a second one."
            scheduler.run_task_in_background(args['task_id'])
            return (
                f"Started task {args['task_id']} (\"{task['name']}\") now, running in the "
                f"background. The result will be delivered over Telegram (if linked) and recorded "
                f"for review shortly - it is not ready yet."
            )
        else:
            return 'Unknown tool'
    except Exception as e:
        log_event(f"[{state.name}] TOOL '{name}' RAISED: {type(e).__name__}: {e}")
        return f"Error running tool '{name}': {type(e).__name__}: {e}. Do not blindly retry, adjust the arguments or approach."
"""
Scheduled tasks - Bardgent's equivalent of Claude Cowork's "Scheduled tasks".

You describe a task once (a prompt) and a cadence (a schedule spec), and
Bardgent runs it automatically from then on - as its own isolated sub-agent
session, exactly like Task() - and delivers the result over Telegram, the
same way a finished Cowork task shows up for you to review.

Storage:
    ~/.bardgent/scheduled_tasks.json   (flat list of task records)

Schedule specs (case-insensitive), typed by the user or the model:
    every 30m | every 2h | every 1d          -> recurring interval
    daily 09:00 | daily 6pm                  -> once a day at that time
    weekly mon 09:00 | weekly friday 6pm      -> once a week
    once 2026-07-20 09:00 | once 6pm          -> a single one-off run
    cron */15 * * * *                        -> standard 5-field cron

Scheduler daemon:
    A detached process (not tied to the interactive terminal) wakes up
    periodically, finds any enabled task whose next_run has passed, and runs
    it via subagents.run_subagent(..., unattended=True) - unattended so a
    dangerous shell command can never block forever waiting on a y/N prompt
    nobody is there to answer.

    The daemon is started automatically when Bardgent launches or when a
    scheduled task is created. It keeps running after you close the REPL /
    terminal (survives SIGHUP). It does *not* survive a full machine reboot
    unless you add a launchd/cron unit yourself.

    PID / lock files:
        ~/.bardgent/scheduler.pid
        ~/.bardgent/scheduler.lock
        ~/.bardgent/schedule_running/<task_id>.pid  (cross-process run lock)

On-demand runs (/schedule run <id>, or the RunScheduledTaskNow tool) use the
exact same executor, just triggered immediately instead of by the clock.
"""

import os
import re
import sys
import json
import uuid
import signal
import threading
import subprocess
from datetime import datetime, timedelta
from pathlib import Path

from rich.panel import Panel

from bardgent import config
from bardgent.config import console, console_lock, log_event
from bardgent.telegram import _load_telegram_chat_id, send_telegram_message

SCHEDULE_FILE = config.GLOBAL_DIR / 'scheduled_tasks.json'
DAEMON_PID_FILE = config.GLOBAL_DIR / 'scheduler.pid'
DAEMON_LOCK_FILE = config.GLOBAL_DIR / 'scheduler.lock'
DAEMON_LOG_FILE = config.GLOBAL_DIR / 'scheduler.log'
RUNNING_DIR = config.GLOBAL_DIR / 'schedule_running'

_WEEKDAYS = {
    'mon': 0, 'monday': 0,
    'tue': 1, 'tues': 1, 'tuesday': 1,
    'wed': 2, 'weds': 2, 'wednesday': 2,
    'thu': 3, 'thur': 3, 'thurs': 3, 'thursday': 3,
    'fri': 4, 'friday': 4,
    'sat': 5, 'saturday': 5,
    'sun': 6, 'sunday': 6,
}

_INTERVAL_RE = re.compile(
    r'^(\d+)\s*(s|sec|secs|seconds|m|min|mins|minutes|h|hr|hrs|hours|d|day|days|w|week|weeks)$'
)
_CLOCK_RE = re.compile(r'^(\d{1,2})(?::(\d{2}))?\s*(am|pm)?$')

MIN_INTERVAL_SECONDS = 60
_CRON_SEARCH_HORIZON_DAYS = 730  # ~2 years; brute-force minute stepping cap


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def _parse_interval_seconds(s):
    m = _INTERVAL_RE.match(s.strip().lower())
    if not m:
        return None
    n = int(m.group(1))
    unit = m.group(2)
    if unit in ('s', 'sec', 'secs', 'seconds'):
        return n
    if unit in ('m', 'min', 'mins', 'minutes'):
        return n * 60
    if unit in ('h', 'hr', 'hrs', 'hours'):
        return n * 3600
    if unit in ('d', 'day', 'days'):
        return n * 86400
    if unit in ('w', 'week', 'weeks'):
        return n * 604800
    return None


def _parse_clock_time(s):
    m = _CLOCK_RE.match(s.strip().lower())
    if not m:
        return None
    hour = int(m.group(1))
    minute = int(m.group(2) or 0)
    ampm = m.group(3)
    if ampm == 'am':
        if hour == 12:
            hour = 0
    elif ampm == 'pm':
        if hour != 12:
            hour += 12
    if not (0 <= hour <= 23 and 0 <= minute <= 59):
        return None
    return hour, minute


def _parse_once_datetime(s):
    s = s.strip()
    tz = datetime.now().astimezone().tzinfo
    now = datetime.now(tz)

    m = re.match(r'^(\d{4}-\d{2}-\d{2})[ T](.+)$', s)
    if m:
        date_part, time_part = m.group(1), m.group(2)
        t = _parse_clock_time(time_part)
        if not t:
            return None
        try:
            d = datetime.strptime(date_part, '%Y-%m-%d')
        except ValueError:
            return None
        return d.replace(hour=t[0], minute=t[1], second=0, microsecond=0, tzinfo=tz)

    # Bare time -> today, or tomorrow if that time has already passed.
    t = _parse_clock_time(s)
    if t:
        candidate = now.replace(hour=t[0], minute=t[1], second=0, microsecond=0)
        if candidate <= now:
            candidate += timedelta(days=1)
        return candidate
    return None


def _expand_cron_field(field, lo, hi):
    values = set()
    for part in field.split(','):
        part = part.strip()
        if not part:
            raise ValueError('empty cron field')
        step = 1
        if '/' in part:
            rng, step_s = part.split('/', 1)
            step = int(step_s)
            if step <= 0:
                raise ValueError('step must be positive')
        else:
            rng = part
        if rng == '*':
            start, end = lo, hi
        elif '-' in rng:
            a, b = rng.split('-', 1)
            start, end = int(a), int(b)
        else:
            start = end = int(rng)
        if start < lo or end > hi or start > end:
            raise ValueError(f'value out of range {lo}-{hi}')
        v = start
        while v <= end:
            values.add(v)
            v += step
    return values


def _validate_cron(fields):
    minute_f, hour_f, day_f, month_f, wday_f = fields
    _expand_cron_field(minute_f, 0, 59)
    _expand_cron_field(hour_f, 0, 23)
    _expand_cron_field(day_f, 1, 31)
    _expand_cron_field(month_f, 1, 12)
    _expand_cron_field(wday_f, 0, 6)


def parse_schedule_spec(spec):
    """Returns (schedule_type, params, error). error is None on success."""
    spec = (spec or '').strip()
    if not spec:
        return None, None, "Empty schedule spec."

    parts = spec.split(None, 1)
    kind = parts[0].lower()
    rest = parts[1].strip() if len(parts) > 1 else ''

    if kind == 'every':
        seconds = _parse_interval_seconds(rest)
        if not seconds:
            return None, None, (
                f"Could not parse interval '{rest}'. Try 'every 30m', 'every 2h', 'every 1d'."
            )
        if seconds < MIN_INTERVAL_SECONDS:
            return None, None, f"Minimum interval is {MIN_INTERVAL_SECONDS} seconds."
        return 'interval', {'seconds': seconds}, None

    if kind == 'daily':
        t = _parse_clock_time(rest)
        if not t:
            return None, None, f"Could not parse time '{rest}'. Try 'daily 09:00' or 'daily 6pm'."
        return 'daily', {'hour': t[0], 'minute': t[1]}, None

    if kind == 'weekly':
        wparts = rest.split(None, 1)
        if len(wparts) != 2:
            return None, None, "Usage: 'weekly <weekday> <HH:MM>', e.g. 'weekly mon 09:00'."
        wday_s, time_s = wparts
        wday = _WEEKDAYS.get(wday_s.lower())
        if wday is None:
            return None, None, f"Unknown weekday '{wday_s}'. Use mon/tue/wed/thu/fri/sat/sun."
        t = _parse_clock_time(time_s)
        if not t:
            return None, None, f"Could not parse time '{time_s}'."
        return 'weekly', {'weekday': wday, 'hour': t[0], 'minute': t[1]}, None

    if kind == 'once':
        dt = _parse_once_datetime(rest)
        if not dt:
            return None, None, f"Could not parse date/time '{rest}'. Try 'once 2026-07-20 09:00'."
        return 'once', {'run_at': dt.isoformat()}, None

    if kind == 'cron':
        fields = rest.split()
        if len(fields) != 5:
            return None, None, "Cron spec must have exactly 5 fields: minute hour day month weekday."
        try:
            _validate_cron(fields)
        except (ValueError, IndexError):
            return None, None, (
                "Invalid cron expression. Fields: minute(0-59) hour(0-23) day(1-31) "
                "month(1-12) weekday(0-6, 0=Sun). Each field: a number, *, a-b, a,b,c, or */n."
            )
        return 'cron', {'expr': rest}, None

    return None, None, (
        f"Unknown schedule type '{kind}'. Use one of: 'every <N><unit>', 'daily <HH:MM>', "
        "'weekly <day> <HH:MM>', 'once <date> <time>', 'cron <5 fields>'."
    )


# ---------------------------------------------------------------------------
# Next-run computation
# ---------------------------------------------------------------------------

def _next_cron_run(expr, after):
    minute_f, hour_f, day_f, month_f, wday_f = expr.split()
    minutes = _expand_cron_field(minute_f, 0, 59)
    hours = _expand_cron_field(hour_f, 0, 23)
    days = _expand_cron_field(day_f, 1, 31)
    months = _expand_cron_field(month_f, 1, 12)
    wdays = _expand_cron_field(wday_f, 0, 6)  # 0 = Sunday, cron convention

    candidate = (after + timedelta(minutes=1)).replace(second=0, microsecond=0)
    limit = after + timedelta(days=_CRON_SEARCH_HORIZON_DAYS)
    while candidate <= limit:
        cron_wday = (candidate.weekday() + 1) % 7  # python Mon=0 -> cron Sun=0
        if (candidate.minute in minutes and candidate.hour in hours and
                candidate.day in days and candidate.month in months and
                cron_wday in wdays):
            return candidate
        candidate += timedelta(minutes=1)
    return None


def compute_next_run(schedule_type, params, after=None):
    tz = datetime.now().astimezone().tzinfo
    after = after or datetime.now(tz)

    if schedule_type == 'once':
        run_at = datetime.fromisoformat(params['run_at'])
        if run_at.tzinfo is None:
            run_at = run_at.replace(tzinfo=tz)
        return run_at if run_at > after else None

    if schedule_type == 'interval':
        return after + timedelta(seconds=params['seconds'])

    if schedule_type == 'daily':
        candidate = after.replace(hour=params['hour'], minute=params['minute'], second=0, microsecond=0)
        if candidate <= after:
            candidate += timedelta(days=1)
        return candidate

    if schedule_type == 'weekly':
        candidate = after.replace(hour=params['hour'], minute=params['minute'], second=0, microsecond=0)
        days_ahead = (params['weekday'] - candidate.weekday()) % 7
        candidate += timedelta(days=days_ahead)
        if candidate <= after:
            candidate += timedelta(days=7)
        return candidate

    if schedule_type == 'cron':
        return _next_cron_run(params['expr'], after)

    return None


# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------

_store_lock = threading.RLock()


def load_tasks():
    with _store_lock:
        if SCHEDULE_FILE.exists():
            try:
                return json.loads(SCHEDULE_FILE.read_text(encoding='utf-8'))
            except (OSError, json.JSONDecodeError):
                return []
        return []


def save_tasks(tasks):
    with _store_lock:
        try:
            config.GLOBAL_DIR.mkdir(exist_ok=True)
            SCHEDULE_FILE.write_text(json.dumps(tasks, indent=2), encoding='utf-8')
        except OSError as e:
            log_event(f"SCHEDULE SAVE FAILED: {e}")


def get_task(task_id):
    for t in load_tasks():
        if t['id'] == task_id:
            return t
    return None


def add_task(prompt, schedule_spec, name=None):
    """Create and persist a new scheduled task. Returns (task, error)."""
    prompt = (prompt or '').strip()
    if not prompt:
        return None, "Task prompt cannot be empty."

    schedule_type, params, err = parse_schedule_spec(schedule_spec)
    if err:
        return None, err

    now = datetime.now().astimezone()
    next_run = compute_next_run(schedule_type, params, after=now)
    if schedule_type == 'once' and next_run is None:
        return None, "That one-time run time is already in the past."

    with _store_lock:
        tasks = load_tasks()
        task_id = f"sched_{uuid.uuid4().hex[:8]}"
        task = {
            'id': task_id,
            'name': (name or prompt.splitlines()[0])[:80],
            'prompt': prompt,
            'schedule_spec': schedule_spec.strip(),
            'schedule_type': schedule_type,
            'params': params,
            'enabled': True,
            'created': now.isoformat(),
            'last_run': None,
            'next_run': next_run.isoformat() if next_run else None,
            'run_count': 0,
            'last_status': None,
            'last_summary': None,
        }
        tasks.append(task)
        save_tasks(tasks)

    log_event(f"SCHEDULE CREATED {task_id}: {schedule_spec!r} -> {prompt[:80]!r}")
    # Keep the detached daemon alive so the new task fires even if the user
    # closes the terminal right after creating it.
    ensure_daemon_running()
    return task, None


def remove_task(task_id):
    with _store_lock:
        tasks = load_tasks()
        new_tasks = [t for t in tasks if t['id'] != task_id]
        if len(new_tasks) == len(tasks):
            return False
        save_tasks(new_tasks)
    log_event(f"SCHEDULE DELETED {task_id}")
    return True


def set_enabled(task_id, enabled):
    with _store_lock:
        tasks = load_tasks()
        for t in tasks:
            if t['id'] == task_id:
                t['enabled'] = enabled
                if enabled and t['schedule_type'] != 'once' and not t.get('next_run'):
                    nxt = compute_next_run(t['schedule_type'], t['params'], after=datetime.now().astimezone())
                    t['next_run'] = nxt.isoformat() if nxt else None
                save_tasks(tasks)
                log_event(f"SCHEDULE {'RESUMED' if enabled else 'PAUSED'} {task_id}")
                if enabled:
                    ensure_daemon_running()
                return True
    return False


def format_dt(iso):
    if not iso:
        return '(none)'
    try:
        dt = datetime.fromisoformat(iso)
        return dt.strftime('%Y-%m-%d %H:%M %Z').strip()
    except ValueError:
        return iso


def list_tasks(tasks=None):
    """Return raw task list for API consumption."""
    tasks = tasks if tasks is not None else load_tasks()
    return tasks


def list_tasks_text(tasks=None):
    tasks = tasks if tasks is not None else load_tasks()
    status = daemon_status()
    if status['running']:
        header = f"Scheduler daemon: running (pid {status['pid']}) — keeps firing after you close the terminal"
    else:
        header = (
            "Scheduler daemon: NOT running — tasks will not fire until Bardgent starts "
            "(or /schedule daemon start). Closing the terminal without a daemon stops schedules."
        )
    if not tasks:
        return header + '\n\n(no scheduled tasks yet. Create one with /schedule <spec> :: <prompt>)'
    lines = [header, '']
    for t in tasks:
        enabled = 'enabled' if t.get('enabled') else 'paused'
        next_run = format_dt(t.get('next_run'))
        last_run = format_dt(t.get('last_run')) if t.get('last_run') else 'never'
        last_status = t.get('last_status') or '-'
        lines.append(
            f"{t['id']}  [{enabled}]  \"{t['name']}\"\n"
            f"    schedule: {t['schedule_spec']}\n"
            f"    next run: {next_run}   last run: {last_run} ({last_status})   "
            f"runs so far: {t.get('run_count', 0)}"
        )
    return '\n\n'.join(lines)


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------

# Scheduled tasks often need a few more tool calls than an ad-hoc Task()
# (e.g. several searches/fetches to put together a news digest), so give
# them more headroom than the default 15 before giving up.
SCHEDULED_TASK_MAX_ITERS = 40

# In-process guard (thread safety within one process). Cross-process safety
# uses RUNNING_DIR marker files so the daemon and a live REPL can't both run
# the same task at once.
_running_task_ids = set()
_running_lock = threading.Lock()


def _running_marker_path(task_id):
    return RUNNING_DIR / f'{task_id}.pid'


def _pid_is_alive(pid):
    if not pid or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        # Exists but we can't signal it.
        pass
    except OSError:
        return False
    return True


def _is_scheduler_daemon_process(pid):
    """True if pid is alive and looks like our --scheduler-daemon process."""
    if not _pid_is_alive(pid):
        return False
    try:
        out = subprocess.check_output(
            ['ps', '-p', str(pid), '-o', 'args='],
            text=True, stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        # ps failed; fall back to bare liveness (best effort).
        return _pid_is_alive(pid)
    if not out:
        return False
    return '--scheduler-daemon' in out


def _try_claim_task_run(task_id):
    """Cross-process exclusive claim for one task run. Returns True if claimed."""
    RUNNING_DIR.mkdir(parents=True, exist_ok=True)
    marker = _running_marker_path(task_id)
    for _ in range(2):
        try:
            fd = os.open(str(marker), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            try:
                os.write(fd, str(os.getpid()).encode('utf-8'))
            finally:
                os.close(fd)
            return True
        except FileExistsError:
            try:
                old_pid = int(marker.read_text(encoding='utf-8').strip())
            except (OSError, ValueError):
                old_pid = None
            if old_pid and _pid_is_alive(old_pid):
                return False
            # Stale marker from a crashed process — remove and retry once.
            try:
                marker.unlink()
            except OSError:
                return False
    return False


def _release_task_run(task_id):
    marker = _running_marker_path(task_id)
    try:
        if not marker.exists():
            return
        try:
            owner = int(marker.read_text(encoding='utf-8').strip())
        except (OSError, ValueError):
            owner = None
        if owner is None or owner == os.getpid():
            marker.unlink()
    except OSError:
        pass


def is_task_running(task_id):
    with _running_lock:
        if task_id in _running_task_ids:
            return True
    marker = _running_marker_path(task_id)
    if not marker.exists():
        return False
    try:
        pid = int(marker.read_text(encoding='utf-8').strip())
    except (OSError, ValueError):
        return False
    if _pid_is_alive(pid):
        return True
    try:
        marker.unlink()
    except OSError:
        pass
    return False


def _execute_scheduled_task(task_id):
    """Run one scheduled task to completion (blocking) and record/deliver
    the result. Safe to call from a background thread or the daemon process.
    No-ops if this task is already running in this process or another."""
    with _running_lock:
        if task_id in _running_task_ids:
            log_event(f"SCHEDULE RUN SKIPPED {task_id}: already running in this process")
            return
        _running_task_ids.add(task_id)

    if not _try_claim_task_run(task_id):
        with _running_lock:
            _running_task_ids.discard(task_id)
        log_event(f"SCHEDULE RUN SKIPPED {task_id}: already running elsewhere")
        return

    try:
        tasks = load_tasks()
        idx = next((i for i, t in enumerate(tasks) if t['id'] == task_id), None)
        if idx is None:
            return
        task = tasks[idx]

        with console_lock:
            console.print(Panel(
                task['prompt'],
                title=f"[bold magenta]SCHEDULED TASK running: {task['name']} ({task['id']})",
                border_style='magenta',
            ))
        log_event(f"SCHEDULE RUN START {task_id}: {task['schedule_spec']!r}")

        status = 'ok'
        try:
            from bardgent.subagents import run_subagent, INCOMPLETE_RESULT
            result = run_subagent(
                task['prompt'], render=False, label=task['name'], unattended=True,
                max_iters=SCHEDULED_TASK_MAX_ITERS,
            )
            if result.strip() == INCOMPLETE_RESULT:
                status = 'incomplete'
                log_event(f"SCHEDULE RUN {task_id}: hit max_iters ({SCHEDULED_TASK_MAX_ITERS}) without finishing")
        except Exception as e:
            status = 'error'
            result = f"Scheduled task failed: {type(e).__name__}: {e}"
            log_event(f"SCHEDULE RUN ERROR {task_id}: {type(e).__name__}: {e}")

        now = datetime.now().astimezone()

        with _store_lock:
            tasks = load_tasks()
            idx = next((i for i, t in enumerate(tasks) if t['id'] == task_id), None)
            if idx is None:
                return  # deleted while it was running
            task = tasks[idx]
            task['last_run'] = now.isoformat()
            task['run_count'] = task.get('run_count', 0) + 1
            task['last_status'] = status
            task['last_summary'] = (result or '')[:500]

            if task['schedule_type'] == 'once':
                task['enabled'] = False
                task['next_run'] = None
            elif task.get('enabled', True):
                nxt = compute_next_run(task['schedule_type'], task['params'], after=now)
                task['next_run'] = nxt.isoformat() if nxt else None

            tasks[idx] = task
            save_tasks(tasks)

        chat_id = _load_telegram_chat_id()
        if chat_id and config.TELEGRAM_BOT_TOKEN:
            header = f"Scheduled task \"{task['name']}\" finished ({status})"
            if not send_telegram_message(result or '(no output)', chat_id, header=header):
                log_event(f"SCHEDULE RUN {task_id}: Telegram delivery failed (see log above)")
        else:
            log_event(f"SCHEDULE RUN {task_id}: no Telegram chat linked (run /telegram to link); result not sent")

        with console_lock:
            console.print(Panel(
                result or '(empty result)',
                title=f"[bold magenta]SCHEDULED TASK finished: {task['name']} ({task['id']}) [{status}]",
                border_style='magenta',
            ))
        log_event(f"SCHEDULE RUN DONE {task_id} status={status}")
    finally:
        _release_task_run(task_id)
        with _running_lock:
            _running_task_ids.discard(task_id)


def run_task_in_background(task_id):
    """Fire off a scheduled task immediately (on-demand), without blocking
    the caller. Used by /schedule run and the RunScheduledTaskNow tool.
    Returns the thread; if the task is already running, _execute_scheduled_task
    will simply no-op once it starts (see is_task_running for checking first)."""
    t = threading.Thread(
        target=_execute_scheduled_task, args=(task_id,),
        daemon=True, name=f'bardgent-schedule-{task_id}',
    )
    t.start()
    return t


# ---------------------------------------------------------------------------
# Scheduler loop (used by the detached daemon process)
# ---------------------------------------------------------------------------

_scheduler_stop = threading.Event()
DEFAULT_CHECK_INTERVAL_SECONDS = 20


def _check_and_run_due_tasks():
    tasks = load_tasks()
    now = datetime.now().astimezone()
    due_ids = []
    for t in tasks:
        if not t.get('enabled'):
            continue
        nr = t.get('next_run')
        if not nr:
            continue
        try:
            next_run_dt = datetime.fromisoformat(nr)
        except ValueError:
            continue
        if next_run_dt <= now:
            due_ids.append(t['id'])
    for task_id in due_ids:
        # Run sequentially: scheduled tasks are typically infrequent, and
        # running them one at a time avoids many sub-agents at once.
        _execute_scheduled_task(task_id)


def _scheduler_loop(check_interval):
    while not _scheduler_stop.is_set():
        try:
            _check_and_run_due_tasks()
        except Exception as e:
            log_event(f"SCHEDULER LOOP ERROR: {type(e).__name__}: {e}")
        # Wait in 1s slices so SIGTERM is acted on promptly even if a single
        # long wait is in progress when the signal arrives.
        remaining = float(check_interval)
        while remaining > 0 and not _scheduler_stop.is_set():
            slice_ = min(1.0, remaining)
            if _scheduler_stop.wait(slice_):
                break
            remaining -= slice_


# ---------------------------------------------------------------------------
# Detached scheduler daemon (survives closing the terminal)
# ---------------------------------------------------------------------------

def _read_daemon_pid():
    try:
        if not DAEMON_PID_FILE.exists():
            return None
        return int(DAEMON_PID_FILE.read_text(encoding='utf-8').strip())
    except (OSError, ValueError):
        return None


def _write_daemon_pid(pid):
    try:
        config.GLOBAL_DIR.mkdir(exist_ok=True)
        DAEMON_PID_FILE.write_text(str(pid), encoding='utf-8')
    except OSError as e:
        log_event(f"SCHEDULER DAEMON: failed to write pid file: {e}")


def _clear_daemon_pid():
    try:
        if DAEMON_PID_FILE.exists():
            DAEMON_PID_FILE.unlink()
    except OSError:
        pass


def daemon_status():
    """Return {'running': bool, 'pid': int|None}."""
    pid = _read_daemon_pid()
    if pid and _is_scheduler_daemon_process(pid):
        return {'running': True, 'pid': pid}
    if pid:
        # Stale pid file or PID reused by an unrelated process.
        _clear_daemon_pid()
    return {'running': False, 'pid': None}


def _daemon_command():
    """Argv to re-enter this package as the long-lived scheduler process."""
    # Prefer the installed console script when available; fall back to
    # `python -m bardgent` so editable installs and source trees both work.
    return [sys.executable, '-m', 'bardgent', '--scheduler-daemon']


def ensure_daemon_running(cwd=None):
    """Start the detached scheduler daemon if it is not already running.

    Returns (ok: bool, message: str). Safe to call from the REPL or tools;
    closing the terminal does not stop a successfully started daemon.
    """
    import time

    status = daemon_status()
    if status['running']:
        return True, f"scheduler daemon already running (pid {status['pid']})"

    if not os.environ.get('GEMINI_API_KEY'):
        # Child also loads ~/.bardgent/.env via config, but if the parent has
        # no key either, the daemon would just spin and fail every run.
        from dotenv import dotenv_values
        env_file = Path.home() / '.bardgent' / '.env'
        file_vals = dotenv_values(env_file) if env_file.exists() else {}
        if not (file_vals.get('GEMINI_API_KEY') or os.environ.get('GEMINI_API_KEY')):
            return False, "cannot start scheduler daemon: GEMINI_API_KEY is not set"

    try:
        config.GLOBAL_DIR.mkdir(exist_ok=True)
        log_f = open(DAEMON_LOG_FILE, 'a', encoding='utf-8')
    except OSError as e:
        return False, f"cannot open scheduler log: {e}"

    cmd = _daemon_command()
    child_env = os.environ.copy()
    # Line-buffered logs when stdout is redirected to a file.
    child_env['PYTHONUNBUFFERED'] = '1'
    popen_kwargs = {
        'stdin': subprocess.DEVNULL,
        'stdout': log_f,
        'stderr': subprocess.STDOUT,
        'close_fds': True,
        'cwd': cwd or os.getcwd(),
        'env': child_env,
    }
    if sys.platform == 'win32':
        # DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP
        popen_kwargs['creationflags'] = 0x00000008 | 0x00000200
    else:
        # New session so closing the terminal (SIGHUP) does not kill the daemon.
        popen_kwargs['start_new_session'] = True

    try:
        proc = subprocess.Popen(cmd, **popen_kwargs)
    except OSError as e:
        try:
            log_f.close()
        except OSError:
            pass
        return False, f"failed to spawn scheduler daemon: {e}"

    # Parent no longer needs the log fd; child has its own dup.
    try:
        log_f.close()
    except OSError:
        pass

    # Give the child a moment to write its pid / take the lock.
    for _ in range(30):
        time.sleep(0.1)
        status = daemon_status()
        if status['running']:
            log_event(f"SCHEDULER DAEMON started pid={status['pid']}")
            return True, f"scheduler daemon started (pid {status['pid']})"
        if proc.poll() is not None:
            # Child exited immediately — surface last log lines if any.
            tail = ''
            try:
                if DAEMON_LOG_FILE.exists():
                    lines = DAEMON_LOG_FILE.read_text(encoding='utf-8', errors='replace').splitlines()
                    tail = '\n'.join(lines[-8:]) if lines else ''
            except OSError:
                pass
            msg = f"scheduler daemon exited immediately (code {proc.returncode})"
            if tail:
                msg += f"\nLast log lines:\n{tail}"
            log_event(f"SCHEDULER DAEMON START FAILED: {msg}")
            return False, msg

    # Process still alive but pid file not written yet — trust the spawn.
    if proc.poll() is None and _is_scheduler_daemon_process(proc.pid):
        _write_daemon_pid(proc.pid)
        log_event(f"SCHEDULER DAEMON started pid={proc.pid} (pid file written by parent)")
        return True, f"scheduler daemon started (pid {proc.pid})"

    return False, "scheduler daemon failed to start"


def stop_daemon(timeout=5.0):
    """Ask the detached scheduler daemon to exit. Returns (ok, message)."""
    import time

    status = daemon_status()
    if not status['running']:
        _clear_daemon_pid()
        return True, "scheduler daemon is not running"

    pid = status['pid']
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        _clear_daemon_pid()
        return True, "scheduler daemon is not running"
    except OSError as e:
        return False, f"failed to signal daemon pid {pid}: {e}"

    deadline = time.time() + timeout
    while time.time() < deadline:
        if not _is_scheduler_daemon_process(pid):
            _clear_daemon_pid()
            log_event(f"SCHEDULER DAEMON stopped pid={pid}")
            return True, f"scheduler daemon stopped (was pid {pid})"
        time.sleep(0.1)

    # Still our daemon — escalate.
    if _is_scheduler_daemon_process(pid):
        try:
            os.kill(pid, signal.SIGKILL)
        except OSError:
            pass
        # Brief wait for SIGKILL to take effect.
        for _ in range(20):
            if not _pid_is_alive(pid):
                break
            time.sleep(0.05)
        _clear_daemon_pid()
        log_event(f"SCHEDULER DAEMON killed pid={pid}")
        return True, f"scheduler daemon force-killed (was pid {pid})"

    _clear_daemon_pid()
    return True, f"scheduler daemon stopped (was pid {pid})"


def _acquire_daemon_lock():
    """Exclusive lock so only one daemon instance runs. Returns open file or None."""
    try:
        config.GLOBAL_DIR.mkdir(exist_ok=True)
        lock_f = open(DAEMON_LOCK_FILE, 'a+', encoding='utf-8')
    except OSError as e:
        log_event(f"SCHEDULER DAEMON: cannot open lock file: {e}")
        return None

    try:
        if sys.platform == 'win32':
            # Best-effort on Windows: rely on pid file alone.
            return lock_f
        import fcntl
        fcntl.flock(lock_f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        return lock_f
    except BlockingIOError:
        lock_f.close()
        log_event("SCHEDULER DAEMON: another instance already holds the lock")
        return None
    except OSError as e:
        lock_f.close()
        log_event(f"SCHEDULER DAEMON: flock failed: {e}")
        return None


def run_daemon_forever(check_interval=DEFAULT_CHECK_INTERVAL_SECONDS):
    """Entry point for `bardgent --scheduler-daemon`.

    Blocks forever (or until SIGTERM/SIGINT), running due tasks on a timer.
    This process is detached from the user's terminal so schedules keep firing
    after the REPL exits.
    """
    lock_f = _acquire_daemon_lock()
    if lock_f is None:
        # Another daemon is already running — exit quietly so ensure_daemon
        # can still report success via the existing pid.
        os._exit(0)

    _write_daemon_pid(os.getpid())
    _scheduler_stop.clear()
    _cleaned = {'done': False}

    def _shutdown(signum=None, frame=None):
        try:
            log_event(f"SCHEDULER DAEMON received signal {signum}; shutting down")
        except Exception:
            pass
        _scheduler_stop.set()

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)

    def _cleanup():
        if _cleaned['done']:
            return
        _cleaned['done'] = True
        _clear_daemon_pid()
        try:
            if sys.platform != 'win32':
                import fcntl
                try:
                    fcntl.flock(lock_f.fileno(), fcntl.LOCK_UN)
                except (OSError, ValueError):
                    pass
            try:
                lock_f.close()
            except (OSError, ValueError):
                pass
        except Exception:
            pass

    log_event(f"SCHEDULER DAEMON ready pid={os.getpid()} interval={check_interval}s")
    try:
        _scheduler_loop(check_interval)
    finally:
        _cleanup()
        try:
            log_event(f"SCHEDULER DAEMON exit pid={os.getpid()}")
        except Exception:
            pass
        # Hard-exit so non-daemon threads / atexit hooks from imported libs
        # cannot keep the process alive after the scheduler loop ends.
        os._exit(0)


# Back-compat aliases used by older call sites / docs.
def start_scheduler(check_interval=DEFAULT_CHECK_INTERVAL_SECONDS):
    """Ensure the detached daemon is running (replaces the old in-process thread)."""
    ok, msg = ensure_daemon_running()
    log_event(f"SCHEDULER start_scheduler: {msg}")
    return ok


def stop_scheduler():
    """No-op for REPL exit: the daemon must keep running after the terminal closes.

    Use stop_daemon() (or /schedule daemon stop) to shut it down deliberately.
    """
    log_event("SCHEDULER stop_scheduler: leaving detached daemon running")
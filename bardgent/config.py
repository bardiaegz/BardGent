"""
Central place for constants, paths, the OpenAI-compatible client, logging,
and anything else every other module needs.

IMPORTANT: `MODEL` is mutated at runtime (via /model). Other modules must
always read it as `config.MODEL` (attribute access) rather than
`from bardgent.config import MODEL`, or they'll keep a stale copy.
"""

import os
import sys
import platform
import logging
import threading
import datetime
import re
from pathlib import Path

from openai import OpenAI
from rich.console import Console
from dotenv import load_dotenv

load_dotenv()
load_dotenv(Path.expanduser(Path("~/.bardgent/.env")))

console = Console()
console_lock = threading.RLock()

python_path = sys.executable
operating_system = platform.platform()
working_directory = os.getcwd()
home_directory = os.path.expanduser('~')

client = OpenAI(
    base_url='https://integrate.api.nvidia.com/v1',
    api_key=os.environ.get('GEMINI_API_KEY', '')
)

MODEL = 'nvidia/nemotron-3-ultra-550b-a55b'
TEMPERATURE = 0.2
MAX_ITERATIONS = 30
# Tool loops produce 2 messages per call (assistant tool_calls + tool result).
# 30 was far too low and mid-turn trims wiped the whole conversation.
MAX_HISTORY_MESSAGES = 120
# Always keep at least this many non-system messages after a trim.
MIN_HISTORY_MESSAGES = 6
MAX_TOOL_OUTPUT = 24_000
BASH_TIMEOUT_SECONDS = 60
# Default wait when the model calls Await() without a timeout.
BASH_AWAIT_DEFAULT_SECONDS = 30
# Hard cap on Await waits so a hung process cannot block the agent forever.
BASH_AWAIT_MAX_SECONDS = 600
# Default Read() window when limit is omitted but offset is set, and max lines
# returned in one Read call to keep tool results manageable.
READ_DEFAULT_LIMIT = 2_000
READ_MAX_LIMIT = 5_000
RESPONSE_TOKEN_RESERVE = 8_192
MAX_HISTORY_TOKENS = 180_000
AUTO_SUMMARY_TOKEN_THRESHOLD = 140_000
CONTEXT_WINDOW_TOKENS = 256_000

# Retry schedule for transient API failures.
MODEL_MAX_RETRIES = 10
MODEL_RETRY_DELAYS = [3, 5, 8, 13, 21, 30, 45, 60, 60, 60]

from openai import (  # noqa: E402  (kept together with the retry schedule above)
    APIError,
    APIConnectionError,
    APITimeoutError,
    RateLimitError,
    InternalServerError,
    BadRequestError,
    AuthenticationError,
    PermissionDeniedError,
    NotFoundError,
    UnprocessableEntityError,
)

# Transient / server-side only. 4xx client errors (bad history, bad schema,
# auth, etc.) must not be retried — they will fail the same way every time.
RETRYABLE_ERRORS = (APIConnectionError, APITimeoutError, RateLimitError, InternalServerError)
NON_RETRYABLE_ERRORS = (
    BadRequestError,
    AuthenticationError,
    PermissionDeniedError,
    NotFoundError,
    UnprocessableEntityError,
)

# ---------------------------------------------------------------------------
# Modes: plan / normal / auto (Claude Code style)
# ---------------------------------------------------------------------------
VALID_MODES = ('plan', 'normal', 'auto')
READONLY_TOOLS = {
    'Read', 'Glob', 'Grep', 'WebSearch', 'Fetch',
    'read_memory', 'list_memory', 'Skill', 'list_skills',
    # Observe background jobs without mutating the project.
    'ListJobs', 'Await',
}

# ---------------------------------------------------------------------------
# Directories
# ---------------------------------------------------------------------------
GLOBAL_DIR = Path.home() / '.bardgent'
GLOBAL_DIR.mkdir(exist_ok=True)

PERMISSIONS_DIR = Path.cwd() / '.bardgent'

SESSION_DIR = GLOBAL_DIR / "sessions"
SESSION_DIR.mkdir(exist_ok=True)
SESSION_PREFIX = "session_"
SUMMARY_PREFIX = '[Conversation summary so far]: '

BACKUP_DIR = GLOBAL_DIR / "backups"
BACKUP_DIR.mkdir(exist_ok=True)
last_backup = {}

MEMORY_FILE = GLOBAL_DIR / 'Bardgent.md'

# Background job log files (Bash run_in_background=true) live here. Must be
# defined after GLOBAL_DIR, since it's a subdirectory of it.
JOBS_DIR = GLOBAL_DIR / 'jobs'
JOBS_DIR.mkdir(exist_ok=True)

CHECKPOINT_REF = 'refs/bardgent/checkpoints'
CHECKPOINT_LOG = PERMISSIONS_DIR / 'checkpoints.json'
CHECKPOINT_INDEX_FILE = PERMISSIONS_DIR / 'checkpoint.index'

# ---------------------------------------------------------------------------
# Telegram
# ---------------------------------------------------------------------------
TELEGRAM_BOT_TOKEN = os.environ.get('TELEGRAM_BOT_TOKEN', '')
TELEGRAM_API_BASE = f'https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}'
TELEGRAM_CHATID_FILE = GLOBAL_DIR / 'telegram_chat_id.json'
TELEGRAM_MAX_LEN = 4000

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_FILE = GLOBAL_DIR / 'bardgent.log'
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
)


def log_event(msg):
    logging.info(msg)


# ---------------------------------------------------------------------------
# <thought>/<think> tag stripping
# ---------------------------------------------------------------------------
THOUGHT_TAG_RE = re.compile(r"<(?:thought|think)>.*?</(?:thought|think)>", re.DOTALL | re.IGNORECASE)
THOUGHT_OPEN_RE = re.compile(r"<(?:thought|think)>", re.IGNORECASE)


def remove_thoughts(text):
    if not text:
        return text
    text = THOUGHT_TAG_RE.sub("", text)
    m = THOUGHT_OPEN_RE.search(text)
    if m:
        text = text[:m.start()]
    return text.strip()


# ---------------------------------------------------------------------------
# Misc runtime constants
# ---------------------------------------------------------------------------
CWD_MARKER = '__BARDGENT_CWD__'
IS_WARP = os.environ.get('TERM_PROGRAM') == 'WarpTerminal'

DATETIME = datetime.datetime.now().astimezone()

SYSTEM_INFO = f"""[CRITICAL SYSTEM INFO]:
- Python Executable Path: {python_path}
- Operating System: {operating_system}
- Current Working Directory: {working_directory}
- User Home Directory: {home_directory}"""
from asyncio import new_event_loop
import logging
import pytz
import random
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Any, cast, Tuple, Dict, Literal

from bupp.src.log_utils import get_ctxt_id, LoggerProxy

# Logger names as top-level constants
AGENT_LOGGER_NAME = "agentlog"
FULL_REQUESTS_LOGGER_NAME = "full_requests"

_LOG_FORMAT = "%(asctime)s:[%(funcName)s:%(lineno)s] - %(message)s"

def converter(timestamp):
    dt = datetime.fromtimestamp(timestamp, tz=pytz.utc)
    return dt.astimezone(pytz.timezone("US/Eastern")).timetuple()

formatter = logging.Formatter(_LOG_FORMAT, datefmt="%H:%M:%S")
formatter.converter = converter

# Application log level type and resolver
LogLevel = Literal["debug", "info", "warning", "error", "critical"]

def _resolve_log_level(level: Optional[LogLevel]) -> int:
    if level is None:
        return logging.INFO
    level_map = {
        "debug": logging.DEBUG,
        "info": logging.INFO,
        "warning": logging.WARNING,
        "error": logging.ERROR,
        "critical": logging.CRITICAL,
    }
    # Defensive: normalize just in case callers pass uppercase
    return level_map.get(level.lower(), logging.INFO)  # type: ignore[attr-defined]

# note: literally exists to filter out Litellm ...
class ExcludeStringsFilter(logging.Filter):
    """
    A logging filter that excludes log records containing any of the specified strings.
    """
    DEFAULT_EXCLUDE_STRS = [
        "LiteLLM",
        "completion()",
        "selected model name",
        "Wrapper: Completed Call"
    ]

    def __init__(self, exclude_strs: List[str] = []):
        super().__init__()
        self.exclude_strs = exclude_strs + self.DEFAULT_EXCLUDE_STRS
    
    def filter(self, record):
        """
        Return False if the log record should be excluded, True otherwise.
        """
        if not self.exclude_strs:
            return True
        
        # Check if any exclude string is in the log message
        log_message = record.getMessage()
        for exclude_str in self.exclude_strs:
            if exclude_str in log_message:
                return False
        return True

# --------------------------------------------------------------------------- #
#  existing helpers (unchanged except for the new formatter objects)
# --------------------------------------------------------------------------- #
def get_file_handler(log_file: str | Path) -> logging.FileHandler:
    """
    Returns a file handler for logging.
    """
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)
    return file_handler

def get_console_handler(exclude_strs: List[str] = []) -> logging.StreamHandler:
    """
    Returns a console handler for logging.
    """
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    console_handler.addFilter(ExcludeStringsFilter(exclude_strs))
    return console_handler

def create_log_dir_or_noop(log_dir: str):
    date_dir = datetime.now().strftime("%Y-%m-%d")
    base_dir = Path(log_dir) / date_dir

    print(f"Creating log dir: {base_dir}")
    base_dir.mkdir(parents=True, exist_ok=True)
    return base_dir

def run_id_dir(base_dir: Path):
    """Returns the next run-id directory"""
    run_id = max((int(p.name) for p in base_dir.iterdir() if p.name.isdigit()), default=-1) + 1
    return base_dir / str(run_id)

class _ThreadFilter(logging.Filter):
    """Accept records only from the thread that created this handler."""
    def __init__(self, thread_id: int):
        super().__init__()
        self._thread_id = thread_id

    def filter(self, record: logging.LogRecord) -> bool:    # noqa: D401
        return record.thread == self._thread_id             # ❶ key line

class AgentFileHandler(logging.FileHandler):
    """
    A self-contained FileHandler that

    • creates the per-run directory tree (<LOG_DIR>/…/<run_id>/)
    • stores   .base_logdir   (Path to that run directory)
    • stores   .log_filepath  (Path to the specific *.log file)
    • auto-adds a _ThreadFilter so each thread gets its own file
    """
    def __init__(
        self,
        eval_name: str,
        base_dir: Path,
        *,
        level: int = logging.INFO,
        thread_id: Optional[int] = None,
        create_run_subdir: bool = True,
        add_thread_filter: bool = True,
        streaming: bool = False,
    ):
        self.thread_id = thread_id or threading.get_ident()

        if create_run_subdir:
            try:
                self.base_logdir = run_id_dir(base_dir)
                self.base_logdir.mkdir()
            except FileExistsError: 
                # race condition with other loggers creating files in same dir, random backoff to avoid 
                # re-conflicts
                time.sleep(0.1 + 0.3 * random.random())
                self.base_logdir = run_id_dir(base_dir)
                self.base_logdir.mkdir()
        else:
            # Write directly under the provided base_dir
            self.base_logdir = base_dir

        # Final log file path
        self.log_filepath = self.base_logdir / f"{eval_name}.log"

        # ── initialise parent FileHandler ───────────────────────────────── #
        # Pass mode='a' and delay=False to open immediately
        # When streaming=True, we'll set the stream to unbuffered
        super().__init__(self.log_filepath, mode='a', encoding="utf-8", delay=False)
        self.setLevel(level)
        self.setFormatter(formatter)

        # Configure unbuffered I/O for streaming mode
        if streaming and self.stream:
            # Reconfigure the stream with buffer size 0 (unbuffered)
            # We need to close and reopen with the correct buffering parameter
            self.stream.close()
            self.stream = open(self.log_filepath, mode='a', encoding='utf-8', buffering=1)  # line buffering

        # Per-thread isolation (optional)
        self._thread_id = self.thread_id
        if add_thread_filter:
            self.addFilter(_ThreadFilter(self.thread_id))

    def emit(self, record):
        """
        Override emit to force flush after each write when in streaming mode
        """
        super().emit(record)
        if hasattr(self, 'stream') and self.stream:
            self.stream.flush()

    def get_log_dirs(self):
        return self.base_logdir, self.log_filepath

# --------------------------------------------------------------------------- #
#  updated helpers
# --------------------------------------------------------------------------- #
def _setup_agent_logger(
    log_dir: str,
    *,
    parent_dir: Optional[Path] = None, # empty path
    name: str = AGENT_LOGGER_NAME,
    level: int = logging.INFO,
    create_run_subdir: bool = True,
    add_thread_filter: bool = True,
    no_console: bool = False,
    streaming: bool = False,
):
    """
    Updated log-directory layout:

        <LOG_DIR>/pentest_bot/<YYYY-MM-DD>/<N>/…

    The optional *subfolder* argument is still supported and, if supplied,
    is placed **inside** the date directory:

        <LOG_DIR>/pentest_bot/<YYYY-MM-DD>/<subfolder>/<N>/…
    
    Args:
        streaming: If True, disables buffering and flushes log output immediately
                   to disk after each log statement. Useful for real-time monitoring
                   but may impact performance.
    """
    base_dir = parent_dir if parent_dir else create_log_dir_or_noop(log_dir)
    thread_id = threading.get_ident()
    
    # Clear all handlers from root logger
    logging.getLogger().handlers.clear()

    # ─────────── Primary logger ────────────────────────────────────────── #
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not no_console and not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
        logger.addHandler(get_console_handler())
    existing_fh = next((
        h for h in logger.handlers
        if isinstance(h, AgentFileHandler) and getattr(h, "_thread_id", None) == thread_id
    ), None)
    if existing_fh is not None:
        fh = existing_fh
        cast(Any, logger)._run_dir = fh.base_logdir          # keep this public attr
    else:
        fh = AgentFileHandler(
            name,
            base_dir,
            level=level,
            thread_id=thread_id,
            create_run_subdir=create_run_subdir,
            add_thread_filter=add_thread_filter,
            streaming=streaming,
        )
        logger.addHandler(fh)
        cast(Any, logger)._run_dir = fh.base_logdir          # keep this public attr

    # ─────────── Secondary logger ("full_requests") ────────────────────── #
    fr_logger = logging.getLogger(FULL_REQUESTS_LOGGER_NAME)
    fr_logger.setLevel(level)
    fr_logger.propagate = False

    if not any(isinstance(h, AgentFileHandler) and h._thread_id == thread_id for h in fr_logger.handlers):
        # create a sibling dir <run>/full_requests/
        run_dir = cast(Path, getattr(logger, "_run_dir"))
        fr_dir = run_dir / "full_requests"
        fr_dir.mkdir(exist_ok=True)

        fr_fh = AgentFileHandler(
            f"{name}_requests",
            fr_dir,
            level=level,
            thread_id=thread_id,
            create_run_subdir=create_run_subdir,
            add_thread_filter=add_thread_filter,
            streaming=streaming,
        )
        fr_logger.addHandler(fr_fh)

    # return logger, fr_logger, fh.get_log_dirs
    return fh.get_log_dirs()

def setup_server_logger(log_dir: str):
    """Initialize a server logger with file handler using run-id directory structure"""
    base_dir = create_log_dir_or_noop(log_dir)
    run_dir = run_id_dir(base_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    
    logger = logging.getLogger(SERVER_LOGGER_NAME)
    logger.setLevel(logging.INFO)
    
    # Clear existing handlers to avoid duplicates
    logger.handlers.clear()
    
    # Create file handler for run_id.log
    log_file = run_dir / f"{run_dir.name}.log"
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(logging.Formatter(
        "%(asctime)s:[%(funcName)s:%(lineno)s] - %(message)s",
        datefmt="%H:%M:%S"
    ))
    logger.addHandler(file_handler)
    logger.addHandler(get_console_handler())

    for h in logger.handlers:
        print(f"SERVER LOGGER HANDLER: ", h)

def get_server_logger():
    return logging.getLogger(SERVER_LOGGER_NAME)

def unified_log():
    agent_log, full_log = get_agent_loggers()
    return LoggerProxy([agent_log, full_log])

def get_agent_loggers():
    return logging.getLogger(AGENT_LOGGER_NAME), logging.getLogger(FULL_REQUESTS_LOGGER_NAME)

# --------------------------------------------------------------------------- #
#  ServerLogFactory: per-engagement loggers and directory layout
# --------------------------------------------------------------------------- #

# static logger names
SERVER_LOGGER_NAME = "serverlog"
PROXY_LOGGER_NAME = "proxylog"
UVICORN_LOG = "uvicorn"
UVICORN_ACCESSS_LOG = "uvicorn.access"
UVICORN_ERROR_LOG = "uvicorn.error"
AGENT_POOL_LOGGER_NAME = "agentpool"

class _ServerLogFactory:
    """
    Singleton application-wide logger factory.
    3-tier file-level logging:

    1. base_dir (server / single-run agents)
        2. timestamp (represents a run instance)
            3. agents (discovery/exploit)

    - tier 1 represents a logical application container for the logs ie. single run of server
    - tier 2 [static] represents log processes where only a single log file is created ie. server logs
    - tier 3 [dynamic] represents log processes where multiple log files may be created ie. mutliple agents running in pool 
    """
    def __init__(self, base_dir: str, *, log_level: Optional[LogLevel] = None) -> None:
        self._base_dir = Path(base_dir)
        self._server_logger: Optional[logging.Logger] = None
        self._parent_logdir: Optional[Path] = None
        self._level: int = _resolve_log_level(log_level)

        self.setup_static_loggers()

    def _get_parent_logdir(self) -> Path:
        # Return cached directory if already created
        if self._parent_logdir is not None:
            return cast(Path, self._parent_logdir)
        
        # Create timestamp directory
        timestamp = datetime.now().strftime("%Y-%m-%d")
        timestamp_dir = self._base_dir / timestamp
        timestamp_dir.mkdir(parents=True, exist_ok=True)
        
        # Find next incremental ID in timestamp directory
        max_incr = 0
        for p in timestamp_dir.iterdir():
            if p.is_dir() and p.name.isdigit():
                max_incr = max(max_incr, int(p.name))
        
        incr_id = max_incr + 1
        _parent_logdir = timestamp_dir / str(incr_id)
        _parent_logdir.mkdir(parents=True, exist_ok=True)
        
        # Ensure subfolders exist
        (_parent_logdir / "discovery_agents").mkdir(exist_ok=True)
        (_parent_logdir / "exploit_agents").mkdir(exist_ok=True)
        
        # Cache the directory
        self._parent_logdir = _parent_logdir
        
        return _parent_logdir

    def _next_numeric_name(self, root: Path) -> str:
        """
        Determine the next numeric filename stem by scanning existing *.log files
        recursively and returning max(N)+1.
        """
        max_num = 0
        for p in root.rglob("*.log"):
            try:
                num = int(p.stem)
                if num > max_num:
                    max_num = num
            except ValueError:
                continue
        return str(max_num + 1)

    # static loggers
    def _create_static_logger(
        self, 
        logger_name: str, 
        log_filepath: Path, 
        no_console: bool = False
    ) -> logging.Logger:
        """Create a static logger with console and file handlers."""
        logger = logging.getLogger(logger_name)
        logger.setLevel(self._level if hasattr(self, "_level") else logging.INFO)
        logger.propagate = False

        # Avoid duplicate handlers if called multiple times
        if not no_console and not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
            logger.addHandler(get_console_handler())

        if not any(isinstance(h, logging.FileHandler) and getattr(h, "baseFilename", None) == str(log_filepath) for h in logger.handlers):
            fh = logging.FileHandler(log_filepath, encoding="utf-8")
            fh.setFormatter(formatter)
            logger.addHandler(fh)

        return logger

    def setup_server_logger(self, logger_name: str):
        """Create or return the server logger (with console + file)."""
        e_dir = self._get_parent_logdir()
        server_log_path = e_dir / "server.log"
        self._create_static_logger(logger_name, server_log_path)

    def setup_proxy_logger(self, logger_name: str):
        """Create or return the proxy logger (with console + file)."""
        e_dir = self._get_parent_logdir()
        proxy_log_path = e_dir / "proxy.log"
        self._create_static_logger(logger_name, proxy_log_path, no_console=True)

        print("DISABLING CONSOLE LOGGING FOR PROXY LOGGER!!!!")

    def setup_uvicorn_logger(self, logger_name: str):
        e_dir = self._get_parent_logdir()
        uvicorn_log_path = e_dir / "server.log"
        self._create_static_logger(logger_name, uvicorn_log_path)
    
    def setup_agent_pool_logger(self, logger_name: str):
        e_dir = self._get_parent_logdir()
        agent_pool_log_path = e_dir / "agent_pool.log"
        self._create_static_logger(logger_name, agent_pool_log_path)

    def _clear_all_handlers(self, logger_name: str):
        logging.getLogger(logger_name).handlers.clear()
        for handler in logging.getLogger(logger_name).handlers:
            logging.getLogger(logger_name).removeHandler(handler)

    def setup_static_loggers(self) -> None:
        """Setup the static loggers."""
        # clear all handlers from loggers first
        self._clear_all_handlers(SERVER_LOGGER_NAME)
        self._clear_all_handlers(PROXY_LOGGER_NAME)
        self._clear_all_handlers(UVICORN_LOG)
        self._clear_all_handlers(UVICORN_ACCESSS_LOG)
        self._clear_all_handlers(UVICORN_ERROR_LOG)
        self._clear_all_handlers(AGENT_POOL_LOGGER_NAME)

        self.setup_server_logger(SERVER_LOGGER_NAME)
        self.setup_proxy_logger(PROXY_LOGGER_NAME)
        self.setup_uvicorn_logger(UVICORN_LOG)
        self.setup_uvicorn_logger(UVICORN_ACCESSS_LOG)
        self.setup_uvicorn_logger(UVICORN_ERROR_LOG)
        self.setup_agent_pool_logger(AGENT_POOL_LOGGER_NAME)

        # should include?
        # self.get_discovery_agent_loggers()
        # self.get_exploit_agent_loggers()

    # dynamic loggers
    def get_discovery_agent_loggers(self, *, no_console: bool = False, streaming: bool = False) -> Tuple[logging.Logger, logging.Logger]:
        """
        Return loggers for a new discovery agent.
        Does not attach handlers; the worker thread should call setup_agent_logger
        with these values.
        """
        e_dir = self._get_parent_logdir()
        discovery_dir = e_dir / "discovery_agents"
        
        name = self._next_numeric_name(discovery_dir)

        _setup_agent_logger(
            log_dir="", 
            parent_dir=discovery_dir, 
            name=name, 
            level=self._level,
            create_run_subdir=False, 
            add_thread_filter=False, 
            no_console=no_console, 
            streaming=streaming
        )
        return logging.getLogger(name), logging.getLogger(FULL_REQUESTS_LOGGER_NAME)
    
    def get_exploit_agent_loggers(self, *, no_console: bool = False, streaming: bool = False) -> Tuple[logging.Logger, logging.Logger]:
        """
        Return loggers for a new exploit agent.
        Does not attach handlers; the worker thread should call setup_agent_logger.
        """
        e_dir = self._get_parent_logdir()
        exploit_dir = e_dir / "exploit_agents"

        name = self._next_numeric_name(exploit_dir)

        _setup_agent_logger(
            log_dir="", parent_dir=exploit_dir, name=name, level=self._level, create_run_subdir=False, add_thread_filter=False, no_console=no_console
        )
        return logging.getLogger(name), logging.getLogger(FULL_REQUESTS_LOGGER_NAME)

    def get_log_dir(self) -> Path:
        return self._get_parent_logdir()

def get_logger_or_default(logger_name: str) -> logging.Logger:
    """
    Return a logger with the given name. If the logger has no handlers,
    attach a console handler to it.
    """
    logger = logging.getLogger(logger_name)
    
    # Check if logger has any handlers
    if not logger.handlers:
        # No handlers, add a console handler
        console_handler = get_console_handler()
        logger.addHandler(console_handler)
        logger.setLevel(logging.INFO)
    
    return logger

_SERVER_LOG_FACTORY_SINGLETON: Optional[_ServerLogFactory] = None

def get_or_init_log_factory(
    base_dir: Optional[str] = None, 
    *, 
    log_level: Optional[LogLevel] = None,
    new: bool = False
) -> _ServerLogFactory:
    """
    Return a singleton ServerLogFactory. If base_dir is provided on first call,
    it sets the base directory; otherwise defaults to ".server_logs/engagements".
    Subsequent calls ignore base_dir and log_level.

    Usage (always use like so):
    log_factory = get_or_init_log_factory(base_dir=SERVER_LOG_DIR, log_level="info")
    agent_logger, full_logger = log_factory.get_exploit_agent_loggers()
    """
    global _SERVER_LOG_FACTORY_SINGLETON
    if _SERVER_LOG_FACTORY_SINGLETON is None or new:
        root = base_dir or ".server_logs/engagements"
        Path(root).mkdir(parents=True, exist_ok=True)
        _SERVER_LOG_FACTORY_SINGLETON = _ServerLogFactory(root, log_level=log_level)
    return _SERVER_LOG_FACTORY_SINGLETON

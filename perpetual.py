#!/usr/bin/env python3
"""
perpetual.py - Advanced mpv Playlist Manager with IPC

Features:
- Organize incoming TV files (sxanni-style) into Show/ folders (no prompts)
- Maintain an mpv 'recent files' playlist via JSON-IPC (append/prune/reorder)
- Default order = oldest first by mtime
- Saves/restores last playing file + position across restarts/quit
- After organizing, per-show S/E retime of mtimes so mtime order matches Season/Episode
- Uses inotify to react to new/updated files (no polling needed)
- Automatic IPC reconnection on failures
- File stability checking before processing
- Metrics tracking and monitoring
- Graceful shutdown with signal handling
- Optional auto-restart of mpv
- Configuration file support (TOML)
"""

import argparse
import atexit
import ctypes
import ctypes.util
import datetime as dt
import json
import logging
import os
import re
import select
import shutil
import signal
import socket
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, List, Set, Tuple, Pattern

# ================= Logging Setup =================
logger = logging.getLogger("perpetual")


def setup_logging(level: str = "INFO", log_file: Optional[str] = None) -> None:
    """
    Configure structured logging with appropriate handlers.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional path to log file
    """
    log_level = getattr(logging, level.upper(), logging.INFO)
    logger.setLevel(log_level)

    formatter = logging.Formatter(
        fmt='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler (optional)
    if log_file:
        try:
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        except Exception as e:
            logger.warning(f"Could not create log file {log_file}: {e}")


# ================= Metrics Tracking =================
@dataclass
class Metrics:
    """Track operational metrics for monitoring."""
    ipc_reconnections: int = 0
    ipc_failures: int = 0
    playlist_syncs: int = 0
    files_organized: int = 0
    files_processed: int = 0
    inotify_events: int = 0
    start_time: float = field(default_factory=time.time)
    last_health_check: float = field(default_factory=time.time)

    def uptime(self) -> float:
        """Return uptime in seconds."""
        return time.time() - self.start_time

    def log_summary(self) -> None:
        """Log metrics summary."""
        logger.info(
            f"Metrics: uptime={self.uptime():.1f}s, "
            f"ipc_reconnects={self.ipc_reconnections}, "
            f"ipc_failures={self.ipc_failures}, "
            f"syncs={self.playlist_syncs}, "
            f"organized={self.files_organized}, "
            f"events={self.inotify_events}"
        )


# Global metrics instance
metrics = Metrics()


# ================= Signal Handling =================
_shutdown_requested = False


def signal_handler(signum: int, frame) -> None:
    """Handle shutdown signals gracefully."""
    global _shutdown_requested
    sig_name = signal.Signals(signum).name
    logger.info(f"Received signal {sig_name}, initiating graceful shutdown...")
    _shutdown_requested = True


def setup_signal_handlers() -> None:
    """Install signal handlers for graceful shutdown."""
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    logger.debug("Signal handlers installed")


def should_shutdown() -> bool:
    """Check if shutdown has been requested."""
    return _shutdown_requested


# ================= Configuration =================
@dataclass
class Config:
    """Application configuration."""
    base_folder: Path
    days: float
    filetypes: List[str]
    socket_path: str
    state_interval: int
    order: str
    prune_playing: bool
    organize: bool
    garbage_words_file: str
    auto_launch_mpv: bool
    auto_restart_mpv: bool
    resume: bool
    resume_file: Path
    resume_allow_stale: bool
    watch_subdirs: bool
    log_level: str
    log_file: Optional[str]
    health_check_interval: int
    file_stability_delay: float
    file_stability_checks: int
    special_cases: Set[str] = field(default_factory=set)
    garbage_words: Set[str] = field(default_factory=set)

    @classmethod
    def from_args_and_file(cls, args: argparse.Namespace, config_file: Optional[Path] = None) -> 'Config':
        """
        Create Config from command-line args and optional config file.

        Args:
            args: Parsed command-line arguments
            config_file: Optional TOML config file path

        Returns:
            Config instance
        """
        config_data = {}

        # Load from config file if provided
        if config_file and config_file.exists():
            try:
                import tomllib
            except ImportError:
                try:
                    import tomli as tomllib
                except ImportError:
                    logger.warning("tomllib/tomli not available, config file support disabled")
                    tomllib = None

            if tomllib:
                try:
                    with open(config_file, 'rb') as f:
                        config_data = tomllib.load(f)
                    logger.info(f"Loaded configuration from {config_file}")
                except Exception as e:
                    logger.error(f"Failed to load config file {config_file}: {e}")

        # Command-line args override config file
        base_folder = Path(args.base).expanduser().resolve()
        resume_file = Path(args.resume_file).expanduser().resolve()

        return cls(
            base_folder=base_folder,
            days=config_data.get('days', args.days),
            filetypes=config_data.get('filetypes', args.ext),
            socket_path=config_data.get('socket_path', args.socket),
            state_interval=config_data.get('state_interval', args.state_interval),
            order=config_data.get('order', args.order),
            prune_playing=config_data.get('prune_playing', args.prune_playing),
            organize=not args.no_organize,
            garbage_words_file=config_data.get('garbage_words_file', args.garbage_words),
            auto_launch_mpv=config_data.get('auto_launch_mpv', args.auto_launch),
            auto_restart_mpv=config_data.get('auto_restart_mpv', args.auto_restart_mpv),
            resume=args.resume,
            resume_file=resume_file,
            resume_allow_stale=config_data.get('resume_allow_stale', args.resume_allow_stale),
            watch_subdirs=config_data.get('watch_subdirs', args.watch_subdirs),
            log_level=config_data.get('log_level', args.log_level),
            log_file=config_data.get('log_file', args.log_file),
            health_check_interval=config_data.get('health_check_interval', args.health_check_interval),
            file_stability_delay=config_data.get('file_stability_delay', args.file_stability_delay),
            file_stability_checks=config_data.get('file_stability_checks', args.file_stability_checks),
        )


# ================= Inotify Wrapper =================
libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)

IN_CLOSE_WRITE = 0x00000008
IN_MOVED_FROM = 0x00000040
IN_MOVED_TO = 0x00000080
IN_DELETE = 0x00000200

WATCH_MASK = IN_CLOSE_WRITE | IN_MOVED_TO | IN_DELETE | IN_MOVED_FROM


class Inotify:
    """Wrapper for Linux inotify file system monitoring."""

    def __init__(self):
        """Initialize inotify instance."""
        self.fd = libc.inotify_init1(0)
        if self.fd < 0:
            e = ctypes.get_errno()
            raise OSError(e, os.strerror(e))
        self.wds: Dict[int, str] = {}
        logger.debug(f"Inotify initialized with fd={self.fd}")

    def add_watch(self, path: str, mask: int = WATCH_MASK) -> int:
        """
        Add a watch for a path.

        Args:
            path: Directory path to watch
            mask: Event mask

        Returns:
            Watch descriptor
        """
        bpath = path.encode('utf-8')
        wd = libc.inotify_add_watch(self.fd, ctypes.c_char_p(bpath), ctypes.c_uint32(mask))
        if wd < 0:
            e = ctypes.get_errno()
            raise OSError(e, os.strerror(e))
        self.wds[wd] = path
        logger.debug(f"Added inotify watch for {path} (wd={wd})")
        return wd

    def read_events(self) -> List[Tuple[int, int, int, str]]:
        """
        Read pending inotify events.

        Returns:
            List of (wd, mask, cookie, name) tuples
        """
        try:
            data = os.read(self.fd, 4096)
        except BlockingIOError:
            return []

        events = []
        i = 0
        while i < len(data):
            wd = int.from_bytes(data[i:i+4], 'little', signed=True)
            mask = int.from_bytes(data[i+4:i+8], 'little')
            cookie = int.from_bytes(data[i+8:i+12], 'little')
            length = int.from_bytes(data[i+12:i+16], 'little')
            name = b""
            if length > 0:
                name = data[i+16:i+16+length].split(b"\0", 1)[0]
            events.append((wd, mask, cookie, name.decode('utf-8')))
            i += 16 + length

        return events

    def close(self) -> None:
        """Close the inotify file descriptor."""
        if self.fd >= 0:
            try:
                os.close(self.fd)
                logger.debug("Inotify fd closed")
            except Exception as e:
                logger.warning(f"Error closing inotify fd: {e}")
            self.fd = -1


# ================= Constants =================
BASE_FOLDER = '/data/videos'
FILETYPES = ["avi", "mkv", "mpeg", "mp4", "m4v", "mpg", "webm", "avif", "ts"]

SPECIAL_CASES = {
    "USA", "FBI", "BBC", "US", "AU", "PL", "IE", "NZ", "FR", "DE", "JP", "UK",
    "QI", "XL", "SAS", "RAF", "WWII", "WPC", "LOL", "VI", "VII", "VIII", "VIIII", "IX", "II", "III", "IV",
    "DCI", "HD", "W1A", "HBO", "100K"
}


def load_garbage_words(filepath: str) -> Set[str]:
    """
    Load garbage words from a file.

    Args:
        filepath: Path to garbage words file (one per line)

    Returns:
        Set of lowercase garbage words
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            words = {line.strip().lower() for line in f if line.strip()}
        logger.info(f"Loaded {len(words)} garbage words from {filepath}")
        return words
    except FileNotFoundError:
        logger.warning(f"Garbage words file not found: {filepath}")
        return set()
    except Exception as e:
        logger.error(f"Error loading garbage words from {filepath}: {e}")
        return set()


# ================= File Organization =================
SE_RE = re.compile(r"\.s(\d{2,3})e(\d{2})\.", re.IGNORECASE)


def make_path_with_test(path: Path) -> bool:
    """
    Create directory path if it doesn't exist.

    Args:
        path: Directory path to create

    Returns:
        True if successful, False otherwise
    """
    try:
        path.mkdir(parents=True, exist_ok=True)
        return True
    except Exception as e:
        logger.error(f"Could not create {path}: {e}")
        return False


def clean_filename(filename: str, garbage_words: Set[str], filetypes: List[str],
                   special_cases: Set[str]) -> Optional[Tuple[str, str]]:
    """
    Parse and clean TV show filename into organized format.

    Args:
        filename: Original filename
        garbage_words: Set of words to filter out
        filetypes: List of valid file extensions
        special_cases: Set of words to keep uppercase

    Returns:
        Tuple of (folder_name, cleaned_filename) or None if parsing fails
    """
    fn = filename.replace('_', '.').replace('-', '.').replace(' ', '.')
    m = re.search(r"(.*?)(s\d{2,3}e\d{2})", fn, re.IGNORECASE)
    if not m:
        return None

    show_raw, season_episode = m.groups()
    title_match = re.search(r"s\d{2,3}e\d{2}\.(.*)", fn, re.IGNORECASE)
    episode_title_raw = title_match.group(1) if title_match else ""
    title_tokens = re.split(r"[._\s]+", episode_title_raw.strip())

    filtered_tokens = []
    filetypes_lower = {e.lower() for e in filetypes}
    for tok in title_tokens:
        low = tok.lower()
        if low in garbage_words or low in filetypes_lower:
            break
        if tok:
            filtered_tokens.append(tok)

    episode_title = ''
    if filtered_tokens:
        formatted = [w.upper() if w.upper() in special_cases else w.capitalize()
                    for w in filtered_tokens]
        episode_title = "." + ".".join(formatted).strip(".")

    show_tokens = re.split(r"[._\s]+", show_raw.strip())
    formatted_show = [w.upper() if w.upper() in special_cases else w.capitalize()
                     for w in show_tokens if w]
    show_name = ".".join(formatted_show).strip(".")

    ext = filename.split(".")[-1].lower()
    folder = show_name.strip()

    if season_episode.upper() != 'S00E00':
        clean = f"{folder}.{season_episode.upper()}{episode_title}.{ext}"
    else:
        clean = f"{folder}{episode_title}.{ext}"

    return folder, clean


def dynamic_episode_pattern(filetypes: List[str]) -> Pattern:
    """
    Create regex pattern for matching TV episode filenames.

    Args:
        filetypes: List of valid file extensions

    Returns:
        Compiled regex pattern
    """
    exts = "|".join(re.escape(e) for e in sorted({e.lower() for e in filetypes}))
    return re.compile(rf'.*(\_|\.)s\d{{1,3}}e\d{{1,2}}.*\.({exts})$', re.IGNORECASE)


def safe_move_auto(source: str, target: str, folder: str) -> None:
    """
    Move file and update metadata if applicable.

    Args:
        source: Source file path
        target: Target file path
        folder: Parent folder path
    """
    folder_p = Path(folder)
    if folder and not folder_p.exists():
        make_path_with_test(folder_p)

    shutil.move(source, target)

    if target.lower().endswith(('.mkv', '.mp4')):
        name, _ = os.path.splitext(os.path.basename(target))
        name_disp = name.replace('.', ' ')
        if target.lower().endswith('.mkv'):
            os.system(f'mkvpropedit "{target}" --edit info --set "title={name_disp}" 2>/dev/null')
        else:
            Path(target).touch()
    else:
        Path(target).touch()


def check_file_stability(filepath: Path, delay: float = 1.0, checks: int = 2) -> bool:
    """
    Check if file has finished being written by verifying stable size.

    Args:
        filepath: Path to file to check
        delay: Delay between checks in seconds
        checks: Number of stability checks to perform

    Returns:
        True if file size is stable, False otherwise
    """
    try:
        last_size = filepath.stat().st_size
        for _ in range(checks):
            time.sleep(delay)
            current_size = filepath.stat().st_size
            if current_size != last_size:
                logger.debug(f"File {filepath} still being written (size changed {last_size} -> {current_size})")
                return False
            last_size = current_size
        return True
    except FileNotFoundError:
        logger.debug(f"File {filepath} disappeared during stability check")
        return False
    except Exception as e:
        logger.warning(f"Error checking file stability for {filepath}: {e}")
        return False


def organize_incoming(base_folder: Path, filetypes: List[str], garbage_words: Set[str],
                     special_cases: Set[str], check_stability: bool = True,
                     stability_delay: float = 1.0, stability_checks: int = 2) -> Set[Path]:
    """
    Organize incoming TV files into show folders.

    Args:
        base_folder: Base directory containing files
        filetypes: List of valid file extensions
        garbage_words: Set of words to filter from titles
        special_cases: Set of words to keep uppercase
        check_stability: Whether to check file write completion
        stability_delay: Delay between stability checks
        stability_checks: Number of stability checks

    Returns:
        Set of show directories that were modified
    """
    moved_shows: Set[Path] = set()
    pattern = dynamic_episode_pattern(filetypes)
    files = [str(f) for f in base_folder.glob('*')
             if f.is_file() and f.stat().st_size > 0 and pattern.match(f.name)]

    for file_ in files:
        file_path = Path(file_)

        # Check if file is still being written
        if check_stability and not check_file_stability(file_path, stability_delay, stability_checks):
            logger.debug(f"Skipping {file_path} - still being written")
            continue

        leaf = os.path.basename(file_)
        result = clean_filename(leaf, garbage_words, filetypes, special_cases)
        if not result:
            continue

        folder, newname = result
        tvshow_path = base_folder / folder
        target = tvshow_path / newname

        if os.path.abspath(file_) != os.path.abspath(target):
            try:
                safe_move_auto(file_, str(target), str(tvshow_path))
                moved_shows.add(tvshow_path)
                logger.info(f"Organized: {file_} -> {target}")
                metrics.files_organized += 1
            except Exception as e:
                logger.warning(f"Failed to organize {file_}: {e}")

    return moved_shows


def parse_se(path: Path) -> Optional[Tuple[int, int]]:
    """
    Parse season and episode numbers from filename.

    Args:
        path: File path

    Returns:
        Tuple of (season, episode) or None if not found
    """
    m = SE_RE.search(path.name + ".")
    if not m:
        return None
    try:
        return (int(m.group(1)), int(m.group(2)))
    except ValueError:
        return None


def is_allowed_ext(path: Path, filetypes: List[str]) -> bool:
    """
    Check if file has an allowed extension.

    Args:
        path: File path
        filetypes: List of valid extensions

    Returns:
        True if extension is allowed
    """
    return path.suffix.lower().lstrip('.') in {e.lower() for e in filetypes}


def retime_show_folder(show_dir: Path, filetypes: List[str]) -> None:
    """
    Adjust file modification times to match season/episode order.

    Args:
        show_dir: Show directory path
        filetypes: List of valid file extensions
    """
    if not show_dir.exists() or not show_dir.is_dir():
        return

    candidates: List[Path] = [
        p for p in show_dir.glob("*")
        if p.is_file() and is_allowed_ext(p, filetypes) and parse_se(p)
    ]

    if len(candidates) < 2:
        return

    candidates.sort(key=lambda p: parse_se(p))
    now = time.time()
    start = now - len(candidates)

    for i, p in enumerate(candidates):
        mtime = start + i
        try:
            os.utime(p, (mtime, mtime))
        except Exception as e:
            logger.warning(f"Failed to retime {p}: {e}")

    logger.info(f"Retimed {show_dir}: {len(candidates)} files ordered by S/E")


# ================= File Scanning =================
def jnow() -> dt.datetime:
    """Get current UTC datetime."""
    return dt.datetime.now(dt.timezone.utc)


def file_is_recent(p: Path, days: float) -> bool:
    """
    Check if file was modified within the specified time window.

    Args:
        p: File path
        days: Number of days for recency window

    Returns:
        True if file is recent
    """
    try:
        m = dt.datetime.fromtimestamp(p.stat().st_mtime, dt.timezone.utc)
    except FileNotFoundError:
        return False
    return (jnow() - m) <= dt.timedelta(days=days)


def scan_recent(dirpath: Path, days: float, exts: List[str], order: str = "oldest") -> List[Path]:
    """
    Scan for recent video files.

    Args:
        dirpath: Directory to scan
        days: Number of days for recency window
        exts: List of valid file extensions
        order: Sort order ("oldest" or "newest")

    Returns:
        Sorted list of recent file paths
    """
    logger.debug(f"Scanning for recent files in {dirpath}")
    exts_set = {e.lower().lstrip('.') for e in exts}
    files = []

    for root, _, fnames in os.walk(dirpath):
        for f in fnames:
            if exts_set and Path(f).suffix.lower().lstrip('.') not in exts_set:
                continue
            p = Path(root) / f
            if file_is_recent(p, days):
                files.append(p.resolve())

    rev = (order == "newest")
    files.sort(key=lambda p: p.stat().st_mtime, reverse=rev)
    logger.info(f"Found {len(files)} recent files")

    return files


# ================= MPV IPC =================
class MpvIPC:
    """MPV JSON IPC client with automatic reconnection."""

    def __init__(self, addr: str, auto_launch: bool = False, auto_restart: bool = False,
                 launch_cmd: Optional[List[str]] = None, timeout: int = 20):
        """
        Initialize MPV IPC connection.

        Args:
            addr: Unix socket path
            auto_launch: Launch mpv if not running
            auto_restart: Restart mpv if it quits
            launch_cmd: Custom mpv launch command
            timeout: Connection timeout in seconds
        """
        self.addr = addr
        self.sock: Optional[socket.socket] = None
        self.auto_launch = auto_launch
        self.auto_restart = auto_restart
        self.launch_cmd = launch_cmd or [
            "/usr/bin/mpv",
            "--idle=yes",
            "--force-window=yes",
            "--really-quiet",
            "--alang=en",
            "--sub-auto=all",
            "--video-unscaled=no",
            "--video-zoom=0",
            "--volume=100",
            "--fullscreen=yes",
            "--fs",
            "--loop-playlist=inf",
            f"--input-ipc-server={addr}",
        ]
        self.timeout = timeout
        self.op_timeout = 2.0  # Shorter timeout for operations
        self.mpv_process: Optional[subprocess.Popen] = None
        self._connect()

    def _launch_mpv(self) -> None:
        """Launch mpv process."""
        if os.path.exists(self.addr):
            try:
                os.unlink(self.addr)
            except Exception:
                pass

        self.mpv_process = subprocess.Popen(
            self.launch_cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        logger.info(f"Launched mpv with PID {self.mpv_process.pid}")

    def _connect(self) -> None:
        """Establish connection to mpv IPC socket."""
        # Close existing socket if any
        if self.sock:
            try:
                self.sock.close()
            except Exception:
                pass
            self.sock = None

        if self.auto_launch or self.auto_restart:
            self._launch_mpv()

        start = time.time()
        last_err = None
        while time.time() - start < self.timeout:
            # Check if shutdown was requested
            if should_shutdown():
                raise ConnectionRefusedError(f"Shutdown requested, aborting connection to {self.addr}")

            try:
                s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                s.settimeout(self.op_timeout)
                s.connect(self.addr)
                self.sock = s
                logger.info(f"Connected to mpv IPC at {self.addr}")
                return
            except Exception as e:
                last_err = e
                # Check again before sleeping
                if not should_shutdown():
                    time.sleep(0.5)

        raise ConnectionRefusedError(f"Could not connect to mpv at {self.addr}: {last_err}")

    def reconnect(self) -> bool:
        """
        Attempt to reconnect to mpv IPC.

        Returns:
            True if reconnection successful
        """
        # Don't reconnect if shutdown requested
        if should_shutdown():
            logger.info("Shutdown requested, skipping reconnection")
            return False

        # Check if socket file exists - if not, mpv quit cleanly
        if not os.path.exists(self.addr):
            if not self.auto_restart:
                logger.info("mpv quit cleanly (socket removed), not reconnecting")
                return False
            logger.info("mpv quit, auto-restart enabled, relaunching...")
        else:
            logger.info("Attempting IPC reconnection...")

        try:
            self._connect()
            metrics.ipc_reconnections += 1
            return True
        except Exception as e:
            logger.error(f"Reconnection failed: {e}")
            metrics.ipc_failures += 1
            return False

    def is_alive(self) -> bool:
        """
        Check if IPC connection is alive.

        Returns:
            True if connection is alive
        """
        if not self.sock:
            return False
        try:
            data = self.sock.recv(1, socket.MSG_PEEK | socket.MSG_DONTWAIT)
            if data == b'':
                return False
            return True
        except BlockingIOError:
            return True
        except (OSError, socket.error):
            return False

    def has_quit_cleanly(self) -> bool:
        """
        Check if mpv has quit cleanly (socket file removed).

        Returns:
            True if mpv quit cleanly
        """
        return not os.path.exists(self.addr)

    def _send(self, payload: Dict) -> None:
        """
        Send JSON command with retry logic.

        Args:
            payload: JSON command payload
        """
        line = json.dumps(payload, separators=(',', ':')) + "\n"
        retry_count = 0
        max_retries = 2

        while retry_count <= max_retries:
            try:
                if not self.sock:
                    raise BrokenPipeError("Socket not connected")
                self.sock.sendall(line.encode('utf-8'))
                return
            except (BrokenPipeError, ConnectionResetError, socket.timeout, OSError) as e:
                logger.warning(f"Send failed: {e}")
                # If mpv quit cleanly, don't retry
                if self.has_quit_cleanly():
                    logger.debug("mpv quit cleanly, aborting send")
                    raise
                if retry_count < max_retries:
                    if self.reconnect():
                        retry_count += 1
                        continue
                raise

    def send(self, *command) -> None:
        """
        Send command to mpv.

        Args:
            *command: Command and arguments
        """
        try:
            self._send({"command": list(command)})
        except Exception as e:
            logger.error(f"Failed to send command {command}: {e}")

    def _recv(self) -> Optional[Dict]:
        """
        Receive JSON response.

        Returns:
            Parsed JSON response or None
        """
        try:
            data = b''
            while True:
                chunk = self.sock.recv(65536)
                if not chunk:
                    break
                data += chunk
                if b"\n" in chunk:
                    break

            if not data:
                return None

            obj = json.loads(data.splitlines()[-1].decode('utf-8'))
            return obj if isinstance(obj, dict) else None
        except socket.timeout:
            logger.warning("Receive timeout")
            return None
        except Exception as e:
            logger.warning(f"Receive error: {e}")
            return None

    def get_property(self, prop: str) -> Optional[any]:
        """
        Get mpv property value.

        Args:
            prop: Property name

        Returns:
            Property value or None
        """
        retry_count = 0
        max_retries = 2

        while retry_count <= max_retries:
            try:
                self._send({"command": ["get_property", prop]})
                resp = self._recv()
                if not resp or resp.get("error") != "success":
                    return None
                return resp.get("data")
            except Exception as e:
                logger.warning(f"get_property({prop}) failed: {e}")
                # If mpv quit cleanly, don't retry
                if self.has_quit_cleanly():
                    logger.debug("mpv quit cleanly, aborting get_property")
                    return None
                if retry_count < max_retries:
                    if self.reconnect():
                        retry_count += 1
                        continue
                return None

    def close(self) -> None:
        """Close IPC connection and optionally terminate mpv."""
        if self.sock:
            try:
                self.sock.close()
                logger.debug("IPC socket closed")
            except Exception:
                pass
            self.sock = None

        if self.mpv_process and self.mpv_process.poll() is None:
            try:
                self.mpv_process.terminate()
                self.mpv_process.wait(timeout=5)
                logger.info("mpv process terminated")
            except Exception as e:
                logger.warning(f"Error terminating mpv: {e}")


# ================= Playlist Management =================
def reconcile_playlist(ipc: MpvIPC, desired_paths: List[Path], keep_current: bool = True) -> None:
    """
    Synchronize mpv playlist with desired file list.

    Args:
        ipc: MPV IPC client
        desired_paths: List of desired file paths
        keep_current: Don't remove currently playing file
    """
    playlist = ipc.get_property("playlist")

    # Validate playlist is a list
    if not isinstance(playlist, list):
        logger.warning(f"Expected playlist to be list, got {type(playlist).__name__}: {playlist}")
        print(playlist)
        return

    idx_by_path = {}

    for i, it in enumerate(playlist):
        if not isinstance(it, dict):
            continue
        fn = it.get("filename")
        if fn:
            try:
                p = str(Path(fn).resolve())
            except Exception:
                p = fn
            idx_by_path[p] = i

    cur_pos = ipc.get_property("playlist-current-pos") or -1

    desired_set = {str(p) for p in desired_paths}
    current_set = set(idx_by_path.keys())

    to_add = [str(p) for p in desired_paths if str(p) not in current_set]
    to_remove_paths = current_set - desired_set

    remove_indices = []
    for path in to_remove_paths:
        idx = idx_by_path.get(path)
        if idx is None:
            continue
        if keep_current and idx == cur_pos:
            continue
        remove_indices.append(idx)

    for idx in sorted(remove_indices, reverse=True):
        ipc.send("playlist-remove", idx)

    for p in to_add:
        ipc.send("loadfile", p, "append-play")

    if to_add or remove_indices:
        logger.debug(f"Playlist: added {len(to_add)}, removed {len(remove_indices)}")
        metrics.playlist_syncs += 1


def reorder_playlist(ipc: MpvIPC, desired_paths: List[Path]) -> None:
    """
    Reorder mpv playlist to match desired order.

    Args:
        ipc: MPV IPC client
        desired_paths: List of paths in desired order
    """
    desired = [str(p) for p in desired_paths]
    pl = ipc.get_property("playlist")

    # Validate playlist is a list
    if not isinstance(pl, list):
        logger.warning(f"Expected playlist to be list, got {type(pl).__name__}: {pl}")
        return

    current = []

    for it in pl:
        if not isinstance(it, dict):
            continue
        fn = it.get("filename")
        if not fn:
            continue
        try:
            current.append(str(Path(fn).resolve()))
        except Exception:
            current.append(fn)

    index_of = {path: i for i, path in enumerate(current)}

    for i, want in enumerate(desired):
        if i < len(current) and current[i] == want:
            continue

        j = index_of.get(want)
        if j is None:
            ipc.send("loadfile", want, "append-play")
            current.append(want)
            index_of[want] = len(current) - 1
            j = index_of[want]

        if j != i:
            ipc.send("playlist-move", j, i)
            item = current.pop(j)
            current.insert(i, item)
            start = min(i, j)
            for k in range(start, len(current)):
                index_of[current[k]] = k


def ensure_playing(ipc: MpvIPC) -> None:
    """
    Ensure mpv is playing if playlist has items.

    Args:
        ipc: MPV IPC client
    """
    count = ipc.get_property("playlist-count") or 0
    if isinstance(count, dict):
        count = count.get("playlist_entry_id", 0)

    cur = ipc.get_property("playlist-current-pos")
    if cur is not None and isinstance(cur, dict):
        cur = cur.get("playlist_entry_id", 0)

    if count > 0:
        if cur is None or cur < 0:
            ipc.send("playlist-play-index", 0)
        ipc.send("set", "pause", "no")
        ipc.send("sub-pos", "-100")


# ================= Resume State =================
def default_resume_path() -> Path:
    """Get default resume state file path."""
    return Path.home() / ".local" / "state" / "perpetual_resume.json"


def save_resume_state(ipc: MpvIPC, resume_path: Path) -> None:
    """
    Save current playback state for resume.

    Args:
        ipc: MPV IPC client
        resume_path: Path to resume state file
    """
    state_dir = resume_path.parent
    state_dir.mkdir(parents=True, exist_ok=True)

    path = ipc.get_property("path")
    pos = ipc.get_property("time-pos")

    if path and isinstance(pos, (int, float)):
        try:
            with open(resume_path, "w", encoding='utf-8') as f:
                json.dump({"path": path, "time": float(pos)}, f)
            logger.debug(f"Saved resume state: {path} @ {pos:.1f}s")
        except Exception as e:
            logger.warning(f"Failed to save resume state: {e}")


def load_resume_state(resume_path: Path) -> Tuple[Optional[str], Optional[float]]:
    """
    Load saved resume state.

    Args:
        resume_path: Path to resume state file

    Returns:
        Tuple of (path, time_position) or (None, None)
    """
    try:
        with open(resume_path, "r", encoding='utf-8') as f:
            data = json.load(f)
        path = data.get("path")
        t = data.get("time")
        if path and isinstance(t, (int, float)):
            return (path, float(t))
    except FileNotFoundError:
        pass
    except Exception as e:
        logger.warning(f"Failed to load resume state: {e}")

    return (None, None)


def apply_resume_if_possible(ipc: MpvIPC, desired_paths: List[Path], resume_path: Path,
                            allow_stale: bool = False) -> None:
    """
    Apply saved resume state if possible.

    Args:
        ipc: MPV IPC client
        desired_paths: List of current file paths
        resume_path: Path to resume state file
        allow_stale: Allow resuming files not in current playlist
    """
    saved_path, saved_time = load_resume_state(resume_path)
    if not saved_path:
        return

    try:
        saved_path = str(Path(saved_path).resolve())
        desired_strs = [str(p) for p in desired_paths]

        if saved_path not in desired_strs and allow_stale:
            ipc.send("loadfile", saved_path, "append-play")
            desired_strs.append(saved_path)

        pl = ipc.get_property("playlist") or []
        idx_map = {}
        for i, it in enumerate(pl):
            if not isinstance(it, dict):
                continue
            fn = it.get("filename")
            if not fn:
                continue
            try:
                p = str(Path(fn).resolve())
            except Exception:
                p = fn
            idx_map[p] = i

        idx = idx_map.get(saved_path)
        if idx is None:
            return

        ipc.send("playlist-play-index", idx)
        ipc.send("seek", saved_time, "absolute", "exact")
        ipc.send("set", "pause", "no")
        ipc.send("sub-pos", "-100")
        logger.info(f"Resumed playback: {saved_path} @ {saved_time:.1f}s")
    except Exception as e:
        logger.warning(f"Failed to apply resume state: {e}")


# ================= Main =================
def resync_playlist(ipc: MpvIPC, config: Config) -> None:
    """
    Full playlist resynchronization.

    Args:
        ipc: MPV IPC client
        config: Application configuration
    """
    desired = scan_recent(config.base_folder, config.days, config.filetypes, order=config.order)
    reconcile_playlist(ipc, desired, keep_current=not config.prune_playing)
    reorder_playlist(ipc, desired)
    ensure_playing(ipc)


def main() -> None:
    """Main application entry point."""
    ap = argparse.ArgumentParser(
        description="Advanced mpv playlist manager with IPC reconnection, file monitoring, and auto-organization"
    )

    # Core settings
    ap.add_argument("--config", type=Path, help="TOML configuration file")
    ap.add_argument("--base", default=BASE_FOLDER, help=f"Base folder (default: {BASE_FOLDER})")
    ap.add_argument("--days", type=float, default=3.0, help="Recent window in days (default: 3)")
    ap.add_argument("--ext", nargs="*", default=FILETYPES, help="Allowed extensions")
    ap.add_argument("--socket", default="/tmp/mpv.sock", help="mpv IPC path")
    ap.add_argument("--order", choices=["newest", "oldest"], default="oldest",
                   help="Sort by mtime (default: oldest)")

    # State management
    ap.add_argument("--state-interval", type=int, default=10,
                   help="Resume-state save interval seconds (default: 10)")
    ap.add_argument("--resume", action="store_true", default=True,
                   help="Resume last file/time if possible (default: on)")
    ap.add_argument("--no-resume", action="store_false", dest="resume", help="Disable resume")
    ap.add_argument("--resume-file", default=str(default_resume_path()),
                   help="Path to resume state JSON")
    ap.add_argument("--resume-allow-stale", action="store_true",
                   help="Resume even if file is outside recent window")

    # Organization
    ap.add_argument("--no-organize", action="store_true", help="Skip organize step")
    ap.add_argument("--garbage-words", default="/data/tvtitle_munge.txt",
                   help="Garbage words file")
    ap.add_argument("--prune-playing", action="store_true",
                   help="Allow pruning currently playing item")

    # Monitoring
    ap.add_argument("--watch-subdirs", action="store_true",
                   help="Also watch 1-level subdirs under base for changes")
    ap.add_argument("--health-check-interval", type=int, default=30,
                   help="IPC health check interval in seconds (default: 30)")

    # MPV control
    ap.add_argument("--auto-launch", action="store_true", help="Launch mpv if IPC not present")
    ap.add_argument("--auto-restart-mpv", action="store_true",
                   help="Automatically restart mpv if it quits unexpectedly")

    # File stability
    ap.add_argument("--file-stability-delay", type=float, default=1.0,
                   help="Delay between file stability checks in seconds (default: 1.0)")
    ap.add_argument("--file-stability-checks", type=int, default=2,
                   help="Number of stability checks before processing file (default: 2)")

    # Logging
    ap.add_argument("--log-level", default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                   help="Logging level (default: INFO)")
    ap.add_argument("--log-file", help="Optional log file path")

    args = ap.parse_args()

    # Setup logging first
    setup_logging(args.log_level, args.log_file)

    # Setup signal handlers
    setup_signal_handlers()

    # Load configuration
    config = Config.from_args_and_file(args, args.config)

    # Load garbage words and special cases
    config.garbage_words = load_garbage_words(config.garbage_words_file)
    config.special_cases = SPECIAL_CASES

    # Validate base folder
    if not config.base_folder.exists():
        logger.error(f"Base folder not found: {config.base_folder}")
        sys.exit(1)

    logger.info("=" * 60)
    logger.info("Perpetual MPV Playlist Manager")
    logger.info("=" * 60)
    logger.info(f"Base folder: {config.base_folder}")
    logger.info(f"Recent window: {config.days} days")
    logger.info(f"Order: {config.order}")
    logger.info(f"Auto-launch: {config.auto_launch_mpv}")
    logger.info(f"Auto-restart: {config.auto_restart_mpv}")
    logger.info(f"Resume: {config.resume}")
    logger.info("=" * 60)

    # Initial organization
    if config.organize:
        logger.info("Running initial organization...")
        moved_shows = organize_incoming(
            config.base_folder, config.filetypes, config.garbage_words,
            config.special_cases, True, config.file_stability_delay,
            config.file_stability_checks
        )
        for show_dir in sorted(moved_shows):
            retime_show_folder(show_dir, config.filetypes)

    # Connect to mpv
    try:
        ipc = MpvIPC(
            config.socket_path,
            auto_launch=config.auto_launch_mpv,
            auto_restart=config.auto_restart_mpv
        )
    except Exception as e:
        logger.error(f"Failed to connect to mpv: {e}")
        sys.exit(1)

    # Initial playlist sync
    desired = scan_recent(config.base_folder, config.days, config.filetypes, order=config.order)
    reconcile_playlist(ipc, desired, keep_current=not config.prune_playing)
    reorder_playlist(ipc, desired)

    if config.resume:
        apply_resume_if_possible(ipc, desired, config.resume_file, allow_stale=config.resume_allow_stale)

    ensure_playing(ipc)

    # Setup inotify
    ino = Inotify()
    ino.add_watch(str(config.base_folder))
    watched_dirs = {str(config.base_folder)}

    if config.watch_subdirs:
        for d in config.base_folder.iterdir():
            if d.is_dir():
                try:
                    ino.add_watch(str(d))
                    watched_dirs.add(str(d))
                except Exception as e:
                    logger.warning(f"Could not watch {d}: {e}")

    # Main event loop
    next_state = time.monotonic() + max(5, config.state_interval)
    next_health_check = time.monotonic() + config.health_check_interval
    next_metrics_log = time.monotonic() + 300  # Log metrics every 5 minutes

    logger.info("Entering main event loop...")

    try:
        while not should_shutdown():
            now = time.monotonic()

            # Periodic IPC health check
            if now >= next_health_check:
                if not ipc.is_alive():
                    # Check if socket file exists - if not, mpv quit cleanly
                    if not os.path.exists(config.socket_path):
                        if config.auto_restart_mpv:
                            logger.info("mpv quit, auto-restarting...")
                            if ipc.reconnect():
                                resync_playlist(ipc, config)
                            else:
                                logger.error("Failed to restart mpv")
                                break
                        else:
                            logger.info("mpv quit cleanly, exiting...")
                            break
                    else:
                        logger.warning("IPC connection appears dead, reconnecting...")
                        if ipc.reconnect():
                            resync_playlist(ipc, config)
                        else:
                            logger.error("Failed to reconnect")
                            # If mpv quit cleanly, exit instead of retrying
                            if ipc.has_quit_cleanly() and not config.auto_restart_mpv:
                                logger.info("mpv quit cleanly, exiting...")
                                break
                next_health_check = now + config.health_check_interval
                metrics.last_health_check = now

            # Periodic metrics logging
            if now >= next_metrics_log:
                metrics.log_summary()
                next_metrics_log = now + 300

            # Calculate select timeout
            timeout = max(0.0, min(
                next_state - now,
                next_health_check - now,
                next_metrics_log - now
            ))

            # Check socket validity
            if not ipc.sock:
                # Check if socket file exists - if not, mpv quit cleanly
                if not os.path.exists(config.socket_path):
                    if config.auto_restart_mpv:
                        logger.info("mpv quit, auto-restarting...")
                        if ipc.reconnect():
                            resync_playlist(ipc, config)
                        else:
                            time.sleep(1)
                            continue
                    else:
                        logger.info("mpv quit cleanly, exiting...")
                        break
                else:
                    logger.error("Socket is None but file exists, attempting reconnect...")
                    if ipc.reconnect():
                        resync_playlist(ipc, config)
                    else:
                        # If reconnect failed and mpv quit cleanly, exit
                        if ipc.has_quit_cleanly() and not config.auto_restart_mpv:
                            logger.info("mpv quit cleanly, exiting...")
                            break
                        time.sleep(1)
                        continue

            # Wait for events
            r, _, _ = select.select([ipc.sock, ino.fd], [], [], timeout)

            if r:
                # MPV socket readable
                if ipc.sock in r:
                    try:
                        data = ipc.sock.recv(1, socket.MSG_PEEK)
                        if not data:
                            # Socket returned EOF - mpv has quit
                            logger.info("mpv exited (socket EOF)")
                            if config.auto_restart_mpv:
                                logger.info("Auto-restarting mpv...")
                                if ipc.reconnect():
                                    resync_playlist(ipc, config)
                                else:
                                    logger.error("Failed to restart mpv")
                                    break
                            else:
                                logger.info("Shutting down as mpv quit cleanly")
                                break
                    except (BlockingIOError, InterruptedError):
                        pass
                    except OSError:
                        # Socket error - mpv has closed IPC
                        logger.info("mpv IPC closed (OSError)")
                        if config.auto_restart_mpv:
                            logger.info("Auto-restarting mpv...")
                            if ipc.reconnect():
                                resync_playlist(ipc, config)
                            else:
                                logger.error("Failed to restart mpv")
                                break
                        else:
                            logger.info("Shutting down as mpv quit cleanly")
                            break

                # Inotify events
                if ino.fd in r:
                    events = ino.read_events()
                    if events:
                        logger.info(f"Received {len(events)} inotify event(s)")
                        metrics.inotify_events += len(events)

                        moved_shows = set()
                        if config.organize:
                            moved_shows = organize_incoming(
                                config.base_folder, config.filetypes, config.garbage_words,
                                config.special_cases, True, config.file_stability_delay,
                                config.file_stability_checks
                            )
                            for show_dir in sorted(moved_shows):
                                retime_show_folder(show_dir, config.filetypes)

                                # Add watch for new directories
                                if config.watch_subdirs and str(show_dir) not in watched_dirs:
                                    try:
                                        ino.add_watch(str(show_dir))
                                        watched_dirs.add(str(show_dir))
                                        logger.info(f"Added watch for {show_dir}")
                                    except Exception as e:
                                        logger.warning(f"Could not watch {show_dir}: {e}")

                        resync_playlist(ipc, config)

            # Periodic resume save
            now = time.monotonic()
            if now >= next_state and config.resume:
                save_resume_state(ipc, config.resume_file)
                # Check if mpv quit during save operation
                if ipc.has_quit_cleanly() and not config.auto_restart_mpv:
                    logger.info("mpv quit cleanly during operation, exiting...")
                    break
                next_state = now + max(5, config.state_interval)

    except (BrokenPipeError, ConnectionResetError) as e:
        logger.warning(f"Lost mpv IPC connection: {e}")
        # Check if mpv quit cleanly before trying to reconnect
        if ipc.has_quit_cleanly() and not config.auto_restart_mpv:
            logger.info("mpv quit cleanly, exiting...")
            # Exit main loop
        elif ipc.reconnect():
            logger.info("Reconnected successfully, continuing...")
            resync_playlist(ipc, config)
        else:
            logger.error("Could not reconnect")
            # Also exit if can't reconnect and mpv quit cleanly
            if ipc.has_quit_cleanly() and not config.auto_restart_mpv:
                logger.info("mpv not available, exiting...")

    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)

    finally:
        cleanup(ipc, config, ino)


def cleanup(ipc: MpvIPC, config: Config, ino: Optional[Inotify] = None) -> None:
    """
    Clean up resources on exit.

    Args:
        ipc: MPV IPC client
        config: Application configuration
        ino: Optional inotify instance
    """
    logger.info("Cleaning up...")

    # Save final state only if mpv is still connected
    if config.resume and ipc.sock and not ipc.has_quit_cleanly():
        try:
            save_resume_state(ipc, config.resume_file)
        except Exception as e:
            logger.debug(f"Could not save final state: {e}")

    # Log final metrics
    metrics.log_summary()

    # Close connections
    if ino:
        ino.close()

    ipc.close()

    logger.info("Shutdown complete")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
perpetual.py (inotify version)
- Organize incoming TV files (sxanni-style) into Show/ folders (no prompts)
- Maintain an mpv 'recent files' playlist via JSON-IPC (append/prune/reorder)
- Default order = oldest first by mtime
- Exits immediately when mpv quits (press 'q')
- Saves/restores last playing file + position across restarts/quit
- After organizing, per-show S/E retime of mtimes so mtime order matches Season/Episode
- Uses inotify to react to new/updated files
- No periodic polling scan needed
"""

import argparse
import datetime as dt
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import time
import select
import ctypes
import ctypes.util
from pathlib import Path

libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)

IN_ACCESS        = 0x00000001
IN_MODIFY        = 0x00000002
IN_ATTRIB        = 0x00000004
IN_CLOSE_WRITE   = 0x00000008
IN_CLOSE_NOWRITE = 0x00000010
IN_OPEN          = 0x00000020
IN_MOVED_FROM    = 0x00000040
IN_MOVED_TO      = 0x00000080
IN_CREATE        = 0x00000100
IN_DELETE        = 0x00000200
IN_DELETE_SELF   = 0x00000400
IN_MOVE_SELF     = 0x00000800

IN_ONLYDIR       = 0x01000000
IN_DONT_FOLLOW   = 0x02000000
IN_EXCL_UNLINK   = 0x04000000
IN_MASK_ADD      = 0x20000000

WATCH_MASK = (
    IN_CLOSE_WRITE | IN_MOVED_TO | IN_CREATE | IN_DELETE | IN_MOVED_FROM
)

class Inotify:
    def __init__(self):
        self.fd = libc.inotify_init1(0)
        if self.fd < 0:
            e = ctypes.get_errno()
            raise OSError(e, os.strerror(e))
        self.wds = {}

    def add_watch(self, path: str, mask: int = WATCH_MASK):
        bpath = path.encode("utf-8")
        wd = libc.inotify_add_watch(self.fd, ctypes.c_char_p(bpath), ctypes.c_uint32(mask))
        if wd < 0:
            e = ctypes.get_errno()
            raise OSError(e, os.strerror(e))
        self.wds[wd] = path
        return wd

    def read_events(self):
        # read a chunk of events
        try:
            data = os.read(self.fd, 4096)
        except BlockingIOError:
            return []
        events = []
        i = 0
        _sz_event = ctypes.sizeof(ctypes.c_int) * 3 + ctypes.sizeof(ctypes.c_uint32)
        while i < len(data):
            # struct inotify_event {
            #   int wd; uint32_t mask; uint32_t cookie; uint32_t len; char name[];
            # }
            wd = int.from_bytes(data[i:i+4], "little", signed=True)
            mask = int.from_bytes(data[i+4:i+8], "little")
            cookie = int.from_bytes(data[i+8:i+12], "little")
            length = int.from_bytes(data[i+12:i+16], "little")
            name = b""
            if length > 0:
                name = data[i+16:i+16+length].split(b"\0", 1)[0]
            events.append((wd, mask, cookie, name.decode("utf-8")))
            i += 16 + length
        return events

BASE_FOLDER = "/data/videos"
FILETYPES  = ["avi","mkv","mpeg","mp4","m4v","mpg","webm","avif","ts"]

SPECIAL_CASES = {
    "USA","FBI","BBC","US","AU","PL","IE","NZ","FR","DE","JP","UK",
    "QI","XL","SAS","RAF","WWII","WPC","LOL","VI","VII","VIII","VIIII","IX","II","III","IV",
    "DCI","HD","W1A","HBO","100K"
}

def load_garbage_words(filepath: str) -> set[str]:
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return {line.strip().lower() for line in f if line.strip()}
    except FileNotFoundError:
        return set()

GARBAGE_WORDS = load_garbage_words("/data/tvtitle_munge.txt")

def make_path_with_test(path: Path) -> bool:
    try:
        path.mkdir(parents=True, exist_ok=True)
        return True
    except Exception as e:
        print(f"Couldn't create {path}: {e}", file=sys.stderr)
        return False

def clean_filename(filename: str):
    fn = filename.replace('_', '.').replace('-', '.').replace(' ', '.')
    m = re.search(r"(.*?)(s\d{2,3}e\d{2})", fn, re.IGNORECASE)
    if not m:
        return None
    show_raw, season_episode = m.groups()
    title_match = re.search(r"s\d{2,3}e\d{2}\.(.*)", fn, re.IGNORECASE)
    episode_title_raw = title_match.group(1) if title_match else ""
    title_tokens = re.split(r"[._\s]+", episode_title_raw.strip())

    filtered_tokens = []
    filetypes_lower = {e.lower() for e in FILETYPES}
    for tok in title_tokens:
        low = tok.lower()
        if low in GARBAGE_WORDS or low in filetypes_lower:
            break
        if tok:
            filtered_tokens.append(tok)

    episode_title = ""
    if filtered_tokens:
        formatted = [w.upper() if w.upper() in SPECIAL_CASES else w.capitalize()
                     for w in filtered_tokens]
        episode_title = "." + ".".join(formatted).strip(".")

    show_tokens = re.split(r"[._\s]+", show_raw.strip())
    formatted_show = [w.upper() if w.upper() in SPECIAL_CASES else w.capitalize()
                      for w in show_tokens if w]
    show_name = ".".join(formatted_show).strip(".")

    extension = filename.split(".")[-1].lower()
    folder = show_name.strip()

    if season_episode.upper() != "S00E00":
        clean = f"{folder}.{season_episode.upper()}{episode_title}.{extension}"
    else:
        clean = f"{folder}{episode_title}.{extension}"
    return folder, clean

def dynamic_episode_pattern(filetypes: list[str]) -> re.Pattern:
    exts = "|".join(re.escape(e) for e in sorted({e.lower() for e in filetypes}))
    return re.compile(rf'.*(\_|\.)s\d{{1,3}}e\d{{1,2}}.*\.({exts})$', re.IGNORECASE)

def safe_move_auto(source: str, target: str, folder: str):
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

def organize_incoming(base_folder: Path, filetypes: list[str]):
    moved_shows: set[Path] = set()
    pattern = dynamic_episode_pattern(filetypes)
    files = [str(f) for f in base_folder.glob('*')
             if f.is_file() and f.stat().st_size > 0 and pattern.match(f.name)]
    for file_ in files:
        leaf = os.path.basename(file_)
        result = clean_filename(leaf)
        if not result:
            continue
        folder, newname = result
        tvshow_path = Path(base_folder) / folder
        target = tvshow_path / newname
        if os.path.abspath(file_) != os.path.abspath(target):
            try:
                safe_move_auto(file_, str(target), str(tvshow_path))
                moved_shows.add(tvshow_path)
                print(f"[organize] {file_} -> {target}")
            except Exception as e:
                print(f"[organize][warn] {file_}: {e}", file=sys.stderr)
    return moved_shows

SE_RE = re.compile(r"\.s(\d{2,3})e(\d{2})\.", re.IGNORECASE)

def parse_se(path: Path):
    m = SE_RE.search(path.name + ".")
    if not m:
        return None
    try:
        s = int(m.group(1)); e = int(m.group(2))
        return (s, e)
    except ValueError:
        return None

def is_allowed_ext(path: Path) -> bool:
    return path.suffix.lower().lstrip('.') in {e.lower() for e in FILETYPES}

def retime_show_folder(show_dir: Path):
    if not show_dir.exists() or not show_dir.is_dir():
        return
    candidates: list[Path] = [
        p for p in show_dir.glob("*") if p.is_file() and is_allowed_ext(p) and parse_se(p)
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
            print(f"[retime][warn] {p}: {e}", file=sys.stderr)
    print(f"[retime] {show_dir} -> {len(candidates)} files S/E-ordered via mtime")

def jnow(): return dt.datetime.now(dt.timezone.utc)

def file_is_recent(p: Path, days: float) -> bool:
    try:
        m = dt.datetime.fromtimestamp(p.stat().st_mtime, dt.timezone.utc)
    except FileNotFoundError:
        return False
    return (jnow() - m) <= dt.timedelta(days=days)

def scan_recent(dirpath: Path, days: float, exts, order: str = "oldest"):
    print(f"[info] scan_recent: {dirpath}")
    exts = {e.lower().lstrip('.') for e in exts}
    files = []
    for root, _, fnames in os.walk(dirpath):
        for f in fnames:
            if exts and Path(f).suffix.lower().lstrip('.') not in exts:
                continue
            p = Path(root) / f
            if file_is_recent(p, days):
                files.append(p.resolve())
    rev = (order == "newest")
    files.sort(key=lambda p: p.stat().st_mtime, reverse=rev)
    return files

class MpvIPC:
    def __init__(self, addr, auto_launch=False, launch_cmd=None, timeout=20):
        self.addr = addr
        self.sock = None
        self.auto_launch = auto_launch
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
        self._connect()

    def _connect(self):
        if self.auto_launch:
            if os.path.exists(self.addr):
                try: os.unlink(self.addr)
                except Exception: pass
            subprocess.Popen(self.launch_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        start = time.time()
        last_err = None
        while time.time() - start < self.timeout:
            try:
                s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                s.settimeout(10.0)
                s.connect(self.addr)
                self.sock = s
                self.sock.settimeout(10.0)
                return
            except Exception as e:
                last_err = e
                time.sleep(0.5)
        raise ConnectionRefusedError(f"Could not connect to mpv at {self.addr}: {last_err}")

    def _send(self, payload):
        line = json.dumps(payload, separators=(',', ':')) + "\n"
        self.sock.sendall(line.encode("utf-8"))

    def send(self, *command):
        self._send({"command": list(command)})

    def _recv(self):
        data = b''
        while True:
            chunk = self.sock.recv(65536)
            if not chunk: break
            data += chunk
            if b"\n" in chunk: break
        if not data: return None
        try:
            obj = json.loads(data.splitlines()[-1].decode("utf-8"))
            return obj if isinstance(obj, dict) else None
        except Exception:
            return None

    def get_property(self, prop):
        self._send({"command": ["get_property", prop]})
        resp = self._recv()
        if not resp or resp.get("error") != "success":
            return None
        return resp.get("data")

def reconcile_playlist(ipc: MpvIPC, desired_paths: list[Path], keep_current=True):
    playlist = ipc.get_property("playlist") or []
    idx_by_path = {}
    for i, it in enumerate(playlist):
        if not isinstance(it, dict):
            continue
        fn = it.get("filename")
        if fn:
            try: p = str(Path(fn).resolve())
            except Exception: p = fn
            idx_by_path[p] = i
    cur_pos = ipc.get_property("playlist-current-pos") or -1

    desired_set = {str(p) for p in desired_paths}
    current_set = set(idx_by_path.keys())

    to_add = [str(p) for p in desired_paths if str(p) not in current_set]
    to_remove_paths = current_set - desired_set

    remove_indices = []
    for path in to_remove_paths:
        idx = idx_by_path.get(path)
        if idx is None: continue
        if keep_current and idx == cur_pos: continue
        remove_indices.append(idx)
    for idx in sorted(remove_indices, reverse=True):
        ipc.send("playlist-remove", idx)

    for p in to_add:
        ipc.send("loadfile", p, "append-play")

def reorder_playlist(ipc: MpvIPC, desired_paths: list[Path]):
    desired = [str(p) for p in desired_paths]
    pl = ipc.get_property("playlist") or []
    print(f"[debug] type pl is '{type(pl)}'")
    current = []
    if type(pl) is int:
        return
    for it in pl:
        if not isinstance(it, dict):
            continue
        fn = it.get("filename")
        if not fn:
            continue
        try: current.append(str(Path(fn).resolve()))
        except Exception: current.append(fn)
    index_of = {path: i for i, path in enumerate(current)}

    print(f"[debug] type desired is '{type(desired)}'")
    for i, want in enumerate(desired):
        if i < len(current) and current[i] == want:
            continue
        j = index_of.get(want)
        if j is None:
            ipc.send("loadfile", want, "append-play")
            current.append(want)
            index_of[want] = len(current)-1
            j = index_of[want]
        if j != i:
            ipc.send("playlist-move", j, i)
            item = current.pop(j)
            current.insert(i, item)
            start = min(i, j)
            for k in range(start, len(current)):
                index_of[current[k]] = k

def ensure_playing(ipc: MpvIPC):
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

# resume state (unchanged)
def default_resume_path() -> Path:
    return Path.home() / ".local" / "state" / "perpetual_resume.json"

def save_resume_state(ipc: MpvIPC, resume_path: Path):
    state_dir = resume_path.parent
    state_dir.mkdir(parents=True, exist_ok=True)
    path = ipc.get_property("path")
    pos  = ipc.get_property("time-pos")
    if path and isinstance(pos, (int, float)):
        try:
            with open(resume_path, "w", encoding="utf-8") as f:
                json.dump({"path": path, "time": float(pos)}, f)
        except Exception as e:
            print(f"[warn] save_resume_state: {e}", file=sys.stderr)

def load_resume_state(resume_path: Path):
    try:
        with open(resume_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        path = data.get("path")
        t    = data.get("time")
        if path and isinstance(t, (int, float)):
            return (path, float(t))
    except FileNotFoundError:
        pass
    except Exception as e:
        print(f"[warn] load_resume_state: {e}", file=sys.stderr)
    return (None, None)

def apply_resume_if_possible(ipc: MpvIPC, desired_paths: list[Path], resume_path: Path, allow_stale=False):
    saved_path, saved_time = load_resume_state(resume_path)
    if not saved_path:
        return
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

def main():
    ap = argparse.ArgumentParser(description="Organize and keep mpv playlist synced to recent files (with resume + inotify).")
    ap = argparse.ArgumentParser(description="Organize and keep mpv playlist synced to recent files (with resume).")
    ap.add_argument("--base", default=BASE_FOLDER, help="Base folder (default: /data/videos)")
    ap.add_argument("--days", type=float, default=3.0, help="Recent window in days (default: 3)")
    ap.add_argument("--ext", nargs="*", default=FILETYPES, help="Allowed extensions")
    ap.add_argument("--socket", default="/tmp/mpv.sock", help="mpv IPC path")
    ap.add_argument("--state-interval", type=int, default=10, help="Resume-state save interval seconds (default: 60)")
    ap.add_argument("--order", choices=["newest","oldest"], default="oldest", help="Sort by mtime (default: oldest)")
    ap.add_argument("--prune-playing", action="store_true", help="Allow pruning currently playing item")
    ap.add_argument("--no-organize", action="store_true", help="Skip organize step")
    ap.add_argument("--garbage-words", default="/data/tvtitle_munge.txt", help="Garbage words file")
    ap.add_argument("--auto-launch", action="store_true", help="Launch mpv if IPC not present")
    ap.add_argument("--resume", action="store_true", default=True, help="Resume last file/time if possible (default: on)")
    ap.add_argument("--no-resume", action="store_false", dest="resume", help="Disable resume")
    ap.add_argument("--resume-file", default=str(default_resume_path()), help="Path to resume state JSON")
    ap.add_argument("--resume-allow-stale", action="store_true", help="Resume even if file is outside the recent window")
    ap.add_argument("--watch-subdirs", action="store_true", help="Also watch 1-level subdirs under base for changes")
    args = ap.parse_args()

    global GARBAGE_WORDS
    GARBAGE_WORDS = load_garbage_words(args.garbage_words)

    base_folder = Path(args.base).expanduser().resolve()
    if not base_folder.exists():
        print(f"Base folder not found: {base_folder}", file=sys.stderr)
        sys.exit(1)

    # initial organize + retime
    if not args.no_organize:
        moved_shows = organize_incoming(base_folder, args.ext)
        for show_dir in sorted(moved_shows):
            retime_show_folder(show_dir)

    ipc = MpvIPC(args.socket, auto_launch=args.auto_launch)
    desired = scan_recent(base_folder, args.days, args.ext, order=args.order)
    reconcile_playlist(ipc, desired, keep_current=not args.prune_playing)
    reorder_playlist(ipc, desired)
    if args.resume:
        apply_resume_if_possible(ipc, desired, Path(args.resume_file), allow_stale=args.resume_allow_stale)
    ensure_playing(ipc)

    # inotify
    ino = Inotify()
    ino.add_watch(str(base_folder))
    if args.watch_subdirs:
        for d in base_folder.iterdir():
            if d.is_dir():
                ino.add_watch(str(d))

    # timers for resume-save only
    next_state = time.monotonic() + max(5, args.state_interval)

    # main loop: wait on mpv.sock and inotify fd
    while True:
        try:
            now = time.monotonic()
            timeout = max(0.0, next_state - now)
            r, _, _ = select.select([ipc.sock, ino.fd], [], [], timeout)

            if r:
                if ipc.sock in r:
                    try:
                        data = ipc.sock.recv(65536)
                        if not data:
                            print("[info] mpv exited — stopping daemon.")
                            break
                    except Exception:
                        print("[info] mpv IPC closed — stopping daemon.")
                        break

                if ino.fd in r:
                    events = ino.read_events()
                    # we got filesystem changes -> re-run pipeline
                    if events:
                        print(f"[inotify] {len(events)} event(s)")
                        moved_shows = set()
                        if not args.no_organize:
                            moved_shows = organize_incoming(base_folder, args.ext)
                            for show_dir in sorted(moved_shows):
                                retime_show_folder(show_dir)

                        desired = scan_recent(base_folder, args.days, args.ext, order=args.order)
                        reconcile_playlist(ipc, desired, keep_current=not args.prune_playing)
                        reorder_playlist(ipc, desired)
                        ensure_playing(ipc)

            # resume save
            now = time.monotonic()
            if now >= next_state and args.resume:
                save_resume_state(ipc, Path(args.resume_file))
                next_state = now + max(5, args.state_interval)

        except KeyboardInterrupt:
            break
        except (BrokenPipeError, ConnectionResetError):
            print("[info] Lost mpv IPC connection — stopping.", file=sys.stderr)
            break
        except Exception as e:
            print(f"[warn] {e}", file=sys.stderr)

if __name__ == "__main__":
    main()

# Perpetual - Advanced MPV Playlist Manager

A robust, production-ready mpv playlist manager with automatic file organization, IPC reconnection, and comprehensive monitoring capabilities.

## Features

### Core Functionality
- **Automatic File Organization**: Organizes TV show files into structured folders (Show/Season format)
- **Intelligent Playlist Management**: Maintains a rolling playlist of recent files via mpv's JSON-IPC
- **Resume Support**: Saves and restores playback position across restarts
- **Season/Episode Ordering**: Automatically adjusts file modification times to maintain correct playback order

### Reliability & Monitoring
- **Automatic IPC Reconnection**: Handles mpv crashes/restarts gracefully with automatic reconnection
- **Health Monitoring**: Periodic health checks with automatic recovery
- **File Stability Detection**: Waits for files to finish being written before processing
- **Metrics Tracking**: Comprehensive operational metrics (reconnections, syncs, events, uptime)
- **Graceful Shutdown**: Proper signal handling (SIGINT, SIGTERM) with state preservation

### Advanced Features
- **Auto-restart MPV**: Optionally restart mpv automatically if it quits
- **Configuration Files**: TOML configuration file support with command-line override
- **Structured Logging**: Professional logging with configurable levels and optional file output
- **Inotify Monitoring**: Real-time file system monitoring (no polling)
- **Dynamic Watch Management**: Automatically monitors newly created directories

## Installation

### Requirements
- Python 3.11+ (for TOML support) or Python 3.8+ with `tomli` package
- mpv media player
- Linux (uses inotify)
- Optional: `mkvpropedit` for MKV metadata updates

### Setup

```bash
# Clone or download the script
cd perpetual

# Make executable
chmod +x perpetual.py

# Optional: Install TOML support for older Python versions
pip install tomli  # Only needed for Python < 3.11
```

## Usage

### Basic Usage

```bash
# Basic usage with default settings
./perpetual.py

# Specify custom base folder
./perpetual.py --base /path/to/videos

# Auto-launch mpv if not running
./perpetual.py --auto-launch

# Auto-restart mpv if it quits
./perpetual.py --auto-restart-mpv

# Enable debug logging
./perpetual.py --log-level DEBUG

# Use configuration file
./perpetual.py --config perpetual.toml
```

### Configuration File

Create `perpetual.toml`:

```toml
base_folder = "/data/videos"
days = 3.0
auto_restart_mpv = true
log_level = "INFO"
log_file = "/var/log/perpetual.log"
```

See `perpetual.toml.example` for all available options.

### Command-Line Options

#### Core Settings
- `--config PATH`: TOML configuration file
- `--base PATH`: Base folder containing videos (default: /data/videos)
- `--days FLOAT`: Recent files window in days (default: 3.0)
- `--ext EXT [EXT ...]`: Allowed file extensions
- `--socket PATH`: mpv IPC socket path (default: /tmp/mpv.sock)
- `--order {newest,oldest}`: Sort order by mtime (default: oldest)

#### State Management
- `--state-interval SECONDS`: Resume state save interval (default: 10)
- `--resume` / `--no-resume`: Enable/disable resume (default: enabled)
- `--resume-file PATH`: Custom resume state file location
- `--resume-allow-stale`: Resume files outside recent window

#### Organization
- `--no-organize`: Skip automatic file organization
- `--garbage-words PATH`: Garbage words file (default: /data/tvtitle_munge.txt)
- `--prune-playing`: Allow removing currently playing file from playlist

#### Monitoring
- `--watch-subdirs`: Monitor subdirectories for changes
- `--health-check-interval SECONDS`: IPC health check interval (default: 30)

#### MPV Control
- `--auto-launch`: Launch mpv if IPC not present
- `--auto-restart-mpv`: Restart mpv if it quits unexpectedly

#### File Stability
- `--file-stability-delay SECONDS`: Delay between stability checks (default: 1.0)
- `--file-stability-checks INT`: Number of checks before processing (default: 2)

#### Logging
- `--log-level {DEBUG,INFO,WARNING,ERROR,CRITICAL}`: Logging level (default: INFO)
- `--log-file PATH`: Optional log file path

## How It Works

### File Organization

1. **Detection**: Monitors base folder for TV show files matching pattern: `*s##e##*.<ext>`
2. **Parsing**: Extracts show name, season, episode from filename
3. **Cleaning**: Removes garbage words, formats titles properly
4. **Organization**: Moves files into `Show.Name/Show.Name.S##E##.Title.ext` structure
5. **Metadata**: Updates MKV titles, adjusts modification times for proper ordering

### Playlist Management

1. **Scanning**: Finds all recent files (within specified day window)
2. **Reconciliation**: Adds new files, removes old files from mpv playlist
3. **Reordering**: Ensures playlist matches desired order (oldest/newest first)
4. **Resume**: Restores last playing position on startup

### IPC Reliability

1. **Health Checks**: Periodic socket health verification using MSG_PEEK
2. **Auto-Reconnection**: Automatic reconnection on timeout/failure (up to 2 retries)
3. **Fast Failure**: 2-second operation timeout for quick failure detection
4. **State Preservation**: Saves resume state before any shutdown

### Monitoring

Metrics tracked:
- IPC reconnections and failures
- Playlist synchronizations
- Files organized
- Inotify events
- Uptime
- Last health check timestamp

Logged every 5 minutes during operation.

## Running as a Service

### Systemd Service

Create `/etc/systemd/system/perpetual.service`:

```ini
[Unit]
Description=Perpetual MPV Playlist Manager
After=network.target

[Service]
Type=simple
User=your-user
Group=your-group
WorkingDirectory=/home/your-user/perpetual
ExecStart=/home/your-user/perpetual/perpetual.py --config /etc/perpetual/perpetual.toml
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
```

Enable and start:

```bash
sudo systemctl daemon-reload
sudo systemctl enable perpetual.service
sudo systemctl start perpetual.service

# View logs
sudo journalctl -u perpetual.service -f
```

## Troubleshooting

### IPC Connection Issues

**Problem**: "Could not connect to mpv"

**Solutions**:
- Ensure mpv is running: `pgrep mpv`
- Check socket exists: `ls -la /tmp/mpv.sock`
- Use `--auto-launch` to start mpv automatically
- Verify socket path matches mpv's `--input-ipc-server` setting

### File Organization Not Working

**Problem**: Files not being organized automatically

**Solutions**:
- Check file naming matches pattern: `*s##e##*.<ext>`
- Verify inotify events are being received (use `--log-level DEBUG`)
- Ensure base folder has proper permissions
- Check garbage words file exists and is readable

### High CPU Usage

**Problem**: Excessive CPU usage

**Solutions**:
- Increase `--health-check-interval` (default: 30s)
- Increase `--state-interval` (default: 10s)
- Use `--log-level WARNING` instead of INFO/DEBUG
- Avoid `--watch-subdirs` if not needed

### Resume Not Working

**Problem**: Playback position not restored

**Solutions**:
- Check resume file exists: `~/.local/state/perpetual_resume.json`
- Verify resume file is writable
- Use `--resume-allow-stale` if file is old
- Check logs for resume errors

## Architecture

### Event Loop

```
┌─────────────────────────────────────────┐
│         Main Event Loop                 │
│  ┌───────────────────────────────────┐ │
│  │ select() on:                      │ │
│  │ - mpv IPC socket                  │ │
│  │ - inotify fd                      │ │
│  │ - timeout (min of all timers)     │ │
│  └───────────────────────────────────┘ │
│                                         │
│  Periodic Tasks:                        │
│  ├─ Health Check (30s)                  │
│  ├─ Resume Save (10s)                   │
│  └─ Metrics Log (300s)                  │
└─────────────────────────────────────────┘
```

### Data Flow

```
File System → inotify → organize_incoming() → scan_recent()
    ↓
reconcile_playlist() → reorder_playlist() → ensure_playing()
    ↓
MPV IPC ← automatic reconnection on failure
```

## Performance

- **Startup**: < 1 second (unless initial organization needed)
- **IPC Operations**: 2-second timeout per operation
- **Reconnection**: ~3-5 seconds (including retries)
- **Memory**: ~20-30 MB typical
- **CPU**: < 1% idle, 2-5% during file processing

## Changelog

### Recent Updates

- ✅ Automatic IPC reconnection on timeout/failure
- ✅ Periodic health checks with automatic recovery
- ✅ File stability detection (prevents processing partial files)
- ✅ Comprehensive metrics tracking
- ✅ Signal handling for graceful shutdown (SIGINT/SIGTERM)
- ✅ Auto-restart mpv option
- ✅ TOML configuration file support
- ✅ Structured logging with file output option
- ✅ Dynamic inotify watch management
- ✅ Full type hints for better IDE support
- ✅ Comprehensive docstrings
- ✅ Inotify cleanup on exit
- ✅ Improved error handling and resilience

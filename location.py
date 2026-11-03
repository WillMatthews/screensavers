#!/usr/bin/env python3
"""Opt-in IP geolocation for lock backgrounds; never overwrites manual location."""
import argparse
import contextlib
import fcntl
import json
import math
import os
from pathlib import Path
import tempfile
import time
import urllib.request


def coordinates(lat, lon):
    if isinstance(lat, bool) or isinstance(lon, bool):
        raise ValueError("invalid coordinates")
    lat, lon = float(lat), float(lon)
    if not (math.isfinite(lat) and math.isfinite(lon)
            and -90 <= lat <= 90 and -180 <= lon <= 180):
        raise ValueError("invalid coordinates")
    return lat, lon


def atomic_write(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            stream.write(content)
        os.replace(name, path)
    finally:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(name)


def resolve(config, cache):
    # Disabled means no network access and no cached automatic coordinates.
    try:
        enabled = (config / "location-mode").read_text().strip() == "auto"
    except OSError:
        enabled = False
    if not enabled:
        return None
    cache.mkdir(parents=True, exist_ok=True)
    with (cache / "location.lock").open("a") as lock:
        # Other outputs can use their manual fallback while one fetch is running.
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return None
        now = time.time()
        cached = {}
        try:
            cached = json.loads((cache / "location.json").read_text())
            if not isinstance(cached, dict):
                raise ValueError("invalid cache")
            age = now - float(cached["checked"])
            if 0 <= age < (3600 if cached.get("coords") else 300):
                return coordinates(*cached["coords"]) if cached.get("coords") else None
        except (OSError, ValueError, TypeError, KeyError):
            pass
        result = None
        try:
            request = urllib.request.Request(
                "https://ipapi.co/json/", headers={"User-Agent": "screensaver-location/1.0"})
            with urllib.request.urlopen(request, timeout=3) as response:
                data = json.loads(response.read(65536))
            result = coordinates(data["latitude"], data["longitude"])
        except (OSError, ValueError, TypeError, KeyError):
            # Don't reuse a stale city after travelling; savers read manual fallback.
            pass
        atomic_write(cache / "location.json", json.dumps({"checked": now, "coords": result}))
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("auto", "manual", "status", "resolve"))
    action = parser.parse_args().action
    config = Path(os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config") / "wallpapers"
    cache = Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache") / "wallpapers"
    if action in ("auto", "manual"):
        atomic_write(config / "location-mode", action + "\n")
        print("Automatic approximate location enabled (ipapi.co; public IP is shared)."
              if action == "auto" else "Manual location enabled; automatic lookups disabled.")
    elif action == "status":
        try:
            mode = (config / "location-mode").read_text().strip()
        except OSError:
            mode = "manual"
        print("auto" if mode == "auto" else "manual")
    else:
        try:
            result = resolve(config, cache)
        except (OSError, ValueError, TypeError):
            result = None
        if result is not None:
            print(*result)


if __name__ == "__main__":
    main()

"""Provenance helpers: git SHA, file hashing, config hashing — attached to every canonical result."""
import hashlib
import json
import subprocess
from pathlib import Path


def git_sha(short: bool = False) -> str:
    try:
        args = ["git", "rev-parse", "--short", "HEAD"] if short else ["git", "rev-parse", "HEAD"]
        return subprocess.check_output(args, cwd=Path(__file__).resolve().parent.parent,
                                        stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def hash_array(arr) -> str:
    import numpy as np
    a = np.ascontiguousarray(arr)
    return hashlib.sha256(a.tobytes()).hexdigest()[:16]


def hash_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def hash_config(cfg: dict) -> str:
    blob = json.dumps(cfg, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def write_manifest(path, record: dict):
    record = dict(record)
    record["git_sha"] = git_sha()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(record, f, indent=2, default=str)

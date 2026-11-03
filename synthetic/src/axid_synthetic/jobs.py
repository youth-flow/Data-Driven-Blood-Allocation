"""Resumable local jobs: completed artifacts are verified; interrupted attempts are preserved."""

from pathlib import Path
import hashlib
import json
import re
from dat_study.io import save_json, read_json, sha256


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def source_fingerprint(root):
    roots = [
        "src",
        "DAT/src",
        "DAT/configs",
        "ETO",
        "QR",
        "SAA",
        "synthetic/src",
        "synthetic/configs",
        "configs",
        "scripts",
    ]
    h = hashlib.sha256()
    for folder in roots:
        for path in sorted((root / folder).rglob("*")):
            if path.is_file() and path.suffix in {".py", ".yaml"} and "__pycache__" not in path.parts:
                h.update(str(path.relative_to(root)).encode())
                h.update(path.read_bytes())
    return h.hexdigest()


def run_job(root, name, signature, action, resume=False):
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", name):
        raise ValueError("Job name must be a simple identifier")
    folder = Path(root) / name
    complete = folder / "complete.json"
    if folder.exists() and not resume:
        raise ValueError(f"Job exists: {name}; use --resume")
    if complete.exists():
        record = read_json(complete)
        if record["signature"] != digest(signature):
            raise ValueError(f"Completed job has a different recipe: {name}")
        for relative, expected in record["artifacts"].items():
            path = (folder / relative).resolve()
            if not path.is_relative_to(folder.resolve()) or sha256(path) != expected:
                raise ValueError(f"Completed job integrity mismatch: {name}")
        return read_json(folder / record["result"])
    folder.mkdir(parents=True, exist_ok=True)
    attempts = list(folder.glob("attempt_*"))
    attempt = folder / f"attempt_{len(attempts):04d}"
    attempt.mkdir()
    save_json(attempt / "signature.json", signature)
    result = action(attempt)
    save_json(attempt / "result.json", result)
    files = {str(p.relative_to(folder)): sha256(p) for p in attempt.rglob("*") if p.is_file()}
    save_json(
        complete,
        {
            "complete": True,
            "signature": digest(signature),
            "result": str((attempt / "result.json").relative_to(folder)),
            "artifacts": files,
        },
    )
    print(json.dumps({"completed_job": name}, ensure_ascii=False), flush=True)
    return result

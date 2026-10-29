"""Fail closed on private artifacts in the index and the entire history being pushed.

Only source/configuration/documentation paths are eligible. Error messages never
include matched credential values or file contents. Uses only the standard library.
"""

import argparse
from pathlib import PurePosixPath
import re
import subprocess
import sys

ROOT_FILES = {"README.md", "CONTRIBUTING.md", "pyproject.toml", ".gitignore", ".gitattributes"}
PREFIXES = ("src/", "tests/", "configs/", "docs/", ".github/workflows/")
MODEL_DIRS = ("DAT", "SAA", "wSAA", "Flat-ETO", "Set-ETO")
PREFIXES += tuple(f"{model}/{part}/" for model in MODEL_DIRS for part in ("src", "tests", "configs", "scripts"))
ROOT_FILES |= {f"{model}/README.md" for model in MODEL_DIRS}
ROOT_FILES |= {"data/数据预处理.py", ".githooks/pre-commit", ".githooks/pre-push"}
SCRIPTS = {"scripts/check_git_privacy.py", "scripts/run_experiments.py"}
SOURCE_EXTENSIONS = {".py", ".md", ".yaml", ".yml", ".toml", ".sh"}
PRIVATE_PARTS = {"__pycache__", "outputs", "results", "artifacts", "checkpoints", ".git"}
SECRET_PATTERN = re.compile(
    rb"gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,}|sk-[A-Za-z0-9_-]{35,}"
    rb"|-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----"
)


def allowed(path):
    p = PurePosixPath(path)
    if p.is_absolute() or ".." in p.parts or "\\" in path:
        return False
    if any(part.startswith((".env", ".private")) or part in PRIVATE_PARTS for part in p.parts):
        return False
    if path in ROOT_FILES or path in SCRIPTS:
        return True
    return path.startswith(PREFIXES) and p.suffix.lower() in SOURCE_EXTENSIONS


def git(*args):
    return subprocess.check_output(["git", *args], stderr=subprocess.PIPE)


def content_problems(label, value):
    problems = []
    if len(value) > 2_000_000 or b"\0" in value:
        problems.append(f"Oversized or binary content: {label}")
    try:
        value.decode("utf-8")
    except UnicodeDecodeError:
        problems.append(f"Non-UTF-8 source content: {label}")
    if SECRET_PATTERN.search(value):
        problems.append(f"Possible credential/private key: {label}")
    return problems


def check_entries(entries, seen_blobs=None):
    problems = []
    checked = {} if seen_blobs is None else seen_blobs
    for mode, oid, path in entries:
        if not allowed(path):
            problems.append(f"Private or unexpected path: {path}")
            continue
        if mode not in ("100644", "100755"):
            problems.append(f"Symlink, submodule or unsupported mode: {path}")
            continue
        if oid not in checked:
            checked[oid] = content_problems(path, git("cat-file", "blob", oid))
        problems.extend(checked[oid])
    return problems


def check(all_tracked=False):
    selected = None
    if not all_tracked:
        selected = set(git("diff", "--cached", "--name-only", "--diff-filter=ACMR", "-z").decode("utf-8").split("\0"))
    entries, problems = [], []
    for record in git("ls-files", "--stage", "-z").split(b"\0"):
        if not record:
            continue
        metadata, path = record.split(b"\t", 1)
        mode, oid, stage = metadata.decode("ascii").split()
        name = path.decode("utf-8")
        if stage != "0":
            problems.append("Unmerged index entry")
        elif selected is None or name in selected:
            entries.append((mode, oid, name))
    return problems + check_entries(entries)


def check_history(revisions):
    commits = git("rev-list", *revisions).decode("ascii").splitlines()
    checked, problems = {}, []
    for commit in commits:
        problems.extend(content_problems(f"commit {commit[:12]}", git("cat-file", "commit", commit)))
        entries = []
        for record in git("ls-tree", "-r", "-z", commit).split(b"\0"):
            if not record:
                continue
            metadata, path = record.split(b"\t", 1)
            mode, _, oid = metadata.decode("ascii").split()
            entries.append((mode, oid, path.decode("utf-8")))
        problems.extend(check_entries(entries, checked))
    return list(dict.fromkeys(problems))


def check_push(lines):
    tips, problems = [], []
    for line in lines:
        parts = line.split()
        if len(parts) != 4:
            problems.append("Invalid pre-push input")
            continue
        local_ref, local_oid, _, _ = parts
        if not re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", local_oid):
            problems.append("Invalid local object ID")
        elif set(local_oid) == {"0"}:
            continue  # A deletion sends no source objects.
        elif not local_ref.startswith(("refs/heads/", "refs/tags/")):
            problems.append("Only explicitly audited branches or tags may be pushed")
        else:
            tips.append(local_oid)
    if tips:
        problems.extend(check_history(sorted(set(tips))))
    return problems


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--all-tracked", action="store_true")
    mode.add_argument("--history", nargs="+", metavar="REVISION")
    mode.add_argument("--pre-push", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.pre_push:
            errors = check_push(sys.stdin)
        elif args.history:
            errors = check_history(args.history)
        else:
            errors = check(args.all_tracked)
    except (subprocess.CalledProcessError, UnicodeError, ValueError, OSError):
        print("Git privacy check could not complete; commit/push blocked.", file=sys.stderr)
        return 1
    for error in dict.fromkeys(errors):
        print(error, file=sys.stderr)
    if errors:
        return 1
    print("Git privacy boundary passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

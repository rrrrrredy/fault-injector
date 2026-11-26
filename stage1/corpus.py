"""Inspect pinned Git sources, group their families, and validate blind ticket text."""

import ast
import hashlib
import io
import itertools
import json
from pathlib import Path
import re
import subprocess
import tokenize
import tomllib


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()


def sha(data):
    return hashlib.sha256(data).hexdigest()


def git(repo, *args):
    return subprocess.check_output(["git", "-C", str(repo), *args])


def slug(row):
    return row["repository"].removeprefix("https://github.com/").removesuffix(".git")


def get_repo(cache, row):
    repo = cache / (slug(row).replace("/", "--") + ".git")
    revision = row.get("reference", {}).get("merged_sha", row["commit_sha"])
    if not repo.exists():
        subprocess.run(["git", "init", "--bare", "--quiet", str(repo)], check=True)
    if subprocess.run(["git", "-C", str(repo), "cat-file", "-e", revision + "^{commit}"],
                      capture_output=True).returncode:
        git(repo, "-c", "http.version=HTTP/1.1", "fetch", "--quiet", "--no-tags", row["repository"], revision)
    if git(repo, "rev-parse", "--is-shallow-repository").strip() != b"false":
        raise ValueError(f"Full history required for {slug(row)}")
    git(repo, "merge-base", "--is-ancestor", row["commit_sha"], revision)
    return repo


def blob(repo, revision, path):
    return git(repo, "cat-file", "blob", f"{revision}:{path}")


def package_name(meta, data):
    if meta.endswith(".toml"):
        name = tomllib.loads(data.decode())["project"]["name"]
    else:
        tree = ast.parse(data)
        name, = [kw.value.value for call in ast.walk(tree)
                 if isinstance(call, ast.Call) and isinstance(call.func, (ast.Name, ast.Attribute))
                 and getattr(call.func, "id", getattr(call.func, "attr", "")) == "setup"
                 for kw in call.keywords if kw.arg == "name" and isinstance(kw.value, ast.Constant)]
    return re.sub(r"[-_.]+", "-", name.lower())


def shingles(data):
    tokens = []
    for token in tokenize.tokenize(io.BytesIO(data).readline):
        if token.type in (tokenize.NAME, tokenize.NUMBER, tokenize.STRING, tokenize.OP):
            tokens.append(token.string if token.type in (tokenize.NAME, tokenize.OP) else tokenize.tok_name[token.type])
    return set(zip(*(tokens[i:] for i in range(7))))


def inspect_source(repo, row):
    revision = row["commit_sha"]
    roots = git(repo, "rev-list", "--max-parents=0", revision).decode().splitlines()
    listed = git(repo, "ls-tree", "-r", "-z", "--name-only", revision, "--", row["core"]).decode().split("\0")
    paths = sorted(p for p in listed if p.endswith(".py") and not {"test", "tests"} & set(Path(p).parts))
    if not paths or not roots:
        raise ValueError(f"No source or root history for {slug(row)}")
    prints = set()
    hashes = {}
    for path in paths:
        data = blob(repo, revision, path)
        hashes[path] = sha(data)
        prints.update(shingles(data))
    meta = blob(repo, revision, row["metadata"])
    notice = blob(repo, revision, row["notice"])
    normalized = b" ".join(notice.lower().split())
    if row["spdx"] != "MIT" or not all(term in normalized for term in
            (b"copyright", b"permission is hereby granted", b"without warranty", b"copies")):
        raise ValueError(f"Review the original license at {slug(row)}/{row['notice']}")
    return dict(roots=sorted(roots), core_sha256=sha(json_bytes(hashes)), core_file_count=len(paths),
                package=package_name(row["metadata"], meta), metadata_sha256=sha(meta),
                notice_sha256=sha(notice)), prints


def assess_pair(left, right, a, b):
    intersect = len(a & b)
    union = len(a | b)
    jaccard = intersect / union if union else 0
    smaller = min(len(a), len(b))
    containment = intersect / smaller if smaller else 0
    reasons = []
    if set(left["roots"]) & set(right["roots"]):
        reasons.append("shared_git_root")
    if left["core_sha256"] == right["core_sha256"]:
        reasons.append("identical_core_tree")
    same_package = left["package"] == right["package"]
    if smaller >= 32 and ((containment >= .82 and jaccard >= .60)
                          or (same_package and containment >= .65 and jaccard >= .40)):
        reasons.append("near_duplicate")
    return reasons, dict(shared_shingles=intersect, jaccard=round(jaccard, 4),
                         containment=round(containment, 4), same_package=same_package)


def group_families(rows, facts, prints):
    names = sorted(facts)
    groups = [{name} for name in names]
    checks = []
    for a, b in itertools.combinations(names, 2):
        reasons, scores = assess_pair(facts[a], facts[b], prints[a], prints[b])
        checks.append(dict(left=a, right=b, reasons=reasons, **scores))
        if reasons:
            left = next(group for group in groups if a in group)
            right = next(group for group in groups if b in group)
            if left is not right:
                left.update(right)
                groups.remove(right)
    assignments = {}
    splits = {s: set() for s in ("train", "eval_unseen", "human_regressions")}
    for group in groups:
        eligible = [r for r in rows if slug(r) in group and r["split"] != "fork_control"]
        if len({r["split"] for r in eligible}) != 1:
            raise ValueError(f"Repository family crosses the split boundary: {sorted(group)}")
        main = min(slug(r) for r in eligible)
        identifier = "family-" + sha(json_bytes(facts[main]["roots"]))[:16]
        for name in group:
            assignments[name] = identifier
        splits[eligible[0]["split"]].add(identifier)
    overlaps = {f"{a}--{b}": sorted(splits[a] & splits[b]) for a, b in itertools.combinations(splits, 2)}
    if any(overlaps.values()):
        raise ValueError(overlaps)
    return assignments, checks, overlaps


LEAKS = {
    "source URL": r"https?://|github\.com",
    "issue or revision": r"(?<!\w)#[0-9]+|\b(?:PR|issue|commit|SHA|pull request)\b|(?<![\w.])[0-9a-f]{7,64}\b",
    "patch or source location": r"@@|(?:^|\n)[+-][^\n]|'''|\x60{3}|\.py\b|\b(?:line|lines)\s+\d+",
    "repair hint": r"_irregular|\b(?:regex|patch|cast|truncate|round down|format specifier|regular expression)\b|\b(?:replace|rewrite|modify|edit|change)\s+(?:the|this)\b",
}


def check_ticket(ticket):
    if set(ticket) != {"symptom", "expected", "reproduce"}:
        raise ValueError("A blind ticket must contain exactly symptom, expected and reproduce.")
    for key, value in ticket.items():
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"Empty ticket field: {key}")
        for label, pattern in LEAKS.items():
            if re.search(pattern, value, re.I):
                raise ValueError(f"Blind ticket {key} contains {label}")

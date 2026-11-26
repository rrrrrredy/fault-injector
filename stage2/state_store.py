"""Atomic completed steps, SQLite online backup, and verified restore to a fresh local directory."""

from contextlib import closing, contextmanager
import hashlib
import json
from pathlib import Path
import shutil
import sqlite3
import tempfile
from zipfile import ZipFile, ZIP_DEFLATED

from .sandbox import drive_roots


def local_db_path(path):
    path = Path(path).resolve()
    if any(path.is_relative_to(root.resolve()) for root in drive_roots()):
        raise PermissionError("SQLite needs a local disk outside Google Drive")
    return path


@contextmanager
def open_state(path):
    path = local_db_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(path, timeout=30)) as con:
        con.execute("PRAGMA journal_mode=WAL")
        con.execute("PRAGMA foreign_keys=ON")
        con.executescript("""
            CREATE TABLE IF NOT EXISTS steps(run TEXT, step INTEGER, metadata TEXT NOT NULL, PRIMARY KEY(run,step));
            CREATE TABLE IF NOT EXISTS rollouts(run TEXT, step INTEGER, idx INTEGER, seed TEXT, completion TEXT,
                result TEXT, PRIMARY KEY(run,step,idx), FOREIGN KEY(run,step) REFERENCES steps(run,step));
        """)
        yield con


def record_step(con, run, step, rollouts, *, metadata=None):
    rollouts = tuple(rollouts)
    if not rollouts or step < 1:
        raise ValueError("A completed step requires its rollouts")
    with con:
        con.execute("BEGIN IMMEDIATE")
        last = con.execute("SELECT coalesce(max(step),0) FROM steps WHERE run=?", (run,)).fetchone()[0]
        if step != last + 1:
            raise ValueError(f"Expected step {last + 1} for {run}, got {step}")
        con.execute(
            "INSERT INTO steps VALUES (?,?,?)", (run, step, json.dumps(metadata or {}, sort_keys=True))
        )
        con.executemany(
            "INSERT INTO rollouts VALUES (?,?,?,?,?,?)",
            (
                (run, step, i, r["seed"], r["completion"], json.dumps(r["result"], sort_keys=True))
                for i, r in enumerate(rollouts)
            ),
        )


def cursor_state(con):
    return {
        "steps": dict(con.execute("SELECT run,max(step) FROM steps GROUP BY run")),
        "rollouts": con.execute("SELECT count(*) FROM rollouts").fetchone()[0],
        "metadata": {
            key: json.loads(value)
            for key, value in con.execute(
                "SELECT s.run,s.metadata FROM steps s JOIN (SELECT run,max(step) AS step FROM steps GROUP BY run) latest USING(run,step)"
            )
        },
    }


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def snapshot(con, archive):
    if con.in_transaction:
        raise ValueError("Commit or roll back the active transaction before calling backup()")
    archive = local_db_path(archive)
    archive.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=archive.parent) as tmp:
        db = Path(tmp) / "state.sqlite"
        with closing(sqlite3.connect(db)) as dest:
            con.backup(dest)
        with ZipFile(Path(tmp) / "state.zip", "w", ZIP_DEFLATED) as z:
            z.write(db, "state.sqlite")
            z.writestr("state.sqlite.sha256", sha(db) + "\n")
        sidecar = archive.with_suffix(archive.suffix + ".sha256")
        staged = Path(tmp) / "state.zip"
        (Path(tmp) / "checksum").write_text(sha(staged) + "  " + archive.name + "\n", encoding="ascii")
        staged.replace(archive)
        (Path(tmp) / "checksum").replace(sidecar)
    return archive


def restore(archive, new_directory):
    archive, new_directory = Path(archive), local_db_path(new_directory)
    expected = archive.with_suffix(archive.suffix + ".sha256").read_text(encoding="ascii").split()[0]
    if sha(archive) != expected:
        raise ValueError("Archive SHA-256 mismatch")
    new_directory.mkdir(parents=True, exist_ok=True)
    if any(new_directory.iterdir()):
        raise ValueError("Restore needs an empty destination")
    target = new_directory / "state.sqlite"
    with tempfile.TemporaryDirectory(dir=new_directory) as tmp:
        db = Path(tmp) / "unpacked.sqlite"
        with ZipFile(archive) as z:
            if sorted(z.namelist()) != ["state.sqlite", "state.sqlite.sha256"]:
                raise ValueError("Unexpected snapshot entries")
            with z.open("state.sqlite") as src, db.open("wb") as dest:
                shutil.copyfileobj(src, dest)
            if sha(db) != z.read("state.sqlite.sha256").decode("ascii").strip():
                raise ValueError("SQLite SHA-256 mismatch")
        with closing(sqlite3.connect(db.as_uri() + "?mode=ro&immutable=1", uri=True)) as con:
            integrity = con.execute("PRAGMA integrity_check").fetchone()[0]
            if integrity != "ok":
                raise ValueError(integrity)
            summary = dict(cursor_state(con), integrity=integrity, db_sha256=sha(db))
        db.replace(target)
    return target, summary

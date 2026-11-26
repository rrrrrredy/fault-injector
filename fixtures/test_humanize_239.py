from pathlib import Path
from runpy import run_path


def test_fractional_bytes():
    # humanize.__init__ re-exports this function; the source checkout has no generated _version.py.
    module = run_path(str(Path.cwd() / "src" / "humanize" / "filesize.py"))
    naturalsize = module["naturalsize"]
    print(f"precision control (3000 bytes): {naturalsize(3000)}")
    actual = naturalsize(1.123456789)
    print(f"issue 239 (1.123456789 bytes): {actual}")
    assert actual == "1 Bytes"

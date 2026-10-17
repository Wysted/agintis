"""Download the sample files listed in samples/_manifest.json that are missing on disk.

The index (index/documents.jsonl) is gitignored and the files are not redistributed: this
script fetches them from spensiones.cl, one request at a time with a pause between requests.

Usage: uv run python case/spensiones/fetch_samples.py [--samples DIR] [--delay SECONDS]
"""

import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path

USER_AGENT = "agintis-corpus-survey/1.0"
DEFAULT_DELAY = 3.5
TIMEOUT = 60


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--samples", type=Path, default=Path(__file__).parent / "samples")
    parser.add_argument("--delay", type=float, default=DEFAULT_DELAY)
    args = parser.parse_args()

    root = args.samples.resolve()
    entries = json.loads((root / "_manifest.json").read_text(encoding="utf-8"))
    failed = 0
    first = True
    for entry in entries:
        target = (root / entry["family"] / Path(entry["path"]).name).resolve()
        if not target.is_relative_to(root):
            print(f"skip {entry['id']}: path outside {root}", file=sys.stderr)
            continue
        if target.exists():
            continue
        if not first:
            time.sleep(args.delay)
        first = False
        request = urllib.request.Request(entry["url"], headers={"User-Agent": USER_AGENT})
        try:
            with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
                body = response.read()
        except OSError as exc:
            print(f"fail {entry['id']}: {exc}", file=sys.stderr)
            failed += 1
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(body)
        print(f"ok   {entry['id']} ({len(body)} bytes)")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

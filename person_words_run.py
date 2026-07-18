#!/usr/bin/env python3
"""Run Stanza, IAHLT, and ensemble person-word analysis on a transcript."""

import argparse
import subprocess
import sys
from pathlib import Path

LAB_DIR = Path(__file__).parent

SCRIPTS = (
    ("Stanza", "person_words_stanza.py"),
    ("IAHLT", "person_words_iahlt.py"),
    ("Ensemble", "person_words_ensemble.py"),
)


def run_script(name, script, transcript):
    print("=" * 60, flush=True)
    print(name, flush=True)
    print("=" * 60, flush=True)
    result = subprocess.run(
        [sys.executable, "-u", str(LAB_DIR / script), transcript],
        check=False,
    )
    if result.returncode != 0:
        return result.returncode
    print(flush=True)
    return 0


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Run person_words_stanza, person_words_iahlt, and "
            "person_words_ensemble on a Hebrew transcript."
        )
    )
    parser.add_argument(
        "transcript",
        help="Path to a Hebrew transcript file",
    )
    args = parser.parse_args()

    transcript = Path(args.transcript)
    if not transcript.is_file():
        print(f"Error: file not found: {transcript}", file=sys.stderr)
        sys.exit(1)

    for name, script in SCRIPTS:
        code = run_script(name, script, str(transcript))
        if code != 0:
            sys.exit(code)


if __name__ == "__main__":
    main()

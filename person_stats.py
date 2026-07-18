#!/usr/bin/env python3
"""Compute first/second/third person word percentages in a Hebrew transcript."""

import argparse
import sys
from pathlib import Path

import stanza


def get_pipeline():
    stanza.download("he", verbose=False)
    return stanza.Pipeline(
        lang="he",
        processors="tokenize,mwt,pos",
        verbose=False,
        download_method=stanza.DownloadMethod.REUSE_RESOURCES,
    )


def parse_person(feats):
    if not feats:
        return None
    for part in feats.split("|"):
        if part.startswith("Person="):
            return part.split("=", 1)[1]
    return None


def count_person_words(doc):
    counts = {1: 0, 2: 0, 3: 0}
    for sent in doc.sentences:
        for word in sent.words:
            person = parse_person(word.feats)
            if person == "1":
                counts[1] += 1
            elif person == "2":
                counts[2] += 1
            elif person == "3":
                counts[3] += 1
    return counts


def format_pct(count, total):
    if total == 0:
        return 0.0
    return 100.0 * count / total


def print_stats(counts):
    total = sum(counts.values())
    labels = {
        1: "First person",
        2: "Second person",
        3: "Third person",
    }

    print(f"Total person words: {total}")
    if total == 0:
        print("No person-marked words found.")
        return

    for person in (1, 2, 3):
        pct = format_pct(counts[person], total)
        print(f"{labels[person]}: {counts[person]} ({pct:.1f}%)")


def load_transcript(path):
    if path == "-":
        return sys.stdin.read()
    return Path(path).read_text(encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(
        description="Report first/second/third person word percentages in Hebrew text."
    )
    parser.add_argument(
        "transcript",
        nargs="?",
        default="-",
        help="Path to a Hebrew transcript file, or '-' for stdin (default: stdin)",
    )
    args = parser.parse_args()

    text = load_transcript(args.transcript).strip()
    if not text:
        print("Error: empty transcript.", file=sys.stderr)
        sys.exit(1)

    nlp = get_pipeline()
    doc = nlp(text)
    counts = count_person_words(doc)
    print_stats(counts)


if __name__ == "__main__":
    main()

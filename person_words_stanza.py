#!/usr/bin/env python3
"""List first/second/third person words in a Hebrew transcript by category."""

import argparse
import sys

from person_stats import format_pct, get_pipeline, load_transcript, parse_person

LABELS = {
    1: "First person",
    2: "Second person",
    3: "Third person",
}

PERSONS = (1, 2, 3)
PERSON_TAGS = {"1", "2", "3"}


def collect_person_words(doc):
    words = {1: [], 2: [], 3: []}
    for sent in doc.sentences:
        for word in sent.words:
            person = parse_person(word.feats)
            if person == "1":
                words[1].append(word)
            elif person == "2":
                words[2].append(word)
            elif person == "3":
                words[3].append(word)
    return words


def print_words_by_category(words):
    total = sum(len(items) for items in words.values())
    print(f"Total person words: {total}")
    if total == 0:
        print("No person-marked words found.")
        return

    for person in PERSONS:
        items = words[person]
        pct = format_pct(len(items), total)
        print()
        print(f"{LABELS[person]} ({len(items)}, {pct:.1f}%)")
        print("-" * 40)
        for word in items:
            feats = word.feats or ""
            print(f"{word.text[::-1]}\t{word.upos}\t{feats}")


def main():
    parser = argparse.ArgumentParser(
        description="List first/second/third person words in Hebrew text by category."
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
    words = collect_person_words(doc)
    print_words_by_category(words)


if __name__ == "__main__":
    main()

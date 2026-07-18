#!/usr/bin/env python3
"""List first/second/third person words using the IAHLT UDPipe model."""

import argparse
import sys

from iahlt_morph import DEFAULT_TOKENIZER_MODEL, analyze_text
from person_stats import load_transcript, parse_person
from person_words_stanza import print_words_by_category


def collect_person_words(tokens):
    words = {1: [], 2: [], 3: []}
    for token in tokens:
        person = parse_person(token.feats)
        if person == "1":
            words[1].append(token)
        elif person == "2":
            words[2].append(token)
        elif person == "3":
            words[3].append(token)
    return words


def main():
    parser = argparse.ArgumentParser(
        description=(
            "List first/second/third person words in Hebrew text using "
            "IAHLT UDPipe morphological analysis."
        )
    )
    parser.add_argument(
        "transcript",
        nargs="?",
        default="-",
        help="Path to a Hebrew transcript file, or '-' for stdin (default: stdin)",
    )
    parser.add_argument(
        "--tokenizer-model",
        default=str(DEFAULT_TOKENIZER_MODEL),
        help="Path to IAHLT tokenizer.udpipe model",
    )
    args = parser.parse_args()

    text = load_transcript(args.transcript).strip()
    if not text:
        print("Error: empty transcript.", file=sys.stderr)
        sys.exit(1)

    try:
        tokens = analyze_text(text, model_path=args.tokenizer_model)
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    words = collect_person_words(tokens)
    print_words_by_category(words)


if __name__ == "__main__":
    main()

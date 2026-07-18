#!/usr/bin/env python3
"""Ensemble first/second/third person words from Stanza and IAHLT UDPipe."""

import argparse
import sys
from dataclasses import dataclass

from iahlt_morph import DEFAULT_TOKENIZER_MODEL, analyze_text
from person_stats import format_pct, get_pipeline, load_transcript, parse_person
from person_words_stanza import LABELS, PERSONS, PERSON_TAGS


@dataclass
class PersonHit:
    text: str
    upos: str
    feats: str
    person: int
    start: int
    end: int


def add_char_spans(text, tokens):
    pos = 0
    spans = []
    for token in tokens:
        idx = text.find(token.text, pos)
        if idx == -1:
            spans.append((None, None))
            continue
        spans.append((idx, idx + len(token.text)))
        pos = idx + len(token.text)
    return spans


def stanza_person_hits(doc):
    hits = []
    for sent in doc.sentences:
        for word in sent.words:
            person = parse_person(word.feats)
            if person not in PERSON_TAGS:
                continue
            hits.append(
                PersonHit(
                    text=word.text,
                    upos=word.upos or "",
                    feats=word.feats or "",
                    person=int(person),
                    start=word.start_char,
                    end=word.end_char,
                )
            )
    return hits


def iahlt_person_hits(text, tokens):
    hits = []
    for token, (start, end) in zip(tokens, add_char_spans(text, tokens)):
        person = parse_person(token.feats)
        if person not in PERSON_TAGS or start is None:
            continue
        hits.append(
            PersonHit(
                text=token.text,
                upos=token.upos,
                feats=token.feats,
                person=int(person),
                start=start,
                end=end,
            )
        )
    return hits


def group_hits(hits):
    grouped = {}
    for hit in hits:
        key = (hit.start, hit.end)
        grouped.setdefault(key, []).append(hit)
    return grouped


def consensus_hits(stanza_hits, iahlt_hits):
    stanza_by_span = group_hits(stanza_hits)
    iahlt_by_span = group_hits(iahlt_hits)
    spans = sorted(set(stanza_by_span) & set(iahlt_by_span))

    results = []
    for span in spans:
        stanza_item = stanza_by_span[span][0]
        iahlt_item = iahlt_by_span[span][0]
        if stanza_item.person != iahlt_item.person:
            continue
        results.append(stanza_item)

    results.sort(key=lambda item: item.start)
    return results


def analyze_ensemble(text, nlp, tokenizer_model=DEFAULT_TOKENIZER_MODEL):
    """Run consensus person analysis. Returns (total_words, consensus_hits)."""
    doc = nlp(text)
    iahlt_tokens = analyze_text(text, model_path=tokenizer_model)

    total_words = sum(len(sent.words) for sent in doc.sentences)
    stanza_hits = stanza_person_hits(doc)
    iahlt_hits = iahlt_person_hits(text, iahlt_tokens)
    return total_words, consensus_hits(stanza_hits, iahlt_hits)


def count_hits(hits):
    counts = {1: 0, 2: 0, 3: 0}
    for hit in hits:
        counts[hit.person] += 1
    return counts


def hits_to_categories(hits):
    words = {1: [], 2: [], 3: []}
    for hit in hits:
        words[hit.person].append(hit)
    return words


def print_model_summary(name, hits):
    counts = count_hits(hits)
    total = sum(counts.values())
    print(f"{name}: {total} person words")
    for person in PERSONS:
        pct = format_pct(counts[person], total)
        print(f"  {LABELS[person]}: {counts[person]} ({pct:.1f}%)")


def print_ensemble_output(hits):
    words = hits_to_categories(hits)
    total = len(hits)
    print(f"\nEnsemble person words: {total}")
    if total == 0:
        print("No ensemble person words found.")
        return

    for person in PERSONS:
        items = words[person]
        pct = format_pct(len(items), total)
        print()
        print(f"{LABELS[person]} ({len(items)}, {pct:.1f}%)")
        print("-" * 40)
        for item in items:
            print(f"{item.text[::-1]}\t{item.upos}\t{item.feats}")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Ensemble first/second/third person words using Stanza and IAHLT UDPipe "
            "(consensus only: both taggers agree on span and person)."
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

    nlp = get_pipeline()
    doc = nlp(text)

    try:
        iahlt_tokens = analyze_text(text, model_path=args.tokenizer_model)
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    stanza_hits = stanza_person_hits(doc)
    iahlt_hits = iahlt_person_hits(text, iahlt_tokens)
    ensemble = consensus_hits(stanza_hits, iahlt_hits)

    print_model_summary("Stanza", stanza_hits)
    print_model_summary("IAHLT", iahlt_hits)
    print_ensemble_output(ensemble)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Batch ensemble person-word stats by label from an Excel spreadsheet."""

import argparse
import sys
from pathlib import Path

import pandas as pd

from iahlt_morph import DEFAULT_TOKENIZER_MODEL, get_pipeline as get_udpipe
from person_stats import get_pipeline
from person_words_ensemble import analyze_ensemble, count_hits

DEFAULT_XLSX = Path(__file__).parent / "audio_segments_with_label.xlsx"
DEFAULT_OUT = Path(__file__).parent / "person_by_label.csv"


def analyze_row(text, nlp, tokenizer_model):
    total_words, hits = analyze_ensemble(text, nlp, tokenizer_model=tokenizer_model)
    counts = count_hits(hits)
    return {
        "total_words": total_words,
        "person_words": len(hits),
        "first_person": counts[1],
        "second_person": counts[2],
        "third_person": counts[3],
    }


def run_batch(xlsx_path, tokenizer_model, limit=None):
    df = pd.read_excel(xlsx_path)
    required = {"transcription", "label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Spreadsheet missing columns: {sorted(missing)}")

    print("Loading Stanza and IAHLT models...", flush=True)
    nlp = get_pipeline()
    get_udpipe(tokenizer_model)  # warm UDPipe cache

    rows = []
    work = df.dropna(subset=["transcription", "label"]).copy()
    work["transcription"] = work["transcription"].astype(str).str.strip()
    work["label"] = work["label"].astype(str).str.strip()
    work = work[(work["transcription"] != "") & (work["label"] != "")]
    if limit is not None:
        work = work.head(limit)

    total = len(work)
    print(f"Analyzing {total} transcripts...", flush=True)

    for i, (_, row) in enumerate(work.iterrows(), start=1):
        text = row["transcription"]
        label = row["label"]
        filename = row.get("filename", "")
        try:
            stats = analyze_row(text, nlp, tokenizer_model)
        except Exception as exc:
            print(f"[{i}/{total}] FAILED {filename}: {exc}", file=sys.stderr, flush=True)
            continue

        stats["label"] = label
        stats["filename"] = filename
        rows.append(stats)
        if i == 1 or i % 10 == 0 or i == total:
            print(
                f"[{i}/{total}] {filename} label={label} "
                f"words={stats['total_words']} person={stats['person_words']}",
                flush=True,
            )

    return pd.DataFrame(rows)


def summarize_by_label(per_row):
    if per_row.empty:
        return pd.DataFrame(
            columns=[
                "label",
                "n_segments",
                "total_words",
                "person_words",
                "first_person",
                "second_person",
                "third_person",
            ]
        )

    summary = (
        per_row.groupby("label", as_index=False)
        .agg(
            n_segments=("filename", "count"),
            total_words=("total_words", "sum"),
            person_words=("person_words", "sum"),
            first_person=("first_person", "sum"),
            second_person=("second_person", "sum"),
            third_person=("third_person", "sum"),
        )
        .sort_values("label")
    )
    return summary


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Run ensemble person-word analysis on all spreadsheet transcripts "
            "and aggregate totals by label."
        )
    )
    parser.add_argument(
        "xlsx",
        nargs="?",
        default=str(DEFAULT_XLSX),
        help="Path to audio_segments_with_label.xlsx",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=str(DEFAULT_OUT),
        help="Output CSV path for label summary (default: person_by_label.csv)",
    )
    parser.add_argument(
        "--per-row-output",
        default=None,
        help="Optional CSV path for per-segment counts",
    )
    parser.add_argument(
        "--tokenizer-model",
        default=str(DEFAULT_TOKENIZER_MODEL),
        help="Path to IAHLT tokenizer.udpipe model",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional max number of rows to process (for testing)",
    )
    args = parser.parse_args()

    xlsx_path = Path(args.xlsx)
    if not xlsx_path.is_file():
        print(f"Error: spreadsheet not found: {xlsx_path}", file=sys.stderr)
        sys.exit(1)

    try:
        per_row = run_batch(xlsx_path, args.tokenizer_model, limit=args.limit)
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    summary = summarize_by_label(per_row)

    out_path = Path(args.output)
    summary.to_csv(out_path, index=False)
    print("\nBy label:")
    print(summary.to_string(index=False))
    print(f"\nWrote {out_path}")

    if args.per_row_output:
        per_row_path = Path(args.per_row_output)
        per_row.to_csv(per_row_path, index=False)
        print(f"Wrote {per_row_path}")


if __name__ == "__main__":
    main()

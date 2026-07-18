# Hebrew Person-Word Analysis

Scripts for counting first-, second-, and third-person words in Hebrew transcripts using an ensemble of [Stanza](https://stanfordnlp.github.io/stanza/) and [IAHLT UDPipe](https://github.com/IAHLT/iahlt_coref_he) morphological tagging. Only tokens where **both** taggers agree on the same span and person label are kept (consensus).

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install stanza pandas openpyxl
pip install ./iahlt_coref_he-main
```

Download the IAHLT models from [Google Drive](https://drive.google.com/file/d/1jNQ3LfjQ0dZp1B3N54KLFy0Qw9-DN-t3/view), extract them, and place:

- `tokenizer.udpipe` → `iahlt_coref_he-main/models/tokenizer.udpipe`
- `coref_model/` → `iahlt_coref_he-main/models/coref_model/`

(The large model files are not included in this repo.)

On first Stanza run, the Hebrew model downloads automatically.

## Run the ensemble on all spreadsheet transcripts

Main entry point for the labeled dataset:

```bash
python person_by_label.py audio_segments_with_label.xlsx
```

This loads Stanza + IAHLT once, runs consensus person tagging on every non-empty `transcription`, and aggregates by `label`.

### Output

| File | Contents |
|------|----------|
| **`person_by_label.csv`** | Per-label totals: segments, total words, person words, 1st / 2nd / 3rd person |
| **`person_by_row.csv`** | Same counts per segment (optional; use `--per-row-output`) |

Default paths (same folder as the scripts):

```text
person_by_label.csv   ← summary by label
```

Custom paths:

```bash
python person_by_label.py audio_segments_with_label.xlsx \
  -o person_by_label.csv \
  --per-row-output person_by_row.csv
```

Smoke-test on a few rows:

```bash
python person_by_label.py --limit 3
```

## Run the ensemble on a single transcript

```bash
python person_words_ensemble.py path/to/transcript.txt
```

Or run Stanza, IAHLT, and ensemble together:

```bash
python person_words_run.py path/to/transcript.txt
```

Single-tagger variants:

```bash
python person_words_stanza.py path/to/transcript.txt
python person_words_iahlt.py path/to/transcript.txt
```

These print person-word lists to **stdout** (not CSV).

## Spreadsheet columns used

From `audio_segments_with_label.xlsx`:

- `transcription` — Hebrew text to analyze
- `label` — group key for aggregation (e.g. `ppd`, `control`)
- `filename` — included in the per-row CSV when requested

Empty or missing transcriptions/labels are skipped.

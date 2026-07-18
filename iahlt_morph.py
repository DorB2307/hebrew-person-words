"""Morphological analysis via IAHLT UDPipe model."""

import sys
from dataclasses import dataclass
from pathlib import Path

from conllu import parse
from ufal.udpipe import Model, Pipeline, ProcessingError

DEFAULT_TOKENIZER_MODEL = (
    Path(__file__).parent / "iahlt_coref_he-main" / "models" / "tokenizer.udpipe"
)

# Keep Model objects alive: UDPipe Pipeline does not retain a Python ref.
_MODEL_CACHE = {}
_PIPELINE_CACHE = {}


@dataclass
class MorphToken:
    text: str
    upos: str
    feats: str


def ensure_model_file(model_path):
    path = Path(model_path)
    if not path.is_file():
        raise FileNotFoundError(f"UDPipe model not found: {path}")

    # Only read the start of the file — full-file UTF-8 decode of a binary
    # model is slow and unnecessary.
    with path.open("rb") as handle:
        head = handle.read(64)
    if head.startswith(b"version https://git-lfs"):
        raise FileNotFoundError(
            f"UDPipe model at {path} is a Git LFS pointer, not the real file. "
            "Download models from "
            "https://drive.google.com/file/d/1jNQ3LfjQ0dZp1B3N54KLFy0Qw9-DN-t3/view "
            "and extract tokenizer.udpipe into iahlt_coref_he-main/models/."
        )


def feats_to_str(feats):
    if feats is None:
        return ""
    if hasattr(feats, "items"):
        return "|".join(f"{key}={value}" for key, value in feats.items())
    return str(feats)


def get_pipeline(model_path=DEFAULT_TOKENIZER_MODEL):
    """Load and cache a UDPipe tokenize+tag pipeline."""
    key = str(Path(model_path).resolve())
    if key not in _PIPELINE_CACHE:
        ensure_model_file(model_path)
        model = Model.load(str(model_path))
        _MODEL_CACHE[key] = model
        _PIPELINE_CACHE[key] = Pipeline(
            model, "tokenize", Pipeline.DEFAULT, Pipeline.NONE, "conllu"
        )
    return _PIPELINE_CACHE[key]


def analyze_text(text, model_path=DEFAULT_TOKENIZER_MODEL, pipeline=None):
    pipe = pipeline or get_pipeline(model_path)
    error = ProcessingError()
    processed = pipe.process(text, error)
    if error.occurred():
        print(f"UDPipe error: {error.message}", file=sys.stderr)
        sys.exit(1)

    tokens = []
    for sentence in parse(processed):
        for token in sentence:
            if not isinstance(token["id"], int):
                continue
            tokens.append(
                MorphToken(
                    text=token["form"],
                    upos=token["upos"] or "",
                    feats=feats_to_str(token["feats"]),
                )
            )
    return tokens

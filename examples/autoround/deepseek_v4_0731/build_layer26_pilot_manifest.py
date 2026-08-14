# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Build immutable calibration and held-out manifests for the layer-26 pilot."""

from __future__ import annotations

import argparse
import json
from itertools import islice
from pathlib import Path

from datasets import load_dataset
from layer26_pilot_evidence import (
    DATASET_SOURCES,
    MODEL_ID,
    MODEL_REVISION,
    atomic_write_json,
    finalize_corpus_manifest,
    select_source_records,
    tokenize_corpus_manifest,
    validate_corpus_manifest,
    validate_token_manifest,
)
from transformers import AutoTokenizer

DEFAULT_SCAN_LIMIT = 4096


def build_corpus_manifest(scan_limit: int) -> dict:
    """Select the fixed category mixture from bounded immutable-source scans."""
    records = []
    for category, source in DATASET_SOURCES.items():
        dataset = load_dataset(
            source["repo_id"],
            revision=source["revision"],
            split=source["split"],
            streaming=True,
        )
        indexed_rows = enumerate(islice(iter(dataset), scan_limit))
        records.extend(
            select_source_records(
                category=category,
                rows=indexed_rows,
                calibration_count=source["calibration_count"],
                held_out_count=source["held_out_count"],
            )
        )
    return finalize_corpus_manifest(records)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build pinned DeepSeek V4 layer-26 pilot manifests"
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--scan-limit", type=int, default=DEFAULT_SCAN_LIMIT)
    parser.add_argument(
        "--identity-lock",
        type=Path,
        default=Path(__file__).with_name("dependency-lock.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.scan_limit < 64:
        raise ValueError("DeepSeek V4 pilot scan limit must be at least 64")

    corpus_manifest = build_corpus_manifest(args.scan_limit)
    validate_corpus_manifest(corpus_manifest)

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
    )
    token_manifest = tokenize_corpus_manifest(corpus_manifest, tokenizer)
    validate_token_manifest(token_manifest)

    with args.identity_lock.open(encoding="utf-8") as file_handle:
        expected_identity = json.load(file_handle)["pilot_identity"]
    actual_identity = {
        "calibration_samples": 128,
        "corpus_manifest_sha256": corpus_manifest["manifest_sha256"],
        "held_out_samples": 32,
        "max_sequence_length": 2048,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "target_layer": "layers.26",
        "token_manifest_sha256": token_manifest["token_manifest_sha256"],
    }
    if actual_identity != expected_identity:
        raise ValueError(
            "DeepSeek V4 pilot rebuilt manifests do not match dependency-lock.json"
        )

    atomic_write_json(args.output_dir / "corpus-manifest.json", corpus_manifest)
    atomic_write_json(args.output_dir / "token-manifest.json", token_manifest)
    print(
        json.dumps(
            {
                "corpus_manifest_sha256": corpus_manifest["manifest_sha256"],
                "token_manifest_sha256": token_manifest["token_manifest_sha256"],
                "calibration_samples": 128,
                "held_out_samples": 32,
                "output_dir": str(args.output_dir.resolve()),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

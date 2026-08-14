# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Validate and summarize matched RTN and AutoRound layer-26 pilot reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from layer26_pilot_evidence import atomic_write_json, sha256_json

_MATCHED_FIELDS = (
    "dependency_commits",
    "checkpoint",
    "token_manifest_sha256",
    "reference_manifest_sha256",
    "target_layer",
    "output_schema",
)


def load_report(path: Path) -> dict:
    """Load an arm report and verify its canonical receipt hash."""
    with path.open(encoding="utf-8") as file_handle:
        report = json.load(file_handle)
    claimed_hash = report.pop("report_sha256", None)
    if claimed_hash != sha256_json(report):
        raise ValueError(f"DeepSeek V4 pilot report checksum mismatch: {path}")
    return {**report, "report_sha256": claimed_hash}


def summarize_reports(baseline: dict, autoround: dict) -> dict:
    """Compare only reports that bind the same source, data, layer, and schema."""
    if baseline.get("arm") != "baseline" or baseline.get("algorithm") != "plain_rtn":
        raise ValueError("DeepSeek V4 pilot baseline report is not plain RTN")
    if autoround.get("arm") != "autoround" or autoround.get("iterations") != 200:
        raise ValueError(
            "DeepSeek V4 pilot AutoRound report has wrong arm or iterations"
        )
    for field in _MATCHED_FIELDS:
        if baseline.get(field) != autoround.get(field):
            raise ValueError(f"DeepSeek V4 pilot reports differ in {field}")

    rtn_metrics = baseline["comparison"]["summary"]
    autoround_metrics = autoround["comparison"]["summary"]
    rtn_mse = rtn_metrics["mse"]
    autoround_mse = autoround_metrics["mse"]
    if rtn_mse <= 0:
        raise ValueError("DeepSeek V4 pilot RTN MSE must be positive")
    payload = {
        "schema_version": 1,
        "target_layer": baseline["target_layer"],
        "baseline_report_sha256": baseline["report_sha256"],
        "autoround_report_sha256": autoround["report_sha256"],
        "token_manifest_sha256": baseline["token_manifest_sha256"],
        "reference_manifest_sha256": baseline["reference_manifest_sha256"],
        "output_schema": baseline["output_schema"],
        "rtn": rtn_metrics,
        "autoround": autoround_metrics,
        "autoround_mse_reduction_fraction": (rtn_mse - autoround_mse) / rtn_mse,
        "timing_seconds": {
            "baseline_quantization": baseline["timing_seconds"]["quantization"],
            "autoround_quantization": autoround["timing_seconds"]["quantization"],
        },
        "resources": {
            "baseline": baseline["resources"],
            "autoround": autoround["resources"],
        },
    }
    return {**payload, "summary_sha256": sha256_json(payload)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize matched DeepSeek V4 layer-26 pilot arms"
    )
    parser.add_argument("--baseline-report", type=Path, required=True)
    parser.add_argument("--autoround-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = summarize_reports(
        load_report(args.baseline_report),
        load_report(args.autoround_report),
    )
    atomic_write_json(args.output, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

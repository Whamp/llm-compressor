# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Evidence contracts for the DeepSeek V4 layer-26 AutoRound pilot."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypedDict, cast

import torch

MODEL_ID = "deepseek-ai/DeepSeek-V4-Flash-0731"
MODEL_REVISION = "7872f01b1d1fe23eabc4c98b48bffcef5a386062"
TARGET_LAYER_PATH = "layers.26"
MAX_SEQUENCE_LENGTH = 2048
CALIBRATION_SAMPLE_COUNT = 128
HELD_OUT_SAMPLE_COUNT = 32


class DatasetSource(TypedDict):
    """Immutable source and quota for one pilot corpus category."""

    repo_id: str
    revision: str
    split: str
    calibration_count: int
    held_out_count: int


DEPENDENCY_COMMITS = {
    "compressed-tensors": "434b5a33f584a8990e91a96e9177ee49c2f2aa31",
    "llm-compressor": "812bb30398327cc5f862018988500fa47378037d",
    "auto-round": "52636d3c725926047e66da752651ca643fd2806b",
}

DATASET_SOURCES: dict[str, DatasetSource] = {
    "coding": {
        "repo_id": "codeparrot/self-instruct-starcoder",
        "revision": "9598805f9276dc0a339e293e9f205ea8bbf41f9f",
        "split": "curated",
        "calibration_count": 32,
        "held_out_count": 8,
    },
    "tool_use": {
        "repo_id": "ashotaslanyan/Nemotron-SFT-OpenCode-v1",
        "revision": "bb286811729fde04cc47b6f4b4588867bdd2a7d9",
        "split": "agent_skills",
        "calibration_count": 32,
        "held_out_count": 8,
    },
    "reasoning": {
        "repo_id": ("Magpie-Align/Magpie-Reasoning-V2-250K-CoT-Deepseek-R1-Llama-70B"),
        "revision": "d78edb811991faba57a3b7226719f3818460e723",
        "split": "train",
        "calibration_count": 32,
        "held_out_count": 8,
    },
    "general_chat": {
        "repo_id": "HuggingFaceH4/ultrachat_200k",
        "revision": "8049631c405ae6576f93f445c6b8166f76f5505a",
        "split": "train_sft",
        "calibration_count": 32,
        "held_out_count": 8,
    },
}

_FORBIDDEN_LEAKAGE_PHRASES = (
    "superjson-error-stack-serialization",
    "error stack serialization",
)


def canonical_json_bytes(value: Any) -> bytes:
    """Serialize evidence as stable UTF-8 JSON for content hashing."""
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def sha256_json(value: Any) -> str:
    """Return the SHA-256 digest of canonical JSON evidence."""
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def sha256_file(path: Path) -> str:
    """Hash a file without loading it wholly into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    """Write a JSON receipt atomically on the destination filesystem."""
    path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8") as file_handle:
            json.dump(value, file_handle, ensure_ascii=False, indent=2, sort_keys=True)
            file_handle.write("\n")
            file_handle.flush()
            os.fsync(file_handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def render_deepseek_v4_messages(messages: Sequence[Mapping[str, str]]) -> str:
    """Render supported chat roles with the pinned DeepSeek V4 encoding."""
    text = "<｜begin▁of▁sentence｜>"
    for message in messages:
        role = message.get("role")
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            raise ValueError("DeepSeek V4 pilot message content must be non-empty text")
        if role == "system":
            text += content
        elif role == "user":
            text += f"<｜User｜>{content}"
        elif role == "assistant":
            reasoning_prefix = (
                "" if content.lstrip().startswith("<think>") else "</think>"
            )
            text += f"<｜Assistant｜>{reasoning_prefix}{content}<｜end▁of▁sentence｜>"
        else:
            raise ValueError(f"DeepSeek V4 pilot message role is unsupported: {role!r}")
    return text


def normalize_source_messages(
    category: str, row: Mapping[str, Any]
) -> list[dict[str, str]]:
    """Map one pinned source row to the pilot's user/assistant message contract."""
    if category == "coding":
        messages = [
            {"role": "user", "content": row.get("instruction")},
            {"role": "assistant", "content": row.get("output")},
        ]
    elif category == "tool_use":
        messages = [
            {"role": "user", "content": row.get("question")},
            {"role": "assistant", "content": row.get("agent_prompt")},
        ]
    elif category == "reasoning":
        messages = [
            {"role": "user", "content": row.get("instruction")},
            {"role": "assistant", "content": row.get("response")},
        ]
    elif category == "general_chat":
        messages = row.get("messages")
    else:
        raise ValueError(f"Unknown DeepSeek V4 pilot category: {category}")

    if not isinstance(messages, list) or len(messages) < 2:
        raise ValueError(f"DeepSeek V4 pilot {category} row has no conversation")
    normalized: list[dict[str, str]] = []
    for message in messages:
        if not isinstance(message, Mapping):
            raise ValueError(f"DeepSeek V4 pilot {category} message is not a mapping")
        role = message.get("role")
        content = message.get("content")
        if not isinstance(role, str) or not isinstance(content, str):
            raise ValueError(
                f"DeepSeek V4 pilot {category} message role/content must be text"
            )
        normalized.append({"role": role, "content": content})
    render_deepseek_v4_messages(normalized)
    return normalized


def contains_acceptance_task_leakage(messages: Sequence[Mapping[str, str]]) -> bool:
    """Reject corpus rows that mention the held-back SuperJSON acceptance task."""
    searchable = "\n".join(str(message.get("content", "")) for message in messages)
    searchable = searchable.casefold()
    return any(phrase in searchable for phrase in _FORBIDDEN_LEAKAGE_PHRASES)


def select_source_records(
    category: str,
    rows: Iterable[tuple[int, Mapping[str, Any]]],
    calibration_count: int,
    held_out_count: int,
) -> list[dict]:
    """Select deterministic disjoint rows from a bounded pinned-source scan."""
    source = DATASET_SOURCES[category]
    candidates = []
    for row_index, row in rows:
        try:
            messages = normalize_source_messages(category, row)
        except ValueError:
            continue
        if contains_acceptance_task_leakage(messages):
            continue
        rendered = render_deepseek_v4_messages(messages)
        source_identity = {
            "repo_id": source["repo_id"],
            "revision": source["revision"],
            "split": source["split"],
            "row_index": row_index,
        }
        identity = {
            **source_identity,
            "rendered_sha256": hashlib.sha256(rendered.encode("utf-8")).hexdigest(),
        }
        candidates.append((sha256_json(source_identity), identity, messages))

    required = calibration_count + held_out_count
    if len(candidates) < required:
        raise ValueError(
            f"DeepSeek V4 pilot {category} source yielded {len(candidates)} valid "
            f"rows; {required} required"
        )

    selected = []
    for selection_index, (_, identity, messages) in enumerate(sorted(candidates)):
        if selection_index >= required:
            break
        partition = "calibration" if selection_index < calibration_count else "held_out"
        selected.append(
            {
                "category": category,
                "partition": partition,
                **identity,
                "messages": messages,
            }
        )
    return selected


def finalize_corpus_manifest(records: Sequence[Mapping[str, Any]]) -> dict:
    """Validate and checksum the complete calibration/held-out corpus manifest."""
    records = [dict(record) for record in records]
    expected_counts = {
        "calibration": CALIBRATION_SAMPLE_COUNT,
        "held_out": HELD_OUT_SAMPLE_COUNT,
    }
    actual_counts = {
        partition: sum(record.get("partition") == partition for record in records)
        for partition in expected_counts
    }
    if actual_counts != expected_counts:
        raise ValueError(
            f"DeepSeek V4 pilot corpus counts mismatch: {actual_counts}; "
            f"expected {expected_counts}"
        )
    for category, source in DATASET_SOURCES.items():
        for partition, count_field in (
            ("calibration", "calibration_count"),
            ("held_out", "held_out_count"),
        ):
            actual_count = sum(
                record.get("category") == category
                and record.get("partition") == partition
                for record in records
            )
            expected_count = source[count_field]
            if actual_count != expected_count:
                raise ValueError(
                    f"DeepSeek V4 pilot {category}/{partition} count is "
                    f"{actual_count}; expected {expected_count}"
                )

    identities = [
        (
            record.get("repo_id"),
            record.get("revision"),
            record.get("split"),
            record.get("row_index"),
        )
        for record in records
    ]
    if len(set(identities)) != len(identities):
        raise ValueError("DeepSeek V4 pilot corpus contains duplicate source rows")

    for record in records:
        category = record.get("category")
        source = DATASET_SOURCES.get(category)
        if source is None:
            raise ValueError(
                f"DeepSeek V4 pilot corpus category is unknown: {category}"
            )
        for field in ("repo_id", "revision", "split"):
            if record.get(field) != source[field]:
                raise ValueError(
                    f"DeepSeek V4 pilot {category} record has wrong {field}"
                )
        raw_messages = record.get("messages")
        if not isinstance(raw_messages, list):
            raise ValueError(f"DeepSeek V4 pilot {category} messages must be a list")
        messages = cast(list[Mapping[str, str]], raw_messages)
        rendered = render_deepseek_v4_messages(messages)
        rendered_hash = hashlib.sha256(rendered.encode("utf-8")).hexdigest()
        if record.get("rendered_sha256") != rendered_hash:
            raise ValueError(
                f"DeepSeek V4 pilot {category} rendered text checksum mismatch"
            )
        if contains_acceptance_task_leakage(messages):
            raise ValueError(
                f"DeepSeek V4 pilot {category} record leaks the acceptance task"
            )

    payload = {
        "schema_version": 1,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "max_sequence_length": MAX_SEQUENCE_LENGTH,
        "counts": actual_counts,
        "records": records,
    }
    return {**payload, "manifest_sha256": sha256_json(payload)}


def validate_corpus_manifest(manifest: Mapping[str, Any]) -> None:
    """Fail closed unless a corpus manifest is internally and externally pinned."""
    manifest = dict(manifest)
    claimed_hash = manifest.pop("manifest_sha256", None)
    if claimed_hash != sha256_json(manifest):
        raise ValueError("DeepSeek V4 pilot corpus manifest checksum mismatch")
    rebuilt = finalize_corpus_manifest(manifest.get("records", []))
    if rebuilt != {**manifest, "manifest_sha256": claimed_hash}:
        raise ValueError("DeepSeek V4 pilot corpus manifest contract mismatch")


def tokenize_corpus_manifest(manifest: Mapping[str, Any], tokenizer: Any) -> dict:
    """Tokenize pinned corpus records and bind exact token IDs into evidence."""
    validate_corpus_manifest(manifest)
    tokenized_records = []
    for record in manifest["records"]:
        rendered = render_deepseek_v4_messages(record["messages"])
        encoded = tokenizer(
            rendered,
            add_special_tokens=False,
            max_length=manifest["max_sequence_length"],
            truncation=True,
            return_attention_mask=True,
        )
        input_ids = encoded["input_ids"]
        attention_mask = encoded["attention_mask"]
        if not input_ids or len(input_ids) != len(attention_mask):
            raise ValueError("DeepSeek V4 pilot tokenization produced invalid lengths")
        tokenized_records.append(
            {
                "category": record["category"],
                "partition": record["partition"],
                "source_row_index": record["row_index"],
                "rendered_sha256": record["rendered_sha256"],
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "token_sha256": sha256_json(
                    {"input_ids": input_ids, "attention_mask": attention_mask}
                ),
            }
        )

    payload = {
        "schema_version": 1,
        "corpus_manifest_sha256": manifest["manifest_sha256"],
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "max_sequence_length": MAX_SEQUENCE_LENGTH,
        "records": tokenized_records,
    }
    return {**payload, "token_manifest_sha256": sha256_json(payload)}


def validate_token_manifest(token_manifest: Mapping[str, Any]) -> None:
    """Fail closed unless all token rows and the aggregate token manifest match."""
    token_manifest = dict(token_manifest)
    claimed_hash = token_manifest.pop("token_manifest_sha256", None)
    if claimed_hash != sha256_json(token_manifest):
        raise ValueError("DeepSeek V4 pilot token manifest checksum mismatch")
    if token_manifest.get("model_id") != MODEL_ID:
        raise ValueError("DeepSeek V4 pilot token manifest model mismatch")
    if token_manifest.get("model_revision") != MODEL_REVISION:
        raise ValueError("DeepSeek V4 pilot token manifest revision mismatch")
    records = token_manifest.get("records", [])
    counts = {
        partition: sum(record.get("partition") == partition for record in records)
        for partition in ("calibration", "held_out")
    }
    expected = {
        "calibration": CALIBRATION_SAMPLE_COUNT,
        "held_out": HELD_OUT_SAMPLE_COUNT,
    }
    if counts != expected:
        raise ValueError(
            f"DeepSeek V4 pilot token counts mismatch: {counts}; expected {expected}"
        )
    for category, source in DATASET_SOURCES.items():
        for partition, count_field in (
            ("calibration", "calibration_count"),
            ("held_out", "held_out_count"),
        ):
            actual_count = sum(
                record.get("category") == category
                and record.get("partition") == partition
                for record in records
            )
            if actual_count != source[count_field]:
                raise ValueError(
                    f"DeepSeek V4 pilot token {category}/{partition} count mismatch"
                )
    for record in records:
        input_ids = record.get("input_ids")
        attention_mask = record.get("attention_mask")
        if not input_ids or len(input_ids) != len(attention_mask):
            raise ValueError("DeepSeek V4 pilot token row has invalid lengths")
        expected_hash = sha256_json(
            {"input_ids": input_ids, "attention_mask": attention_mask}
        )
        if record.get("token_sha256") != expected_hash:
            raise ValueError("DeepSeek V4 pilot token row checksum mismatch")


@dataclass(frozen=True)
class LayerOutputError:
    """Additive numerical-error terms for one held-out layer output."""

    element_count: int
    squared_error_sum: float
    reference_square_sum: float
    candidate_square_sum: float
    cross_product_sum: float
    max_absolute_error: float

    @classmethod
    def compare(
        cls, reference: torch.Tensor, candidate: torch.Tensor
    ) -> LayerOutputError:
        """Compare one candidate output against an exact-shape BF16 reference."""
        if reference.shape != candidate.shape:
            raise ValueError(
                "DeepSeek V4 pilot layer output shape mismatch: "
                f"{tuple(candidate.shape)} != {tuple(reference.shape)}"
            )
        if reference.dtype != torch.bfloat16 or candidate.dtype != torch.bfloat16:
            raise ValueError(
                "DeepSeek V4 pilot layer outputs must both have BF16 dtype"
            )
        reference_float = reference.float()
        candidate_float = candidate.float()
        difference = candidate_float - reference_float
        return cls(
            element_count=reference.numel(),
            squared_error_sum=difference.square().sum().item(),
            reference_square_sum=reference_float.square().sum().item(),
            candidate_square_sum=candidate_float.square().sum().item(),
            cross_product_sum=(reference_float * candidate_float).sum().item(),
            max_absolute_error=difference.abs().max().item(),
        )


def summarize_layer_output_errors(errors: Sequence[LayerOutputError]) -> dict:
    """Aggregate held-out output errors by element rather than by sequence."""
    if not errors:
        raise ValueError("DeepSeek V4 pilot has no held-out output errors")
    element_count = sum(error.element_count for error in errors)
    squared_error_sum = sum(error.squared_error_sum for error in errors)
    reference_square_sum = sum(error.reference_square_sum for error in errors)
    candidate_square_sum = sum(error.candidate_square_sum for error in errors)
    cross_product_sum = sum(error.cross_product_sum for error in errors)
    denominator = math.sqrt(reference_square_sum * candidate_square_sum)
    return {
        "sample_count": len(errors),
        "element_count": element_count,
        "mse": squared_error_sum / element_count,
        "normalized_mse": squared_error_sum / reference_square_sum,
        "cosine_similarity": cross_product_sum / denominator,
        "max_absolute_error": max(error.max_absolute_error for error in errors),
    }

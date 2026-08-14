from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest
import torch
from compressed_tensors.utils import match_targets

EXAMPLE_DIR = Path(__file__).parents[2] / "examples" / "autoround" / "deepseek_v4_0731"
sys.path.insert(0, str(EXAMPLE_DIR))

from layer26_pilot_evidence import (  # noqa: E402
    DATASET_SOURCES,
    LayerOutputError,
    finalize_corpus_manifest,
    render_deepseek_v4_messages,
    select_source_records,
    sha256_json,
    summarize_layer_output_errors,
    tokenize_corpus_manifest,
    validate_corpus_manifest,
    validate_token_manifest,
)
from summarize_layer26_pilot import summarize_reports  # noqa: E402


def _source_row(category: str, index: int) -> dict:
    user = f"{category} user {index}"
    assistant = f"{category} assistant {index}"
    if category == "coding":
        return {"instruction": user, "output": assistant}
    if category == "tool_use":
        return {"question": user, "agent_prompt": assistant}
    if category == "reasoning":
        return {"instruction": user, "response": assistant}
    if category == "general_chat":
        return {
            "messages": [
                {"role": "user", "content": user},
                {"role": "assistant", "content": assistant},
            ]
        }
    raise AssertionError(f"unexpected category {category}")


def _complete_manifest() -> dict:
    records = []
    for category, source in DATASET_SOURCES.items():
        count = source["calibration_count"] + source["held_out_count"]
        rows = [(index, _source_row(category, index)) for index in range(count + 5)]
        records.extend(
            select_source_records(
                category,
                rows,
                source["calibration_count"],
                source["held_out_count"],
            )
        )
    return finalize_corpus_manifest(records)


class _FakeTokenizer:
    def __call__(self, text, **_kwargs):
        input_ids = [ord(character) % 127 for character in text[:32]]
        return {
            "input_ids": input_ids,
            "attention_mask": [1] * len(input_ids),
        }


class _MixedDtypePrefix(torch.nn.Module):
    """Minimal prefix with FP32 activations entering a BF16 layer."""

    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList(
            [torch.nn.Identity() for _ in range(26)]
            + [torch.nn.Linear(4, 4, bias=False, dtype=torch.bfloat16)]
        )

    def forward(self, input_ids, attention_mask, use_cache=False):
        del attention_mask, use_cache
        hidden_states = torch.ones(
            (*input_ids.shape, 4),
            dtype=torch.float32,
            device="cpu",
        )
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        return hidden_states


def _load_runner_module():
    module_path = EXAMPLE_DIR / "run_layer26_autoround_pilot.py"
    specification = importlib.util.spec_from_file_location(
        "layer26_runner", module_path
    )
    module = importlib.util.module_from_spec(specification)
    assert specification.loader is not None
    specification.loader.exec_module(module)
    return module


def _arm_report(arm: str, mse: float) -> dict:
    algorithm = "plain_rtn" if arm == "baseline" else "autoround"
    return {
        "arm": arm,
        "algorithm": algorithm,
        "iterations": 0 if arm == "baseline" else 200,
        "dependency_commits": {"a": "1"},
        "checkpoint": {"index_sha256": "index"},
        "token_manifest_sha256": "tokens",
        "reference_manifest_sha256": "reference",
        "target_layer": "layers.26",
        "output_schema": {"module_counts": {"W2_G256": 512, "W4_G128": 256}},
        "comparison": {
            "summary": {
                "mse": mse,
                "normalized_mse": mse / 2,
                "cosine_similarity": 0.9,
                "max_absolute_error": 1.0,
            }
        },
        "timing_seconds": {"quantization": 1.0},
        "resources": {},
        "report_sha256": f"{arm}-hash",
    }


def test_render_deepseek_v4_messages_preserves_reasoning_prefix():
    reasoning = render_deepseek_v4_messages(
        [
            {"role": "user", "content": "question"},
            {"role": "assistant", "content": "<think>reasoning</think>answer"},
        ]
    )
    direct = render_deepseek_v4_messages(
        [
            {"role": "user", "content": "question"},
            {"role": "assistant", "content": "answer"},
        ]
    )

    assert "<｜Assistant｜><think>" in reasoning
    assert "</think><think>" not in reasoning
    assert "<｜Assistant｜></think>answer" in direct


def test_render_deepseek_v4_messages_rejects_unsupported_role():
    with pytest.raises(ValueError, match="role is unsupported"):
        render_deepseek_v4_messages(
            [
                {"role": "user", "content": "question"},
                {"role": "tool", "content": "result"},
            ]
        )


def test_select_source_records_is_deterministic_and_disjoint():
    rows = [(index, _source_row("coding", index)) for index in range(20)]
    first = select_source_records("coding", rows, 4, 2)
    second = select_source_records("coding", reversed(rows), 4, 2)

    assert first == second
    assert [record["partition"] for record in first].count("calibration") == 4
    assert [record["partition"] for record in first].count("held_out") == 2
    assert len({record["row_index"] for record in first}) == 6


def test_select_source_records_is_independent_of_rendering_content():
    original_rows = [(index, _source_row("coding", index)) for index in range(20)]
    changed_rows = [
        (
            index,
            {
                **row,
                "output": f"changed rendering content {index}",
            },
        )
        for index, row in original_rows
    ]

    original = select_source_records("coding", original_rows, 4, 2)
    changed = select_source_records("coding", changed_rows, 4, 2)

    assert [record["row_index"] for record in original] == [
        record["row_index"] for record in changed
    ]


def test_select_source_records_excludes_acceptance_task_leakage():
    rows = [(index, _source_row("coding", index)) for index in range(10)]
    rows[0][1]["instruction"] = "Fix superjson-error-stack-serialization"

    selected = select_source_records("coding", rows, 4, 2)

    assert all(record["row_index"] != 0 for record in selected)


def test_corpus_and_token_manifests_are_checksum_bound():
    corpus_manifest = _complete_manifest()
    validate_corpus_manifest(corpus_manifest)
    token_manifest = tokenize_corpus_manifest(corpus_manifest, _FakeTokenizer())
    validate_token_manifest(token_manifest)

    corpus_manifest["records"][0]["messages"][0]["content"] = "tampered"
    with pytest.raises(ValueError, match="manifest checksum mismatch"):
        validate_corpus_manifest(corpus_manifest)

    token_manifest["records"][0]["input_ids"][0] += 1
    with pytest.raises(ValueError, match="manifest checksum mismatch"):
        validate_token_manifest(token_manifest)


def test_corpus_manifest_enforces_category_partition_counts():
    manifest = _complete_manifest()
    records = manifest["records"]
    coding_index = next(
        index
        for index, record in enumerate(records)
        if record["category"] == "coding" and record["partition"] == "calibration"
    )
    reasoning_record = next(
        record
        for record in records
        if record["category"] == "reasoning" and record["partition"] == "calibration"
    )
    replacement = dict(reasoning_record)
    replacement["row_index"] += 10_000
    records[coding_index] = replacement

    with pytest.raises(ValueError, match="coding/calibration count"):
        finalize_corpus_manifest(records)


def test_capture_layer26_output_autocasts_mixed_dtype_prefix():
    runner = _load_runner_module()

    output = runner.capture_layer26_output(
        _MixedDtypePrefix(),
        {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1]},
    )

    assert output.shape == (1, 3, 4)
    assert output.dtype == torch.bfloat16


def test_layer_output_error_aggregates_by_element():
    reference_a = torch.tensor([1.0, 2.0], dtype=torch.bfloat16)
    candidate_a = torch.tensor([2.0, 2.0], dtype=torch.bfloat16)
    reference_b = torch.tensor([1.0], dtype=torch.bfloat16)
    candidate_b = torch.tensor([3.0], dtype=torch.bfloat16)

    summary = summarize_layer_output_errors(
        [
            LayerOutputError.compare(reference_a, candidate_a),
            LayerOutputError.compare(reference_b, candidate_b),
        ]
    )

    assert summary["element_count"] == 3
    assert summary["mse"] == pytest.approx(5 / 3)
    assert summary["normalized_mse"] == pytest.approx(5 / 6)
    assert summary["max_absolute_error"] == 2.0


def test_layer_output_error_rejects_wrong_shape_or_dtype():
    reference = torch.ones(2, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="shape mismatch"):
        LayerOutputError.compare(reference, torch.ones(3, dtype=torch.bfloat16))
    with pytest.raises(ValueError, match="BF16 dtype"):
        LayerOutputError.compare(reference, torch.ones(2, dtype=torch.float32))


def test_summarize_reports_requires_matched_evidence():
    baseline = _arm_report("baseline", mse=2.0)
    autoround = _arm_report("autoround", mse=1.0)

    summary = summarize_reports(baseline, autoround)

    assert summary["autoround_mse_reduction_fraction"] == pytest.approx(0.5)
    assert summary["summary_sha256"] == sha256_json(
        {key: value for key, value in summary.items() if key != "summary_sha256"}
    )

    autoround["token_manifest_sha256"] = "different"
    with pytest.raises(ValueError, match="differ in token_manifest_sha256"):
        summarize_reports(baseline, autoround)


def test_prepare_reference_outputs_reuses_verified_manifest(tmp_path, monkeypatch):
    module = _load_runner_module()
    reference_dir = tmp_path / "reference"
    reference_dir.mkdir()
    (reference_dir / "reference-manifest.json").write_text("{}")
    expected = {"manifest_sha256": "verified"}

    monkeypatch.setattr(
        module,
        "load_reference_manifest",
        lambda path: expected if path == reference_dir else None,
    )
    monkeypatch.setattr(
        module,
        "write_reference_outputs",
        lambda *args, **kwargs: pytest.fail("verified references must be reused"),
    )

    manifest, reused = module.prepare_reference_outputs(
        torch.nn.Linear(1, 1), [], reference_dir
    )

    assert manifest == expected
    assert reused is True


def test_run_quantization_streams_oversized_sequential_weights(monkeypatch):
    module = _load_runner_module()
    captured = {}

    def capture_oneshot(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(module, "oneshot", capture_oneshot)
    module.run_quantization(
        model=torch.nn.Linear(1, 1),
        tokenizer=object(),
        calibration_dataset=[],
        iterations=0,
        device_ids="0,1",
    )

    assert captured["sequential_keep_onloaded_weights"] is False


def test_dependency_revision_allows_only_exact_or_lock_only_child():
    module = _load_runner_module()
    lock_path = "examples/autoround/deepseek_v4_0731/dependency-lock.json"

    assert module.is_allowed_dependency_revision(
        "compressed-tensors", "expected", "expected", "parent", []
    )
    assert module.is_allowed_dependency_revision(
        "llm-compressor", "runtime", "lock-child", "runtime", [lock_path]
    )
    assert not module.is_allowed_dependency_revision(
        "llm-compressor",
        "runtime",
        "changed-child",
        "runtime",
        [lock_path, "src/llmcompressor/entrypoints/oneshot.py"],
    )
    assert not module.is_allowed_dependency_revision(
        "auto-round", "runtime", "lock-child", "runtime", [lock_path]
    )


def test_verify_bf16_checkpoint_requires_exact_dense_inventory(tmp_path):
    module = _load_runner_module()
    model_path = tmp_path / "model"
    model_path.mkdir()
    (model_path / "config.json").write_text("{}", encoding="utf-8")
    shard_name = "model-00001-of-00001.safetensors"
    (model_path / shard_name).touch()
    weight_map = {
        f"tensor.{index}": shard_name
        for index in range(module.BF16_OUTPUT_TENSOR_COUNT)
    }
    (model_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": weight_map}),
        encoding="utf-8",
    )

    evidence = module.verify_bf16_checkpoint(model_path)
    assert evidence["tensor_count"] == module.BF16_OUTPUT_TENSOR_COUNT
    assert evidence["shard_count"] == 1

    (model_path / "config.json").write_text(
        '{"quantization_config": {"quant_method": "fp8"}}',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="remains quantized"):
        module.verify_bf16_checkpoint(model_path)


def test_verify_pilot_identity_rejects_different_token_manifest(tmp_path):
    module = _load_runner_module()
    corpus_manifest = _complete_manifest()
    token_manifest = tokenize_corpus_manifest(corpus_manifest, _FakeTokenizer())
    lock = {
        "pilot_identity": {
            "calibration_samples": 128,
            "corpus_manifest_sha256": token_manifest["corpus_manifest_sha256"],
            "held_out_samples": 32,
            "max_sequence_length": 2048,
            "model_id": token_manifest["model_id"],
            "model_revision": token_manifest["model_revision"],
            "target_layer": "layers.26",
            "token_manifest_sha256": "different",
        }
    }
    lock_path = tmp_path / "lock.json"
    lock_path.write_text(json.dumps(lock), encoding="utf-8")

    with pytest.raises(ValueError, match="identity does not match"):
        module.verify_pilot_identity(lock_path, token_manifest)


def test_projection_specific_recipe_is_layer26_only():
    module = _load_runner_module()
    recipe = module.projection_specific_recipe(iterations=200, device_ids="0,1")

    assert recipe.iters == 200
    assert recipe.device_ids == "0,1"
    assert recipe.disable_opt_rtn is True
    assert recipe.config_groups["layer26_gate_up"].weights.num_bits == 2
    assert recipe.config_groups["layer26_gate_up"].weights.group_size == 256
    assert recipe.config_groups["layer26_down"].weights.num_bits == 4
    assert recipe.config_groups["layer26_down"].weights.group_size == 128

    gate_up_targets = recipe.config_groups["layer26_gate_up"].targets
    down_targets = recipe.config_groups["layer26_down"].targets
    linear = torch.nn.Linear(1, 1)
    assert match_targets("layers.26.mlp.experts.127.gate_proj", linear, gate_up_targets)
    assert match_targets("layers.26.mlp.experts.127.up_proj", linear, gate_up_targets)
    assert match_targets("layers.26.mlp.experts.127.down_proj", linear, down_targets)
    assert not match_targets(
        "layers.25.mlp.experts.127.gate_proj", linear, gate_up_targets
    )
    assert not match_targets(
        "layers.26.mlp.shared_experts.gate_proj", linear, gate_up_targets
    )

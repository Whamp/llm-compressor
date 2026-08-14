# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Run restartable BF16/RTN and calibrated AutoRound layer-26 pilot arms."""

from __future__ import annotations

import argparse
import contextlib
import importlib
import importlib.metadata
import json
import resource
import subprocess
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any, cast

import torch
from compressed_tensors.quantization import QuantizationArgs, QuantizationScheme
from datasets import Dataset
from layer26_pilot_evidence import (
    HELD_OUT_SAMPLE_COUNT,
    TARGET_LAYER_PATH,
    LayerOutputError,
    atomic_write_json,
    sha256_file,
    sha256_json,
    summarize_layer_output_errors,
    validate_token_manifest,
)
from safetensors.torch import load_file, save_file
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modifiers.autoround import AutoRoundModifier, fix_batch_if_needed
from llmcompressor.utils import get_main_device, load_context

BF16_OUTPUT_TENSOR_COUNT = 36_599
TARGET_LAYER_INDEX = 26
AUTOROUND_ITERATIONS = 200


@contextlib.contextmanager
def mixed_device_bf16_autocast() -> Iterator[None]:
    """Autocast CPU-offloaded prefix work and CUDA target-layer work to BF16."""
    with contextlib.ExitStack() as stack:
        stack.enter_context(torch.autocast("cpu", dtype=torch.bfloat16))
        if torch.cuda.is_available():
            stack.enter_context(torch.autocast("cuda", dtype=torch.bfloat16))
        yield


def load_json(path: Path) -> dict:
    """Read one UTF-8 JSON evidence file."""
    with path.open(encoding="utf-8") as file_handle:
        return json.load(file_handle)


def is_allowed_dependency_revision(
    name: str,
    expected_commit: str,
    actual_commit: str,
    parent_commit: str,
    changed_paths: list[str],
) -> bool:
    """Allow an exact revision or llm-compressor's single lock-only child."""
    lock_only_path = "examples/autoround/deepseek_v4_0731/dependency-lock.json"
    return actual_commit == expected_commit or (
        name == "llm-compressor"
        and parent_commit == expected_commit
        and changed_paths == [lock_only_path]
    )


def verify_dependency_lock(
    lock_path: Path, repository_paths: dict[str, Path]
) -> dict[str, str]:
    """Require imported packages from clean repositories at exact revisions."""
    lock = load_json(lock_path)
    if lock.get("schema_version") != 1:
        raise ValueError("DeepSeek V4 pilot dependency lock schema is unsupported")
    expected_keys = {"compressed-tensors", "llm-compressor", "auto-round"}
    repositories = lock.get("repositories")
    if not isinstance(repositories, dict) or set(repositories) != expected_keys:
        raise ValueError("DeepSeek V4 pilot dependency lock has wrong repositories")
    if set(repository_paths) != expected_keys:
        raise ValueError("DeepSeek V4 pilot repository paths are incomplete")

    module_names = {
        "compressed-tensors": "compressed_tensors",
        "llm-compressor": "llmcompressor",
        "auto-round": "auto_round",
    }
    commits = {}
    for name, expected_commit in repositories.items():
        repository = repository_paths[name].resolve()
        actual_commit = subprocess.check_output(
            ["git", "-C", str(repository), "rev-parse", "HEAD"],
            text=True,
        ).strip()
        if actual_commit != expected_commit:
            parent_commit = subprocess.check_output(
                ["git", "-C", str(repository), "rev-parse", "HEAD^"],
                text=True,
            ).strip()
            changed_paths = subprocess.check_output(
                [
                    "git",
                    "-C",
                    str(repository),
                    "diff",
                    "--name-only",
                    f"{expected_commit}..{actual_commit}",
                ],
                text=True,
            ).splitlines()
            if not is_allowed_dependency_revision(
                name,
                expected_commit,
                actual_commit,
                parent_commit,
                changed_paths,
            ):
                raise ValueError(
                    f"DeepSeek V4 pilot {name} revision mismatch: "
                    f"{actual_commit} is not exact or an allowed lock-only child "
                    f"of {expected_commit}"
                )
        status = subprocess.check_output(
            ["git", "-C", str(repository), "status", "--porcelain"],
            text=True,
        )
        if status:
            raise ValueError(f"DeepSeek V4 pilot {name} repository is dirty")
        module_file = importlib.import_module(module_names[name]).__file__
        if module_file is None:
            raise ValueError(f"DeepSeek V4 pilot {name} import has no file path")
        module_path = Path(module_file).resolve()
        if not module_path.is_relative_to(repository):
            raise ValueError(
                f"DeepSeek V4 pilot {name} import is outside its pinned repository"
            )
        commits[name] = actual_commit

    required_versions = lock.get("required_versions", {})
    for package_name, expected_version in required_versions.items():
        actual_version = importlib.metadata.version(package_name)
        if actual_version != expected_version:
            raise ValueError(
                f"DeepSeek V4 pilot {package_name} version mismatch: "
                f"{actual_version} != {expected_version}"
            )
    return commits


def verify_pilot_identity(lock_path: Path, token_manifest: dict) -> None:
    """Bind the exact tokenized corpus and experiment shape to the lock."""
    expected = load_json(lock_path).get("pilot_identity")
    records = token_manifest.get("records", [])
    actual = {
        "calibration_samples": sum(
            record.get("partition") == "calibration" for record in records
        ),
        "corpus_manifest_sha256": token_manifest.get("corpus_manifest_sha256"),
        "held_out_samples": sum(
            record.get("partition") == "held_out" for record in records
        ),
        "max_sequence_length": token_manifest.get("max_sequence_length"),
        "model_id": token_manifest.get("model_id"),
        "model_revision": token_manifest.get("model_revision"),
        "target_layer": TARGET_LAYER_PATH,
        "token_manifest_sha256": token_manifest.get("token_manifest_sha256"),
    }
    if actual != expected:
        raise ValueError("DeepSeek V4 pilot identity does not match dependency lock")


def verify_bf16_checkpoint(model_path: Path) -> dict:
    """Reject a derived checkpoint with stale quantization metadata or inventory."""
    config_path = model_path / "config.json"
    index_path = model_path / "model.safetensors.index.json"
    if not config_path.is_file() or not index_path.is_file():
        raise ValueError("DeepSeek V4 pilot BF16 checkpoint metadata is incomplete")

    config = load_json(config_path)
    if "quantization_config" in config:
        raise ValueError("DeepSeek V4 pilot BF16 checkpoint remains quantized")
    if "expert_dtype" in config:
        raise ValueError("DeepSeek V4 pilot BF16 checkpoint retains expert_dtype")
    index = load_json(index_path)
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or len(weight_map) != BF16_OUTPUT_TENSOR_COUNT:
        count = len(weight_map) if isinstance(weight_map, dict) else None
        raise ValueError(
            f"DeepSeek V4 pilot BF16 tensor count is {count}; "
            f"expected {BF16_OUTPUT_TENSOR_COUNT}"
        )
    missing_shards = sorted(
        shard_name
        for shard_name in set(weight_map.values())
        if not (model_path / shard_name).is_file()
    )
    if missing_shards:
        raise ValueError(
            f"DeepSeek V4 pilot BF16 checkpoint is missing {missing_shards[0]}"
        )
    return {
        "config_sha256": sha256_file(config_path),
        "index_sha256": sha256_file(index_path),
        "tensor_count": len(weight_map),
        "shard_count": len(set(weight_map.values())),
    }


def build_token_dataset(token_manifest: dict, partition: str) -> Dataset:
    """Create an already-tokenized dataset from one immutable manifest partition."""
    records = [
        {
            "input_ids": record["input_ids"],
            "attention_mask": record["attention_mask"],
        }
        for record in token_manifest["records"]
        if record["partition"] == partition
    ]
    return Dataset.from_list(records).map(fix_batch_if_needed)


def projection_specific_recipe(iterations: int, device_ids: str) -> AutoRoundModifier:
    """Build the approved layer-26 W2 gate/up and W4 down recipe."""
    prefix = r"re:^layers\.26\.mlp\.experts\.\d+\."
    return AutoRoundModifier(
        iters=iterations,
        enable_torch_compile=True,
        batch_size=1,
        device_ids=device_ids,
        disable_opt_rtn=True,
        config_groups={
            "layer26_gate_up": QuantizationScheme(
                targets=[prefix + r"(gate|up)_proj$"],
                weights=QuantizationArgs(
                    num_bits=2,
                    strategy="group",
                    group_size=256,
                ),
            ),
            "layer26_down": QuantizationScheme(
                targets=[prefix + r"down_proj$"],
                weights=QuantizationArgs(
                    num_bits=4,
                    strategy="group",
                    group_size=128,
                ),
            ),
        },
    )


def load_layer26_prefix_model(
    model_path: Path,
    offload_folder: Path,
    cpu_memory: str,
) -> tuple[torch.nn.Module, float]:
    """Load genuine BF16 weights and retain only the prefix through layer 26."""
    load_start = time.perf_counter()
    with load_context():
        causal_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            dtype=torch.bfloat16,
            device_map="auto_offload",
            max_memory={"cpu": cpu_memory},
            offload_folder=offload_folder,
        )
    backbone = causal_model.model
    backbone.layers = torch.nn.ModuleList(
        list(backbone.layers[: TARGET_LAYER_INDEX + 1])
    )
    backbone.name_or_path = str(model_path)
    backbone.config._name_or_path = str(model_path)
    backbone.eval()
    del causal_model
    return backbone, time.perf_counter() - load_start


def capture_layer26_output(model: torch.nn.Module, sample: dict) -> torch.Tensor:
    """Run the genuine prefix and capture the BF16 layer-26 hidden state."""
    captured_outputs = []

    def capture_output(_module: torch.nn.Module, _args: tuple, output: Any) -> None:
        if not isinstance(output, torch.Tensor):
            raise TypeError("DeepSeek V4 pilot layer 26 returned a non-tensor output")
        captured_outputs.append(output.detach().to("cpu", dtype=torch.bfloat16))

    layers = cast(torch.nn.ModuleList, getattr(model, "layers"))
    layer = layers[TARGET_LAYER_INDEX]
    device = get_main_device()
    model_inputs = {
        key: torch.tensor([value], dtype=torch.long, device=device)
        for key, value in sample.items()
        if key in {"input_ids", "attention_mask"}
    }
    with (
        layer.register_forward_hook(capture_output),
        torch.inference_mode(),
        mixed_device_bf16_autocast(),
    ):
        model(**model_inputs, use_cache=False)
    if len(captured_outputs) != 1:
        raise ValueError(
            f"DeepSeek V4 pilot captured {len(captured_outputs)} layer outputs; "
            "expected exactly one"
        )
    output = captured_outputs[0]
    if output.dtype != torch.bfloat16:
        raise ValueError("DeepSeek V4 pilot layer output is not BF16")
    return output.contiguous()


def write_reference_outputs(
    model: torch.nn.Module,
    held_out_records: list[dict],
    reference_dir: Path,
) -> dict:
    """Write checksum-bound held-out BF16 layer outputs."""
    reference_dir.mkdir(parents=True, exist_ok=True)
    entries = []
    for index, record in enumerate(held_out_records):
        output = capture_layer26_output(model, record)
        output_path = reference_dir / f"layer26-reference-{index:03d}.safetensors"
        save_file({"layer26_output": output}, output_path)
        entries.append(
            {
                "index": index,
                "file": output_path.name,
                "sha256": sha256_file(output_path),
                "shape": list(output.shape),
                "dtype": str(output.dtype),
                "token_sha256": record["token_sha256"],
            }
        )
    payload = {
        "schema_version": 1,
        "target_layer": TARGET_LAYER_PATH,
        "sample_count": len(entries),
        "entries": entries,
    }
    manifest = {**payload, "manifest_sha256": sha256_json(payload)}
    atomic_write_json(reference_dir / "reference-manifest.json", manifest)
    return manifest


def load_reference_manifest(reference_dir: Path) -> dict:
    """Verify every held-out BF16 reference before candidate comparison."""
    manifest = load_json(reference_dir / "reference-manifest.json")
    claimed_hash = manifest.pop("manifest_sha256", None)
    if claimed_hash != sha256_json(manifest):
        raise ValueError("DeepSeek V4 pilot reference manifest checksum mismatch")
    if manifest.get("sample_count") != HELD_OUT_SAMPLE_COUNT:
        raise ValueError("DeepSeek V4 pilot reference sample count mismatch")
    for entry in manifest["entries"]:
        path = reference_dir / entry["file"]
        if not path.is_file() or sha256_file(path) != entry["sha256"]:
            raise ValueError(
                f"DeepSeek V4 pilot reference file mismatch: {entry['file']}"
            )
    return {**manifest, "manifest_sha256": claimed_hash}


def compare_candidate_outputs(
    model: torch.nn.Module,
    held_out_records: list[dict],
    reference_dir: Path,
) -> dict:
    """Compare candidate layer outputs with exact held-out BF16 references."""
    reference_manifest = load_reference_manifest(reference_dir)
    errors = []
    per_sample = []
    for entry, record in zip(reference_manifest["entries"], held_out_records):
        if entry["token_sha256"] != record["token_sha256"]:
            raise ValueError("DeepSeek V4 pilot held-out token identity mismatch")
        reference = load_file(reference_dir / entry["file"])["layer26_output"]
        candidate = capture_layer26_output(model, record)
        error = LayerOutputError.compare(reference, candidate)
        errors.append(error)
        per_sample.append(
            {
                "index": entry["index"],
                "token_sha256": entry["token_sha256"],
                **error.__dict__,
            }
        )
    return {
        "reference_manifest_sha256": reference_manifest["manifest_sha256"],
        "summary": summarize_layer_output_errors(errors),
        "per_sample": per_sample,
    }


def summarize_output_schema(model: torch.nn.Module) -> dict:
    """Prove all 256 experts expose the requested projection-specific schema."""
    counts: dict[str, int] = {}
    layers = cast(torch.nn.ModuleList, getattr(model, "layers"))
    layer = layers[TARGET_LAYER_INDEX]
    for name, module in layer.named_modules():
        scheme = getattr(module, "quantization_scheme", None)
        weights = getattr(scheme, "weights", None)
        if weights is None:
            continue
        key = f"W{weights.num_bits}_G{weights.group_size}"
        counts[key] = counts.get(key, 0) + 1
    expected = {"W2_G256": 512, "W4_G128": 256}
    if counts != expected:
        raise ValueError(
            f"DeepSeek V4 pilot output schema mismatch: {counts}; expected {expected}"
        )
    return {"target_layer": TARGET_LAYER_PATH, "module_counts": counts}


def resource_snapshot() -> dict:
    """Capture process and per-device peak allocation evidence."""
    gpu_peaks = []
    if torch.accelerator.is_available():
        device_module = torch.get_device_module()
        gpu_peaks = [
            {
                "device": index,
                "max_memory_allocated_bytes": torch.accelerator.max_memory_allocated(
                    index
                ),
                "max_memory_reserved_bytes": device_module.max_memory_reserved(index),
            }
            for index in range(torch.accelerator.device_count())
        ]
    return {
        "max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "gpu_peaks": gpu_peaks,
    }


def run_quantization(
    model: torch.nn.Module,
    tokenizer: Any,
    calibration_dataset: Dataset,
    iterations: int,
    device_ids: str,
) -> float:
    """Quantize exactly layer 26 through llm-compressor's public oneshot seam."""
    quantization_start = time.perf_counter()
    with mixed_device_bf16_autocast():
        oneshot(
            model=model,
            processor=tokenizer,
            dataset=calibration_dataset,
            recipe=projection_specific_recipe(iterations, device_ids),
            pipeline="sequential",
            sequential_targets=[TARGET_LAYER_PATH],
            sequential_targets_per_subgraph=1,
            batch_size=1,
            max_seq_length=2048,
            num_calibration_samples=128,
            shuffle_calibration_samples=False,
            moe_calibrate_all_experts=True,
            propagate_error=False,
            clear_sparse_session=True,
        )
    return time.perf_counter() - quantization_start


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the DeepSeek V4 layer-26 BF16/RTN or AutoRound pilot arm"
    )
    parser.add_argument("arm", choices=("baseline", "autoround"))
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--token-manifest", type=Path, required=True)
    parser.add_argument("--dependency-lock", type=Path, required=True)
    parser.add_argument("--compressed-tensors-repo", type=Path, required=True)
    parser.add_argument("--llm-compressor-repo", type=Path, required=True)
    parser.add_argument("--auto-round-repo", type=Path, required=True)
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--offload-folder", type=Path, required=True)
    parser.add_argument("--cpu-memory", default="340GiB")
    parser.add_argument("--device-ids", default="0,1")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_start = time.perf_counter()
    dependency_commits = verify_dependency_lock(
        args.dependency_lock,
        {
            "compressed-tensors": args.compressed_tensors_repo,
            "llm-compressor": args.llm_compressor_repo,
            "auto-round": args.auto_round_repo,
        },
    )
    checkpoint_evidence = verify_bf16_checkpoint(args.model_path)
    token_manifest = load_json(args.token_manifest)
    validate_token_manifest(token_manifest)
    verify_pilot_identity(args.dependency_lock, token_manifest)
    calibration_dataset = build_token_dataset(token_manifest, "calibration")
    held_out_records = [
        record
        for record in token_manifest["records"]
        if record["partition"] == "held_out"
    ]

    if torch.accelerator.is_available():
        for index in range(torch.accelerator.device_count()):
            torch.accelerator.reset_peak_memory_stats(index)

    model, load_seconds = load_layer26_prefix_model(
        args.model_path,
        args.offload_folder,
        args.cpu_memory,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    if args.arm == "baseline":
        reference_start = time.perf_counter()
        reference_manifest = write_reference_outputs(
            model,
            held_out_records,
            args.reference_dir,
        )
        reference_seconds = time.perf_counter() - reference_start
        iterations = 0
    else:
        reference_manifest = load_reference_manifest(args.reference_dir)
        reference_seconds = 0.0
        iterations = AUTOROUND_ITERATIONS

    quantization_seconds = run_quantization(
        model,
        tokenizer,
        calibration_dataset,
        iterations,
        args.device_ids,
    )
    output_schema = summarize_output_schema(model)
    comparison_start = time.perf_counter()
    comparison = compare_candidate_outputs(
        model,
        held_out_records,
        args.reference_dir,
    )
    comparison_seconds = time.perf_counter() - comparison_start

    report = {
        "schema_version": 1,
        "arm": args.arm,
        "algorithm": "plain_rtn" if args.arm == "baseline" else "autoround",
        "iterations": iterations,
        "target_layer": TARGET_LAYER_PATH,
        "dependency_commits": dependency_commits,
        "installed_versions": {
            name: importlib.metadata.version(name)
            for name in ("compressed-tensors", "llmcompressor", "auto-round")
        },
        "checkpoint": checkpoint_evidence,
        "token_manifest_sha256": token_manifest["token_manifest_sha256"],
        "reference_manifest_sha256": reference_manifest["manifest_sha256"],
        "output_schema": output_schema,
        "comparison": comparison,
        "timing_seconds": {
            "model_load": load_seconds,
            "reference_capture": reference_seconds,
            "quantization": quantization_seconds,
            "candidate_comparison": comparison_seconds,
            "total": time.perf_counter() - run_start,
        },
        "resources": resource_snapshot(),
    }
    report["report_sha256"] = sha256_json(report)
    atomic_write_json(args.report, report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

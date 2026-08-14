from contextlib import contextmanager

import llmcompressor.pipelines.sequential.pipeline as sequential_pipeline
from llmcompressor.args.dataset_arguments import DatasetArguments


def test_sequential_keep_onloaded_weights_defaults_true():
    assert DatasetArguments().sequential_keep_onloaded_weights is True


def test_sequential_weight_residency_context_honors_opt_out(monkeypatch):
    entries = []

    @contextmanager
    def record_disable_offloading():
        entries.append("entered")
        yield

    monkeypatch.setattr(
        sequential_pipeline, "disable_offloading", record_disable_offloading
    )

    with sequential_pipeline._sequential_weight_residency_context(False):
        pass
    assert entries == []

    with sequential_pipeline._sequential_weight_residency_context(True):
        pass
    assert entries == ["entered"]

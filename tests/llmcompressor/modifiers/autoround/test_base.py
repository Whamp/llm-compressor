from contextlib import nullcontext
from unittest.mock import MagicMock, patch

import pytest
import torch
from auto_round.schemes import PRESET_SCHEMES as AR_PRESET_SCHEMES
from auto_round.schemes import QuantizationScheme as ARQuantizationScheme
from compressed_tensors.offload import disable_onloading
from compressed_tensors.offload.cache.cpu import CPUCache
from compressed_tensors.offload.cache.disk import DiskCache
from compressed_tensors.offload.module import offload_module
from compressed_tensors.quantization import QuantizationArgs, QuantizationScheme
from torch import nn

from llmcompressor.core import Event, EventType, State
from llmcompressor.modifiers.autoround import AutoRoundModifier
from llmcompressor.modifiers.autoround.base import (
    _freeze_model_parameters,
    _wrap_decoding_layer,
    suspend_offloading,
)


class _FakeDecoderLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(64, 64)
        self.k_proj = nn.Linear(64, 64)


class _MixedFakeDecoderLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(128, 128)
        self.o_proj = nn.Linear(128, 128)
        self.up_proj = nn.Linear(128, 128)


class _NonLeafOffloadedParameterModel(nn.Module):
    def __init__(self):
        super().__init__()
        source = nn.Parameter(torch.ones(4, dtype=torch.float32))
        converted_parameter = source.to(torch.bfloat16)
        self._parameters = CPUCache.from_mapping(
            {"converted_parameter": converted_parameter},
            onload_device="cpu",
        )


def test_qparam_processing_preserves_unquantized_scale_parameter():
    module = nn.Module()
    expected_scale = nn.Parameter(torch.ones(3), requires_grad=False)
    module.register_parameter("scale", expected_scale)
    modifier = AutoRoundModifier()

    saved_qparams = modifier._preprocess_qparams(module)
    modifier._postprocess_qparams(module, saved_qparams)

    assert saved_qparams == {}
    assert module.scale is expected_scale


def test_suspend_offloading_restores_disk_cache_directory(tmp_path):
    module = nn.Linear(4, 4)
    offload_module(
        module,
        onload_device="cpu",
        offload_device="disk",
        offload_dir=tmp_path,
    )

    with suspend_offloading(module):
        assert not isinstance(module._parameters, DiskCache)

    assert isinstance(module._parameters, DiskCache)
    assert module._parameters.offload_dir == tmp_path.resolve()
    assert module.weight.shape == (4, 4)


def test_preprocess_qparams_reads_direct_disk_cache_values(tmp_path):
    module = nn.Linear(4, 4)
    offload_module(
        module,
        onload_device="cpu",
        offload_device="disk",
        offload_dir=tmp_path,
    )
    module.quantization_scheme = QuantizationScheme(
        targets=["Linear"],
        weights=QuantizationArgs(num_bits=4, strategy="group", group_size=4),
    )
    expected_scale = torch.ones(4, dtype=torch.float32)
    with disable_onloading():
        module.register_parameter(
            "weight_scale",
            nn.Parameter(expected_scale.clone(), requires_grad=False),
        )

    saved_qparams = AutoRoundModifier()._preprocess_qparams(module)

    assert torch.equal(saved_qparams[""]["weight_scale"], expected_scale)
    assert "weight_scale" not in module._parameters


def test_freeze_model_parameters_detaches_offload_cache_backing_tensor():
    model = _NonLeafOffloadedParameterModel()
    cache = model._parameters
    backing_parameter = cache.offloaded_values["converted_parameter"]
    assert not backing_parameter.is_leaf
    assert backing_parameter.requires_grad

    _freeze_model_parameters(model)

    assert backing_parameter.is_leaf
    assert not backing_parameter.requires_grad
    assert model.converted_parameter.is_leaf
    assert not model.converted_parameter.requires_grad


@pytest.mark.parametrize(
    "sequential_target",
    [
        "model.layers.26",
        r"re:^model\.layers\.26$",
        "_FakeDecoderLayer",
    ],
)
def test_sequential_target_selects_one_decoder_layer(sequential_target):
    modifier = AutoRoundModifier(
        ignore=["lm_head"],
        iters=10,
        scheme="W4A16",
    )
    selected_layer = _FakeDecoderLayer()
    selected_layer._tmp_name = "model.layers.26"
    unselected_layer = _FakeDecoderLayer()
    unselected_layer._tmp_name = "model.layers.25"
    modifier._sequential_targets = [sequential_target]

    assert modifier._is_decoding_layer(selected_layer)
    if sequential_target != "_FakeDecoderLayer":
        assert not modifier._is_decoding_layer(unselected_layer)


def test_on_sequential_epoch_end_passes_all_modules():
    """Verify that on_sequential_epoch_end passes all modules to apply_autoround
    without filtering. Regression test for a bug where an is_module_quantized
    filter silently dropped decoder layers, causing autoround to be a no-op."""
    modifier = AutoRoundModifier(
        ignore=["lm_head"],
        iters=10,
        scheme="W4A16",
    )
    state = MagicMock(spec=State)
    event = Event(type_=EventType.SEQUENTIAL_EPOCH_END)
    modules = [_FakeDecoderLayer(), nn.Linear(64, 64)]

    with (
        patch.object(AutoRoundModifier, "apply_autoround") as mock_apply,
        patch.object(AutoRoundModifier, "post_autoround_cleanup"),
    ):
        modifier.on_sequential_epoch_end(state, event, modules=modules)
        mock_apply.assert_called_once_with(state, modules)


@pytest.mark.parametrize(
    ("scheme_name", "expected_bits"),
    [
        ("W2A16", 2),
        ("W3A16", 3),
        ("W5A16", 5),
        ("W6A16", 6),
        ("W7A16", 7),
        ("w2a16", 2),
        ("w7a16", 7),
    ],
)
def test_mapping_config_to_autoround_supports_weight_only_wna16_schemes(
    scheme_name, expected_bits
):
    modifier = AutoRoundModifier(
        ignore=["lm_head"],
        iters=0,
        scheme=scheme_name,
    )

    mapped = modifier._mapping_config_to_autoround()

    if scheme_name.upper() in AR_PRESET_SCHEMES:
        assert mapped == scheme_name.upper()
    else:
        assert isinstance(mapped, ARQuantizationScheme)
        assert mapped.bits == expected_bits
        assert mapped.sym is True
        assert mapped.group_size == 128
        assert mapped.data_type == "int"
        assert mapped.act_bits == 16
        assert mapped.act_group_size is None
        assert mapped.act_sym is None
        assert mapped.act_dynamic is None
        assert mapped.act_data_type is None


def test_mapping_config_to_autoround_uses_fallback_for_w7a16():
    assert "W7A16" not in AR_PRESET_SCHEMES

    modifier = AutoRoundModifier(
        ignore=["lm_head"],
        iters=0,
        scheme="W7A16",
    )

    mapped = modifier._mapping_config_to_autoround()

    assert isinstance(mapped, ARQuantizationScheme)
    assert mapped.bits == 7
    assert mapped.group_size == 128


def test_build_layer_config_for_autoround_supports_mixed_weight_only_schemes():
    modifier = AutoRoundModifier(
        ignore=["lm_head"],
        iters=0,
        config_groups={
            "attention": QuantizationScheme(
                targets=["q_proj", "o_proj"],
                weights=QuantizationArgs(num_bits=2, strategy="group", group_size=128),
            ),
            "mlp": QuantizationScheme(
                targets=["up_proj"],
                weights=QuantizationArgs(num_bits=4, strategy="group", group_size=128),
            ),
        },
    )
    layer = _MixedFakeDecoderLayer()
    modifier.initialize_quantization(layer)

    wrapped = _wrap_decoding_layer(layer)
    layer_config = modifier._build_layer_config_for_autoround(wrapped)

    assert "model.layers.0.up_proj" in layer_config
    assert layer_config["model.layers.0.up_proj"]["bits"] == 4
    assert layer_config["model.layers.0.up_proj"]["group_size"] == 128
    assert "model.layers.0.q_proj" not in layer_config
    assert "model.layers.0.o_proj" not in layer_config


def test_build_layer_config_for_autoround_supports_mxfp4_activation_groups():
    modifier = AutoRoundModifier(
        ignore=["lm_head"],
        iters=0,
        scheme="MXFP4",
    )
    layer = _FakeDecoderLayer()
    modifier.initialize_quantization(layer)

    wrapped = _wrap_decoding_layer(layer)
    layer_config = modifier._build_layer_config_for_autoround(wrapped)

    assert layer_config == {}


def test_update_device_map_for_dp_uses_current_rank_device():
    modifier = AutoRoundModifier(ignore=["lm_head"], iters=0, scheme="W4A16")
    ar_kwargs = {}

    with (
        patch("torch.distributed.is_initialized", return_value=True),
        patch(
            "llmcompressor.modifiers.autoround.base.get_local_gpu_group_size",
            return_value=1,
        ),
        patch("torch.accelerator.is_available", return_value=True),
        patch("torch.accelerator.current_device_index", return_value=1),
        patch("torch.accelerator.current_accelerator") as mock_accelerator,
    ):
        mock_accelerator.return_value.type = "cuda"
        modifier._update_device_map_for_dp(ar_kwargs)

    assert ar_kwargs["device_map"] == "cuda:1"


def test_apply_autoround_passes_moved_inputs_to_quantize_block():
    modifier = AutoRoundModifier(ignore=["lm_head"], iters=0, scheme="W4A16")
    layer = _FakeDecoderLayer()
    layer._tmp_name = "decoder"
    modifier._sequential_targets = [layer.__class__.__name__]
    modifier._all_module_input[layer._tmp_name] = [((torch.ones(1),), {})]

    state = MagicMock(spec=State)
    state.model.name_or_path = "stub-model"
    state.model.config = MagicMock()

    autoround = MagicMock()
    autoround.quantize_block.return_value = (None, None)

    with (
        patch.object(
            AutoRoundModifier, "_mapping_config_to_autoround", return_value="W4A16"
        ),
        patch.object(
            AutoRoundModifier, "_build_layer_config_for_autoround", return_value={}
        ),
        patch.object(AutoRoundModifier, "get_unquantized_layer_names", return_value=[]),
        patch.object(AutoRoundModifier, "_preprocess_qparams", return_value={}),
        patch.object(AutoRoundModifier, "_postprocess_qparams"),
        patch.object(
            AutoRoundModifier, "_unwrapper_quantized_layer", side_effect=lambda m: m
        ),
        patch.object(AutoRoundModifier, "_update_device_map_for_dp"),
        patch(
            "llmcompressor.modifiers.autoround.base.align_module_device",
            return_value=nullcontext(),
        ),
        patch(
            "llmcompressor.modifiers.autoround.base.suspend_offloading",
            return_value=nullcontext(),
        ),
        patch(
            "llmcompressor.modifiers.autoround.base.get_local_gpu_group_size",
            return_value=2,
        ),
        patch(
            "llmcompressor.modifiers.autoround.base.get_main_device",
            return_value=torch.device("meta"),
        ),
        patch(
            "llmcompressor.modifiers.autoround.base.AutoRound", return_value=autoround
        ),
    ):
        modifier.apply_autoround(state, [layer])

    quantize_inputs = autoround.quantize_block.call_args.kwargs["inputs"]
    assert quantize_inputs[0][0][0][0].device.type == "meta"

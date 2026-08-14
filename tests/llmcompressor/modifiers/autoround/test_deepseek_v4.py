from types import SimpleNamespace

import pytest
import torch
from compressed_tensors.quantization import QuantizationArgs, QuantizationScheme
from torch.utils.data import DataLoader
from transformers.models.deepseek_v4.configuration_deepseek_v4 import (
    DeepseekV4Config,
)
from transformers.models.deepseek_v4.modeling_deepseek_v4 import (
    DeepseekV4ForCausalLM,
)

from llmcompressor import oneshot
from llmcompressor.modifiers.autoround import AutoRoundModifier


def _tiny_deepseek_v4_model(tmp_path):
    config = DeepseekV4Config(
        vocab_size=256,
        hidden_size=256,
        moe_intermediate_size=128,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=128,
        q_lora_rank=128,
        num_experts_per_tok=2,
        n_routed_experts=4,
        n_shared_experts=1,
        max_position_embeddings=512,
        layer_types=["compressed_sparse_attention"],
        mlp_layer_types=["moe"],
        o_groups=2,
        o_lora_rank=64,
        index_n_heads=4,
        index_head_dim=32,
        index_topk=8,
        sliding_window=16,
        num_nextn_predict_layers=0,
        partial_rotary_factor=0.25,
        compress_rates={
            "compressed_sparse_attention": 4,
            "heavily_compressed_attention": 8,
        },
        compress_rope_theta=10_000,
        hc_mult=2,
        dtype="bfloat16",
    )
    config.save_pretrained(tmp_path)
    model = DeepseekV4ForCausalLM(config).to(torch.bfloat16)
    model.name_or_path = str(tmp_path)
    model.config._name_or_path = str(tmp_path)
    return model


def _projection_specific_autoround_recipe():
    return AutoRoundModifier(
        iters=1,
        enable_torch_compile=False,
        batch_size=1,
        config_groups={
            "routed_gate_up": QuantizationScheme(
                targets=[r"re:^model\.layers\.0\.mlp\.experts\.\d+\.(gate|up)_proj$"],
                weights=QuantizationArgs(
                    num_bits=2,
                    strategy="group",
                    group_size=256,
                ),
            ),
            "routed_down": QuantizationScheme(
                targets=[r"re:^model\.layers\.0\.mlp\.experts\.\d+\.down_proj$"],
                weights=QuantizationArgs(
                    num_bits=4,
                    strategy="group",
                    group_size=128,
                ),
            ),
        },
    )


@pytest.mark.integration
def test_deepseek_v4_projection_specific_autoround_one_layer(tmp_path):
    """A real DeepSeek V4 layer completes calibrated W2/W4 AutoRound tuning."""
    model = _tiny_deepseek_v4_model(tmp_path)
    calibration_records = []
    for length in range(2, 11):
        input_ids = torch.arange(length, dtype=torch.long).unsqueeze(0)
        calibration_records.append(
            {"input_ids": input_ids, "attention_mask": torch.ones_like(input_ids)}
        )
    calibration_data = DataLoader(calibration_records, batch_size=None)
    processor = SimpleNamespace(save_pretrained=lambda *_args, **_kwargs: None)

    quantized_model = oneshot(
        model=model,
        processor=processor,
        dataset=calibration_data,
        recipe=_projection_specific_autoround_recipe(),
        pipeline="sequential",
        sequential_targets=["model.layers.0"],
        sequential_targets_per_subgraph=1,
        batch_size=1,
        max_seq_length=10,
        num_calibration_samples=9,
        shuffle_calibration_samples=False,
        moe_calibrate_all_experts=True,
        propagate_error=False,
        clear_sparse_session=True,
    )

    expert_schemes = {
        name: module.quantization_scheme.weights
        for name, module in quantized_model.named_modules()
        if ".experts." in name
        and getattr(module, "quantization_scheme", None) is not None
    }
    assert len(expert_schemes) == 12
    for name, weights in expert_schemes.items():
        expected = (4, 128) if name.endswith("down_proj") else (2, 256)
        assert (weights.num_bits, weights.group_size) == expected

# DeepSeek V4 0731 calibrated layer-26 pilot

This pilot answers one bounded question: does 200-iteration AutoRound calibration
reduce held-out decoder-layer output error versus matching plain RTN for the same
projection-specific WNA16 schema?

It uses the official pinned `deepseek-ai/DeepSeek-V4-Flash-0731` checkpoint, first
converted to genuine BF16 by `DeepseekV4Dequantizer`. Raw packed FP4/FP8 values must
never be passed to `oneshot` or described as full precision.

## Fixed experiment

| Field | Value |
| --- | --- |
| Source | `deepseek-ai/DeepSeek-V4-Flash-0731` |
| Revision | `7872f01b1d1fe23eabc4c98b48bffcef5a386062` |
| Decoder layer | `layers.26` |
| Gate/up | symmetric W2, group size 256 |
| Down | symmetric W4, group size 128 |
| AutoRound | 200 iterations, batch size 1, Torch compilation enabled |
| Baseline | same schema, `iters=0`, plain RTN (`disable_opt_rtn=True`) |
| Calibration | 128 sequences × at most 2,048 tokens |
| Held out | 32 disjoint sequences × at most 2,048 tokens |
| Corpus manifest | `5f8508ba96234f12a2bdfb7d8c1fa0ccebde4209c3a2f7fb1216c32bf0a1133a` |
| Token manifest | `646de977abf3677d71978bc31ad8bcb65660e53d33d7cd9628a2ee3f1c6784af` |

The corpus is split evenly across coding, tool use, reasoning, and general chat. It
contains 128,668 real tokens; 17 of 160 examples reach the 2,048-token truncation
ceiling. Shorter examples remain unpadded and retain their exact attention masks.
Every source repository and revision is pinned. The builder scans a fixed 4,096-row
prefix, deterministically selects rows by content identity, rejects the held-back
SuperJSON task phrases, and writes both rendered-text and exact-token receipts.

This is a layer-output method screen, not an end-to-end model quality result.

## Capacity contract

The source checkpoint is about 155 GiB and the BF16 output about 530 GiB. Use at
least 1 TB of local storage and keep the Hugging Face cache and BF16 output on that
same volume. Do not create a second source copy. The conversion uses one worker to
bound host memory.

The intended pilot host has two H200 141 GB GPUs and at least 370 GB host RAM.
`device_map="auto_offload"` deliberately maps weights only to CPU and disk, preserving
both GPUs for AutoRound's layer dispatch. With the default 340 GiB CPU limit, roughly
190 GiB of the 530 GiB BF16 checkpoint may also be materialized in the offload folder.
The runner retains only the model prefix ending at layer 26 after loading, but it still
needs the complete BF16 checkpoint because the model loader validates and resolves the
full index.

## 1. Reconstruct the exact source trees

Clone the three repositories next to one another, check out the commits in
[`dependency-lock.json`](dependency-lock.json), and install all three editable into
one Python 3.12 environment. Use Torch 2.13.0 with CUDA 13.0 for the intended H200
host. The runner rejects dirty repositories, wrong commits, and imports resolving
outside those repositories.

```shell
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python \
  --index-url https://download.pytorch.org/whl/cu130 \
  'torch==2.13.0' 'torchvision==0.28.0'
uv pip install --python .venv/bin/python \
  'datasets==5.0.1' 'safetensors==0.8.0' 'transformers==5.14.1' \
  -e ./compressed-tensors -e ./auto-round -e ./llm-compressor
```

Keep logs and generated evidence outside all three Git worktrees so the required
clean-tree check remains meaningful.

## 2. Convert the official source to BF16

Set all caches under the large local volume. The example assumes `/pilot`.

```shell
export HF_HOME=/pilot/hf-home
export HF_HUB_CACHE=/pilot/hf-home/hub
export HF_XET_CACHE=/pilot/hf-home/xet
mkdir -p /pilot/{bf16,evidence,offload,logs}
```

Run the converter from the pinned compressed-tensors checkout and retain its log:

```shell
script -qefc '.venv/bin/python compressed-tensors/examples/convert_checkpoint/deepseek_v4_0731_mixed_example.py' \
  /pilot/logs/convert.log
```

Move or configure the example's output as `/pilot/bf16/DeepSeek-V4-Flash-0731-BF16`.
After conversion, retain the pinned source revision, conversion log, derived
`config.json`, and `model.safetensors.index.json`, then delete the 155 GiB packed source
cache before model loading. Keeping it alongside the BF16 checkpoint and the expected
disk-offload materialization can exhaust a 1 TB boot volume. The arm runner then
requires:

- no `quantization_config`;
- no source-only `expert_dtype`;
- exactly 36,599 indexed output tensors;
- every indexed BF16 shard present.

## 3. Build immutable calibration manifests

Run this off the billable host when possible. Copy the resulting small directory to
the pilot host unchanged.

```shell
cd llm-compressor/examples/autoround/deepseek_v4_0731
../../../.venv/bin/python build_layer26_pilot_manifest.py \
  --output-dir /pilot/evidence/data \
  | tee /pilot/logs/build-manifest.log
```

The builder writes `corpus-manifest.json` and `token-manifest.json`. Both are
checksum-bound and include source row identities. Do not edit either file.

## 4. Run the BF16 reference plus RTN baseline

The baseline arm first writes the 32 held-out BF16 layer outputs, then applies plain
RTN to the same layer and measures element-weighted MSE, normalized MSE, cosine
similarity, and maximum absolute error.

```shell
python run_layer26_autoround_pilot.py baseline \
  --model-path /pilot/bf16/DeepSeek-V4-Flash-0731-BF16 \
  --token-manifest /pilot/evidence/data/token-manifest.json \
  --dependency-lock dependency-lock.json \
  --compressed-tensors-repo /pilot/src/compressed-tensors \
  --llm-compressor-repo /pilot/src/llm-compressor \
  --auto-round-repo /pilot/src/auto-round \
  --reference-dir /pilot/evidence/reference \
  --report /pilot/evidence/baseline-report.json \
  --offload-folder /pilot/offload/baseline \
  --cpu-memory 340GiB \
  --device-ids 0,1 \
  2>&1 | tee /pilot/logs/baseline.log
```

Do not begin the calibrated arm unless the baseline report exists, validates, and
reports exactly 512 W2/group-256 gate/up modules plus 256 W4/group-128 down modules.

## 5. Run calibrated AutoRound

Start this arm in a fresh process so it reloads unmodified BF16 weights.

```shell
python run_layer26_autoround_pilot.py autoround \
  --model-path /pilot/bf16/DeepSeek-V4-Flash-0731-BF16 \
  --token-manifest /pilot/evidence/data/token-manifest.json \
  --dependency-lock dependency-lock.json \
  --compressed-tensors-repo /pilot/src/compressed-tensors \
  --llm-compressor-repo /pilot/src/llm-compressor \
  --auto-round-repo /pilot/src/auto-round \
  --reference-dir /pilot/evidence/reference \
  --report /pilot/evidence/autoround-report.json \
  --offload-folder /pilot/offload/autoround \
  --cpu-memory 340GiB \
  --device-ids 0,1 \
  2>&1 | tee /pilot/logs/autoround.log
```

## 6. Produce the matched result

```shell
python summarize_layer26_pilot.py \
  --baseline-report /pilot/evidence/baseline-report.json \
  --autoround-report /pilot/evidence/autoround-report.json \
  --output /pilot/evidence/pilot-summary.json \
  | tee /pilot/logs/summarize.log
```

The summarizer refuses different dependency commits, checkpoint identities, token
manifests, BF16 references, layer paths, or output schemas. Preserve the five JSON
evidence files, dependency lock, conversion/model metadata, and logs before deleting
any rented host.

A lower held-out MSE is evidence for layer-26 block reconstruction only. It does not
justify full-checkpoint quantization until downstream task quality, runtime loading,
kernel dispatch, and end-to-end performance are separately demonstrated.

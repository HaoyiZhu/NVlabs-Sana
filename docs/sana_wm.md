<p align="center" style="border-radius: 10px">
  <img src="../asset/sana-wm-logo.png" width="70%" alt="SANA-WM Logo"/>
</p>

# 🌍 SANA-WM: Efficient Minute-Scale World Modeling with Hybrid Linear Diffusion Transformer

<div align="center">
  <a href="https://nvlabs.github.io/Sana/WM"><img src="https://img.shields.io/static/v1?label=Project&message=Github&color=blue&logo=github-pages"></a> &ensp;
  <a href="https://arxiv.org/abs/2605.15178"><img src="https://img.shields.io/static/v1?label=Arxiv&message=Sana-WM&color=red&logo=arxiv"></a> &ensp;
  <a href="https://huggingface.co/Efficient-Large-Model/SANA-WM_bidirectional"><img src="https://img.shields.io/static/v1?label=HF%20Weights&message=SANA-WM&color=yellow&logo=huggingface"></a> &ensp;
</div>

<div align="center">
  <video src="https://nvlabs.github.io/Sana/WM/media/videos/hero_reel_v9.mp4#t=1" autoplay playsinline controls muted loop width="90%" onloadedmetadata="this.currentTime=1;this.playbackRate=2"></video>
</div>

## 📽️ About SANA-WM

**SANA-WM** is an efficient 2.6 B-parameter open-source world model trained natively for one-minute video generation. It synthesises 720p, minute-scale videos with precise 6-DoF camera control, paired with an LTX-2 sink-bidirectional Euler refiner for high-fidelity decoding.

Core contributions:

- **Hybrid Linear Attention** — frame-wise Gated DeltaNet combined with softmax attention every $N$-th block for memory-efficient long-context modelling.
- **Dual-Branch Camera Control** — independent main and camera branches enable precise per-frame trajectory adherence (6 DoF).
- **Two-Stage Generation Pipeline** — a long-video refiner stitched on top of Stage-1 latents improves quality and temporal consistency.
- **Robust Annotation Pipeline** — metric-scale 6-DoF camera poses extracted from public corpora yield spatiotemporally consistent action supervision.

SANA-WM completes pre-training in 15 days on 64 H100s and generates a 60s 720p clip on a single GPU; the distilled variant runs on an RTX 5090 with NVFP4 quantisation.

> **Note** This release ships two inference paths: a **bidirectional** pipeline
> (full-sequence Stage 1 + sink-bidirectional refiner) and a **streaming**
> pipeline (chunk-causal distilled Stage 1 + chunk-causal refiner + causal-VAE
> decoder, all overlapped on three CUDA streams and written progressively to
> MP4). Streaming weights are released under
> [`SANA-WM_streaming`](https://huggingface.co/Efficient-Large-Model/SANA-WM_streaming).

## ⚙️ Environment Setup

```bash
bash ./environment_setup.sh sana
conda activate sana
```

## 🏃 Inference

All Stage-1 / Stage-2 weights, the VAE, and the LTX-2 Gemma text encoder are
fetched on first use from
[`Efficient-Large-Model/SANA-WM_bidirectional`](https://huggingface.co/Efficient-Large-Model/SANA-WM_bidirectional)
— no manual download required.

### Example 1 — image + prompt + action string

```bash
python inference_video_scripts/inference_sana_wm.py \
  --image      asset/sana_wm/demo_0.png \
  --prompt     asset/sana_wm/demo_0.txt \
  --action     "w-80,jw-40,w-40,lw-60,w-100" \
  --translation_speed 0.055 \
  --rotation_speed_deg 1.2 \
  --num_frames 321 \
  --output_dir results/sana_wm_demo
```

Action DSL: each segment is `<keys>-<frames>` joined by commas. Movement keys
`w` (forward), `a` (strafe left), `s` (back), `d` (strafe right) translate
on the world horizontal plane; rotation keys `i` (pitch up), `k` (pitch
down), `j` (yaw left), `l` (yaw right) act in the camera's local frame.
`none-N` holds the pose for `N` frames.

### Example 2 — image + prompt + camera trajectory (`.npy`)

```bash
python inference_video_scripts/inference_sana_wm.py \
  --image      asset/sana_wm/demo_0.png \
  --prompt     asset/sana_wm/demo_0.txt \
  --camera     asset/sana_wm/demo_0_pose.npy \
  --intrinsics asset/sana_wm/demo_0_intrinsics.npy \
  --num_frames 321 \
  --output_dir results/sana_wm_demo
```

`--camera` is a NumPy `.npy` of shape `(F, 4, 4)` (camera-to-world
matrices); `--intrinsics` is `.npy` of shape `(3, 3)`, `(F, 3, 3)`, or
`(4,) = (fx, fy, cx, cy)` in input-image pixels. If `--intrinsics` is
omitted we estimate it from `--image` with Pi3X and abort if the
resulting FOV is outside `[25°, 120°]`.

### Lower memory

For tight VRAM budgets, opt in to lazy-load + CPU offload:

```bash
... --offload_vae --offload_refiner
```

### Streaming inference

The streaming pipeline replaces all three full-sequence stages with chunk-causal
variants and emits one decoded chunk per AR block straight into a progressive
MP4. Stage 1 runs the 4-step distilled student (CFG-baked-in, runs at
`cfg_scale=1`), the refiner runs chunk-causal AR with a sliding KV window, and
the causal LTX-2 VAE decodes chunk-by-chunk.

Drop the streaming weights into `pretrained_models/sana_wm_streaming/` (DiT,
refiner, causal VAE, and YAML are all under the
[`SANA-WM_streaming`](https://huggingface.co/Efficient-Large-Model/SANA-WM_streaming)
HF repo) and run:

```bash
python inference_video_scripts/inference_sana_wm_streaming.py \
  --image       asset/sana_wm/demo_0.png \
  --prompt      asset/sana_wm/demo_0.txt \
  --action      "w-80,dw-40,w-80,aw-40" \
  --num_frames  241 \                       # 15 s @ 16 fps
  --intrinsics  asset/sana_wm/demo_0_intrinsics.npy \
  --output_dir  results/sana_wm_streaming
```

> **Low-precision / small-GPU inference.** Add `--stage1_precision` and
> `--refiner_precision` (`bf16` default, `fp8` for Hopper+, `fp4` for Blackwell) to
> cut peak memory (bf16 ~47 GB → fp4 ~29 GB) and stream realtime on an RTX 5090. See
> [Quantized Inference](#-quantized-inference-fp8--fp4) below.

Output lands at `results/sana_wm_streaming/<name>_streaming.mp4` and grows in
place — you can watch it while inference continues. Reaches **~0.93× realtime
on a single H100** after a one-time `torch.compile` warmup (~3 min cold, ~30 s
warm cache; the warmup amortises across runs that reuse the same shapes).

All speed-critical knobs are baked into the script as defaults — `torch.compile`
on the VAE decoder and refiner transformer (`max-autotune-no-cudagraphs` mode),
flash-only SDPA, Inductor `coordinate_descent_tuning` + `epilogue_fusion`, cuDNN
benchmark, and the expandable CUDA allocator. There is no slow/fast toggle; the
script is the fast config.

Overrides for advanced use:

- `--streaming_root <path>` — directory holding `sana_dit/`, `ltx2_causal_vae/`,
  `refiner_diffusers/`, `gemma3_12b/`, and the YAML (default
  `pretrained_models/sana_wm_streaming`).
- `--config / --model_path / --causal_vae_path / --refiner_root /
  --refiner_gemma_root` — point at non-default weight paths.
- `--num_frame_per_block` (default 3, must match the checkpoint's
  `chunk_size`), `--denoising_step_list` (default
  `"1000,960,889,727,0"`), `--refiner_block_size` (3), `--refiner_kv_max_frames`
  (11) — change the canonical recipe at your own quality risk.

## ⚡ Quantized Inference (fp8 / fp4)

Streaming inference supports **per-component low-precision compute** so you can trade
a little speed for a large memory reduction and run on smaller GPUs (e.g. an RTX
5090). The **stage-1 DiT** and the **LTX-2 refiner** can each run in `bf16` (default),
`fp8`, or `fp4`, chosen independently:

```bash
--stage1_precision  {bf16,fp8,fp4}    # stage-1 DiT
--refiner_precision {bf16,fp8,fp4}    # LTX-2 refiner
```

| precision | format | hardware | notes |
|-----------|--------|----------|-------|
| `bf16` | — (default) | any | reference quality, largest memory |
| `fp8`  | FP8 W8A8 (TE block scaling) | **Hopper (H100) or Blackwell** | broad HW support; ~half the weight memory of bf16 |
| `fp4`  | NVFP4 W4A4 | **Blackwell only** (sm_100 / sm_120) | lowest memory — the config that fits a 32 GB 5090 |

Quantization targets the self-attention, cross-attention and FFN linears, scoped per
transformer block (`fp8`/`fp4` use the same layers — only the numeric format differs).
Embedders and the camera branch stay in bf16.

### Requirements

fp8/fp4 need NVIDIA [Transformer Engine](https://github.com/NVIDIA/TransformerEngine)
≥ 2.x (for the `Float8BlockScaling` / `NVFP4BlockScaling` recipes). **`environment_setup.sh`
installs it by default** (skip with `SANA_SKIP_TE=1` for a faster bf16-only install).
If TE is missing, the precision flags exit early with this install hint:

```bash
pip install --no-build-isolation 'transformer_engine[pytorch]'   # CUDA toolkit from environment_setup.sh required to build
```

bf16 needs nothing beyond the standard install.

### Usage

```bash
# Hopper (H100): FP8
--stage1_precision fp8 --refiner_precision fp8
# Blackwell (GB200 / RTX 5090): NVFP4 (lowest memory)
--stage1_precision fp4 --refiner_precision fp4
```

> For realtime throughput, also enable the optimized kernels via
> `scripts/benchmark_sana_wm_5090_realtime.sh` (torch.compile VAE, flash-attn refiner,
> fp8 KV cache, etc.); the precision flags compose with it. On aarch64 hosts pass
> `--streaming_encoder libx264` (no NVENC).

### Benchmarks

**Speed = steady-state ×realtime of SANA-WM compute only** (`discard` mode excludes the
MP4 encode / host copy; the one-time `torch.compile` warmup and Pi3X estimation are
excluded). **Peak memory is hardware-independent** (identical tensors on H100 / GB200 /
5090). Two refiner KV-window settings:

**`--refiner_kv_max_frames 11` — default (quality)**

| stage-1 / refiner | peak GPU memory | H100 (Hopper) | GB200 (Blackwell) |
|-------------------|:---------------:|:-------------:|:-----------------:|
| bf16 / bf16       | **47.3 GB**     | 1.09×         | 1.27×             |
| fp8 / fp8         | **35.4 GB**     | ~1.00× *(est.)* | 1.16×           |
| fp4 / fp4         | **29.4 GB**     | n/a (Blackwell only) | 1.16×        |

**`--refiner_kv_max_frames 2` — peak speed / lowest memory**

| stage-1 / refiner | peak GPU memory | H100 (Hopper) | GB200 (Blackwell) |
|-------------------|:---------------:|:-------------:|:-----------------:|
| bf16 / bf16       | **37.4 GB**     | 1.26×         | 1.57×             |
| fp8 / fp8         | **25.4 GB**     | ~1.05× *(est.)* | 1.32×           |
| fp4 / fp4         | **25.0 GB**     | n/a (Blackwell only) | 1.25×        |

> ⚠️ **`kv_max_frames=2` degrades generation quality** — the small refiner KV window
> causes visible temporal **flicker**. It is faster / smaller, but the default **`11`**
> is recommended; `kv=2` is shown only to bound the speed/memory envelope.

Notes:
- Quantization is a **memory** optimization, not a speed one — the stage-1/refiner GEMMs
  are latency-bound, so the TE quant path is marginally *slower* than bf16 (bf16 is
  fastest). The payoff is the memory drop that lets it run on a 5090.
- At `kv=11`, **fp4/fp4 (29 GB) saves ~6 GB over fp8/fp8 (35 GB)** — that gap is the
  refiner weights (W4A4 vs W8A8); fp4 is what fits a 32 GB 5090. (At `kv=2` the two are
  closer, ~25 GB, because activation/VAE peaks dominate.)
- *(est.)* only the H100 fp8 cells are extrapolated (H100 bf16 × the GB200 fp8/bf16
  ratio); everything else is measured. fp8 W8A8 is a supported Hopper feature.

#### 🟢 Runs realtime on an RTX 5090 (32 GB)

bf16 (47 GB) does **not** fit a 5090. At the recommended `kv=11`, **`fp4/fp4` (29 GB)
fits** the 32 GB budget and runs **at/above realtime** on Blackwell (1.16× on GB200, the
5090's closest proxy):

```bash
# RTX 5090 (Blackwell), quality (recommended): ~29 GB, ~realtime
--stage1_precision fp4 --refiner_precision fp4 --refiner_kv_max_frames 11
```

(`fp8/fp8` at `kv=11` is ~35 GB — too large for a 5090; use it on Hopper/larger GPUs, or
mix `--stage1_precision fp8 --refiner_precision fp4` to keep the refiner small.)

### Picking a precision

- **Quality / any GPU with ≥48 GB:** `bf16` / `bf16` (default).
- **RTX 5090 (Blackwell, 32 GB):** `fp4` / `fp4` — the config that fits at `kv=11`.
- **Hopper (H100, ≥40 GB):** `fp8` / `fp8` (fp4 needs Blackwell; on H100 memory isn't the
  constraint, so fp8 is the quantized option).
- **Mixing is allowed:** e.g. `--stage1_precision fp8 --refiner_precision fp4` keeps the
  stage-1 backbone less noisy (fp8 flickers less than fp4 on stage-1) while the refiner
  fp4 does most of the memory saving.

> **Long rollouts (>~20 s):** the autoregressive backbone accumulates drift over very
> long clips (motion "jumps", independent of precision — present in bf16 too). Keep clips
> ≤ ~15–20 s or use gentler camera trajectories; this is a model-horizon property, not a
> quantization artifact.

## 🎛️ Argument Reference

| Argument | Format / Default |
|------------------------|----------------------------------------------------------------------------------------|
| `--image` | First-frame RGB image. Aspect-preserving resized + center-cropped to 704×1280. |
| `--prompt` | UTF-8 text file with the conditioning prompt. |
| `--camera` | `(F, 4, 4)` `.npy` camera-to-world matrices. Mutually exclusive with `--action`. |
| `--action` | WASD/IJKL DSL. Rolled out via `action_string_to_c2w` to a `(F+1, 4, 4)` trajectory. |
| `--translation_speed` | Per-frame translation magnitude (default `0.05`). |
| `--rotation_speed_deg` | Per-frame rotation magnitude in degrees (default `1.2`). |
| `--intrinsics` | Optional `.npy` of shape `(3, 3)`, `(F, 3, 3)`, or `(4,)`. Pi3X-estimated if omitted. |
| `--num_frames` | Total frames to generate (default `161`; the demos above use `321`). |
| `--fps` | Output mp4 frame rate (default `16`). |
| `--step` | Stage-1 DiT sampling steps (default `60`). |
| `--cfg_scale` | Classifier-free-guidance scale (default `5.0`). |
| `--flow_shift` | Override the scheduler's `inference_flow_shift`. |
| `--no_refiner` | Skip the LTX-2 refiner and decode Stage-1 latents with the Sana VAE (faster, lower quality). |
| `--refiner_root` | LTX-2 refiner root containing `transformer/` and `connectors/`. |
| `--no_action_overlay` | Skip the WASD + joystick overlay on the output video. |
| `--offload_vae` | Move the VAE to CPU between encode / decode steps. |
| `--offload_refiner` | Lazy-load the LTX-2 refiner only when needed; release afterwards. |
| `--sampling_algo` | `flow_euler_ltx` (default, bidirectional). For streaming use the dedicated `inference_sana_wm_streaming.py`. |
| `--stage1_precision` | Stage-1 DiT precision: `bf16` (default) / `fp8` (Hopper+) / `fp4` (Blackwell). See [Quantized Inference](#-quantized-inference-fp8--fp4). |
| `--refiner_precision` | LTX-2 refiner precision: `bf16` (default) / `fp8` / `fp4`. |

## 📁 HF Repository Layout

`Efficient-Large-Model/SANA-WM_bidirectional`:

| Component | Path | Size |
|------------------------------------|---------------------------------------------|-------:|
| Sana DiT (Stage 1) | `dit/sana_wm_1600m_720p.safetensors` | 10 GB |
| LTX-2 VAE (diffusers) | `vae/` | 2 GB |
| LTX-2 refiner (Stage 2) | `refiner/{transformer,connectors}/` | 38 GB |
| Gemma text encoder for the refiner | `refiner/text_encoder/` | 46 GB |
| Inference config | `config.yaml` | — |

`Efficient-Large-Model/SANA-WM_streaming` (streaming variant):

| Component | Path |
|------------------------------------|----------------------------------------------|
| Chunk-causal Sana DiT (distilled) | `sana_dit/model.pt` |
| Causal LTX-2 VAE | `ltx2_causal_vae/` |
| Chunk-causal LTX-2 refiner | `refiner_diffusers/{transformer,connectors}/` |
| Inference config | `sana_wm_streaming_1600m_720p.yaml` |

The Sana text encoder (`gemma-2-2b-it`) is fetched separately from
`Efficient-Large-Model/gemma-2-2b-it`.

## 📝 BibTeX

```bibtex
@misc{zhu2026sanawm,
      title={SANA-WM: Efficient Minute-Scale World Modeling with Hybrid Linear Diffusion Transformer},
      author={Haoyi Zhu and Haozhe Liu and Yuyang Zhao and Tian Ye and Junsong Chen and Jincheng Yu and Tong He and Song Han and Enze Xie},
      year={2026},
      eprint={2605.15178},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2605.15178},
}
```

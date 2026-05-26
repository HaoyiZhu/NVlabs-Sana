# Copyright 2024 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

"""Cached chunk-causal attention blocks for self-forcing KV-cache inference.

These modules wrap the existing ChunkCausal attention blocks, adding KV-cache
support for autoregressive (one-chunk-at-a-time) inference.  No existing
chunk-causal code is modified.

Two caching strategies are used depending on the attention type:

* **GDN blocks** (``CachedChunkCausalGDN``): Cache the forward-scan recurrent
  state ``(S_kv, S_z)`` — a pair of ``(B, H, D, D)`` and ``(B, H, D, 1)``
  matrices that encode all past context.  The backward scan runs only on the
  current chunk (per-chunk isolated, same as training).

* **Softmax blocks** (``CachedChunkCausalSoftmaxAttn``): Cache post-RoPE K, V
  tensors from past chunks and concatenate with the current chunk for SDPA.

Camera wrappers (``Cached*UCPESinglePathLiteLA``) mirror these strategies for
the camera branch:
  - GDN camera: cache the single-path delta-rule state ``cam_S_kv``.
  - Softmax camera: cache post-UCPE-transform K, V for SDPA concatenation.

Cache slot layout (10 slots per block, same structure as scheduler):

.. list-table::
   :header-rows: 1

   * - Slot
     - GDN blocks
     - Softmax blocks
   * - 0
     - S_kv state (B,H,D,D)
     - k post-RoPE (B,H,N,D)
   * - 1
     - S_z state (B,H,D,1)
     - v (B,H,N,D)
   * - 2
     - cam_S_kv state (B,H_c,D_c,D_c)
     - cam_k post-UCPE (B,H_c,N,D_c)
   * - 3
     - None
     - cam_v post-UCPE (B,H_c,N,D_c)
   * - 4
     - ShortConv K state (BS,K-1,C)
     - None (no conv for softmax)
   * - 5
     - tconv state (handled by CachedGLUMBConvTemp)
     - tconv state
   * - 6
     - type flag: tensor([1.0])
     - type flag: tensor([0.0])
   * - 7-9
     - reserved
     - reserved
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from fla.modules import ShortConvolution

from diffusion.model.registry import ATTENTION_BLOCKS

from .sana_camctrl_blocks import _maybe_drop_cam_branch, prepare_prope_fns
from .sana_gdn_blocks import (
    _HAS_FLEX_ATTENTION,
    GDN,
    ChunkCausalGDN,
    ChunkCausalSoftmaxAttn,
    _forward_softmax_attn,
    _get_chunk_causal_block_mask,
    flip_and_shift,
)

if _HAS_FLEX_ATTENTION:
    from .sana_gdn_blocks import _flex_attention

from .sana_gdn_camctrl_blocks import (
    ChunkCausalGDNUCPESinglePathLiteLA,
    _forward_cam_branch_softmax,
    _prepare_cam_qkv_softmax,
    _SoftmaxUCPESinglePathLiteLA,
)

# ---------------------------------------------------------------------------
# Cache slot indices (must match scheduler constants)
# ---------------------------------------------------------------------------

_SLOT_FWD_KV = 0
_SLOT_FWD_Z = 1
_SLOT_CAM = 2
_SLOT_CAM_AUX = 3
_SLOT_SHORTCONV = 4
_SLOT_TCONV = 5  # NOTE: CachedGLUMBConvTemp actually writes to kv_cache[-1] (slot 9), not slot 5!
_SLOT_TYPE_FLAG = 6

_TYPE_STATE = 1.0  # GDN: state-based cache
_TYPE_CONCAT = 0.0  # Softmax: concat-based cache


def _slice_rope_to_current_chunk(rotary_emb: torch.Tensor, current_n: int) -> torch.Tensor:
    """Slice rotary embedding freqs to the trailing `current_n` token positions.

    When ``sink_token=true``, upstream rope is built for sink + current chunk
    positions (covers ``frame_index.numel()`` frames). But q/k inside the
    cached chunk-causal attention only cover the current chunk — sink K is
    either pre-rotated in S_kv (linear attn) or pre-rotated in kv_cache K
    (softmax attn). Slicing the trailing portion of ``rotary_emb`` aligns it
    with current-chunk q/k. If sizes already match (e.g. rolling_rope path
    that generates rope only for the current chunk's frame range), this is a
    no-op.
    """
    rope_n = rotary_emb.shape[-2]
    if rope_n == current_n:
        return rotary_emb
    if rope_n < current_n:
        raise RuntimeError(
            f"rotary_emb has {rope_n} positions, smaller than current chunk's " f"{current_n}; cannot slice."
        )
    return rotary_emb[..., -current_n:, :]


# ---------------------------------------------------------------------------
# Helper: GDN forward scan from initial state
# ---------------------------------------------------------------------------


def _gdn_forward_scan_from_state(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_rot: torch.Tensor,
    k_rot: torch.Tensor,
    beta: torch.Tensor,
    decay: torch.Tensor,
    S_kv_init: torch.Tensor,
    S_z_init: torch.Tensor,
    T: int,
    S: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run forward GDN scan starting from cached recurrent state.

    Args:
        q, k, v, q_rot, k_rot: ``(B, H, D, N)`` where ``N = T * S``.
        beta: ``(B, H, T, S)`` or ``(B, H, T)``.
        decay: ``(B, H, T)``.
        S_kv_init: ``(B, H, D, D)`` — KV state from previous chunks.
        S_z_init: ``(B, H, D, 1)`` — Z state from previous chunks.
        T, S: frame and spatial counts for the current chunk.

    Returns:
        (num_fwd, den_fwd, S_kv_final, S_z_final)
        num_fwd: ``(B, H, D, N)``
        den_fwd: ``(B, H, 1, N)``
        S_kv_final, S_z_final: final states for next chunk.
    """
    B, H, D, N = q.shape

    def to_frame(x: torch.Tensor) -> torch.Tensor:
        return x.view(B, H, D, T, S).permute(0, 1, 3, 2, 4)  # (B,H,T,D,S)

    q_f = to_frame(q)
    k_f = to_frame(k)
    v_f = to_frame(v)
    q_rot_f = to_frame(q_rot)
    k_rot_f = to_frame(k_rot)

    if beta.ndim == 4:
        beta_f = beta.unsqueeze(3)  # (B,H,T,1,S)
    else:
        beta_f = beta.view(B, H, T, 1, 1)
    decay_f = decay.view(B, H, T, 1, 1)

    I = torch.eye(D, device=q.device, dtype=q.dtype).view(1, 1, 1, D, D)
    target_z = 1.0

    # Build transition matrices for all T frames in parallel.
    k_rot_beta = k_rot_f * beta_f
    W_kv = decay_f * (I - torch.matmul(k_rot_beta, k_rot_f.transpose(-1, -2)))
    U_kv = torch.matmul(v_f * beta_f, k_rot_f.transpose(-1, -2))

    k_beta = k_f * beta_f
    W_z = decay_f * (I - torch.matmul(k_beta, k_f.transpose(-1, -2)))
    U_z = target_z * k_beta.sum(dim=-1, keepdim=True)

    # Recurrent scan from initial state (T is small: 3-4 frames).
    S_kv = S_kv_init
    S_z = S_z_init
    out_kv: list[torch.Tensor] = []
    out_z: list[torch.Tensor] = []
    for t in range(T):
        S_kv = torch.matmul(S_kv, W_kv[:, :, t]) + U_kv[:, :, t]
        S_z = torch.matmul(W_z[:, :, t], S_z) + U_z[:, :, t]
        out_kv.append(S_kv)
        out_z.append(S_z)

    S_kv_all = torch.stack(out_kv, dim=2)  # (B, H, T, D, D)
    S_z_all = torch.stack(out_z, dim=2)  # (B, H, T, D, 1)

    # Output projection.
    num = torch.matmul(S_kv_all, q_rot_f)  # (B, H, T, D, S)
    # Element-wise formulation avoids cuBLAS GEMV batch-size sensitivity (see
    # torch_chunk_sana_gdn for detailed comment).
    den = (S_z_all * q_f).sum(dim=-2, keepdim=True)  # (B, H, T, 1, S)

    def to_flat(t: torch.Tensor, d: int) -> torch.Tensor:
        return t.permute(0, 1, 3, 2, 4).reshape(B, H, d, N)

    return to_flat(num, D), to_flat(den, 1), S_kv, S_z


# ---------------------------------------------------------------------------
# Helper: GDN backward scan for a single chunk
# ---------------------------------------------------------------------------


def _gdn_backward_scan_single_chunk(
    module: GDN,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_rot: torch.Tensor,
    k_rot: torch.Tensor,
    beta: torch.Tensor,
    decay: torch.Tensor,
    T: int,
    S: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Backward (anti-causal) GDN scan for one isolated chunk.

    Mirrors ``ChunkCausalGDN.forward``'s backward scan but for a single chunk
    with no boundary masking needed.

    Args:
        module: GDN instance (provides ``update_rule_func``, ``recall_gate``, ``eps``).
        q, k, v, q_rot, k_rot: ``(B, H, D, N)`` where ``N = T * S``.
        beta: ``(B, H, T, S)`` or ``(B, H, T)``.
        decay: ``(B, H, T)``.

    Returns:
        (num_bwd, den_bwd) each ``(B, H, D, N)`` and ``(B, H, 1, N)``.
    """
    B, H, D, N = q.shape
    recall_gate = module.recall_gate
    if getattr(module, "fp32_attention", True):
        recall_gate = recall_gate.float()

    def to_time(t: torch.Tensor) -> torch.Tensor:
        return t.view(B, H, D, T, S).permute(0, 1, 3, 2, 4)

    def from_time(t: torch.Tensor) -> torch.Tensor:
        return t.permute(0, 1, 3, 2, 4).reshape(B, H, t.shape[3], N)

    q_T = to_time(q)
    k_T = to_time(k)
    v_T = to_time(v)
    q_rot_T = to_time(q_rot)
    k_rot_T = to_time(k_rot)

    # Flip for backward.
    q_bwd = torch.flip(q_T, dims=[2])
    q_rot_bwd = torch.flip(q_rot_T, dims=[2])

    k_bwd = flip_and_shift(k_T, dim=2, shift_val=0.0)
    v_bwd = flip_and_shift(v_T, dim=2, shift_val=0.0)
    k_rot_bwd = flip_and_shift(k_rot_T, dim=2, shift_val=0.0)
    beta_bwd = flip_and_shift(beta, dim=2, shift_val=0.0)
    decay_bwd = flip_and_shift(decay, dim=2, shift_val=1.0)

    # Run backward scan using the module's update rule.
    num_bwd_flipped, den_bwd_flipped = module.update_rule_func(
        from_time(q_bwd),
        from_time(k_bwd),
        from_time(v_bwd),
        from_time(q_rot_bwd),
        from_time(k_rot_bwd),
        beta_bwd,
        decay_bwd,
        recall_gate=recall_gate,
        eps=module.eps,
        return_components=True,
    )

    # Flip back.
    def flip_back(tensor: torch.Tensor) -> torch.Tensor:
        d_actual = tensor.shape[2]
        t_struct = tensor.view(B, H, d_actual, T, S)
        return torch.flip(t_struct, dims=[3]).reshape(B, H, d_actual, N)

    return flip_back(num_bwd_flipped), flip_back(den_bwd_flipped)


# ---------------------------------------------------------------------------
# Helper: Camera single-path forward scan from initial state
# ---------------------------------------------------------------------------


def _cam_forward_scan_from_state(
    q_rot: torch.Tensor,
    k_rot: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    decay: torch.Tensor,
    cam_S_kv_init: torch.Tensor,
    T: int,
    S: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Numerator-only delta-rule recurrence from cached state.

    Args:
        q_rot, k_rot, v: ``(B, H, D, N)`` — UCPE-transformed camera tensors.
        beta: ``(B, H, T, S)`` or ``(B, H, T)``.
        decay: ``(B, H, T)``.
        cam_S_kv_init: ``(B, H, D, D)`` — camera state from previous chunks.
        T, S: frame / spatial counts.

    Returns:
        (out_fwd, cam_S_kv_final)
        out_fwd: ``(B, H, D, N)`` — forward scan output.
    """
    B, H, D, N = q_rot.shape

    def to_frame(x: torch.Tensor) -> torch.Tensor:
        return x.view(B, H, D, T, S).permute(0, 1, 3, 2, 4)

    q_rot_f = to_frame(q_rot)
    k_rot_f = to_frame(k_rot)
    v_f = to_frame(v)

    if beta.ndim == 4:
        beta_f = beta.unsqueeze(3)
    else:
        beta_f = beta.view(B, H, T, 1, 1)
    decay_f = decay.view(B, H, T, 1, 1)

    I = torch.eye(D, device=q_rot.device, dtype=q_rot.dtype).view(1, 1, 1, D, D)

    # Transition matrices (same algebra as torch_chunk_cam_single_path_delta_rule).
    k_rot_beta = k_rot_f * beta_f
    W_kv = decay_f * (I - torch.matmul(k_rot_beta, k_rot_f.transpose(-1, -2)))
    U_kv = torch.matmul(v_f * beta_f, k_rot_f.transpose(-1, -2))

    state = cam_S_kv_init
    out_list: list[torch.Tensor] = []
    for t in range(T):
        state = torch.matmul(state, W_kv[:, :, t]) + U_kv[:, :, t]
        out_list.append(torch.matmul(state, q_rot_f[:, :, t]))

    out = torch.stack(out_list, dim=2)  # (B, H, T, D, S)
    out_flat = out.permute(0, 1, 3, 2, 4).reshape(B, H, D, N)
    return out_flat, state


# ---------------------------------------------------------------------------
# Helper: Camera backward scan for single chunk
# ---------------------------------------------------------------------------


def _cam_backward_scan_single_chunk(
    module: object,
    q_cam_trans: torch.Tensor,
    k_cam_trans: torch.Tensor,
    v_cam_trans: torch.Tensor,
    beta: torch.Tensor,
    decay: torch.Tensor,
    T: int,
    S: int,
) -> torch.Tensor:
    """Per-chunk isolated backward scan for camera single-path delta rule.

    Args:
        module: Camera wrapper instance (provides ``_run_cam_single_path``).
        q_cam_trans, k_cam_trans, v_cam_trans: ``(B, H, D, N)``.
        beta, decay: frame-level gates.

    Returns:
        out_bwd: ``(B, H, D, N)`` — backward scan output.
    """
    B = q_cam_trans.shape[0]
    H_heads = q_cam_trans.shape[1]
    D_head = q_cam_trans.shape[2]
    N = q_cam_trans.shape[3]

    def to_time(t: torch.Tensor) -> torch.Tensor:
        return t.view(B, H_heads, D_head, T, S).permute(0, 1, 3, 2, 4)

    q_rot_T = to_time(q_cam_trans)
    k_rot_T = to_time(k_cam_trans)
    v_T = to_time(v_cam_trans)

    # Flip for backward.
    q_bwd = torch.flip(q_rot_T, dims=[2])
    k_bwd = flip_and_shift(k_rot_T, dim=2, shift_val=0.0)
    v_bwd = flip_and_shift(v_T, dim=2, shift_val=0.0)
    beta_bwd = flip_and_shift(beta, dim=2, shift_val=0.0)
    decay_bwd = flip_and_shift(decay, dim=2, shift_val=1.0)

    chunk_N = T * S

    def _flat(t: torch.Tensor) -> torch.Tensor:
        return t.permute(0, 1, 3, 2, 4).reshape(B, H_heads, D_head, chunk_N)

    out_bwd_flat = module._run_cam_single_path(
        _flat(q_bwd),
        _flat(k_bwd),
        _flat(v_bwd),
        beta_bwd,
        decay_bwd,
    )
    out_bwd_t = out_bwd_flat.view(B, H_heads, D_head, T, S)
    out_bwd = torch.flip(out_bwd_t, dims=[3]).reshape(B, H_heads, D_head, N)
    return out_bwd


# ---------------------------------------------------------------------------
# Helper: Cached temporal short convolution
# ---------------------------------------------------------------------------


def _cached_temporal_short_conv(
    x: torch.Tensor,
    conv: ShortConvolution,
    HW: tuple[int, int, int],
    conv_cache: torch.Tensor | None,
    save_cache: bool,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Short conv with cached left context: forward-cached + backward-isolated.

    Mirrors ``ChunkCausalGDN._apply_temporal_short_conv`` but replaces the
    global forward causal conv with a cache-aware version.

    Uses the same ``ShortConvolution.forward()`` (Triton/CUDA backend) as the
    training path for bit-exact numerical parity.  The only difference is that
    the forward causal pass prepends cached left context instead of starting
    from zeros.

    Args:
        x: ``(B, N, C)`` where ``N = T * S``.
        conv: FLA ``ShortConvolution`` (depthwise causal Conv1d).
        HW: ``(T, H, W)``.
        conv_cache: ``(B*S, K-1, C)`` from previous chunk, or None.
        save_cache: Whether to return a new cache for the next chunk.

    Returns:
        (output, new_cache): output ``(B, N, C)``, new_cache ``(B*S, K-1, C)`` or None.
    """
    T, H, W = HW
    S = H * W
    B_orig, N, C = x.shape
    dtype_in = x.dtype
    K = conv.weight.shape[-1]

    # Reshape to temporal: (B*S, T, C).
    x_t = x.reshape(B_orig, T, S, C).permute(0, 2, 1, 3).contiguous().reshape(B_orig * S, T, C)

    # --- Forward causal conv with cache ---
    # Use ShortConvolution.forward() (Triton/CUDA kernel) for exact numerical
    # parity with ChunkCausalGDN._apply_temporal_short_conv.
    if conv_cache is not None:
        # Prepend cached left context and run full causal conv, then slice.
        x_fwd_in = torch.cat([conv_cache.to(x_t.dtype), x_t], dim=1)
        y_fwd_full, _ = conv(x_fwd_in)
        y_fwd = y_fwd_full[:, K - 1 :, :]  # drop positions from cached prefix
    else:
        y_fwd, _ = conv(x_t)

    # --- Backward conv (isolated within current chunk) ---
    # Same as ChunkCausalGDN._backward_causal_conv_per_chunk for a single chunk:
    # flip → causal conv → flip back.
    y_bwd_flipped, _ = conv(x_t.flip(1))
    y_bwd = y_bwd_flipped.flip(1)  # (B*S, T, C)

    # --- Center tap ---
    w_center = conv.weight[:, 0, -1]  # (C,)
    center_term = x_t * w_center.unsqueeze(0).unsqueeze(0)

    y = y_fwd + y_bwd - center_term

    # Save cache: last K-1 timesteps of the conv INPUT (for next chunk's left context).
    new_cache: torch.Tensor | None = None
    if save_cache and K > 1:
        new_cache = x_t[:, -(K - 1) :, :].detach().clone()

    # Reshape back to (B, N, C).
    y = y.reshape(B_orig, S, T, C).permute(0, 2, 1, 3).reshape(B_orig, N, C)
    if y.dtype != dtype_in:
        y = y.to(dtype_in)
    return y, new_cache


# ===================================================================
# CachedChunkCausalGDN
# ===================================================================


@ATTENTION_BLOCKS.register_module()
class CachedChunkCausalGDN(ChunkCausalGDN):
    """Cached chunk-causal GDN for self-forcing inference.

    When ``kv_cache`` is present in kwargs, runs a state-based cached forward
    scan instead of the full-sequence scan.  The backward scan runs only on
    the current chunk (per-chunk isolated).

    When ``kv_cache`` is absent, delegates to ``ChunkCausalGDN.forward``.
    """

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        HW: tuple[int, int, int] | None = None,
        rotary_emb: torch.Tensor | None = None,
        block_mask: torch.Tensor | None = None,
        apply_output_gate: bool = True,
        chunk_size: int | None = None,
        chunk_split_strategy: str = "uniform",
        chunk_index: list[int] | None = None,
        **kwargs: object,
    ) -> torch.Tensor | tuple[torch.Tensor, list]:
        kv_cache = kwargs.get("kv_cache", None)
        save_kv_cache = kwargs.get("save_kv_cache", False)

        if kv_cache is None:
            # Training / non-cached: delegate to parent.
            return super().forward(
                x,
                mask=mask,
                HW=HW,
                rotary_emb=rotary_emb,
                block_mask=block_mask,
                apply_output_gate=apply_output_gate,
                chunk_size=chunk_size,
                chunk_split_strategy=chunk_split_strategy,
                chunk_index=chunk_index,
                **kwargs,
            )

        # ---- Cached inference path ----
        del mask, block_mask
        if HW is None:
            raise ValueError("HW (T, H, W) must be provided.")

        B, N, C = x.shape
        T, H, W = HW
        S = H * W

        # 1. QKV projection.
        qkv = self.qkv(x).reshape(B, N, 3, self.heads, self.dim)
        q, k, v = qkv.unbind(2)

        # 2. Short conv on K (with cache).
        if self.conv_k is not None:
            k_flat = k.reshape(B, N, C)
            k_flat, new_conv_cache = _cached_temporal_short_conv(
                k_flat, self.conv_k, HW, kv_cache[_SLOT_SHORTCONV], save_kv_cache
            )
            k = k_flat.reshape(B, N, self.heads, self.dim)
            if save_kv_cache:
                kv_cache[_SLOT_SHORTCONV] = new_conv_cache

        # 3. QK norm + ReLU kernel + K scaling.
        q = self.q_norm(q.reshape(B, N, C)).reshape(B, N, self.heads, self.dim)
        k = self.k_norm(k.reshape(B, N, C)).reshape(B, N, self.heads, self.dim)
        q = self.kernel_func(q)
        k = self.kernel_func(k)
        k_scale = (self.dim**-0.5) * (S**-0.5)
        k = k * k_scale

        # 4. Permute to (B, H, D, N).
        q = q.permute(0, 2, 3, 1)
        k = k.permute(0, 2, 3, 1)
        v = v.permute(0, 2, 3, 1)

        # 5. RoPE (applied to current chunk positions only).
        # When sink_token is enabled, upstream rope is built for sink + current
        # frame positions (covers `frame_index.numel()` frames). Linear-attn
        # cache stores S_kv as RNN state post-rope — sink's contribution is
        # already pre-rotated. q/k here are current-chunk-only, so slice the
        # last `q.shape[-1]` positions out of rotary_emb.
        if rotary_emb is not None:
            rotary_emb_cur = _slice_rope_to_current_chunk(rotary_emb, q.shape[-1])
            q_rot = self._apply_rotary_emb(q, rotary_emb_cur)
            k_rot = self._apply_rotary_emb(k, rotary_emb_cur)
        else:
            q_rot = q
            k_rot = k

        # 6. Frame gates.
        precomputed_gates = kwargs.get("precomputed_gates", None)
        if precomputed_gates is not None:
            beta, decay = precomputed_gates
        else:
            beta, decay = self._compute_frame_gates(x, HW)

        # 7. Cast to FP32 for recurrence stability.
        dtype_orig = x.dtype
        recall_gate = self.recall_gate
        if getattr(self, "fp32_attention", True):
            q = q.float()
            k = k.float()
            v = v.float()
            q_rot = q_rot.float()
            k_rot = k_rot.float()
            beta = beta.float()
            decay = decay.float()
            recall_gate = recall_gate.float()

        # 8. Get cached forward-scan state (or start from zeros).
        S_kv_prev = kv_cache[_SLOT_FWD_KV]
        S_z_prev = kv_cache[_SLOT_FWD_Z]
        _is_first_chunk = S_kv_prev is None

        # 9. Forward scan — always use the same @torch.compile path
        #    (update_rule_func = torch_chunk_sana_gdn) for both output AND
        #    state extraction.  This guarantees bit-exact parity with the
        #    non-cached ChunkCausalGDN.forward baseline for all chunks.
        scan_kwargs: dict = dict(
            recall_gate=recall_gate,
            eps=self.eps,
            return_components=True,
            return_state=save_kv_cache,
        )
        if not _is_first_chunk:
            scan_kwargs["S_kv_init"] = S_kv_prev.to(q.dtype)
            scan_kwargs["S_z_init"] = S_z_prev.to(q.dtype)

        scan_out = self.update_rule_func(
            q,
            k,
            v,
            q_rot,
            k_rot,
            beta,
            decay,
            **scan_kwargs,
        )

        if save_kv_cache:
            num_fwd, den_fwd, S_kv_final, S_z_final = scan_out
            kv_cache[_SLOT_FWD_KV] = S_kv_final.detach().clone()
            kv_cache[_SLOT_FWD_Z] = S_z_final.detach().clone()
            kv_cache[_SLOT_TYPE_FLAG] = torch.tensor([_TYPE_STATE], device=x.device)
        else:
            num_fwd, den_fwd = scan_out

        # 10. Backward scan (per-chunk isolated).
        num_bwd, den_bwd = _gdn_backward_scan_single_chunk(self, q, k, v, q_rot, k_rot, beta, decay, T, S)

        # 11. Combine forward + backward.
        total_num = num_fwd + num_bwd
        total_den = den_fwd + den_bwd
        out = total_num / (total_den + self.eps)

        if getattr(self, "fp32_attention", True) and dtype_orig != torch.float32:
            out = out.to(dtype_orig)

        out = out.permute(0, 3, 1, 2).reshape(B, N, C)

        if apply_output_gate:
            out = self._apply_output_gate(out, x)
            out = self.proj(out.to(self.proj.weight.dtype))
            return out, kv_cache
        # When called from camera wrapper (apply_output_gate=False), return raw.
        return out, kv_cache


# ===================================================================
# Shared helper: SDPA with optional chunk-causal masking
# ===================================================================


def _sdpa_maybe_chunk_causal(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    need_chunk_mask: bool,
    T: int,
    S: int,
    chunk_size: int | None,
    chunk_index: list[int] | None,
    chunk_split_strategy: str,
    device: torch.device,
) -> torch.Tensor:
    """Run SDPA with chunk-causal masking when needed, or plain SDPA otherwise.

    Replicates the exact masking logic from ``_forward_softmax_attn`` and
    ``_forward_cam_branch_softmax`` so that the cached path produces
    bit-exact results for the first chunk (no cached state).
    """
    if need_chunk_mask:
        from diffusion.utils.chunk_utils import normalize_chunk_index

        chunk_boundaries, _ = normalize_chunk_index(
            chunk_index,
            T,
            chunk_size,
            chunk_split_strategy,
        )
        q_len = T * S
        kv_len = T * S

        if _HAS_FLEX_ATTENTION:
            block_mask = _get_chunk_causal_block_mask(
                chunk_boundaries,
                S,
                q_len,
                kv_len,
                0,
                device,
            )
            q_pad = (128 - q_len % 128) % 128
            kv_pad = (128 - kv_len % 128) % 128
            if q_pad > 0:
                q = F.pad(q, (0, 0, 0, q_pad))
            if kv_pad > 0:
                k = F.pad(k, (0, 0, 0, kv_pad))
                v = F.pad(v, (0, 0, 0, kv_pad))
            out = _flex_attention(q, k, v, block_mask=block_mask)
            if q_pad > 0:
                out = out[:, :, :q_len, :]
            return out

        # Fallback: per-chunk loop with head_dim padding for FlashAttention.
        D = q.shape[-1]
        _need_pad = D not in (32, 64, 128, 256) and D < 256
        if _need_pad:
            _pad_to = 128 if D <= 128 else 256
            _pad_size = _pad_to - D
            q = F.pad(q, (0, _pad_size))
            k = F.pad(k, (0, _pad_size))
            v = F.pad(v, (0, _pad_size))
        out_chunks: list[torch.Tensor] = []
        for ci in range(len(chunk_boundaries) - 1):
            c_start = chunk_boundaries[ci]
            c_end = chunk_boundaries[ci + 1]
            q_chunk = q[:, :, c_start * S : c_end * S, :]
            out_chunk = F.scaled_dot_product_attention(
                q_chunk,
                k[:, :, : c_end * S, :],
                v[:, :, : c_end * S, :],
            )
            out_chunks.append(out_chunk)
        out = torch.cat(out_chunks, dim=2)
        if _need_pad:
            out = out[..., :D]
        return out

    # Standard path: full SDPA (all cached tokens are causally prior).
    D = q.shape[-1]
    _need_pad = D not in (32, 64, 128, 256) and D < 256
    if _need_pad:
        _pad_to = 128 if D <= 128 else 256
        _pad_size = _pad_to - D
        q = F.pad(q, (0, _pad_size))
        k = F.pad(k, (0, _pad_size))
        v = F.pad(v, (0, _pad_size))
    out = F.scaled_dot_product_attention(q, k, v)
    if _need_pad:
        out = out[..., :D]
    return out


# ===================================================================
# CachedChunkCausalSoftmaxAttn
# ===================================================================


@ATTENTION_BLOCKS.register_module()
class CachedChunkCausalSoftmaxAttn(ChunkCausalSoftmaxAttn):
    """Cached chunk-causal softmax attention for self-forcing inference.

    Caches post-RoPE K, V from past chunks.  During inference, prepends
    cached K, V to the current chunk for full-history SDPA (no mask needed
    since all cached tokens are causally prior).
    """

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        HW: tuple[int, int, int] | None = None,
        rotary_emb: torch.Tensor | None = None,
        block_mask: torch.Tensor | None = None,
        apply_output_gate: bool = True,
        chunk_size: int | None = None,
        chunk_split_strategy: str = "uniform",
        chunk_index: list[int] | None = None,
        **kwargs: object,
    ) -> torch.Tensor | tuple[torch.Tensor, list]:
        kv_cache = kwargs.get("kv_cache", None)
        save_kv_cache = kwargs.get("save_kv_cache", False)

        if kv_cache is None:
            return super().forward(
                x,
                mask=mask,
                HW=HW,
                rotary_emb=rotary_emb,
                block_mask=block_mask,
                apply_output_gate=apply_output_gate,
                chunk_size=chunk_size,
                chunk_split_strategy=chunk_split_strategy,
                chunk_index=chunk_index,
                **kwargs,
            )

        del mask, block_mask
        if HW is None:
            raise ValueError("HW (T, H, W) must be provided.")

        B, N, C = x.shape
        T, H, W = HW
        S = H * W

        # QKV projection.
        qkv = self.qkv(x).reshape(B, N, 3, self.heads, self.dim)
        q, k, v = qkv.unbind(2)

        # QK norm.
        q = self.q_norm(q.reshape(B, N, C)).reshape(B, N, self.heads, self.dim)
        k = self.k_norm(k.reshape(B, N, C)).reshape(B, N, self.heads, self.dim)

        # RoPE (position-correct for current chunk).
        # Same slicing as the linear-attn path above: upstream rope may cover
        # sink + current under sink_token=true; cached K is post-rope so sink
        # K was already rotated when it was the current chunk.
        if rotary_emb is not None:
            q_perm = q.permute(0, 2, 3, 1)  # (B, H, D, N)
            k_perm = k.permute(0, 2, 3, 1)
            rotary_emb_cur = _slice_rope_to_current_chunk(rotary_emb, q_perm.shape[-1])
            q_perm = GDN._apply_rotary_emb(q_perm, rotary_emb_cur)
            k_perm = GDN._apply_rotary_emb(k_perm, rotary_emb_cur)
            q = q_perm.permute(0, 3, 1, 2)  # (B, N, H, D)
            k = k_perm.permute(0, 3, 1, 2)

        # (B, N, H, D) -> (B, H, N, D) for SDPA.
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.reshape(B, N, self.heads, self.dim).transpose(1, 2)

        dtype_orig = x.dtype

        # SDPA requires bf16/fp16.
        if q.dtype == torch.float32:
            q, k, v = q.bfloat16(), k.bfloat16(), v.bfloat16()

        # Read cached K, V from previous chunks BEFORE overwriting with current.
        cached_k = kv_cache[_SLOT_FWD_KV]
        cached_v = kv_cache[_SLOT_FWD_Z]

        # Save current chunk's K, V (overwrites cache slots).
        if save_kv_cache:
            kv_cache[_SLOT_FWD_KV] = k.detach().clone()
            kv_cache[_SLOT_FWD_Z] = v.detach().clone()
            kv_cache[_SLOT_TYPE_FLAG] = torch.tensor([_TYPE_CONCAT], device=x.device)
        if cached_k is not None:
            k = torch.cat([cached_k.to(k.dtype), k], dim=2)
            v = torch.cat([cached_v.to(v.dtype), v], dim=2)

        # Cached path always processes ONE chunk per forward — see comment in
        # original code. Chunk-causal mask not needed; cache enforces causality.
        _need_chunk_mask = False

        out = _sdpa_maybe_chunk_causal(
            q,
            k,
            v,
            need_chunk_mask=_need_chunk_mask,
            T=T,
            S=S,
            chunk_size=chunk_size,
            chunk_index=chunk_index,
            chunk_split_strategy=chunk_split_strategy,
            device=x.device,
        )

        if out.dtype != dtype_orig:
            out = out.to(dtype_orig)

        out = out.transpose(1, 2).reshape(B, N, C)

        if apply_output_gate:
            out = self._apply_output_gate(out, x)
            out = self.proj(out.to(dtype_orig))
            return out, kv_cache
        return out, kv_cache


# ===================================================================
# CachedChunkCausalGDNUCPESinglePathLiteLA (GDN + camera wrapper)
# ===================================================================


@ATTENTION_BLOCKS.register_module()
class CachedChunkCausalGDNUCPESinglePathLiteLA(ChunkCausalGDNUCPESinglePathLiteLA):
    """Cached variant of ChunkCausalGDNUCPESinglePathLiteLA.

    Main branch: uses ``CachedChunkCausalGDN`` logic (state-based cache).
    Camera branch: uses cached camera delta-rule state.
    """

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        HW: tuple[int, int, int] | None = None,
        rotary_emb: torch.Tensor | None = None,
        block_mask: torch.Tensor | None = None,
        camera_conditions: torch.Tensor | None = None,
        chunk_size: int | None = None,
        **kwargs: object,
    ) -> torch.Tensor | tuple[torch.Tensor, list]:
        kv_cache = kwargs.pop("kv_cache", None)
        save_kv_cache = kwargs.pop("save_kv_cache", False)

        if kv_cache is None:
            return super().forward(
                x,
                mask=mask,
                HW=HW,
                rotary_emb=rotary_emb,
                block_mask=block_mask,
                camera_conditions=camera_conditions,
                chunk_size=chunk_size,
                **kwargs,
            )

        # ---- Cached inference path ----
        if self.cam_debug_ratios:
            self.reset_cam_debug_stats()

        if HW is None:
            raise ValueError("HW (T, H, W) must be provided.")

        B, N, _ = x.shape
        T, H, W = HW
        H * W

        # Pre-compute shared gates once.
        precomputed_gates = self._compute_frame_gates(x, HW)

        # -- Main branch (cached GDN) --
        main_result = CachedChunkCausalGDN.forward(
            self,
            x,
            mask=mask,
            HW=HW,
            rotary_emb=rotary_emb,
            block_mask=block_mask,
            apply_output_gate=False,
            chunk_size=chunk_size,
            precomputed_gates=precomputed_gates,
            kv_cache=kv_cache,
            save_kv_cache=save_kv_cache,
            **kwargs,
        )
        main_raw, kv_cache = main_result

        # -- Camera branch (cached) --
        cam_contrib: torch.Tensor | int = 0
        camera_conditions = _maybe_drop_cam_branch(
            camera_conditions,
            kwargs.get("cam_branch_drop_prob", 0.0),
            self.training,
            x.device,
        )
        if camera_conditions is not None:
            cam_raw = self._cached_cam_branch(
                x,
                HW,
                camera_conditions,
                rotary_emb,
                kv_cache,
                save_kv_cache,
                precomputed_gates,
                chunk_size=chunk_size,
                **kwargs,
            )
            cam_contrib = self.out_proj_cam(cam_raw)

        combined = main_raw + cam_contrib
        combined = self._apply_output_gate(combined, x)
        output = self.proj(combined.to(self.proj.weight.dtype))
        return output, kv_cache

    def _cached_cam_branch(
        self,
        x: torch.Tensor,
        HW: tuple[int, int, int],
        camera_conditions: torch.Tensor,
        rotary_emb: torch.Tensor | None,
        kv_cache: list,
        save_kv_cache: bool,
        precomputed_gates: tuple | None,
        **kwargs: object,
    ) -> torch.Tensor:
        """Camera branch with cached delta-rule state."""
        B, N, _ = x.shape
        T, H, W = HW
        S = H * W
        dtype_orig = x.dtype

        token_valid_mask, beta_valid_mask, decay_valid_mask = self._prepare_frame_valid_masks(
            kwargs.get("frame_valid_mask", None), B=B, T=T, S=S, device=x.device, dtype=x.dtype
        )

        # Prepare UCPE-transformed camera QKV for current chunk.
        # Pass the camera K conv cache so that _prepare_cam_qkv uses the
        # cached temporal short conv (with left context from the previous
        # chunk) instead of the non-cached version that starts from zeros.
        cam_conv_cache_val = kv_cache[_SLOT_CAM_AUX]
        cam_cache_out = [None]  # mutable container for returning conv cache
        cam_kwargs = dict(kwargs)
        cam_kwargs["_cam_conv_cache_info"] = (cam_conv_cache_val, save_kv_cache, cam_cache_out)
        q_cam, _, v_cam_trans, q_cam_trans, k_cam_trans, apply_fn_o, inflation_sq = self._prepare_cam_qkv(
            x, HW, camera_conditions, rotary_emb, token_valid_mask=token_valid_mask, **cam_kwargs
        )
        if save_kv_cache and cam_cache_out[0] is not None:
            kv_cache[_SLOT_CAM_AUX] = cam_cache_out[0]
        if token_valid_mask is not None:
            m = token_valid_mask.view(B, 1, 1, N)
            q_cam = q_cam * m
            v_cam_trans = v_cam_trans * m
            q_cam_trans = q_cam_trans * m
            k_cam_trans = k_cam_trans * m

        if precomputed_gates is not None:
            beta, decay = precomputed_gates
        else:
            beta, decay = self._compute_frame_gates(x, HW)

        # Dynamic beta discounting.
        inflation_sq_spatial = inflation_sq.view(B, self.cam_heads, T, S)
        frame_inflation_sq = inflation_sq_spatial.mean(dim=-1)
        if beta.ndim == 3:
            beta = beta / frame_inflation_sq.clamp_min(1.0)
        elif beta.ndim == 4:
            beta = beta / frame_inflation_sq.unsqueeze(-1).clamp_min(1.0)

        if beta_valid_mask is not None:
            beta = beta * beta_valid_mask.to(beta.dtype)
        if decay_valid_mask is not None:
            decay_m = decay_valid_mask.to(decay.dtype)
            decay = decay * decay_m + (1.0 - decay_m)

        # Forward scan from cached camera state.
        # Use the same compiled scan function (_run_cam_single_path →
        # torch_chunk_cam_single_path_delta_rule) as the parent's
        # _forward_cam_branch to guarantee bit-exact numerical parity.
        cam_S_kv_prev = kv_cache[_SLOT_CAM]
        # Note: cam_S_kv_prev is stored as FP32 from the compiled scan.
        # _run_cam_single_path handles FP32 casting internally, so pass as-is
        # to avoid a lossy bfloat16 round-trip.

        cam_fwd_result = self._run_cam_single_path(
            q_cam_trans,
            k_cam_trans,
            v_cam_trans,
            beta,
            decay,
            S_kv_init=cam_S_kv_prev,
            return_state=save_kv_cache,
        )
        if save_kv_cache:
            out_fwd, cam_S_kv_final = cam_fwd_result
            kv_cache[_SLOT_CAM] = cam_S_kv_final.detach().clone()
        else:
            out_fwd = cam_fwd_result

        # Backward scan (per-chunk isolated).
        out_bwd = _cam_backward_scan_single_chunk(self, q_cam_trans, k_cam_trans, v_cam_trans, beta, decay, T, S)

        out = out_fwd + out_bwd

        if getattr(self, "fp32_attention", True) and dtype_orig != torch.float32:
            out = out.to(dtype_orig)
        if token_valid_mask is not None:
            out = out * token_valid_mask.view(B, 1, 1, N).to(out.dtype)

        out_before_fn_o = out
        out = apply_fn_o(out.transpose(-1, -2)).transpose(-1, -2).contiguous()
        self._maybe_record_cam_output_stats(out_before_fn_o, out, token_valid_mask=token_valid_mask)
        out = out.reshape(B, self.cam_dim, N).permute(0, 2, 1)
        if token_valid_mask is not None:
            out = out * token_valid_mask.view(B, N, 1).to(out.dtype)
        return out


# ===================================================================
# CachedSoftmaxUCPESinglePathLiteLA (Softmax + camera wrapper)
# ===================================================================


@ATTENTION_BLOCKS.register_module()
class CachedSoftmaxUCPESinglePathLiteLA(_SoftmaxUCPESinglePathLiteLA):
    """Cached variant of softmax + UCPE camera attention.

    Main branch: caches post-RoPE K, V (concat across chunks).
    Camera branch: caches post-UCPE-transform K, V (concat across chunks).
    """

    def __init__(self, *args, conv_kernel_size: int = 0, **kwargs):
        super().__init__(*args, conv_kernel_size=0, **kwargs)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        HW: tuple[int, int, int] | None = None,
        rotary_emb: torch.Tensor | None = None,
        block_mask: torch.Tensor | None = None,
        camera_conditions: torch.Tensor | None = None,
        chunk_size: int | None = None,
        **kwargs: object,
    ) -> torch.Tensor | tuple[torch.Tensor, list]:
        kv_cache = kwargs.pop("kv_cache", None)
        save_kv_cache = kwargs.pop("save_kv_cache", False)

        if kv_cache is None:
            return super().forward(
                x,
                mask=mask,
                HW=HW,
                rotary_emb=rotary_emb,
                block_mask=block_mask,
                camera_conditions=camera_conditions,
                chunk_size=chunk_size,
                **kwargs,
            )

        if HW is None:
            raise ValueError("HW must be provided.")
        if self.cam_debug_ratios:
            self.reset_cam_debug_stats()

        B, N, C = x.shape

        # -- Main branch (cached softmax) --
        main_result = CachedChunkCausalSoftmaxAttn.forward(
            self,
            x,
            mask=mask,
            HW=HW,
            rotary_emb=rotary_emb,
            block_mask=block_mask,
            apply_output_gate=False,
            chunk_size=chunk_size,
            kv_cache=kv_cache,
            save_kv_cache=save_kv_cache,
            **kwargs,
        )
        main_raw, kv_cache = main_result

        # -- Camera branch (cached softmax SDPA) --
        cam_contrib: torch.Tensor | int = 0
        camera_conditions = _maybe_drop_cam_branch(
            camera_conditions,
            kwargs.get("cam_branch_drop_prob", 0.0),
            self.training,
            x.device,
        )
        if camera_conditions is not None:
            cam_raw = self._cached_cam_branch_softmax(
                x,
                HW,
                camera_conditions,
                rotary_emb,
                kv_cache,
                save_kv_cache,
                chunk_size=chunk_size,
                **kwargs,
            )
            cam_contrib = self.out_proj_cam(cam_raw)

        combined = main_raw + cam_contrib
        combined = self._apply_output_gate(combined, x)
        output = self.proj(combined.to(x.dtype))
        return output, kv_cache

    def _cached_cam_branch_softmax(
        self,
        x: torch.Tensor,
        HW: tuple[int, int, int],
        camera_conditions: torch.Tensor,
        rotary_emb: torch.Tensor | None,
        kv_cache: list,
        save_kv_cache: bool,
        **kwargs: object,
    ) -> torch.Tensor:
        """Camera branch: softmax SDPA with cached K, V concatenation."""
        B, N, _ = x.shape
        T, H, W = HW
        S = H * W

        token_valid_mask, _, _ = self._prepare_frame_valid_masks(
            kwargs.get("frame_valid_mask", None), B=B, T=T, S=S, device=x.device, dtype=x.dtype
        )

        # Prepare UCPE-transformed camera QKV for current chunk.
        q_cam_trans, k_cam_trans, v_cam_trans, apply_fn_o = _prepare_cam_qkv_softmax(
            self, x, HW, camera_conditions, rotary_emb, token_valid_mask=token_valid_mask, **kwargs
        )
        if token_valid_mask is not None:
            m = token_valid_mask.view(B, 1, 1, N)
            q_cam_trans, k_cam_trans, v_cam_trans = q_cam_trans * m, k_cam_trans * m, v_cam_trans * m

        # (B, H, D, N) -> (B, H, N, D) for SDPA.
        q_sdpa = q_cam_trans.transpose(-1, -2)
        k_sdpa = k_cam_trans.transpose(-1, -2)
        v_sdpa = v_cam_trans.transpose(-1, -2)

        dtype_orig = x.dtype
        if q_sdpa.dtype == torch.float32:
            q_sdpa, k_sdpa, v_sdpa = q_sdpa.bfloat16(), k_sdpa.bfloat16(), v_sdpa.bfloat16()

        # Read cached cam K, V from previous chunks BEFORE overwriting.
        cached_cam_k = kv_cache[_SLOT_CAM]
        cached_cam_v = kv_cache[_SLOT_CAM_AUX]

        # Save current chunk's cam K, V (overwrites cache slots).
        if save_kv_cache:
            kv_cache[_SLOT_CAM] = k_sdpa.detach().clone()
            kv_cache[_SLOT_CAM_AUX] = v_sdpa.detach().clone()
        if cached_cam_k is not None:
            k_sdpa = torch.cat([cached_cam_k.to(k_sdpa.dtype), k_sdpa], dim=2)
            v_sdpa = torch.cat([cached_cam_v.to(v_sdpa.dtype), v_sdpa], dim=2)

        # Cached path always processes ONE chunk per forward — cache enforces
        # chunk causality, no in-forward mask needed. Force False to skip the
        # flex_attention BlockMask path and use plain SDPA. (See sibling change
        # in self.forward for the same rationale.)
        chunk_size = kwargs.get("chunk_size", None)
        _need_chunk_mask = False

        out = _sdpa_maybe_chunk_causal(
            q_sdpa,
            k_sdpa,
            v_sdpa,
            need_chunk_mask=_need_chunk_mask,
            T=T,
            S=S,
            chunk_size=chunk_size,
            chunk_index=kwargs.get("chunk_index", None),
            chunk_split_strategy=kwargs.get("chunk_split_strategy", "uniform"),
            device=x.device,
        )

        # (B, H, N, D) -> (B, H, D, N).
        out = out.transpose(-1, -2)
        if out.dtype != dtype_orig:
            out = out.to(dtype_orig)
        if token_valid_mask is not None:
            out = out * token_valid_mask.view(B, 1, 1, N).to(out.dtype)

        out = apply_fn_o(out.transpose(-1, -2)).transpose(-1, -2).contiguous()
        out = out.reshape(B, self.cam_dim, N).permute(0, 2, 1)
        if token_valid_mask is not None:
            out = out * token_valid_mask.view(B, N, 1).to(out.dtype)
        return out

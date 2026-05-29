#!/usr/bin/env python3
"""Compare Sana-WM sampled-frame NPZ files produced by the benchmark wrapper."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


def _resolve_npz(path: Path) -> Path:
    if path.is_dir():
        path = path / "result.json"
    if path.name == "result.json":
        data = json.loads(path.read_text())
        sample_path = data.get("sample_frames_path")
        if not sample_path:
            raise SystemExit(f"{path} does not contain sample_frames_path")
        sample = Path(sample_path)
        if not sample.is_absolute():
            if not sample.exists():
                sample = path.parent / sample
        return sample
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("reference", type=Path, help="Reference NPZ, result.json, or benchmark output directory.")
    parser.add_argument("candidate", type=Path, help="Candidate NPZ, result.json, or benchmark output directory.")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    args = parser.parse_args()

    reference = _resolve_npz(args.reference)
    candidate = _resolve_npz(args.candidate)
    ref_npz = np.load(reference)
    cand_npz = np.load(candidate)
    ref = ref_npz["frames"].astype(np.float32)
    cand = cand_npz["frames"].astype(np.float32)
    ref_indices = ref_npz["frame_indices"]
    cand_indices = cand_npz["frame_indices"]

    if ref.shape != cand.shape:
        raise SystemExit(f"shape mismatch: {reference} {ref.shape} vs {candidate} {cand.shape}")
    if not np.array_equal(ref_indices, cand_indices):
        raise SystemExit(f"frame index mismatch: {ref_indices.tolist()} vs {cand_indices.tolist()}")

    diff = np.abs(ref - cand)
    mse = float(np.mean((ref - cand) ** 2))
    psnr = float("inf") if mse == 0.0 else 20.0 * math.log10(255.0 / math.sqrt(mse))
    per_frame_mae = [float(np.mean(np.abs(ref[i] - cand[i]))) for i in range(ref.shape[0])]
    result = {
        "reference": str(reference),
        "candidate": str(candidate),
        "shape": list(ref.shape),
        "frame_indices": ref_indices.tolist(),
        "mae": float(diff.mean()),
        "max_abs": int(diff.max()),
        "mse": mse,
        "psnr": psnr,
        "per_frame_mae": per_frame_mae,
    }

    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
        return
    print("__CODEX_SAMPLE_FRAME_COMPARE__")
    print(f"reference={result['reference']}")
    print(f"candidate={result['candidate']}")
    print(f"shape={tuple(result['shape'])}")
    print(f"frame_indices={result['frame_indices']}")
    print(
        "mae={mae:.6f} max_abs={max_abs} mse={mse:.6f} psnr={psnr:.3f}".format(
            **result
        )
    )
    print("per_frame_mae=" + ",".join(f"{v:.6f}" for v in per_frame_mae))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Create a compact visual contact sheet for two Sana-WM sampled-frame NPZ files."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from compare_sana_wm_sample_frames import _resolve_npz


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("reference", type=Path, help="Reference NPZ, result.json, or benchmark output directory.")
    parser.add_argument("candidate", type=Path, help="Candidate NPZ, result.json, or benchmark output directory.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frames", default="0,64,120,160", help="Comma-separated sampled frame indices to show.")
    parser.add_argument("--width", type=int, default=320, help="Thumbnail width.")
    parser.add_argument("--diff-scale", type=float, default=8.0, help="Amplification factor for absolute-diff row.")
    args = parser.parse_args()

    ref_npz = np.load(_resolve_npz(args.reference))
    cand_npz = np.load(_resolve_npz(args.candidate))
    ref = ref_npz["frames"]
    cand = cand_npz["frames"]
    frame_indices = ref_npz["frame_indices"]
    if ref.shape != cand.shape:
        raise SystemExit(f"shape mismatch: {ref.shape} vs {cand.shape}")
    if not np.array_equal(frame_indices, cand_npz["frame_indices"]):
        raise SystemExit("frame index mismatch")

    requested = [int(v.strip()) for v in args.frames.split(",") if v.strip()]
    positions = []
    for frame_idx in requested:
        matches = np.where(frame_indices == frame_idx)[0]
        if len(matches) == 0:
            raise SystemExit(f"frame {frame_idx} is not present; available={frame_indices.tolist()}")
        positions.append(int(matches[0]))

    thumb_w = int(args.width)
    thumb_h = round(ref.shape[1] * thumb_w / ref.shape[2])
    label_h = 26
    sheet = Image.new("RGB", (len(positions) * thumb_w, 3 * (thumb_h + label_h)), "white")
    draw = ImageDraw.Draw(sheet)

    for col, pos in enumerate(positions):
        ref_img = Image.fromarray(ref[pos]).resize((thumb_w, thumb_h), Image.Resampling.LANCZOS)
        cand_img = Image.fromarray(cand[pos]).resize((thumb_w, thumb_h), Image.Resampling.LANCZOS)
        diff = np.abs(ref[pos].astype(np.int16) - cand[pos].astype(np.int16)).astype(np.float32)
        diff = np.clip(diff * float(args.diff_scale), 0, 255).astype(np.uint8)
        diff_img = Image.fromarray(diff).resize((thumb_w, thumb_h), Image.Resampling.LANCZOS)
        for row, (label, image) in enumerate(
            (("reference", ref_img), ("candidate", cand_img), (f"abs diff x{args.diff_scale:g}", diff_img))
        ):
            x = col * thumb_w
            y = row * (thumb_h + label_h)
            sheet.paste(image, (x, y + label_h))
            draw.text((x + 6, y + 6), f"{label} frame {int(frame_indices[pos])}", fill=(0, 0, 0))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(args.output, quality=92)
    print(args.output)


if __name__ == "__main__":
    main()

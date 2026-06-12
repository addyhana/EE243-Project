import argparse
from pathlib import Path

import os
import cv2
import numpy as np
import torch
from PIL import Image

from sam2.build_sam import build_sam2_video_predictor


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--video_dir", type=str, required=True)
    parser.add_argument("--first_mask", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="results/baseline")

    parser.add_argument("--checkpoint", type=str, default=os.environ["SAM2_CHECKPOINT"])
    parser.add_argument("--config", type=str, default="configs/sam2.1/sam2.1_hiera_s.yaml")

    return parser.parse_args()


def save_mask(mask, path):
    mask = mask.astype(np.uint8) * 255
    Image.fromarray(mask).save(path)


def make_overlay(frame, mask, alpha=0.5):
    frame = np.array(frame).astype(np.uint8)

    color_mask = np.zeros_like(frame)
    color_mask[mask > 0] = np.array([255, 0, 0], dtype=np.uint8)

    overlay = (1 - alpha) * frame + alpha * color_mask
    return overlay.astype(np.uint8)


def main():
    args = parse_args()

    video_dir = Path(args.video_dir)
    output_dir = Path(args.output_dir)

    masks_dir = output_dir / "masks"
    overlays_dir = output_dir / "overlays"

    masks_dir.mkdir(parents=True, exist_ok=True)
    overlays_dir.mkdir(parents=True, exist_ok=True)

    frame_paths = sorted(
        list(video_dir.glob("*.jpg")) +
        list(video_dir.glob("*.jpeg")) +
        list(video_dir.glob("*.png"))
    )

    if len(frame_paths) == 0:
        raise RuntimeError(f"No frames found in {video_dir}")

    print(f"Found {len(frame_paths)} frames")

    first_mask = np.array(Image.open(args.first_mask))

    if first_mask.ndim == 3:
        first_mask = first_mask[:, :, 0]

    first_mask = first_mask > 0

    print("First mask shape:", first_mask.shape)

    predictor = build_sam2_video_predictor(
        args.config,
        args.checkpoint,
    )

    inference_state = predictor.init_state(
        video_path=str(video_dir)
    )

    obj_id = 1

    predictor.add_new_mask(
        inference_state=inference_state,
        frame_idx=0,
        obj_id=obj_id,
        mask=first_mask,
    )

    video_segments = {}

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
            masks = {}

            for i, current_obj_id in enumerate(out_obj_ids):
                mask = (out_mask_logits[i] > 0.0).detach().cpu().numpy()
                mask = np.squeeze(mask)
                masks[current_obj_id] = mask

            video_segments[out_frame_idx] = masks

    for frame_idx, frame_path in enumerate(frame_paths):
        frame = Image.open(frame_path).convert("RGB")

        if frame_idx in video_segments and obj_id in video_segments[frame_idx]:
            mask = video_segments[frame_idx][obj_id]
        else:
            mask = np.zeros((frame.height, frame.width), dtype=bool)

        save_mask(mask, masks_dir / f"{frame_idx:05d}.png")

        overlay = make_overlay(frame, mask)
        Image.fromarray(overlay).save(overlays_dir / f"{frame_idx:05d}.png")

    print(f"Saved masks to {masks_dir}")
    print(f"Saved overlays to {overlays_dir}")


if __name__ == "__main__":
    main()
import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from sam2.build_sam import build_sam2_video_predictor

#ABDI CHANGE###########
#RUN ON CPU WHILE CUDA NOT AVAILABLE
from contextlib import nullcontext
#######################

#ARGS
def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--video_dir", type=str, required=True)
    parser.add_argument("--first_mask", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="outputs_sam2")

    parser.add_argument("--checkpoint", type=str, default="./checkpoints/sam2.1_hiera_small.pt")
    parser.add_argument("--config", type=str, default="configs/sam2.1/sam2.1_hiera_s.yaml")

    #ABDI CHANGE
    # parser.add_argument("--checkpoint", type=str, default=r"C:\Users\abdin\OneDrive\Desktop\RockVision\EE243-Project\sam2\checkpoints\sam2.1_hiera_base_plus.pt")
    # parser.add_argument("--config", type=str, default="configs/sam2.1/sam2.1_hiera_b+.yaml")

    return parser.parse_args()

#SAVES MASK TO DISK
def save_mask(mask, path):
    mask = mask.astype(np.uint8) * 255
    Image.fromarray(mask).save(path)

#CREATES IMAGE OVERLAY ONTOP OG FRAME
def make_overlay(frame, mask, alpha=0.5):
    frame = np.array(frame).astype(np.uint8)

    color_mask = np.zeros_like(frame)
    color_mask[mask > 0] = np.array([255, 0, 0], dtype=np.uint8)

    overlay = (1 - alpha) * frame + alpha * color_mask
    return overlay.astype(np.uint8)


############################################################################################
#NOISE FUNCTION#############################################################################
############################################################################################
import numpy as np

# .05 means a pixel has 5% chance of being selected
# noise level means the amount of pixels kept
def add_noise_to_mask(mask, noise_level=0.05, seed=0):
    
    #Seed!
    rng = np.random.default_rng(seed)

    #convert mask to boolean
    mask_bool = mask > 0

    #if rgb channel, combine channels into one mask
    if mask_bool.ndim == 3:
        mask_bool = np.any(mask_bool[:, :, :3], axis=2)

    #generate random vals from 1 to 0 for every pixel in mask
    random_noise = rng.random(mask_bool.shape) < noise_level

    #keep only pixels in the OG mask
    noisy_mask = random_noise & mask_bool

    return noisy_mask

############################################################################################
############################################################################################
############################################################################################


############################################################################################
#MODIFY NOISE LEVEL#########################################################################
############################################################################################
NOISELEVEL = .50
#HOW MUCH IS KEPT
############################################################################################
############################################################################################
############################################################################################

def main():
    args = parse_args()

    #COMMAND LINE INPUTS
    video_dir = Path(args.video_dir)
    output_dir = Path(args.output_dir)

    masks_dir = output_dir / "masks"
    overlays_dir = output_dir / "overlays"

    masks_dir.mkdir(parents=True, exist_ok=True)
    overlays_dir.mkdir(parents=True, exist_ok=True)

    #LOAD FRAMES
    frame_paths = sorted(
        list(video_dir.glob("*.jpg")) +
        list(video_dir.glob("*.jpeg")) +
        list(video_dir.glob("*.png"))
    )

    if len(frame_paths) == 0:
        raise RuntimeError(f"No frames found in {video_dir}")

    print(f"Found {len(frame_paths)} frames")

    #LOAD FIRST MASK
    first_mask = np.array(Image.open(args.first_mask))

    if first_mask.ndim == 3:
        first_mask = first_mask[:, :, 0]

    first_mask = first_mask > 0

    ############################################################################################
    #ADD NOISE TO IMAGE#########################################################################
    ############################################################################################

    clean_first_mask = first_mask.copy()

    first_mask = add_noise_to_mask(
        first_mask,
        noise_level=NOISELEVEL,
        seed=0
    )

    save_mask(clean_first_mask, output_dir / "first_mask_clean.png")
    save_mask(first_mask, output_dir / "first_mask_noisy.png")

    ############################################################################################
    ############################################################################################
    ############################################################################################

    print("First mask shape:", first_mask.shape)

    #ABDI CHANGE###########
    #USES GPU IF AVAILABLE IF NOT CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #######################

    #BUILD SAM2 VIDEO PREDICTOR (CONFIG, CHECKPOINT, DEVICE)
    predictor = build_sam2_video_predictor(
        args.config,
        args.checkpoint,
        device=device,
    )

    #LOAD VIDEO FRAMES
    inference_state = predictor.init_state(
        video_path=str(video_dir)
    )

    obj_id = 1

    # GIV FIRST MASK TO KNOW WHAT OBJECT TO TRACK
    predictor.add_new_mask(
        inference_state=inference_state,
        frame_idx=0,
        obj_id=obj_id,
        mask=first_mask,
    )

    video_segments = {}

    #ABDI CHANGE###########
    # IF RUNNING ON CPU USE NULLCONTEXT
    autocast_context = (
        torch.autocast("cuda", dtype=torch.bfloat16)
        if device.type == "cuda"
        else nullcontext()
    )

    #RUN SAM2 IN INFERENCE MODE WITHOUT TRAINING GRADIENTS
    with torch.inference_mode(), autocast_context:
    #######################

        #RUNS SAM2 THROUGH THE VIDEO AND GETS PREDICTED MASKS FOR EACH FRAME
        for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
            masks = {}

            #CREATES DIC FOR MASKS IN THE CURRENT FRAME
            for i, current_obj_id in enumerate(out_obj_ids):
                mask = (out_mask_logits[i] > 0.0).detach().cpu().numpy()
                mask = np.squeeze(mask)
                masks[current_obj_id] = mask

            video_segments[out_frame_idx] = masks

    #FOR EACH FRAME, SAVE
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
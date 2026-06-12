import os
import torch
from sam2.build_sam import build_sam2_video_predictor

checkpoint = os.environ["SAM2_CHECKPOINT"]
model_cfg = "configs/sam2.1/sam2.1_hiera_s.yaml"

predictor = build_sam2_video_predictor(model_cfg, checkpoint)

print("SAM2 loaded")
print("CUDA:", torch.cuda.is_available())
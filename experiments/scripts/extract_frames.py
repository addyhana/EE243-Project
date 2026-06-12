from pathlib import Path

import cv2
import numpy as np 

VIDEO_PATH = "exp2video.mp4"

OUTPUT_DIR = "experiments/input_frames/experiment2"

# seconds 
START_TIME = 40.0
END_TIME = 50.0

NUM_FRAMES = 60


def main():
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(VIDEO_PATH)

    fps = cap.get(cv2.CAP_PROP_FPS)

    start_frame = int(START_TIME * fps)
    end_frame = int(END_TIME * fps)

    frame_indices = np.linspace(
        start_frame,
        end_frame,
        NUM_FRAMES,
        dtype=int
    )

    saved = 0

    for i, frame_idx in enumerate(frame_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))

        success, frame = cap.read()

        if not success:
            print(f"Failed to read frame at index {frame_idx}")
            continue

        save_path = output_dir / f"{i:05d}.jpg"

        cv2.imwrite(str(save_path), frame)

        saved += 1

    cap.release()

    print(f"Saved {saved} frames to {output_dir}")


if __name__ == "__main__":
    main()
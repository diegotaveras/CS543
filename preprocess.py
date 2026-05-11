import argparse
from pathlib import Path
from typing import Optional, Sequence, Union

import cv2
import numpy as np


PathLike = Union[str, Path]


def _save_frames(
    frames: Sequence[np.ndarray],
    timestamps: Sequence[float],
    output_dir: PathLike,
) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for index, (frame_rgb, timestamp) in enumerate(zip(frames, timestamps)):
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        frame_path = output_path / f"frame_{index:04d}_{timestamp:.2f}s.png"
        cv2.imwrite(str(frame_path), frame_bgr)


def preprocess_video(
    video_path: PathLike,
    sample_every_seconds: float = 2.0,
    output_dir: Optional[PathLike] = None,
) -> np.ndarray:
    """Sample an MP4 video every `sample_every_seconds` seconds.

    Returns:
        A NumPy array of RGB frames with shape (num_frames, height, width, 3).
    """
    if sample_every_seconds <= 0:
        raise ValueError("sample_every_seconds must be greater than 0.")

    path = Path(video_path)
    if path.suffix.lower() != ".mp4":
        raise ValueError("preprocess_video expects an .mp4 file.")
    if not path.exists():
        raise FileNotFoundError(f"Video file not found: {path}")

    video = cv2.VideoCapture(str(path))
    try:
        if not video.isOpened():
            raise ValueError(f"Unable to open video file: {path}")

        fps = video.get(cv2.CAP_PROP_FPS)
        frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        if fps <= 0 or frame_count <= 0:
            raise ValueError(f"Unable to read FPS/frame count from video: {path}")

        frame_step = max(int(round(fps * sample_every_seconds)), 1)
        frames = []
        timestamps = []

        for frame_index in range(0, frame_count, frame_step):
            video.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            success, frame_bgr = video.read()
            if not success:
                continue

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
            timestamps.append(frame_index / fps)

        if not frames:
            raise ValueError(f"No frames could be sampled from video: {path}")

        if output_dir is not None:
            _save_frames(frames, timestamps, output_dir)

        return np.stack(frames, axis=0)
    finally:
        video.release()


def preprocess(video_path: PathLike, output_dir: Optional[PathLike] = None) -> np.ndarray:
    """Default preprocessing entry point for MP4 videos."""
    return preprocess_video(video_path, output_dir=output_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Sample RGB frames from an MP4 video.")
    parser.add_argument("video_path", type=str, help="Path to an MP4 video file.")
    parser.add_argument(
        "--sample-every-seconds",
        type=float,
        default=2.0,
        help="Seconds between sampled frames. Defaults to 2.0.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="sampled_frames",
        help="Folder where sampled frames will be saved as PNGs.",
    )
    args = parser.parse_args()

    frames = preprocess_video(
        args.video_path,
        sample_every_seconds=args.sample_every_seconds,
        output_dir=args.output_dir,
    )
    print(f"frames shape: {frames.shape}")
    print(f"frames dtype: {frames.dtype}")
    print(f"saved frames to: {Path(args.output_dir).resolve()}")


if __name__ == "__main__":
    main()

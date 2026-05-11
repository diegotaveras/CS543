import argparse
from pathlib import Path
from typing import Optional, Sequence, Union

import cv2
import numpy as np


PathLike = Union[str, Path]
SsimThreshold = Union[float, str]


def _save_frames(
    frames: Sequence[np.ndarray],
    timestamps: Sequence[float],
    output_dir: PathLike,
    frame_indices: Optional[Sequence[int]] = None,
) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if frame_indices is None:
        frame_indices = list(range(len(frames)))

    for index, (frame_rgb, timestamp, frame_index) in enumerate(zip(frames, timestamps, frame_indices)):
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        frame_path = output_path / f"frame_{index:04d}_src{frame_index:06d}_{timestamp:.2f}s.png"
        cv2.imwrite(str(frame_path), frame_bgr)


def _validate_video_path(video_path: PathLike) -> Path:
    path = Path(video_path)
    if path.suffix.lower() != ".mp4":
        raise ValueError("Expected an .mp4 file.")
    if not path.exists():
        raise FileNotFoundError(f"Video file not found: {path}")
    return path


def _open_video(video_path: PathLike):
    path = _validate_video_path(video_path)
    video = cv2.VideoCapture(str(path))
    if not video.isOpened():
        video.release()
        raise ValueError(f"Unable to open video file: {path}")

    fps = video.get(cv2.CAP_PROP_FPS)
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps <= 0 or frame_count <= 0:
        video.release()
        raise ValueError(f"Unable to read FPS/frame count from video: {path}")

    return path, video, fps, frame_count


def _read_rgb_frame(video, frame_index: int) -> Optional[np.ndarray]:
    video.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    success, frame_bgr = video.read()
    if not success:
        return None
    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)


def _ssim(frame_a: np.ndarray, frame_b: np.ndarray) -> float:
    from skimage.metrics import structural_similarity as ssim

    gray_a = cv2.cvtColor(frame_a, cv2.COLOR_RGB2GRAY)
    gray_b = cv2.cvtColor(frame_b, cv2.COLOR_RGB2GRAY)
    return float(ssim(gray_a, gray_b, data_range=255))


def _resolve_ssim_threshold(
    threshold: SsimThreshold,
    similarities: Sequence[float],
    percentile: float,
) -> float:
    if isinstance(threshold, str):
        if threshold != "auto":
            raise ValueError("ssim_threshold must be a number or 'auto'.")
        if not 0 <= percentile <= 100:
            raise ValueError("ssim_percentile must be between 0 and 100.")
        if not similarities:
            return 1.0
        return float(np.percentile(np.array(similarities), percentile))

    threshold = float(threshold)
    if not -1.0 <= threshold <= 1.0:
        raise ValueError("ssim_threshold must be between -1.0 and 1.0, or 'auto'.")
    return threshold


def preprocess_video_uniform(
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

    path, video, fps, frame_count = _open_video(video_path)
    try:
        frame_step = max(int(round(fps * sample_every_seconds)), 1)
        frames = []
        timestamps = []
        frame_indices = []

        for frame_index in range(0, frame_count, frame_step):
            frame_rgb = _read_rgb_frame(video, frame_index)
            if frame_rgb is None:
                continue

            frames.append(frame_rgb)
            timestamps.append(frame_index / fps)
            frame_indices.append(frame_index)

        if not frames:
            raise ValueError(f"No frames could be sampled from video: {path}")

        if output_dir is not None:
            _save_frames(frames, timestamps, output_dir, frame_indices)

        return np.stack(frames, axis=0)
    finally:
        video.release()


def preprocess_video_ssim(
    video_path: PathLike,
    ssim_threshold: SsimThreshold = 0.90,
    ssim_frame_step: int = 20,
    ssim_percentile: float = 25.0,
    output_dir: Optional[PathLike] = None,
) -> np.ndarray:
    """Sample keyframes by comparing every Nth frame with SSIM."""
    if ssim_frame_step <= 0:
        raise ValueError("ssim_frame_step must be greater than 0.")

    path, video, fps, frame_count = _open_video(video_path)
    try:
        first_frame = _read_rgb_frame(video, 0)
        if first_frame is None:
            raise ValueError(f"No frames could be sampled from video: {path}")

        frames = [first_frame]
        timestamps = [0.0]
        frame_indices = [0]
        previous_kept_frame = first_frame
        candidates = []
        consecutive_similarities = []

        for frame_index in range(ssim_frame_step, frame_count, ssim_frame_step):
            frame_rgb = _read_rgb_frame(video, frame_index)
            if frame_rgb is None:
                continue

            candidates.append((frame_index, frame_rgb))
            previous_candidate = first_frame if len(candidates) == 1 else candidates[-2][1]
            consecutive_similarities.append(_ssim(previous_candidate, frame_rgb))

        resolved_threshold = _resolve_ssim_threshold(
            ssim_threshold,
            consecutive_similarities,
            ssim_percentile,
        )

        for frame_index, frame_rgb in candidates:
            similarity = _ssim(previous_kept_frame, frame_rgb)
            if similarity < resolved_threshold:
                frames.append(frame_rgb)
                timestamps.append(frame_index / fps)
                frame_indices.append(frame_index)
                previous_kept_frame = frame_rgb

        if output_dir is not None:
            _save_frames(frames, timestamps, output_dir, frame_indices)

        return np.stack(frames, axis=0)
    finally:
        video.release()


def preprocess_video(
    video_path: PathLike,
    sample_every_seconds: float = 2.0,
    output_dir: Optional[PathLike] = None,
) -> np.ndarray:
    """Backward-compatible uniform video preprocessing entry point."""
    return preprocess_video_uniform(
        video_path,
        sample_every_seconds=sample_every_seconds,
        output_dir=output_dir,
    )


def preprocess(video_path: PathLike, output_dir: Optional[PathLike] = None) -> np.ndarray:
    """Default preprocessing entry point for MP4 videos."""
    return preprocess_video(video_path, output_dir=output_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Sample RGB frames from an MP4 video.")
    parser.add_argument("video_path", type=str, help="Path to an MP4 video file.")
    parser.add_argument(
        "--sampling-mode",
        choices=["uniform", "ssim"],
        default="uniform",
        help="Frame sampling strategy.",
    )
    parser.add_argument(
        "--sample-every-seconds",
        type=float,
        default=2.0,
        help="Seconds between sampled frames. Defaults to 2.0.",
    )
    parser.add_argument(
        "--ssim-threshold",
        default="0.90",
        help="Keep an SSIM candidate frame when similarity is below this value, or use 'auto'.",
    )
    parser.add_argument(
        "--ssim-percentile",
        type=float,
        default=25.0,
        help="Percentile used to choose the threshold when --ssim-threshold auto.",
    )
    parser.add_argument(
        "--ssim-frame-step",
        type=int,
        default=20,
        help="Compare every Nth video frame in SSIM mode.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="sampled_frames",
        help="Folder where sampled frames will be saved as PNGs.",
    )
    args = parser.parse_args()

    if args.sampling_mode == "uniform":
        output_dir = Path(args.output_dir) / f"uniform_{args.sample_every_seconds:.2f}s"
        frames = preprocess_video_uniform(
            args.video_path,
            sample_every_seconds=args.sample_every_seconds,
            output_dir=output_dir,
        )
    else:
        threshold_label = (
            f"auto_p{args.ssim_percentile:.0f}"
            if args.ssim_threshold == "auto"
            else f"threshold{float(args.ssim_threshold):.2f}"
        )
        output_dir = Path(args.output_dir) / f"ssim_step{args.ssim_frame_step}_{threshold_label}"
        frames = preprocess_video_ssim(
            args.video_path,
            ssim_threshold=args.ssim_threshold,
            ssim_frame_step=args.ssim_frame_step,
            ssim_percentile=args.ssim_percentile,
            output_dir=output_dir,
        )
    print(f"frames shape: {frames.shape}")
    print(f"frames dtype: {frames.dtype}")
    print(f"saved frames to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()

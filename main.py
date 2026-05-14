import argparse
import subprocess
import sys
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

import requests
import cv2
import torch
from PIL import Image

SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE_DIR = SCRIPT_DIR.parent
ML_FASTVLM_DIR = WORKSPACE_DIR / "ml-fastvlm"

if str(ML_FASTVLM_DIR) not in sys.path:
    sys.path.insert(1, str(ML_FASTVLM_DIR))

from llava.constants import (
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    IMAGE_TOKEN_INDEX,
)
from llava.conversation import conv_templates
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from preprocess import SsimThreshold, preprocess_video_ssim, preprocess_video_uniform


DEFAULT_MODEL_PATH = ML_FASTVLM_DIR / "checkpoints/llava-fastvithd_0.5b_stage3"
DEFAULT_PROMPT_PATH = SCRIPT_DIR / "prompt.txt"
FALLBACK_PROMPT_PATH = ML_FASTVLM_DIR / "prompt.txt"
DEFAULT_OPENROUTER_MODEL = "openai/gpt-oss-120b:free"
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"


@dataclass
class FrameInput:
    image: Image.Image
    timestamp: Optional[float]


@dataclass
class FrameCaption:
    frame_index: int
    timestamp: Optional[float]
    text: str


@dataclass
class IntervalCaptions:
    start: FrameCaption
    end: FrameCaption
    additional: List[FrameCaption]


@dataclass
class LoadedModel:
    tokenizer: object
    model: object
    image_processor: object


def load_default_prompt() -> str:
    for prompt_path in (DEFAULT_PROMPT_PATH, FALLBACK_PROMPT_PATH):
        if prompt_path.exists():
            prompt = prompt_path.read_text(encoding="utf-8").strip()
            if prompt:
                return prompt
    return "Describe the image in one sentence."


DEFAULT_PROMPT = load_default_prompt()


def load_openrouter_api_key() -> str:
    if os.environ.get("OPENROUTER_API_KEY"):
        return os.environ["OPENROUTER_API_KEY"]

    for env_path in (SCRIPT_DIR / ".env", ML_FASTVLM_DIR / ".env"):
        if not env_path.exists():
            continue
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue

            key, value = line.split("=", 1)
            if key.strip() == "OPENROUTER_API_KEY":
                return value.strip().strip("\"'")

    raise RuntimeError("OPENROUTER_API_KEY was not found in the environment, cs543/.env, or ml-fastvlm/.env.")


def build_event_summary_prompt(frame_outputs: List[str]) -> str:
    observations = "\n\n".join(frame_outputs)
    return (
        "You are summarizing a video from chronological key-frame observations.\n"
        "Each observation includes the timestamp of the source frame.\n"
        "Merge repeated adjacent observations into continuous events when appropriate.\n"
        "Use the frame timestamps to estimate when events happen.\n"
        "You can reasonably infer actions that happen between different captions to better summarize.\n"
        "Output:\n"
        "1. A concise chronological event summary.\n"
        "2. Your event rows should be output in this exact format: [start_time, end_time]: <summary>\n"
        f"Frame observations:\n{observations}"
    )


def build_enriched_event_summary_prompt(interval_outputs: List[str]) -> str:
    observations = "\n\n".join(interval_outputs)
    return (
        "You are summarizing a video from chronological key-frame intervals.\n"
        "Each interval is start-inclusive and end-exclusive.\n"
        "Each interval includes the original key-frame caption at the start timestamp.\n"
        "The end timestamp is only a boundary marker; its key-frame caption belongs to the next interval.\n"
        "Some intervals also include Additional captions sampled uniformly inside the interval.\n"
        "Use Additional captions to better understand what happens between keyframes.\n"
        "In the final summary, preserve the original key-frame timestamp ranges for events.\n"
        "Do not output separate events only for Additional captions unless they change the interval interpretation.\n"
        "Merge repeated adjacent observations into continuous events when appropriate.\n"
        "You can reasonably infer actions that happen between different captions to better summarize.\n"
        "Output:\n"
        "1. A concise chronological event summary using the original key-frame timestamp ranges.\n"
        "2. Your event rows should be output in this exact format: [start_time, end_time]: <summary>\n"
        f"Key-frame intervals:\n{observations}"
    )


def summarize_events_with_openrouter(
    frame_outputs: List[str],
    model_name: str,
    output_path: Path,
    use_enriched_intervals: bool = False,
) -> str:
    api_key = load_openrouter_api_key()
    payload = {
        "model": model_name,
        "messages": [
            {
                "role": "user",
                "content": (
                    build_enriched_event_summary_prompt(frame_outputs)
                    if use_enriched_intervals
                    else build_event_summary_prompt(frame_outputs)
                ),
            }
        ],
        "temperature": 0,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost",
        "X-Title": "FastVLM Video Event Summary",
    }

    response = requests.post(OPENROUTER_API_URL, headers=headers, json=payload, timeout=120)
    response.raise_for_status()
    data = response.json()
    summary = data["choices"][0]["message"]["content"].strip()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(summary + "\n", encoding="utf-8")
    return summary


def first_sentence(text: str) -> str:
    first_line = text.strip().splitlines()[0].strip()
    match = re.search(r"(.+?[.!?])(?:\s|$)", first_line)
    if match:
        return match.group(1).strip()
    return first_line


def choose_device(requested_device: str) -> str:
    if requested_device != "auto":
        return requested_device
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_fastvlm_checkpoint(model_path: Path, model_base: Optional[str], device: str, label: str) -> LoadedModel:
    model_name = get_model_name_from_path(str(model_path))
    print(f"Loading {label} model from: {model_path}", flush=True)
    tokenizer, model, image_processor, _ = load_pretrained_model(
        str(model_path),
        model_base,
        model_name,
        device=device,
    )
    return LoadedModel(tokenizer=tokenizer, model=model, image_processor=image_processor)


def build_prompt(prompt: str, model_config, conv_mode: str) -> str:
    if model_config.mm_use_im_start_end:
        question = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + prompt
    else:
        question = DEFAULT_IMAGE_TOKEN + "\n" + prompt

    conversation = conv_templates[conv_mode].copy()
    conversation.append_message(conversation.roles[0], question)
    conversation.append_message(conversation.roles[1], None)
    return conversation.get_prompt()


def parse_timestamp_from_frame_path(path: Path) -> Optional[float]:
    match = re.search(r"_([0-9]+(?:\.[0-9]+)?)s\.(?:png|jpe?g)$", path.name, re.IGNORECASE)
    if match:
        return float(match.group(1))
    return None


def format_timestamp(timestamp: Optional[float]) -> str:
    if timestamp is None:
        return "unknown"
    return f"{timestamp:.2f}s"


def get_video_duration(video_path: Path) -> float:
    video = cv2.VideoCapture(str(video_path))
    try:
        if not video.isOpened():
            raise ValueError(f"Unable to open video file: {video_path}")

        fps = video.get(cv2.CAP_PROP_FPS)
        frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        if fps <= 0 or frame_count <= 0:
            raise ValueError(f"Unable to read FPS/frame count from video: {video_path}")

        return frame_count / fps
    finally:
        video.release()


def chunk_time_bounds(duration: float, chunk_index: int, chunk_count: int) -> tuple[float, float]:
    if chunk_count <= 0:
        raise ValueError("chunk_count must be greater than 0.")
    if chunk_index < 0 or chunk_index >= chunk_count:
        raise ValueError("chunk_index must be in [0, chunk_count).")

    chunk_duration = duration / chunk_count
    return chunk_index * chunk_duration, (chunk_index + 1) * chunk_duration


def filter_frames_to_chunk(
    frames: List[FrameInput],
    duration: float,
    chunk_index: int,
    chunk_count: int,
) -> List[FrameInput]:
    start_time, end_time = chunk_time_bounds(duration, chunk_index, chunk_count)
    is_last_chunk = chunk_index == chunk_count - 1
    chunk_frames = []

    for frame in frames:
        if frame.timestamp is None:
            raise ValueError("Video chunking requires timestamped frames.")
        if frame.timestamp < start_time:
            continue
        if is_last_chunk:
            if frame.timestamp <= end_time:
                chunk_frames.append(frame)
        elif frame.timestamp < end_time:
            chunk_frames.append(frame)

    return chunk_frames


def save_frame_inputs(frames: List[FrameInput], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for index, frame in enumerate(frames):
        timestamp = "unknown" if frame.timestamp is None else f"{frame.timestamp:.2f}s"
        frame.image.save(output_dir / f"frame_{index:04d}_{timestamp}.png")


def write_chunk_output(output_path: Path, summary_inputs: List[str]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n\n".join(summary_inputs) + ("\n" if summary_inputs else ""), encoding="utf-8")


def add_optional_arg(command: List[str], name: str, value) -> None:
    if value is not None:
        command.extend([name, str(value)])


def build_chunk_worker_command(args, chunk_index: int, chunk_count: int, chunk_output: Path) -> List[str]:
    chunk_frames_root = Path(args.frames_output_dir).expanduser() / "chunk_workers" / f"chunk_{chunk_index:04d}"
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        args.input_path,
        "--model-path",
        args.model_path,
        "--prompt",
        args.prompt,
        "--conv-mode",
        args.conv_mode,
        "--sampling-mode",
        args.sampling_mode,
        "--sample-every-seconds",
        str(args.sample_every_seconds),
        "--ssim-threshold",
        str(args.ssim_threshold),
        "--ssim-frame-step",
        str(args.ssim_frame_step),
        "--ssim-percentile",
        str(args.ssim_percentile),
        "--frames-output-dir",
        str(chunk_frames_root),
        "--device",
        args.device,
        "--temperature",
        str(args.temperature),
        "--num-beams",
        str(args.num_beams),
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--additionalCaptions",
        str(args.additionalCaptions),
        "--no-event-summary",
        "--chunk-count",
        str(chunk_count),
        "--chunk-index",
        str(chunk_index),
        "--chunk-output",
        str(chunk_output),
    ]
    add_optional_arg(command, "--model-base", args.model_base)
    add_optional_arg(command, "--key_frame_model", args.key_frame_model)
    add_optional_arg(command, "--uniform_frame_model", args.uniform_frame_model)
    add_optional_arg(command, "--top-p", args.top_p)
    add_optional_arg(command, "--limit-frames", args.limit_frames)
    if args.raw_output:
        command.append("--raw-output")
    return command


def run_video_chunk_workers(args) -> None:
    input_path = Path(args.input_path).expanduser()
    if not input_path.is_file() or input_path.suffix.lower() != ".mp4":
        raise ValueError("--videoChunks requires an MP4 video input.")
    if args.videoChunks <= 1:
        raise ValueError("--videoChunks must be greater than 1.")

    chunk_output_dir = Path(args.chunk_output_dir).expanduser()
    chunk_output_dir.mkdir(parents=True, exist_ok=True)

    if args.ssim_plot:
        if args.sampling_mode != "ssim":
            print("Skipping SSIM distribution plot because --sampling-mode is not ssim.", flush=True)
        else:
            plot_dir = sampling_output_dir(
                Path(args.frames_output_dir).expanduser(),
                args.sampling_mode,
                args.sample_every_seconds,
                args.ssim_threshold,
                args.ssim_frame_step,
                args.ssim_percentile,
            )
            plot_path = plot_dir / "ssim_distribution.png"
            preprocess_video_ssim(
                input_path,
                ssim_threshold=args.ssim_threshold,
                ssim_frame_step=args.ssim_frame_step,
                ssim_percentile=args.ssim_percentile,
                ssim_plot_path=plot_path,
            )
            print(f"Saved SSIM distribution plot to: {plot_path.resolve()}", flush=True)

    print(f"Starting {args.videoChunks} chunk worker process(es)...", flush=True)
    processes = []
    for chunk_index in range(args.videoChunks):
        chunk_output = chunk_output_dir / f"chunk_{chunk_index:04d}.txt"
        command = build_chunk_worker_command(args, chunk_index, args.videoChunks, chunk_output)
        print(f"Launching chunk {chunk_index:04d}: {chunk_output}", flush=True)
        processes.append((chunk_index, chunk_output, subprocess.Popen(command)))

    failures = []
    for chunk_index, chunk_output, process in processes:
        return_code = process.wait()
        if return_code != 0:
            failures.append((chunk_index, return_code))
        else:
            print(f"Chunk {chunk_index:04d} complete: {chunk_output}", flush=True)

    if failures:
        failure_text = ", ".join(f"chunk {index} exited {code}" for index, code in failures)
        raise RuntimeError(f"One or more chunk workers failed: {failure_text}")

    print(f"All chunk workers complete. Chunk outputs saved to: {chunk_output_dir.resolve()}", flush=True)

    chunk_outputs = []
    for chunk_index in range(args.videoChunks):
        chunk_output = chunk_output_dir / f"chunk_{chunk_index:04d}.txt"
        if not chunk_output.exists():
            continue

        chunk_text = chunk_output.read_text(encoding="utf-8").strip()
        if chunk_text:
            chunk_outputs.append(f"Chunk {chunk_index:04d}\n{chunk_text}")

    if args.no_event_summary:
        print("Skipping OpenRouter summary because --no-event-summary was set.", flush=True)
        return
    if not chunk_outputs:
        print("Skipping OpenRouter summary because no chunk captions were generated.", flush=True)
        return

    summary_output = (
        Path(args.summary_output).expanduser()
        if args.summary_output
        else chunk_output_dir / "event_summary.txt"
    )
    print(f"Summarizing all chunks with OpenRouter model: {args.openrouter_model}", flush=True)
    summary = summarize_events_with_openrouter(
        chunk_outputs,
        args.openrouter_model,
        summary_output,
        use_enriched_intervals=args.additionalCaptions > 0,
    )
    print("\nEvent summary:", flush=True)
    print(summary, flush=True)
    print(f"Saved event summary to: {summary_output.resolve()}", flush=True)


def read_video_frame_at_timestamp(video_path: Path, timestamp: float) -> FrameInput:
    video = cv2.VideoCapture(str(video_path))
    try:
        if not video.isOpened():
            raise ValueError(f"Unable to open video file: {video_path}")

        fps = video.get(cv2.CAP_PROP_FPS)
        frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        if fps <= 0 or frame_count <= 0:
            raise ValueError(f"Unable to read FPS/frame count from video: {video_path}")

        frame_index = min(max(int(round(timestamp * fps)), 0), frame_count - 1)
        video.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        success, frame_bgr = video.read()
        if not success:
            raise ValueError(f"Unable to read frame at {timestamp:.2f}s from video: {video_path}")

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        return FrameInput(Image.fromarray(frame_rgb).convert("RGB"), frame_index / fps)
    finally:
        video.release()


def sample_additional_interval_frames(
    video_path: Path,
    keyframes: List[FrameInput],
    additional_captions: int,
) -> List[List[FrameInput]]:
    if additional_captions <= 0:
        return [[] for _ in range(max(len(keyframes) - 1, 0))]

    interval_frames = []
    for start_frame, end_frame in zip(keyframes, keyframes[1:]):
        if start_frame.timestamp is None or end_frame.timestamp is None:
            raise ValueError("Additional captions require timestamped keyframes.")
        if end_frame.timestamp <= start_frame.timestamp:
            interval_frames.append([])
            continue

        step = (end_frame.timestamp - start_frame.timestamp) / (additional_captions + 1)
        interval_frames.append(
            [
                read_video_frame_at_timestamp(video_path, start_frame.timestamp + step * index)
                for index in range(1, additional_captions + 1)
            ]
        )

    return interval_frames


def format_frame_caption(caption: FrameCaption) -> str:
    return f"frame {caption.frame_index:04d} timestamp={format_timestamp(caption.timestamp)}: {caption.text}"


def format_interval_captions(interval_index: int, interval: IntervalCaptions) -> str:
    lines = [
        (
            f"Key interval {interval_index:04d} "
            f"[{format_timestamp(interval.start.timestamp)} - {format_timestamp(interval.end.timestamp)}]"
        ),
        f"Start key caption: {format_frame_caption(interval.start)}",
    ]

    if interval.additional:
        lines.append("Additional captions:")
        for caption in interval.additional:
            lines.append(f"  timestamp={format_timestamp(caption.timestamp)}: {caption.text}")
    else:
        lines.append("Additional captions: none")

    lines.append(
        (
            "End keyframe timestamp: "
            f"{format_timestamp(interval.end.timestamp)} "
            "(caption excluded; this keyframe starts the next interval)"
        )
    )
    return "\n".join(lines)


def load_frames_from_video(
    video_path: Path,
    sampling_mode: str,
    sample_every_seconds: float,
    ssim_threshold: SsimThreshold,
    ssim_frame_step: int,
    ssim_percentile: float,
    output_dir: Optional[Path],
    ssim_plot_path: Optional[Path],
) -> List[FrameInput]:
    if sampling_mode == "uniform":
        frame_array, (timestamps, _) = preprocess_video_uniform(
            video_path,
            sample_every_seconds=sample_every_seconds,
            output_dir=output_dir,
            return_metadata=True,
        )
    elif sampling_mode == "ssim":
        frame_array, (timestamps, _) = preprocess_video_ssim(
            video_path,
            ssim_threshold=ssim_threshold,
            ssim_frame_step=ssim_frame_step,
            ssim_percentile=ssim_percentile,
            output_dir=output_dir,
            ssim_plot_path=ssim_plot_path,
            return_metadata=True,
        )
    else:
        raise ValueError(f"Unknown sampling mode: {sampling_mode}")
    return [
        FrameInput(Image.fromarray(frame).convert("RGB"), timestamp)
        for frame, timestamp in zip(frame_array, timestamps)
    ]


def load_frames_from_dir(frames_dir: Path) -> List[FrameInput]:
    image_paths = sorted(
        path for path in frames_dir.iterdir() if path.suffix.lower() in {".png", ".jpg", ".jpeg"}
    )
    if not image_paths:
        raise ValueError(f"No PNG/JPG frames found in: {frames_dir}")
    return [
        FrameInput(Image.open(path).convert("RGB"), parse_timestamp_from_frame_path(path))
        for path in image_paths
    ]


def sampling_output_dir(
    output_root: Path,
    sampling_mode: str,
    sample_every_seconds: float,
    ssim_threshold: SsimThreshold,
    ssim_frame_step: int,
    ssim_percentile: float,
) -> Path:
    if sampling_mode == "uniform":
        return output_root / f"uniform_{sample_every_seconds:.2f}s"
    if sampling_mode == "ssim":
        threshold_label = (
            f"auto_p{ssim_percentile:.0f}"
            if ssim_threshold == "auto"
            else f"threshold{float(ssim_threshold):.2f}"
        )
        return output_root / f"ssim_step{ssim_frame_step}_{threshold_label}"
    raise ValueError(f"Unknown sampling mode: {sampling_mode}")


def load_input_frames(
    input_path: Path,
    sampling_mode: str,
    sample_every_seconds: float,
    ssim_threshold: SsimThreshold,
    ssim_frame_step: int,
    ssim_percentile: float,
    output_dir: Optional[Path],
    ssim_plot_path: Optional[Path],
) -> List[FrameInput]:
    if input_path.is_dir():
        return load_frames_from_dir(input_path)
    if input_path.suffix.lower() == ".mp4":
        return load_frames_from_video(
            input_path,
            sampling_mode,
            sample_every_seconds,
            ssim_threshold,
            ssim_frame_step,
            ssim_percentile,
            output_dir,
            ssim_plot_path,
        )
    raise ValueError("Input must be an .mp4 video file or a folder of sampled PNG/JPG frames.")


def iter_inference(
    frames: Iterable[FrameInput],
    tokenizer,
    model,
    image_processor,
    prompt: str,
    conv_mode: str,
    device: str,
    temperature: float,
    top_p: float,
    num_beams: int,
    max_new_tokens: int,
    limit_frames: Optional[int],
    first_sentence_only: bool,
):
    prompt_text = build_prompt(prompt, model.config, conv_mode)
    input_ids = tokenizer_image_token(
        prompt_text,
        tokenizer,
        IMAGE_TOKEN_INDEX,
        return_tensors="pt",
    ).unsqueeze(0).to(torch.device(device))

    model.generation_config.pad_token_id = tokenizer.pad_token_id

    for frame_index, frame in enumerate(frames):
        if limit_frames is not None and frame_index >= limit_frames:
            break

        image = frame.image
        print(
            f"Running inference on frame {frame_index:04d} at {format_timestamp(frame.timestamp)}...",
            flush=True,
        )
        image_tensor = process_images([image], image_processor, model.config)[0]
        image_tensor = image_tensor.unsqueeze(0).half().to(torch.device(device))

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor,
                image_sizes=[image.size],
                do_sample=temperature > 0,
                temperature=temperature,
                top_p=top_p,
                num_beams=num_beams,
                max_new_tokens=max_new_tokens,
                use_cache=True,
            )

        output = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        if first_sentence_only:
            output = first_sentence(output)
        yield frame_index, frame.timestamp, output


def main() -> None:
    parser = argparse.ArgumentParser(description="Run FastVLM inference on sampled video frames.")
    parser.add_argument("input_path", type=str, help="Path to an MP4 video or folder of sampled frames.")
    parser.add_argument("--model-path", type=str, default=str(DEFAULT_MODEL_PATH))
    parser.add_argument(
        "--key_frame_model",
        type=str,
        default=None,
        help="Optional checkpoint path used only for keyframe captions. Requires --uniform_frame_model.",
    )
    parser.add_argument(
        "--uniform_frame_model",
        type=str,
        default=None,
        help="Optional checkpoint path used only for additional uniform interval captions. Requires --key_frame_model.",
    )
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--prompt", type=str, default=DEFAULT_PROMPT)
    parser.add_argument("--conv-mode", type=str, default="qwen_2")
    parser.add_argument("--sampling-mode", choices=["uniform", "ssim"], default="uniform")
    parser.add_argument("--sample-every-seconds", type=float, default=2.0)
    parser.add_argument("--ssim-threshold", default="0.90")
    parser.add_argument("--ssim-frame-step", type=int, default=20)
    parser.add_argument("--ssim-percentile", type=float, default=25.0)
    parser.add_argument(
        "--ssim-plot",
        action="store_true",
        help="Save a PNG plot of the SSIM distribution and selected threshold in SSIM mode.",
    )
    parser.add_argument("--frames-output-dir", type=str, default="sampled_frames")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "mps", "cuda", "cpu"])
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--num-beams", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--limit-frames", type=int, default=None)
    parser.add_argument(
        "--videoChunks",
        type=int,
        default=1,
        help="Split an MP4 into this many subprocess workers. Each worker loads its own model.",
    )
    parser.add_argument(
        "--chunk-output-dir",
        type=str,
        default="chunk_outputs",
        help="Directory where subprocess chunk caption files are written.",
    )
    parser.add_argument("--chunk-count", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--chunk-index", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--chunk-output", type=str, default=None, help=argparse.SUPPRESS)
    parser.add_argument(
        "--additionalCaptions",
        type=int,
        default=0,
        help="Number of uniformly sampled additional captions inside each keyframe interval.",
    )
    parser.add_argument("--openrouter-model", type=str, default=DEFAULT_OPENROUTER_MODEL)
    parser.add_argument("--summary-output", type=str, default=None)
    parser.add_argument(
        "--no-event-summary",
        action="store_true",
        help="Skip the OpenRouter event-summary step.",
    )
    parser.add_argument(
        "--raw-output",
        action="store_true",
        help="Print the full generated text instead of trimming to one sentence.",
    )
    args = parser.parse_args()

    if args.videoChunks > 1 and args.chunk_count is None:
        run_video_chunk_workers(args)
        return

    input_path = Path(args.input_path).expanduser()
    model_path = Path(args.model_path).expanduser()
    key_frame_model_path = Path(args.key_frame_model).expanduser() if args.key_frame_model else None
    uniform_frame_model_path = Path(args.uniform_frame_model).expanduser() if args.uniform_frame_model else None
    split_model_mode = key_frame_model_path is not None or uniform_frame_model_path is not None
    frames_output_root = Path(args.frames_output_dir).expanduser()
    frames_output_dir = sampling_output_dir(
        frames_output_root,
        args.sampling_mode,
        args.sample_every_seconds,
        args.ssim_threshold,
        args.ssim_frame_step,
        args.ssim_percentile,
    )
    summary_output = (
        Path(args.summary_output).expanduser()
        if args.summary_output
        else frames_output_dir / "event_summary.txt"
    )
    device = choose_device(args.device)

    if split_model_mode and (key_frame_model_path is None or uniform_frame_model_path is None):
        raise ValueError("--key_frame_model and --uniform_frame_model must be provided together.")
    if split_model_mode:
        if not key_frame_model_path.exists():
            raise FileNotFoundError(f"Keyframe model checkpoint not found: {key_frame_model_path}")
        if not uniform_frame_model_path.exists():
            raise FileNotFoundError(f"Uniform-frame model checkpoint not found: {uniform_frame_model_path}")
    elif not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
    if args.limit_frames is not None and args.limit_frames <= 0:
        raise ValueError("--limit-frames must be greater than 0.")
    is_chunk_worker = args.chunk_count is not None or args.chunk_index is not None
    if is_chunk_worker:
        if args.chunk_count is None or args.chunk_index is None:
            raise ValueError("--chunk-count and --chunk-index must be provided together.")
        if args.chunk_count <= 0:
            raise ValueError("--chunk-count must be greater than 0.")
        if args.chunk_index < 0 or args.chunk_index >= args.chunk_count:
            raise ValueError("--chunk-index must be in [0, --chunk-count).")
        if not input_path.is_file() or input_path.suffix.lower() != ".mp4":
            raise ValueError("Chunk workers require an MP4 video input.")
    if args.additionalCaptions < 0:
        raise ValueError("--additionalCaptions must be greater than or equal to 0.")
    if args.additionalCaptions > 0 and not input_path.is_file():
        raise ValueError("--additionalCaptions requires an MP4 video input, not a frame folder.")
    if device == "cpu":
        print("Warning: running FastVLM on CPU will be very slow. Use --device mps or --device cuda when available.", flush=True)

    print(f"Loading frames from: {input_path}", flush=True)
    preprocessing_output_dir = None if is_chunk_worker else frames_output_dir
    frames = load_input_frames(
        input_path,
        args.sampling_mode,
        args.sample_every_seconds,
        args.ssim_threshold,
        args.ssim_frame_step,
        args.ssim_percentile,
        preprocessing_output_dir,
        frames_output_dir / "ssim_distribution.png" if args.ssim_plot and args.sampling_mode == "ssim" else None,
    )
    if args.ssim_plot and args.sampling_mode != "ssim":
        print("Skipping SSIM distribution plot because --sampling-mode is not ssim.", flush=True)
    elif args.ssim_plot and not is_chunk_worker:
        print(f"Saved SSIM distribution plot to: {(frames_output_dir / 'ssim_distribution.png').resolve()}", flush=True)
    if is_chunk_worker:
        duration = get_video_duration(input_path)
        start_time, end_time = chunk_time_bounds(duration, args.chunk_index, args.chunk_count)
        frames = filter_frames_to_chunk(frames, duration, args.chunk_index, args.chunk_count)
        print(
            (
                f"Chunk {args.chunk_index:04d}/{args.chunk_count:04d} "
                f"time range [{start_time:.2f}s, {end_time:.2f}s"
                f"{']' if args.chunk_index == args.chunk_count - 1 else ')'}"
            ),
            flush=True,
        )
        save_frame_inputs(frames, frames_output_dir)
    print(f"Loaded {len(frames)} frame(s).", flush=True)
    if input_path.suffix.lower() == ".mp4":
        print(f"Saved sampled frames to: {frames_output_dir.resolve()}", flush=True)
    if is_chunk_worker and not frames:
        if args.chunk_output:
            write_chunk_output(Path(args.chunk_output).expanduser(), [])
        print("Chunk has no frames; skipping model load.", flush=True)
        return

    disable_torch_init()
    if split_model_mode:
        keyframe_model = load_fastvlm_checkpoint(key_frame_model_path, args.model_base, device, "keyframe")
        if uniform_frame_model_path == key_frame_model_path:
            uniform_model = keyframe_model
        else:
            uniform_model = load_fastvlm_checkpoint(
                uniform_frame_model_path,
                args.model_base,
                device,
                "uniform interval",
            )
    else:
        keyframe_model = load_fastvlm_checkpoint(model_path, args.model_base, device, "FastVLM")
        uniform_model = keyframe_model

    keyframes = frames[:args.limit_frames] if args.limit_frames is not None else frames
    if args.limit_frames is not None:
        print(f"Using first {len(keyframes)} frame(s) after --limit-frames.", flush=True)

    key_captions = []
    for frame_index, timestamp, output in iter_inference(
        keyframes,
        keyframe_model.tokenizer,
        keyframe_model.model,
        keyframe_model.image_processor,
        args.prompt,
        args.conv_mode,
        device,
        args.temperature,
        args.top_p,
        args.num_beams,
        args.max_new_tokens,
        None,
        not args.raw_output,
    ):
        caption = FrameCaption(frame_index, timestamp, output)
        key_captions.append(caption)
        frame_output = format_frame_caption(caption)
        print(frame_output, flush=True)

    summary_inputs = [format_frame_caption(caption) for caption in key_captions]
    use_enriched_intervals = False

    if args.additionalCaptions > 0 and len(keyframes) > 1:
        print(
            f"Sampling {args.additionalCaptions} additional caption frame(s) per keyframe interval...",
            flush=True,
        )
        interval_frame_groups = sample_additional_interval_frames(
            input_path,
            keyframes,
            args.additionalCaptions,
        )

        intervals = []
        for interval_index, additional_frames in enumerate(interval_frame_groups):
            additional_captions = []
            for extra_index, timestamp, output in iter_inference(
                additional_frames,
                uniform_model.tokenizer,
                uniform_model.model,
                uniform_model.image_processor,
                args.prompt,
                args.conv_mode,
                device,
                args.temperature,
                args.top_p,
                args.num_beams,
                args.max_new_tokens,
                None,
                not args.raw_output,
            ):
                caption = FrameCaption(extra_index, timestamp, output)
                additional_captions.append(caption)
                print(
                    (
                        f"interval {interval_index:04d} additional "
                        f"{extra_index:04d} timestamp={format_timestamp(timestamp)}: {output}"
                    ),
                    flush=True,
                )

            intervals.append(
                IntervalCaptions(
                    start=key_captions[interval_index],
                    end=key_captions[interval_index + 1],
                    additional=additional_captions,
                )
            )

        summary_inputs = [
            format_interval_captions(interval_index, interval)
            for interval_index, interval in enumerate(intervals)
        ]
        use_enriched_intervals = True
    elif args.additionalCaptions > 0:
        print("Skipping additional captions because fewer than 2 keyframes were generated.", flush=True)

    if is_chunk_worker:
        if args.chunk_output is not None:
            chunk_output = Path(args.chunk_output).expanduser()
            write_chunk_output(chunk_output, summary_inputs)
            print(f"Saved chunk captions to: {chunk_output.resolve()}", flush=True)
        return

    if not args.no_event_summary:
        if not summary_inputs:
            print("Skipping event summary because no frame outputs were generated.", flush=True)
            return

        print(f"Summarizing events with OpenRouter model: {args.openrouter_model}", flush=True)
        summary = summarize_events_with_openrouter(
            summary_inputs,
            args.openrouter_model,
            summary_output,
            use_enriched_intervals,
        )
        print("\nEvent summary:", flush=True)
        print(summary, flush=True)
        print(f"Saved event summary to: {summary_output.resolve()}", flush=True)


if __name__ == "__main__":
    main()

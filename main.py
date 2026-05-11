import argparse
import re
from pathlib import Path
from typing import Iterable, List, Optional

import torch
from PIL import Image

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


DEFAULT_MODEL_PATH = Path(__file__).resolve().parent / "checkpoints/llava-fastvithd_7b_stage3"
DEFAULT_PROMPT = "Describe the image in one sentence."


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


def build_prompt(prompt: str, model_config, conv_mode: str) -> str:
    if model_config.mm_use_im_start_end:
        question = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + prompt
    else:
        question = DEFAULT_IMAGE_TOKEN + "\n" + prompt

    conversation = conv_templates[conv_mode].copy()
    conversation.append_message(conversation.roles[0], question)
    conversation.append_message(conversation.roles[1], None)
    return conversation.get_prompt()


def load_frames_from_video(
    video_path: Path,
    sampling_mode: str,
    sample_every_seconds: float,
    ssim_threshold: SsimThreshold,
    ssim_frame_step: int,
    ssim_percentile: float,
    output_dir: Path,
) -> List[Image.Image]:
    if sampling_mode == "uniform":
        frame_array = preprocess_video_uniform(
            video_path,
            sample_every_seconds=sample_every_seconds,
            output_dir=output_dir,
        )
    elif sampling_mode == "ssim":
        frame_array = preprocess_video_ssim(
            video_path,
            ssim_threshold=ssim_threshold,
            ssim_frame_step=ssim_frame_step,
            ssim_percentile=ssim_percentile,
            output_dir=output_dir,
        )
    else:
        raise ValueError(f"Unknown sampling mode: {sampling_mode}")
    return [Image.fromarray(frame).convert("RGB") for frame in frame_array]


def load_frames_from_dir(frames_dir: Path) -> List[Image.Image]:
    image_paths = sorted(
        path for path in frames_dir.iterdir() if path.suffix.lower() in {".png", ".jpg", ".jpeg"}
    )
    if not image_paths:
        raise ValueError(f"No PNG/JPG frames found in: {frames_dir}")
    return [Image.open(path).convert("RGB") for path in image_paths]


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
    output_dir: Path,
) -> List[Image.Image]:
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
        )
    raise ValueError("Input must be an .mp4 video file or a folder of sampled PNG/JPG frames.")


def iter_inference(
    frames: Iterable[Image.Image],
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

    for frame_index, image in enumerate(frames):
        if limit_frames is not None and frame_index >= limit_frames:
            break

        print(f"Running inference on frame {frame_index:04d}...", flush=True)
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
        yield frame_index, output


def main() -> None:
    parser = argparse.ArgumentParser(description="Run FastVLM inference on sampled video frames.")
    parser.add_argument("input_path", type=str, help="Path to an MP4 video or folder of sampled frames.")
    parser.add_argument("--model-path", type=str, default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--prompt", type=str, default=DEFAULT_PROMPT)
    parser.add_argument("--conv-mode", type=str, default="qwen_2")
    parser.add_argument("--sampling-mode", choices=["uniform", "ssim"], default="uniform")
    parser.add_argument("--sample-every-seconds", type=float, default=2.0)
    parser.add_argument("--ssim-threshold", default="0.90")
    parser.add_argument("--ssim-frame-step", type=int, default=20)
    parser.add_argument("--ssim-percentile", type=float, default=25.0)
    parser.add_argument("--frames-output-dir", type=str, default="sampled_frames")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "mps", "cuda", "cpu"])
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--num-beams", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--limit-frames", type=int, default=None)
    parser.add_argument(
        "--raw-output",
        action="store_true",
        help="Print the full generated text instead of trimming to one sentence.",
    )
    args = parser.parse_args()

    input_path = Path(args.input_path).expanduser()
    model_path = Path(args.model_path).expanduser()
    frames_output_root = Path(args.frames_output_dir).expanduser()
    frames_output_dir = sampling_output_dir(
        frames_output_root,
        args.sampling_mode,
        args.sample_every_seconds,
        args.ssim_threshold,
        args.ssim_frame_step,
        args.ssim_percentile,
    )
    device = choose_device(args.device)

    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
    if args.limit_frames is not None and args.limit_frames <= 0:
        raise ValueError("--limit-frames must be greater than 0.")
    if device == "cpu":
        print("Warning: running FastVLM on CPU will be very slow. Use --device mps or --device cuda when available.", flush=True)

    print(f"Loading frames from: {input_path}", flush=True)
    frames = load_input_frames(
        input_path,
        args.sampling_mode,
        args.sample_every_seconds,
        args.ssim_threshold,
        args.ssim_frame_step,
        args.ssim_percentile,
        frames_output_dir,
    )
    print(f"Loaded {len(frames)} frame(s).", flush=True)
    if input_path.suffix.lower() == ".mp4":
        print(f"Saved sampled frames to: {frames_output_dir.resolve()}", flush=True)

    disable_torch_init()
    model_name = get_model_name_from_path(str(model_path))
    print(f"Loading model from: {model_path}", flush=True)
    tokenizer, model, image_processor, _ = load_pretrained_model(
        str(model_path),
        args.model_base,
        model_name,
        device=device,
    )

    for frame_index, output in iter_inference(
        frames,
        tokenizer,
        model,
        image_processor,
        args.prompt,
        args.conv_mode,
        device,
        args.temperature,
        args.top_p,
        args.num_beams,
        args.max_new_tokens,
        args.limit_frames,
        not args.raw_output,
    ):
        print(f"frame {frame_index:04d}: {output}", flush=True)


if __name__ == "__main__":
    main()

import argparse
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional


VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".webm"}


@dataclass(frozen=True)
class ClipSpec:
    line_number: int
    video_id: str
    start_time: str
    end_time: str


def parse_time_value(value: str) -> float:
    try:
        return float(value)
    except ValueError as error:
        raise ValueError(f"Expected a numeric timestamp, got {value!r}") from error


def parse_msg_line(line: str, line_number: int) -> Optional[ClipSpec]:
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        return None

    parts = stripped.split()
    if len(parts) == 3:
        video_id, start_time, end_time = parts
    elif len(parts) == 2 and "_" in parts[1]:
        video_id = parts[0]
        start_time, end_time = parts[1].split("_", 1)
    else:
        raise ValueError(
            f"Line {line_number}: expected 'video_id start end' or 'video_id start_end', got {stripped!r}"
        )

    start_seconds = parse_time_value(start_time)
    end_seconds = parse_time_value(end_time)
    if end_seconds <= start_seconds:
        raise ValueError(
            f"Line {line_number}: end_time must be greater than start_time, got {start_time} -> {end_time}"
        )

    return ClipSpec(line_number, video_id, start_time, end_time)


def load_clip_specs(msg_path: Path) -> List[ClipSpec]:
    specs = []
    for line_number, line in enumerate(msg_path.read_text(encoding="utf-8").splitlines(), start=1):
        spec = parse_msg_line(line, line_number)
        if spec is not None:
            specs.append(spec)
    return specs


def video_id_from_filename(path: Path) -> Optional[str]:
    media_match = re.search(r"_Media_(.+)_\d{3}_[^_]+(?: \(\d+\))?$", path.stem)
    if media_match:
        return media_match.group(1)

    parts = path.stem.split("_")
    if len(parts) < 3:
        return None
    return parts[-3]


def build_video_index(video_dir: Path) -> Dict[str, Path]:
    index: Dict[str, Path] = {}
    duplicates: Dict[str, List[Path]] = {}

    for path in sorted(video_dir.iterdir()):
        if not path.is_file() or path.suffix.lower() not in VIDEO_EXTENSIONS:
            continue

        video_id = video_id_from_filename(path)
        if video_id is None:
            continue

        if video_id in index:
            duplicates.setdefault(video_id, [index[video_id]]).append(path)
            continue

        index[video_id] = path

    for video_id, paths in duplicates.items():
        print(
            f"Warning: found {len(paths)} files for video_id {video_id}; using {paths[0].name}",
            flush=True,
        )

    return index


def safe_time_for_filename(value: str) -> str:
    return value.replace(":", "-").replace("/", "-")


def output_path_for_clip(output_dir: Path, spec: ClipSpec) -> Path:
    start = safe_time_for_filename(spec.start_time)
    end = safe_time_for_filename(spec.end_time)
    return output_dir / f"{spec.video_id}_{start}_{end}.mp4"


def ffmpeg_command(
    source_path: Path,
    output_path: Path,
    start_time: str,
    end_time: str,
    reencode: bool,
    overwrite: bool,
) -> List[str]:
    duration = parse_time_value(end_time) - parse_time_value(start_time)
    command = ["ffmpeg", "-hide_banner", "-loglevel", "error"]
    command.append("-y" if overwrite else "-n")
    command.extend(["-ss", start_time, "-i", str(source_path), "-t", f"{duration:.6f}", "-map", "0"])

    if reencode:
        command.extend(["-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-c:a", "aac"])
    else:
        command.extend(["-c", "copy"])

    command.extend(["-movflags", "+faststart", str(output_path)])
    return command


def extract_clip(
    source_path: Path,
    output_path: Path,
    spec: ClipSpec,
    reencode: bool,
    overwrite: bool,
    dry_run: bool,
) -> bool:
    if output_path.is_file() and output_path.stat().st_size > 0 and not overwrite:
        print(f"Skipping existing clip: {output_path}", flush=True)
        return False

    command = ffmpeg_command(
        source_path,
        output_path,
        spec.start_time,
        spec.end_time,
        reencode=reencode,
        overwrite=overwrite,
    )

    if dry_run:
        print(" ".join(command), flush=True)
        return True

    output_path.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(command)
    if result.returncode != 0:
        print(f"Failed to write {output_path}", flush=True)
        return False

    if output_path.exists() and output_path.stat().st_size > 0:
        print(f"Wrote {output_path}", flush=True)
        return True

    print(f"Failed to write {output_path}: output file is missing or empty", flush=True)
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract video clips listed in msg.txt.")
    parser.add_argument("--msg-path", type=Path, default=Path("dataset/msg.txt"))
    parser.add_argument("--video-dir", type=Path, default=Path("dataset/video"))
    parser.add_argument("--output-dir", type=Path, default=Path("dataset/clips"))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--reencode",
        action="store_true",
        help="Re-encode clips for more accurate cuts. Default uses fast stream copy.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    specs = load_clip_specs(args.msg_path)
    video_index = build_video_index(args.video_dir)

    created = 0
    missing = []
    for spec in specs:
        source_path = video_index.get(spec.video_id)
        if source_path is None:
            missing.append(spec.video_id)
            print(f"Warning: line {spec.line_number}: no source video found for {spec.video_id}", flush=True)
            continue

        output_path = output_path_for_clip(args.output_dir, spec)
        if extract_clip(source_path, output_path, spec, args.reencode, args.overwrite, args.dry_run):
            created += 1

    unique_missing = sorted(set(missing))
    print(
        (
            f"Done. requested={len(specs)} created_or_planned={created} "
            f"missing_rows={len(missing)} missing_video_ids={len(unique_missing)}"
        ),
        flush=True,
    )
    if unique_missing:
        print("Missing video IDs: " + ", ".join(unique_missing), flush=True)


if __name__ == "__main__":
    main()

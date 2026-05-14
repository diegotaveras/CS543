import argparse
import subprocess
import sys
from pathlib import Path


DEFAULT_MAIN_ARGS = [
    "--device",
    "mps",
    "--sampling-mode",
    "ssim",
    "--ssim-threshold",
    "auto",
    "--additionalCaptions",
    "2",
    "--videoChunks",
    "10",
]


def summary_path_for_clip(summaries_dir: Path, clip_path: Path) -> Path:
    return summaries_dir / f"{clip_path.stem}.txt"


def command_for_clip(
    main_path: Path,
    clip_path: Path,
    summaries_dir: Path,
    runs_dir: Path,
) -> list[str]:
    clip_run_dir = runs_dir / clip_path.stem
    return [
        sys.executable,
        str(main_path),
        str(clip_path),
        *DEFAULT_MAIN_ARGS,
        "--summary-output",
        str(summary_path_for_clip(summaries_dir, clip_path)),
        "--chunk-output-dir",
        str(clip_run_dir / "chunk_outputs"),
        "--frames-output-dir",
        str(clip_run_dir / "sampled_frames"),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run FastVLM summaries for all clips.")
    parser.add_argument("--clips-dir", type=Path, default=Path("dataset/clips"))
    parser.add_argument("--summaries-dir", type=Path, default=Path("dataset/summaries"))
    parser.add_argument("--runs-dir", type=Path, default=Path("dataset/summary_runs"))
    parser.add_argument("--main-path", type=Path, default=Path("main.py"))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    clips = sorted(args.clips_dir.glob("*.mp4"))
    if not clips:
        raise FileNotFoundError(f"No .mp4 clips found in {args.clips_dir}")

    args.summaries_dir.mkdir(parents=True, exist_ok=True)
    args.runs_dir.mkdir(parents=True, exist_ok=True)

    completed = 0
    skipped = 0
    failed = 0

    for index, clip_path in enumerate(clips, start=1):
        summary_path = summary_path_for_clip(args.summaries_dir, clip_path)
        if summary_path.exists() and summary_path.stat().st_size > 0 and not args.overwrite:
            print(f"[{index}/{len(clips)}] Skipping existing summary: {summary_path}", flush=True)
            skipped += 1
            continue

        command = command_for_clip(args.main_path, clip_path, args.summaries_dir, args.runs_dir)
        print(f"[{index}/{len(clips)}] Summarizing {clip_path}", flush=True)

        if args.dry_run:
            print(" ".join(command), flush=True)
            completed += 1
            continue

        result = subprocess.run(command)
        if result.returncode != 0:
            print(f"Failed: {clip_path}", flush=True)
            failed += 1
            continue

        completed += 1

    print(
        f"Done. clips={len(clips)} completed={completed} skipped={skipped} failed={failed}",
        flush=True,
    )
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

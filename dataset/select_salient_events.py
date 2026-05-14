import argparse
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class SummaryInterval:
    start: float
    end: float
    summary: str


def load_jsonl_by_vid(path: Path) -> Dict[str, Dict[str, Any]]:
    rows_by_vid: Dict[str, Dict[str, Any]] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue

        row = json.loads(line)
        vid = row.get("vid")
        if not vid:
            raise ValueError(f"Line {line_number}: missing 'vid'")
        if vid in rows_by_vid:
            print(f"Warning: duplicate vid {vid}; keeping first row", flush=True)
            continue
        rows_by_vid[vid] = row

    return rows_by_vid


def average_score(score_triplet: List[float]) -> float:
    if not score_triplet:
        raise ValueError("saliency score triplet cannot be empty")
    return sum(score_triplet) / len(score_triplet)


def row_average_saliency(row: Dict[str, Any]) -> float:
    saliency_scores = row.get("saliency_scores", [])
    if not saliency_scores:
        return 0.0

    per_clip_scores = [average_score(scores) for scores in saliency_scores]
    return sum(per_clip_scores) / len(per_clip_scores)


def choose_highest_saliency_clip(row: Dict[str, Any], rng: random.Random) -> tuple[int, float]:
    relevant_clip_ids = row.get("relevant_clip_ids", [])
    saliency_scores = row.get("saliency_scores", [])
    if len(relevant_clip_ids) != len(saliency_scores):
        raise ValueError(
            f"vid {row.get('vid')}: relevant_clip_ids and saliency_scores have different lengths"
        )
    if not relevant_clip_ids:
        raise ValueError(f"vid {row.get('vid')}: no relevant_clip_ids")

    clip_scores = [
        (int(clip_id), average_score(scores))
        for clip_id, scores in zip(relevant_clip_ids, saliency_scores)
    ]
    max_score = max(score for _, score in clip_scores)
    tied_clip_ids = [clip_id for clip_id, score in clip_scores if score == max_score]
    return rng.choice(tied_clip_ids), max_score


def parse_summary_intervals(summary_path: Path) -> List[SummaryInterval]:
    intervals = []
    pattern = re.compile(
        r"^\[\s*([0-9]+(?:\.[0-9]+)?)\s*s?\s*(?:[–-]|,)\s*([0-9]+(?:\.[0-9]+)?)\s*s?\s*\]\s*:\s*(.+)$"
    )

    for line_number, line in enumerate(summary_path.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue

        match = pattern.match(stripped)
        if not match:
            print(f"Warning: {summary_path}:{line_number}: could not parse summary interval", flush=True)
            continue

        start = float(match.group(1))
        end = float(match.group(2))
        summary = match.group(3).strip()
        intervals.append(SummaryInterval(start, end, summary))

    return intervals


def interval_distance(start_a: float, end_a: float, start_b: float, end_b: float) -> float:
    if start_a <= end_b and start_b <= end_a:
        return 0.0
    if end_a < start_b:
        return start_b - end_a
    return start_a - end_b


def closest_summary_interval(
    intervals: List[SummaryInterval],
    selected_clip_id: int,
    segment_seconds: float,
) -> SummaryInterval:
    selected_start = selected_clip_id * segment_seconds
    selected_end = selected_start + segment_seconds
    selected_center = (selected_start + selected_end) / 2

    def sort_key(interval: SummaryInterval) -> tuple[float, float]:
        distance = interval_distance(selected_start, selected_end, interval.start, interval.end)
        interval_center = (interval.start + interval.end) / 2
        return distance, abs(interval_center - selected_center)

    return min(intervals, key=sort_key)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Match each clip/query saliency row to the closest generated event summary."
    )
    parser.add_argument("--clips-dir", type=Path, default=Path("dataset/clips"))
    parser.add_argument("--summaries-dir", type=Path, default=Path("dataset/summaries"))
    parser.add_argument("--jsonl-path", type=Path, default=Path("dataset/test.jsonl"))
    parser.add_argument("--output-path", type=Path, default=Path("dataset/salient_event_matches.jsonl"))
    parser.add_argument("--segment-seconds", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    rows_by_vid = load_jsonl_by_vid(args.jsonl_path)
    clip_paths = sorted(args.clips_dir.glob("*.mp4"))

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    skipped = 0

    with args.output_path.open("w", encoding="utf-8") as output_file:
        for clip_path in clip_paths:
            vid = clip_path.stem
            row = rows_by_vid.get(vid)
            if row is None:
                print(f"Warning: no test.jsonl row found for {vid}", flush=True)
                skipped += 1
                continue

            summary_path = args.summaries_dir / f"{vid}.txt"
            if not summary_path.exists():
                print(f"Warning: no summary file found for {vid}", flush=True)
                skipped += 1
                continue

            intervals = parse_summary_intervals(summary_path)
            if not intervals:
                print(f"Warning: no parseable summary intervals found for {vid}", flush=True)
                skipped += 1
                continue

            selected_clip_id, _ = choose_highest_saliency_clip(row, rng)
            matched_interval = closest_summary_interval(
                intervals,
                selected_clip_id,
                args.segment_seconds,
            )

            output_row = {
                "filename": clip_path.name,
                "query": row["query"],
                "event_summary": matched_interval.summary,
                "average_saliency_score": row_average_saliency(row),
            }
            output_file.write(json.dumps(output_row, ensure_ascii=False) + "\n")
            written += 1

    print(f"Done. written={written} skipped={skipped} output={args.output_path}", flush=True)


if __name__ == "__main__":
    main()

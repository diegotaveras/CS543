import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import torch
from sentence_transformers import SentenceTransformer, util


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def embed_texts(
    texts: List[str],
    model: SentenceTransformer,
    batch_size: int,
) -> torch.Tensor:
    return model.encode(
        texts,
        batch_size=batch_size,
        convert_to_tensor=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )


def choose_device(requested_device: str) -> torch.device:
    if requested_device == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(requested_device)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute SentenceTransformer cosine similarity for query/event summaries."
    )
    parser.add_argument("--input-path", type=Path, default=Path("dataset/salient_event_matches.jsonl"))
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("dataset/sentence_transformer_similarity_scores.jsonl"),
    )
    parser.add_argument("--model-name", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    rows = load_jsonl(args.input_path)
    if not rows:
        raise ValueError(f"No rows found in {args.input_path}")

    device = choose_device(args.device)
    model = SentenceTransformer(args.model_name, device=str(device))

    queries = [row["query"] for row in rows]
    summaries = [row["event_summary"] for row in rows]

    query_embeddings = embed_texts(queries, model, args.batch_size)
    summary_embeddings = embed_texts(summaries, model, args.batch_size)
    similarities = util.cos_sim(query_embeddings, summary_embeddings).diagonal()

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    with args.output_path.open("w", encoding="utf-8") as output_file:
        for row, similarity in zip(rows, similarities.tolist()):
            output_row = dict(row)
            output_row["sentence_transformer_model"] = args.model_name
            output_row["sentence_transformer_cosine_similarity"] = similarity
            output_file.write(json.dumps(output_row, ensure_ascii=False) + "\n")

    mean_similarity = float(similarities.mean().item())
    print(
        f"Done. rows={len(rows)} mean_sentence_transformer_cosine_similarity={mean_similarity:.6f} output={args.output_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()

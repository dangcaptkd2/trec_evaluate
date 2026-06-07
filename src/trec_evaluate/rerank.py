from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from .cache import key_for, load_json, save_json


@dataclass(frozen=True)
class Candidate:
    doc_id: str
    score: float
    text: str = ""


class CrossEncoderReranker:
    def __init__(self, model_name: str, cache_dir: str | Path = "cache/cross_encoder", batch_size: int = 16, max_length: int = 512):
        self.model_name = model_name
        self.cache_dir = Path(cache_dir)
        self.batch_size = batch_size
        self.max_length = max_length
        self._model = None

    def _load_model(self):
        if self._model is not None:
            return self._model
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        model = AutoModelForSequenceClassification.from_pretrained(self.model_name, trust_remote_code=True)
        model.to(device)
        model.eval()
        self._model = (torch, device, tokenizer, model)
        return self._model

    def score(
        self,
        query: str,
        candidates: list[Candidate],
        progress: Callable[[int, int], None] | None = None,
    ) -> list[float]:
        scores: list[float | None] = [None] * len(candidates)
        missing: list[tuple[int, Candidate, Path]] = []
        for idx, candidate in enumerate(candidates):
            cache_path = self.cache_dir / self.model_name.replace("/", "__") / f"{key_for(query, candidate.doc_id, candidate.text)}.json"
            cached = load_json(cache_path)
            if cached is not None and "score" in cached:
                scores[idx] = float(cached["score"])
            else:
                missing.append((idx, candidate, cache_path))

        if missing:
            torch, device, tokenizer, model = self._load_model()
            for start in range(0, len(missing), self.batch_size):
                batch = missing[start : start + self.batch_size]
                pairs = [[query, item[1].text] for item in batch]
                with torch.no_grad():
                    encoded = tokenizer(
                        pairs,
                        truncation=True,
                        padding=True,
                        return_tensors="pt",
                        max_length=self.max_length,
                    )
                    encoded = {k: v.to(device) for k, v in encoded.items()}
                    logits = model(**encoded).logits
                    if logits.ndim == 2 and logits.shape[1] > 1:
                        batch_scores = logits[:, -1].detach().cpu().tolist()
                    else:
                        batch_scores = logits.reshape(-1).detach().cpu().tolist()
                for (idx, candidate, cache_path), score in zip(batch, batch_scores):
                    scores[idx] = float(score)
                    save_json(cache_path, {"doc_id": candidate.doc_id, "model": self.model_name, "score": float(score)})
                if progress is not None:
                    progress(min(start + len(batch), len(missing)), len(missing))

        return [float(score or 0.0) for score in scores]


def rerank_candidates(
    query: str,
    candidates: list[Candidate],
    reranker: CrossEncoderReranker,
    window: int,
    keep: int | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> list[Candidate]:
    head = candidates[:window]
    tail = candidates[window:]
    scores = reranker.score(query, head, progress=progress)
    reranked = [
        Candidate(doc_id=candidate.doc_id, score=score, text=candidate.text)
        for candidate, score in zip(head, scores)
    ]
    reranked.sort(key=lambda item: item.score, reverse=True)
    if keep is not None:
        return reranked[:keep] + reranked[keep:] + tail
    return reranked + tail

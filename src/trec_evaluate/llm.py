from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import time
from typing import Any

from .cache import key_for, load_json, save_json
from .rerank import Candidate


PROMPT = """You are a clinical trial eligibility expert. Evaluate if this patient matches the trial eligibility criteria.

PATIENT PROFILE:
{query}

TRIAL:
{trial}

Rate the match on a scale of 0.0 to 1.0 where:
- 1.0 = Patient clearly meets all inclusion criteria and does not meet any exclusion criteria
- 0.8-0.9 = Patient likely eligible with minor uncertainties
- 0.5-0.7 = Some criteria match but significant uncertainties or potential exclusions
- 0.0-0.4 = Patient likely does not meet key criteria

Respond only with JSON:
{{"score": <float between 0.0 and 1.0>, "label": "eligible|ineligible|unknown", "reasoning": "<1-2 sentences, maximum 50 words>"}}
"""


class LlmReranker:
    def __init__(
        self,
        provider: str = "openai",
        model: str = "gpt-4.1-nano",
        cache_dir: str | Path = "cache/llm",
        temperature: float = 0.0,
        max_retries: int = 3,
        max_workers: int = 1,
    ):
        self.provider = provider
        self.model = model
        self.cache_dir = Path(cache_dir)
        self.temperature = temperature
        self.max_retries = max_retries
        self.max_workers = max(1, max_workers)

    def score(
        self,
        query: str,
        candidates: list[Candidate],
        progress: Callable[[int, int], None] | None = None,
    ) -> list[float]:
        scores: list[float | None] = [None] * len(candidates)
        missing: list[tuple[int, Candidate]] = []
        for idx, candidate in enumerate(candidates):
            cache_path = self._cache_path(query, candidate)
            cached = load_json(cache_path)
            if cached is not None and "score" in cached:
                scores[idx] = float(cached["score"])
            else:
                missing.append((idx, candidate))

        if missing:
            if self.max_workers == 1 or len(missing) == 1:
                for done, (idx, candidate) in enumerate(missing, start=1):
                    scores[idx] = self._score_one_uncached(query, candidate)
                    if progress is not None:
                        progress(done, len(missing))
            else:
                workers = min(self.max_workers, len(missing))
                with ThreadPoolExecutor(max_workers=workers) as executor:
                    futures = {
                        executor.submit(self._score_one_uncached, query, candidate): idx
                        for idx, candidate in missing
                    }
                    for done, future in enumerate(as_completed(futures), start=1):
                        scores[futures[future]] = future.result()
                        if progress is not None:
                            progress(done, len(missing))

        return [float(score or 0.0) for score in scores]

    def _score_one(self, query: str, candidate: Candidate) -> float:
        cache_path = self._cache_path(query, candidate)
        cached = load_json(cache_path)
        if cached is not None and "score" in cached:
            return float(cached["score"])

        return self._score_one_uncached(query, candidate)

    def _score_one_uncached(self, query: str, candidate: Candidate) -> float:
        cache_path = self._cache_path(query, candidate)
        prompt = PROMPT.format(query=query, trial=candidate.text)
        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                data = self._call(prompt)
                score = float(data["score"])
                save_json(cache_path, {"doc_id": candidate.doc_id, "model": self.model, **data})
                return score
            except Exception as e:
                last_error = e
                time.sleep(2**attempt)
        raise RuntimeError(f"LLM scoring failed for {candidate.doc_id}: {last_error}") from last_error

    def _cache_path(self, query: str, candidate: Candidate) -> Path:
        return self.cache_dir / self.provider / self.model.replace("/", "__") / f"{key_for(query, candidate.doc_id, candidate.text)}.json"

    def _call(self, prompt: str) -> dict[str, Any]:
        if self.provider in {"openai", "vllm"}:
            from openai import OpenAI

            kwargs: dict[str, str] = {}
            if self.provider == "vllm":
                kwargs["base_url"] = os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1")
                kwargs["api_key"] = os.getenv("VLLM_API_KEY", "EMPTY")
            elif os.getenv("OPENAI_BASE_URL"):
                kwargs["base_url"] = os.environ["OPENAI_BASE_URL"]
            client = OpenAI(**kwargs)
            response = client.chat.completions.create(
                model=self.model,
                temperature=self.temperature,
                response_format={"type": "json_object"},
                messages=[{"role": "user", "content": prompt}],
            )
            content = response.choices[0].message.content or "{}"
            return json.loads(content)

        if self.provider == "gemini":
            from google import genai

            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                raise RuntimeError("GEMINI_API_KEY is required for Gemini reranking")
            client = genai.Client(api_key=api_key)
            response = client.models.generate_content(model=self.model, contents=prompt)
            text = (response.text or "{}").strip()
            if text.startswith("```"):
                text = text.strip("`")
                text = text.removeprefix("json").strip()
            return json.loads(text)

        raise ValueError(f"Unsupported LLM provider: {self.provider}")


def llm_rerank_candidates(
    query: str,
    candidates: list[Candidate],
    reranker: LlmReranker,
    window: int,
    keep: int | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> list[Candidate]:
    head = candidates[:window]
    tail = candidates[window:]
    scores = reranker.score(query, head, progress=progress)
    reranked = [Candidate(doc_id=c.doc_id, score=s, text=c.text) for c, s in zip(head, scores)]
    reranked.sort(key=lambda item: item.score, reverse=True)
    if keep is not None:
        return reranked[:keep] + reranked[keep:] + tail
    return reranked + tail

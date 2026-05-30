from __future__ import annotations

import json
import os
from pathlib import Path
import time
from typing import Any

from .cache import key_for, load_json, save_json
from .rerank import Candidate


PROMPT = """You are ranking clinical trials for a TREC Clinical Trials patient topic.
Score the trial for patient-trial matching usefulness.
Return only JSON: {"score": number, "label": "eligible|excluded|unknown", "reason": "short reason"}.

Patient topic:
{query}

Trial:
{trial}
"""


class LlmReranker:
    def __init__(
        self,
        provider: str = "openai",
        model: str = "gpt-4.1-nano",
        cache_dir: str | Path = "cache/llm",
        temperature: float = 0.0,
        max_retries: int = 3,
    ):
        self.provider = provider
        self.model = model
        self.cache_dir = Path(cache_dir)
        self.temperature = temperature
        self.max_retries = max_retries

    def score(self, query: str, candidates: list[Candidate]) -> list[float]:
        return [self._score_one(query, candidate) for candidate in candidates]

    def _score_one(self, query: str, candidate: Candidate) -> float:
        cache_path = self.cache_dir / self.provider / self.model.replace("/", "__") / f"{key_for(query, candidate.doc_id, candidate.text)}.json"
        cached = load_json(cache_path)
        if cached is not None and "score" in cached:
            return float(cached["score"])

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

    def _call(self, prompt: str) -> dict[str, Any]:
        if self.provider == "openai":
            from openai import OpenAI

            kwargs: dict[str, str] = {}
            if os.getenv("OPENAI_BASE_URL"):
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


def llm_rerank_candidates(query: str, candidates: list[Candidate], reranker: LlmReranker, window: int, keep: int | None = None) -> list[Candidate]:
    head = candidates[:window]
    tail = candidates[window:]
    scores = reranker.score(query, head)
    reranked = [Candidate(doc_id=c.doc_id, score=s, text=c.text) for c, s in zip(head, scores)]
    reranked.sort(key=lambda item: item.score, reverse=True)
    if keep is not None:
        return reranked[:keep] + reranked[keep:] + tail
    return reranked + tail

from __future__ import annotations

import json
import os
from pathlib import Path
import time
from typing import Any

from .cache import key_for, load_json, save_json


PROMPT = """You expand clinical trial search queries for BM25 retrieval.

Extract only clinical concepts explicitly present in the patient topic, then add safe lexical variants.
Do not infer missing patient facts. Do not add biomarkers, stages, drugs, or treatments unless they are explicitly mentioned.

Return only JSON in this format:
{{
  "diseases": ["synonyms, abbreviations, spelling variants"],
  "biomarkers": ["explicit biomarkers and aliases"],
  "treatments": ["explicit treatments and aliases"],
  "demographics": ["explicit age/sex/population terms"],
  "other_terms": ["other explicit clinical concepts and aliases"],
  "related_concepts": ["closely related search terms, no new patient facts"]
}}

Patient topic:
{query}
"""


class QueryExpander:
    def __init__(
        self,
        provider: str = "openai",
        model: str = "gpt-4.1-nano",
        cache_dir: str | Path = "cache/query_expansion",
        temperature: float = 0.0,
        max_retries: int = 3,
        max_terms_per_category: int = 8,
        use_related_concepts: bool = True,
    ):
        self.provider = provider
        self.model = model
        self.cache_dir = Path(cache_dir)
        self.temperature = temperature
        self.max_retries = max_retries
        self.max_terms_per_category = max_terms_per_category
        self.use_related_concepts = use_related_concepts

    def expand(self, topic_id: str, query: str) -> str:
        cache_path = self.cache_dir / self.provider / self.model.replace("/", "__") / f"{topic_id}_{key_for(query)}.json"
        cached = load_json(cache_path)
        if cached is not None and "expanded_query" in cached:
            return str(cached["expanded_query"])

        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                data = self._call(PROMPT.format(query=query))
                terms = self._terms_from_response(data)
                expanded_query = self._build_query(query, terms)
                save_json(
                    cache_path,
                    {
                        "topic_id": topic_id,
                        "provider": self.provider,
                        "model": self.model,
                        "original_query": query,
                        "expanded_terms": terms,
                        "expanded_query": expanded_query,
                    },
                )
                return expanded_query
            except Exception as e:
                last_error = e
                time.sleep(2**attempt)
        raise RuntimeError(f"Query expansion failed for topic {topic_id}: {last_error}") from last_error

    def _terms_from_response(self, data: dict[str, Any]) -> dict[str, list[str]]:
        categories = ["diseases", "biomarkers", "treatments", "demographics", "other_terms"]
        if self.use_related_concepts:
            categories.append("related_concepts")

        terms: dict[str, list[str]] = {}
        for category in categories:
            raw_values = data.get(category, [])
            if not isinstance(raw_values, list):
                continue
            clean_values: list[str] = []
            seen: set[str] = set()
            for value in raw_values:
                term = " ".join(str(value).split())
                key = term.lower()
                if term and key not in seen:
                    clean_values.append(term)
                    seen.add(key)
                if len(clean_values) >= self.max_terms_per_category:
                    break
            if clean_values:
                terms[category] = clean_values
        return terms

    def _build_query(self, query: str, terms: dict[str, list[str]]) -> str:
        expanded_lines = [query.strip()]
        flat_terms: list[str] = []
        seen: set[str] = set()
        for values in terms.values():
            for term in values:
                key = term.lower()
                if key not in seen:
                    flat_terms.append(term)
                    seen.add(key)
        if flat_terms:
            expanded_lines.append("expanded_terms: " + " ".join(flat_terms))
        return "\n".join(line for line in expanded_lines if line).strip()

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
                raise RuntimeError("GEMINI_API_KEY is required for Gemini query expansion")
            client = genai.Client(api_key=api_key)
            response = client.models.generate_content(model=self.model, contents=prompt)
            text = (response.text or "{}").strip()
            if text.startswith("```"):
                text = text.strip("`")
                text = text.removeprefix("json").strip()
            return json.loads(text)

        raise ValueError(f"Unsupported query expansion provider: {self.provider}")

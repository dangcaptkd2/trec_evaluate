from __future__ import annotations

import json
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen


class ElasticsearchHttpClient:
    def __init__(self, base_url: str = "http://localhost:9200"):
        self.base_url = base_url.rstrip("/")

    def request(self, method: str, path: str, body: dict[str, Any] | None = None) -> dict[str, Any]:
        url = f"{self.base_url}{path}"
        data = json.dumps(body).encode("utf-8") if body is not None else None
        req = Request(url, data=data, method=method, headers={"Content-Type": "application/json"})
        try:
            with urlopen(req, timeout=120) as resp:
                raw = resp.read()
        except HTTPError as e:
            detail = e.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Elasticsearch HTTP {e.code} for {method} {path}: {detail}") from e
        except URLError as e:
            raise RuntimeError(f"Cannot connect to Elasticsearch at {self.base_url}: {e.reason}") from e
        return json.loads(raw.decode("utf-8")) if raw else {}

    def text(self, path: str) -> str:
        try:
            with urlopen(f"{self.base_url}{path}", timeout=30) as resp:
                return resp.read().decode("utf-8")
        except URLError as e:
            raise RuntimeError(f"Cannot connect to Elasticsearch at {self.base_url}: {e.reason}") from e

    def indices(self) -> str:
        return self.text("/_cat/indices?v")

    def mapping(self, index: str) -> dict[str, Any]:
        return self.request("GET", f"/{quote(index)}/_mapping")

    def count(self, index: str) -> int:
        return int(self.request("GET", f"/{quote(index)}/_count").get("count", 0))

    def sample(self, index: str) -> dict[str, Any]:
        body = {"size": 1, "query": {"match_all": {}}}
        hits = self.request("GET", f"/{quote(index)}/_search", body).get("hits", {}).get("hits", [])
        return hits[0] if hits else {}

    def search(self, index: str, query_text: str, fields: list[str], size: int) -> list[dict[str, Any]]:
        query_text = " ".join((query_text or "").split())
        if not query_text:
            return []
        body = {
            "size": size,
            "_source": True,
            "query": {
                "multi_match": {
                    "query": query_text,
                    "fields": fields,
                    "type": "best_fields",
                    "operator": "or",
                }
            },
        }
        hits = self.request("GET", f"/{quote(index)}/_search", body).get("hits", {}).get("hits", [])
        return hits


def discover_text_fields(mapping: dict[str, Any], configured_fields: list[str]) -> list[str]:
    properties: dict[str, Any] = {}
    for index_mapping in mapping.values():
        properties = index_mapping.get("mappings", {}).get("properties", {})
        if properties:
            break
    text_fields = {name for name, spec in properties.items() if spec.get("type") in {"text", "keyword"}}
    fields = [field for field in configured_fields if field.split("^", 1)[0] in text_fields]
    return fields or configured_fields


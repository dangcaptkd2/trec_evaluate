from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv


DEFAULT_CONFIG_PATH = Path("configs/trec2023.yaml")


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    load_dotenv()
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    es_url = os.getenv("ELASTICSEARCH_URL")
    if es_url:
        config.setdefault("elasticsearch", {})["url"] = es_url

    llm_model = os.getenv("LLM_MODEL")
    if llm_model:
        config.setdefault("llm", {})["model"] = llm_model

    return config


def resolve_path(path: str | Path, base: str | Path = ".") -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    return Path(base) / p


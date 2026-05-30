from __future__ import annotations

from pathlib import Path
from urllib.request import Request, urlopen


def download_file(url: str, output_path: str | Path, overwrite: bool = False) -> Path:
    path = Path(output_path)
    if path.exists() and not overwrite:
        return path

    path.parent.mkdir(parents=True, exist_ok=True)
    req = Request(url, headers={"User-Agent": "trec-evaluate/0.1"})
    with urlopen(req, timeout=60) as resp:
        data = resp.read()
    path.write_bytes(data)
    return path

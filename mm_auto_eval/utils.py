from __future__ import annotations

import base64
import csv
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


@dataclass
class Sample:
    id: str
    image_path: str
    subject_description: str
    text_anchor: str
    target_text: str
    relation: str
    prompt: str


def load_dotenv_if_present() -> None:
    try:
        from dotenv import load_dotenv  # type: ignore

        preserve = (os.getenv("DOTENV_PRESERVE_ENV") or "").strip().lower() in {"1", "true", "yes", "y"}
        load_dotenv(override=not preserve)
    except Exception:
        pass


def b64_image(path: Path) -> str:
    data = path.read_bytes()
    return base64.b64encode(data).decode("ascii")


def json_dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False)


def normalize_text(s: str) -> str:
    """Remove whitespace; casefold for Latin. Used for matching OCR text with target_text."""
    if s is None:
        return ""
    s = str(s).strip()
    s = "".join(ch for ch in s if not ch.isspace())
    return s.casefold()


def read_samples(path: Path) -> List[Sample]:
    if not path.exists():
        raise SystemExit(f"Missing input file: {path}")

    ext = path.suffix.lower()
    rows: List[Dict[str, Any]] = []
    if ext == ".jsonl":
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if isinstance(obj, dict):
                rows.append(obj)
    elif ext == ".json":
        obj = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(obj, dict) and isinstance(obj.get("samples"), list):
            rows = [r for r in obj["samples"] if isinstance(r, dict)]
        elif isinstance(obj, dict) and isinstance(obj.get("cases"), list):
            rows = [r for r in obj["cases"] if isinstance(r, dict)]
        elif isinstance(obj, list):
            rows = [r for r in obj if isinstance(r, dict)]
        else:
            raise SystemExit(f"Unsupported JSON shape: {path}")
    elif ext == ".csv":
        with path.open("r", encoding="utf-8", newline="") as f:
            for r in csv.DictReader(f):
                rows.append(dict(r))
    else:
        raise SystemExit(f"Unsupported input extension: {ext} (use json/jsonl/csv)")

    out: List[Sample] = []
    for i, r in enumerate(rows):
        sid = r.get("id") or r.get("case_id") or r.get("sample_id") or str(i)
        image_path = r.get("image_path") or r.get("image") or r.get("path")
        if not image_path:
            continue
        out.append(
            Sample(
                id=str(sid),
                image_path=str(image_path),
                subject_description=str(r.get("subject_description") or ""),
                text_anchor=str(r.get("text_anchor") or ""),
                target_text=str(r.get("target_text") or ""),
                relation=str(r.get("relation") or "unknown"),
                prompt=str(r.get("prompt") or ""),
            )
        )
    return out


def load_done_ids(results_path: Path) -> set[str]:
    if not results_path.exists():
        return set()
    done: set[str] = set()
    with results_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if isinstance(obj, dict) and isinstance(obj.get("id"), str):
                done.add(obj["id"])
    return done


def write_replace_by_id(results_path: Path, record: Dict[str, Any]) -> None:
    sid = record["id"]
    kept: List[str] = []
    if results_path.exists():
        for line in results_path.read_text(encoding="utf-8").splitlines():
            raw = line.strip()
            if not raw:
                continue
            try:
                obj = json.loads(raw)
            except Exception:
                kept.append(raw)
                continue
            if isinstance(obj, dict) and obj.get("id") == sid:
                continue
            kept.append(raw)
    kept.append(json_dumps(record))
    results_path.write_text("\n".join(kept) + "\n", encoding="utf-8")


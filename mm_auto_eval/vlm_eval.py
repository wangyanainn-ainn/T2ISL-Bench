from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, Tuple

from .prompts import VLM_SSP_SLR_PROMPT
from .utils import b64_image, load_dotenv_if_present


def _openai_client():
    load_dotenv_if_present()
    from openai import OpenAI  # type: ignore
    import httpx  # type: ignore

    base_url = (os.getenv("API_BASE_URL") or "").strip() or None
    # For evaluation (chat/vision), prefer OPENAI_API_KEY first.
    # IMAGE_API_KEY is often provisioned for image generation only.
    api_key = (
        (os.getenv("OPENAI_API_KEY") or "").strip()
        or (os.getenv("IMAGE_API_KEY") or "").strip()
        or (os.getenv("VLLM_API_KEY") or "").strip()
    )
    if not api_key:
        raise RuntimeError("Missing IMAGE_API_KEY / OPENAI_API_KEY / VLLM_API_KEY in env/.env")

    timeout_s = float(os.getenv("AUDIT_TIMEOUT_SECONDS", "180"))
    connect_s = float(os.getenv("AUDIT_CONNECT_TIMEOUT_SECONDS", "30"))
    http_client = httpx.Client(timeout=httpx.Timeout(timeout_s, connect=connect_s))
    return OpenAI(api_key=api_key, base_url=base_url, http_client=http_client)


def _extract_json_object(text: str) -> Dict[str, Any]:
    text = text or ""
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return json.loads(text[start : end + 1])
    raise ValueError(f"Model did not return valid JSON. content_prefix={text[:200]!r}")


def _is_retryable_error(e: Exception) -> bool:
    # We intentionally avoid importing OpenAI exception classes here because
    # different gateways / SDK versions may vary. Use class-name heuristics.
    name = e.__class__.__name__
    if name in {
        "APIConnectionError",
        "APITimeoutError",
        "RateLimitError",
        "InternalServerError",
        "ServiceUnavailableError",
    }:
        return True
    # httpx/network-ish errors often end up here
    s = repr(e)
    if "Connection error" in s or "ConnectTimeout" in s or "ReadTimeout" in s or "timed out" in s:
        return True
    return False


def call_ssp_slr(
    image_path: str,
    subject_description: str,
    text_anchor: str,
    target_text: str,
    relation: str,
    prompt: str,
) -> Tuple[Dict[str, Any], str]:
    model = (os.getenv("AUDIT_MODEL") or os.getenv("MM_EVAL_MODEL") or "gemini-3-pro").strip()
    client = _openai_client()
    max_retries = int((os.getenv("AUDIT_MAX_RETRIES") or "3").strip() or "3")
    retry_sleep_s = float((os.getenv("AUDIT_RETRY_SLEEP_SECONDS") or "2").strip() or "2")

    msg = VLM_SSP_SLR_PROMPT.format(
        subject_description=subject_description,
        text_anchor=text_anchor,
        target_text=target_text,
        relation=relation,
        prompt=prompt,
    )
    image_b64 = b64_image(__import__("pathlib").Path(image_path))
    last_err: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": msg},
                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}},
                        ],
                    }
                ],
                temperature=0,
            )
            raw = resp.choices[0].message.content or ""
            parsed = _extract_json_object(raw)
            return parsed, raw
        except Exception as e:
            last_err = e
            if attempt >= max_retries or not _is_retryable_error(e):
                raise
            time.sleep(retry_sleep_s)
    raise RuntimeError(f"unreachable: exhausted retries but no result; last_err={last_err!r}")


import os
import base64

import json
import re
from io import BytesIO
import httpx
from openai import OpenAI

# Qwen image models use DashScope HTTP API (not OpenAI images.generate).
QWEN_IMAGE_MODEL_PREFIX = "qwen-image"
DOUBAO_IMAGE_MODEL_PREFIX = "doubao-"


def get_image_models():
    """
    Returns a list of image model names to try (in order).
    Configure with IMAGE_MODELS="modelA,modelB" or IMAGE_MODEL="modelA".
    """
    models_env = (os.getenv("IMAGE_MODELS") or "").strip()
    if models_env:
        return [m.strip() for m in models_env.split(",") if m.strip()]

    model = (os.getenv("IMAGE_MODEL") or "").strip() or "gpt-image-1.5"
    return [model]


def _select_image_api_key() -> str:
    """Return the first usable key: IMAGE_API_KEY, then OPENAI_API_KEY, then VLLM_API_KEY."""
    img_key = (os.getenv("IMAGE_API_KEY") or "").strip()
    if img_key and "your" not in img_key.lower():
        return img_key

    openai_key = (os.getenv("OPENAI_API_KEY") or "").strip()
    if openai_key and "your" not in openai_key.lower():
        return openai_key

    vllm_key = (os.getenv("VLLM_API_KEY") or "").strip()
    if vllm_key:
        try:
            vllm_key.encode("ascii")
            if "your" not in vllm_key.lower():
                return vllm_key
        except UnicodeEncodeError:
            pass

    return ""


def _normalize_openai_compat_base_url(base_url: str) -> str:
    """Strip whitespace; callers should set API_BASE_URL to a proper OpenAI-compatible origin (usually ending in /v1)."""
    return base_url.strip()


def _httpx_follow_redirects() -> bool:
    """Default True. Set IMAGE_HTTPX_FOLLOW_REDIRECTS=0 to disable if a gateway returns HTML on redirects."""
    raw = (os.getenv("IMAGE_HTTPX_FOLLOW_REDIRECTS") or "").strip().lower()
    if raw in {"0", "false", "no", "n"}:
        return False
    return True


def generate_image(prompt, model=None):

    base_url = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
    if isinstance(base_url, str):
        base_url = base_url.strip()
    # Guardrail: pasted .env sometimes concatenates two variables into API_BASE_URL.
    if isinstance(base_url, str) and "IMAGE_MODEL=" in base_url:
        raise RuntimeError(
            "API_BASE_URL looks corrupted: it contains 'IMAGE_MODEL='. "
            "In .env, API_BASE_URL and IMAGE_MODEL must be on separate lines."
        )
    # SDK appends /images/generations; strip if user already included that path.
    if isinstance(base_url, str):
        b = base_url.strip().rstrip("/")
        suffix = "/images/generations"
        if b.lower().endswith(suffix):
            base_url = b[: -len(suffix)]
    base_url = _normalize_openai_compat_base_url(base_url)
    api_key = _select_image_api_key()

    timeout_s = float(os.getenv("OPENAI_TIMEOUT_SECONDS", "60"))
    connect_s = float(os.getenv("HTTP_CONNECT_TIMEOUT_SECONDS", "45"))
    trust_env = os.getenv("OPENAI_TRUST_ENV", "").strip().lower() in {"1", "true", "yes", "y"}
    default_headers = _load_default_headers()
    follow_redirects = _httpx_follow_redirects()

    # Separate connect vs read timeouts for long-running image requests.
    http_timeout = httpx.Timeout(
        connect=min(connect_s, timeout_s),
        read=timeout_s,
        write=min(connect_s, timeout_s),
        pool=min(connect_s, timeout_s),
    )
    http_client = httpx.Client(
        timeout=http_timeout,
        trust_env=trust_env,
        follow_redirects=follow_redirects,
    )
    client = OpenAI(
        api_key=api_key,
        base_url=base_url,
        http_client=http_client,
        default_headers=default_headers,
    )

    _debug = os.getenv("DEBUG_IMAGE_ENV", "").strip().lower() in {"1", "true", "yes", "y"}
    if _debug:
        root = base_url.strip().rstrip("/") if isinstance(base_url, str) else ""
        print(
            f"[DEBUG_IMAGE_ENV] OpenAI SDK base_url={base_url!r} -> "
            f"images API will POST to {root}/images/generations "
            f"httpx_follow_redirects={follow_redirects}",
            flush=True,
        )

    model_name = model or get_image_models()[0]
    model_name_lower = model_name.strip().lower()

    # Doubao: use Volcengine Ark when ARK_API_KEY is set; otherwise OpenAI-compatible images API on API_BASE_URL.
    if model_name_lower.startswith(DOUBAO_IMAGE_MODEL_PREFIX):
        if (os.getenv("ARK_API_KEY") or "").strip():
            if _debug:
                ark = (os.getenv("ARK_BASE_URL") or "https://ark.cn-beijing.volces.com/api/v3").strip()
                print(
                    f"[DEBUG_IMAGE_ENV] Doubao via Volcengine Ark SDK "
                    f"(ARK_BASE_URL={ark!r}) model={model_name!r}",
                    flush=True,
                )
            return _generate_image_doubao_ark(prompt, model_name, http_client=http_client)

    # Qwen / DashScope branch
    if QWEN_IMAGE_MODEL_PREFIX in model_name_lower:
        return _generate_image_qwen(prompt, model_name, http_client=http_client)

    # Multimodal image via /chat/completions: Gemini *image* models (google extra_body) and Nano-banana-2.
    if ("gemini" in model_name_lower and "image" in model_name_lower) or "nano-banana" in model_name_lower:
        msg = [{"role": "user", "content": prompt}]
        if "gemini" in model_name_lower and "image" in model_name_lower:
            aspect_ratio = (os.getenv("IMAGE_ASPECT_RATIO") or "1:1").strip()
            resp = client.chat.completions.create(
                model=model_name,
                messages=msg,
                extra_body={"google": {"image_config": {"aspect_ratio": aspect_ratio}}},
                timeout=timeout_s,
            )
        else:
            resp = client.chat.completions.create(
                model=model_name,
                messages=msg,
                timeout=timeout_s,
            )
        return _extract_image_bytes_from_chat_response(resp, http_client=http_client)

    # gpt-image-1 / 1.5: some gateways reject response_format; use server default (b64 or URL).
    size = os.getenv("IMAGE_SIZE", "1024x1024")
    if model_name_lower.startswith("gpt-image-1"):
        resp = client.images.generate(
            model=model_name,
            prompt=prompt,
            size=size,
            timeout=timeout_s,
            # Omit response_format for compatibility with strict gateways
        )
    elif model_name_lower.startswith(DOUBAO_IMAGE_MODEL_PREFIX):
        # OpenAI-compatible Doubao route: omit response_format to avoid slow reads on some gateways.
        omit_size = os.getenv("DOUBAO_IMAGES_OMIT_SIZE", "").strip().lower() in {
            "1",
            "true",
            "yes",
            "y",
        }
        if _debug:
            root = base_url.strip().rstrip("/") if isinstance(base_url, str) else ""
            print(
                f"[DEBUG_IMAGE_ENV] About to call images.generate read_timeout_s={timeout_s} "
                f"connect_timeout_s={connect_s} POST {root}/images/generations "
                f"omit_size={omit_size}",
                flush=True,
            )
        if omit_size:
            resp = client.images.generate(
                model=model_name,
                prompt=prompt,
                timeout=timeout_s,
            )
        else:
            resp = client.images.generate(
                model=model_name,
                prompt=prompt,
                size=size,
                timeout=timeout_s,
            )
    elif model_name_lower.startswith("wan"):
        # WAN image models: images/generations; many gateways reject forced b64_json.
        if _debug:
            root = base_url.strip().rstrip("/") if isinstance(base_url, str) else ""
            print(
                f"[DEBUG_IMAGE_ENV] WAN images.generate POST {root}/images/generations size={size!r}",
                flush=True,
            )
        resp = client.images.generate(
            model=model_name,
            prompt=prompt,
            size=size,
            timeout=timeout_s,
        )
    else:
        resp = client.images.generate(
            model=model_name,
            prompt=prompt,
            size=size,
            response_format="b64_json",
            timeout=timeout_s,
        )
    return _extract_image_bytes_from_images_response(resp, http_client=http_client)


def _http_get_image_url(http_client: httpx.Client, url: str, **kwargs):
    """Fetch image bytes from URL; always follow redirects (CDN redirect chains)."""
    return http_client.get(url, follow_redirects=True, **kwargs)


def _normalize_to_png_if_needed(image_bytes: bytes) -> bytes:
    """
    Some gateways return JPEG bytes in b64 while callers expect .png files.
    Convert JPEG/WebP to PNG bytes for consistent downstream file extension.
    """
    if not image_bytes:
        return image_bytes
    is_png = image_bytes.startswith(b"\x89PNG\r\n\x1a\n")
    if is_png:
        return image_bytes
    is_jpeg = image_bytes.startswith(b"\xff\xd8\xff")
    is_webp = image_bytes.startswith(b"RIFF") and b"WEBP" in image_bytes[:16]
    if not (is_jpeg or is_webp):
        return image_bytes
    try:
        from PIL import Image  # type: ignore

        im = Image.open(BytesIO(image_bytes))
        out = BytesIO()
        im.save(out, format="PNG")
        return out.getvalue()
    except Exception:
        # If conversion fails, keep original bytes instead of hard-failing.
        return image_bytes


def _extract_image_bytes_from_images_response(resp, http_client: httpx.Client) -> bytes:
    """
    Extract image bytes from OpenAI-style images.generate responses.

    Some gateways return:
    - OpenAI SDK objects (resp.data[0].b64_json / url)
    - Plain dict (already json)
    - JSON string (resp is str)
    """
    # 1) Normalize to a Python object we can traverse.
    obj = resp
    if isinstance(resp, str) and resp.strip():
        try:
            obj = json.loads(resp)
        except Exception:
            obj = {"raw": resp}
    else:
        try:
            if hasattr(resp, "model_dump"):
                obj = resp.model_dump()
        except Exception:
            obj = resp

    # 2) Try OpenAI SDK attribute path first.
    try:
        data0 = resp.data[0]  # type: ignore[attr-defined]
        b64 = getattr(data0, "b64_json", None) or getattr(data0, "image_base64", None)
        if isinstance(b64, str) and b64.strip():
            b64 = b64.strip()
            if b64.startswith("data:"):
                b64 = b64.split(",", 1)[-1]
            return _normalize_to_png_if_needed(base64.b64decode(b64))
        url = getattr(data0, "url", None)
        if isinstance(url, str) and url.strip().startswith("http"):
            r = _http_get_image_url(http_client, url.strip())
            r.raise_for_status()
            return r.content
    except Exception:
        pass

    # 3) Generic traversal: find b64 or url anywhere (common gateway variants).
    b64 = _find_first_str_by_keys(obj, {"b64_json", "image_base64", "base64", "b64"})
    if b64:
        if b64.startswith("data:"):
            b64 = b64.split(",", 1)[-1]
        return _normalize_to_png_if_needed(base64.b64decode(b64))

    url = _find_first_str_by_keys(obj, {"url", "image_url"})
    if url and url.startswith("http"):
        r = _http_get_image_url(http_client, url)
        r.raise_for_status()
        return _normalize_to_png_if_needed(r.content)

    snippet = ""
    try:
        snippet = str(obj)[:300]
    except Exception:
        snippet = "<unprintable>"
    raise RuntimeError(f"Could not extract image bytes from images.generate response. prefix={snippet!r}")


def _generate_image_doubao_ark(prompt: str, model_name: str, http_client: httpx.Client) -> bytes:
    """
    Volcengine Ark image API (Doubao / Seedream). Requires ARK_API_KEY.

    Install: pip install "volcengine-python-sdk[ark]"
    """
    api_key = (os.getenv("ARK_API_KEY") or "").strip()
    if not api_key or "your" in api_key.lower():
        raise RuntimeError(
            "Doubao/Ark image model requires ARK_API_KEY. "
            "Get key from Volcengine Ark console."
        )

    base_url = (os.getenv("ARK_BASE_URL") or "https://ark.cn-beijing.volces.com/api/v3").strip().rstrip("/")

    # Ark size is typically 1K / 2K; override with ARK_IMAGE_SIZE.
    size = (os.getenv("ARK_IMAGE_SIZE") or "").strip()
    if not size:
        size_env = (os.getenv("IMAGE_SIZE") or "").strip().lower()
        # Map IMAGE_SIZE hint to 1K/2K unless ARK_IMAGE_SIZE is set.
        if "2048" in size_env or "2k" in size_env:
            size = "2K"
        else:
            size = "1K"

    output_format = (os.getenv("ARK_OUTPUT_FORMAT") or "png").strip().lower()
    response_format = (os.getenv("ARK_RESPONSE_FORMAT") or "url").strip().lower()
    watermark = (os.getenv("ARK_WATERMARK") or "false").strip().lower() in {"1", "true", "yes", "y"}

    try:
        from volcenginesdkarkruntime import Ark  # type: ignore[import-untyped]
    except Exception as e:
        raise RuntimeError(
            "Missing dependency for Doubao/Ark. Install with: pip install \"volcengine-python-sdk[ark]\""
        ) from e

    ark_client = Ark(base_url=base_url, api_key=api_key)
    try:
        resp = ark_client.images.generate(
            model=model_name,
            prompt=prompt,
            size=size,
            output_format=output_format,
            response_format=response_format,
            watermark=watermark,
        )
    except Exception as e:
        msg = str(e)
        # Unactivated Ark models often surface as ModelNotOpen (sometimes 404).
        if "ModelNotOpen" in msg or "has not activated the model" in msg:
            raise RuntimeError(
                f"Ark model not activated for this account: {model_name}. "
                "Please activate the model service in the Ark Console, or switch IMAGE_MODEL to an activated model id."
            ) from e
        raise

    # Prefer URL; fall back to inline base64 fields if present.
    try:
        data0 = resp.data[0]
    except Exception as e:
        raise RuntimeError(f"Unexpected Ark image response structure: {e}") from e

    url = getattr(data0, "url", None)
    if isinstance(url, str) and url.strip().startswith("http"):
        r = _http_get_image_url(http_client, url.strip())
        r.raise_for_status()
        return r.content

    b64 = getattr(data0, "b64_json", None) or getattr(data0, "image_base64", None) or getattr(data0, "base64", None)
    if isinstance(b64, str) and b64.strip():
        if b64.startswith("data:"):
            b64 = b64.split(",", 1)[-1]
        return base64.b64decode(b64.strip())

    raise RuntimeError("Could not extract image bytes from Ark images.generate response")


def _generate_image_qwen(prompt: str, model_name: str, http_client: httpx.Client) -> bytes:
    """Alibaba DashScope multimodal image generation. Use DASHSCOPE_API_KEY or QWEN_API_KEY."""
    api_key = (os.getenv("DASHSCOPE_API_KEY") or os.getenv("QWEN_API_KEY") or "").strip()
    if not api_key or "your" in api_key.lower() or "sk-" not in api_key:
        raise RuntimeError(
            "Qwen image model requires DASHSCOPE_API_KEY or QWEN_API_KEY. "
            "Get key from: https://bailian.console.aliyun.com or Aliyun Model Studio."
        )

    base_url = (os.getenv("DASHSCOPE_BASE_URL") or "https://dashscope.aliyuncs.com").strip().rstrip("/")
    # Regions: CN dashscope.aliyuncs.com ; intl dashscope-intl.aliyuncs.com
    endpoint = f"{base_url}/api/v1/services/aigc/multimodal-generation/generation"

    # DashScope expects size like "1024*1024" (asterisk).
    size_env = (os.getenv("IMAGE_SIZE") or "1024x1024").strip().replace("x", "*").replace("×", "*")

    payload = {
        "model": model_name,
        "input": {
            "messages": [
                {
                    "role": "user",
                    "content": [{"text": prompt}],
                }
            ]
        },
        "parameters": {
            "size": size_env,
            "n": 1,
            "prompt_extend": os.getenv("QWEN_PROMPT_EXTEND", "false").strip().lower() in {"1", "true", "yes", "y"},
            "watermark": False,
        },
    }
    negative = (os.getenv("QWEN_NEGATIVE_PROMPT") or "").strip()
    if negative:
        payload["parameters"]["negative_prompt"] = negative

    timeout_s = float(os.getenv("OPENAI_TIMEOUT_SECONDS", "120"))
    r = http_client.post(
        endpoint,
        json=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        timeout=timeout_s,
    )
    r.raise_for_status()
    data = r.json()

    if data.get("code"):
        raise RuntimeError(f"Qwen API error: {data.get('code')} - {data.get('message', '')}")

    try:
        content = data["output"]["choices"][0]["message"]["content"]
        img_item = content[0] if isinstance(content, list) else content
        image_url = img_item.get("image") if isinstance(img_item, dict) else None
    except (KeyError, IndexError, TypeError) as e:
        raise RuntimeError(f"Unexpected Qwen response structure: {e}") from e

    if not image_url or not str(image_url).startswith("http"):
        raise RuntimeError("Qwen response did not contain image URL")

    img_r = _http_get_image_url(http_client, image_url, timeout=timeout_s)
    img_r.raise_for_status()
    return img_r.content


def _extract_image_bytes_from_chat_response(resp, http_client: httpx.Client) -> bytes:
    try:
        d = resp.model_dump()
    except Exception:
        d = {}

    b64 = _find_first_str_by_keys(d, {"b64_json", "image_base64", "base64", "b64"})
    if b64:
        if b64.startswith("data:"):
            b64 = b64.split(",", 1)[-1]
        return base64.b64decode(b64)

    url = _find_first_str_by_keys(d, {"url", "image_url"})
    if url and url.startswith("http"):
        r = _http_get_image_url(http_client, url)
        r.raise_for_status()
        return r.content

    # Many gateways put the image in markdown: ![image](data:image/png;base64,...)
    content = None
    try:
        content = resp.choices[0].message.content
    except Exception:
        content = None
    if isinstance(content, str) and content.strip():
        src = _extract_markdown_image_src(content)
        if src:
            if src.startswith("data:"):
                b = src.split(",", 1)[-1]
                return base64.b64decode(b)
            if src.startswith("http"):
                r = _http_get_image_url(http_client, src)
                r.raise_for_status()
                return r.content

        c = content.strip()
        if c.startswith("data:"):
            b = c.split(",", 1)[-1]
            return base64.b64decode(b)

    snippet = ""
    if isinstance(content, str):
        snippet = content.strip().replace("\n", "\\n")[:300]
    raise RuntimeError(f"Could not extract image bytes from chat.completions response. content_prefix={snippet!r}")


def _find_first_str_by_keys(obj, keys: set) -> str | None:
    if obj is None:
        return None
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(k, str) and k in keys and isinstance(v, str) and v.strip():
                return v.strip()
        for v in obj.values():
            found = _find_first_str_by_keys(v, keys)
            if found:
                return found
    elif isinstance(obj, list):
        for it in obj:
            found = _find_first_str_by_keys(it, keys)
            if found:
                return found
    return None


_MD_IMAGE_RE = re.compile(r"!\[[^\]]*]\(([^)]+)\)")


def _extract_markdown_image_src(text: str) -> str | None:
    m = _MD_IMAGE_RE.search(text)
    if not m:
        return None
    return m.group(1).strip().strip('"').strip("'")


def _load_default_headers() -> dict:
    raw = (os.getenv("API_DEFAULT_HEADERS") or "").strip()
    if not raw:
        return {}
    try:
        val = json.loads(raw)
    except Exception as e:
        raise RuntimeError(f"Invalid API_DEFAULT_HEADERS JSON: {e}") from e
    if not isinstance(val, dict) or not all(isinstance(k, str) for k in val.keys()):
        raise RuntimeError("API_DEFAULT_HEADERS must be a JSON object with string keys")
    return val
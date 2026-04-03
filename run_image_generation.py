import json
import os
import sys
import time
from pathlib import Path

from benchmark.image_generator import generate_image, get_image_models


def _load_dotenv_for_image_script():
    """Load `.env` so IMAGE_MODEL and related vars apply (override=True unless DOTENV_PRESERVE_ENV is set)."""
    try:
        from dotenv import load_dotenv

        # DOTENV_PRESERVE_ENV=1: prefer existing OS env over `.env`
        preserve = os.getenv("DOTENV_PRESERVE_ENV", "").strip().lower() in {
            "1",
            "true",
            "yes",
            "y",
        }
        load_dotenv(override=not preserve)
    except Exception:
        pass


def _ensure_long_read_timeout_for_doubao():
    """Raise HTTP read timeout for slow OpenAI-compatible image backends (e.g. some doubao/wan routes)."""
    if os.getenv("DOUBAO_NO_TIMEOUT_BUMP", "").strip().lower() in {"1", "true", "yes", "y"}:
        return
    single = (os.getenv("IMAGE_MODEL") or "").strip().lower()
    multi = (os.getenv("IMAGE_MODELS") or "").lower()
    is_doubao = single.startswith("doubao-") or "doubao-" in multi
    is_wan = single.startswith("wan") or "wan" in multi
    if not is_doubao and not is_wan:
        return
    try:
        floor = float((os.getenv("DOUBAO_MIN_READ_TIMEOUT_SECONDS") or "900").strip())
    except ValueError:
        floor = 900.0
    try:
        raw = (os.getenv("OPENAI_TIMEOUT_SECONDS") or "").strip()
        cur = float(raw) if raw else 0.0
    except ValueError:
        cur = 0.0
    if cur < floor:
        os.environ["OPENAI_TIMEOUT_SECONDS"] = str(int(floor) if floor == int(floor) else floor)


def main():
    _load_dotenv_for_image_script()
    _ensure_long_read_timeout_for_doubao()

    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
        sys.stderr.reconfigure(encoding="utf-8", errors="backslashreplace")
    except Exception:
        pass

    # DEBUG_IMAGE_ENV=1: print non-secret config (no key values).
    if os.getenv("DEBUG_IMAGE_ENV", "").strip().lower() in {"1", "true", "yes", "y"}:
        for k in (
            "API_BASE_URL",
            "IMAGE_MODEL",
            "IMAGE_MODELS",
            "BENCHMARK_CASES_PATH",
            "IMAGE_OUT_DIR",
            "ONLY_CASE_ID",
            "MAX_CASES_PER_RUN",
        ):
            print(f"[DEBUG_IMAGE_ENV] {k}={os.getenv(k)!r}", flush=True)
        ik = bool((os.getenv("IMAGE_API_KEY") or "").strip())
        vk = bool((os.getenv("VLLM_API_KEY") or "").strip())
        ok = bool((os.getenv("OPENAI_API_KEY") or "").strip())
        print(
            f"[DEBUG_IMAGE_ENV] keys present: IMAGE_API_KEY={ik}, VLLM_API_KEY={vk}, OPENAI_API_KEY={ok}",
            flush=True,
        )
        print(
            "[DEBUG_IMAGE_ENV] image key order: IMAGE_API_KEY > OPENAI_API_KEY > VLLM_API_KEY",
            flush=True,
        )

    # CASES_JSON_PATH: legacy alias for BENCHMARK_CASES_PATH.
    cases_path = Path(
        (
            os.getenv("BENCHMARK_CASES_PATH")
            or os.getenv("CASES_JSON_PATH")
            or "benchmark_cases/text2sceneleakagebench_cases_merged_v1.json"
        )
    )
    _out = (os.getenv("IMAGE_OUT_DIR") or "").strip()
    out_dir = Path(_out if _out else "outputs/images")
    _err = (os.getenv("ERROR_OUT_DIR") or "").strip()
    err_dir = Path(_err if _err else "outputs/errors")
    only_case_id = (os.getenv("ONLY_CASE_ID") or "").strip()
    max_retries = int(os.getenv("IMAGE_MAX_RETRIES", "2"))
    retry_sleep_s = float(os.getenv("IMAGE_RETRY_SLEEP_SECONDS", "2"))
    # MAX_CASES_PER_RUN: cap API requests per run (0 = no cap). Skipped existing files do not count.
    max_cases_per_run = int(os.getenv("MAX_CASES_PER_RUN", "0") or "0")

    model_override = (os.getenv("IMAGE_MODEL") or "").strip()
    models = [model_override] if model_override else get_image_models()
    if not models:
        raise RuntimeError("No image models configured (set IMAGE_MODEL or IMAGE_MODELS)")

    with cases_path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    if isinstance(raw, dict) and isinstance(raw.get("cases"), list):
        cases = raw["cases"]
    elif isinstance(raw, list):
        cases = raw
    else:
        raise RuntimeError(f"Unsupported benchmark JSON shape in {cases_path}")

    normalized = []
    for c in cases:
        if not isinstance(c, dict):
            continue
        cid = c.get("case_id")
        prompt = c.get("prompt")
        if not prompt:
            lang = (os.getenv("TEXT2SCENE_PROMPT_LANG") or "zh").strip().lower()
            if lang == "en":
                prompt = c.get("prompt_en") or c.get("prompt_zh")
            else:
                prompt = c.get("prompt_zh") or c.get("prompt_en")
        if cid and prompt:
            normalized.append({"case_id": cid, "prompt": prompt})
    cases = normalized

    out_dir.mkdir(parents=True, exist_ok=True)
    err_dir.mkdir(parents=True, exist_ok=True)

    if only_case_id:
        print(
            f"[run_image_generation] ONLY_CASE_ID is set -> running a single case: {only_case_id!r}",
            flush=True,
        )
        cases = [c for c in cases if c.get("case_id") == only_case_id]
        if not cases:
            raise RuntimeError(f"ONLY_CASE_ID not found in cases: {only_case_id}")

    total = len(cases)
    fail_count = 0
    requested_count = 0  # API calls only (not skipped existing outputs)
    # IMAGE_OUT_NAME_SUFFIX: e.g. _zh / _en when batching languages into one folder
    out_name_suffix = (os.getenv("IMAGE_OUT_NAME_SUFFIX") or "").strip()

    for i, case in enumerate(cases, start=1):
        # Stop early when MAX_CASES_PER_RUN is reached.
        if max_cases_per_run and requested_count >= max_cases_per_run:
            print(
                f"Reached MAX_CASES_PER_RUN={max_cases_per_run}, stop at case index {i-1}/{total}.",
                flush=True,
            )
            break

        case_id = case["case_id"]
        prompt = case["prompt"]

        # Reserve multi-model support: save under outputs/images/<model>/<case_id>.png
        case_counted = False  # one count per case even with model fallbacks
        for model in models:
            model_dir = out_dir / model
            model_dir.mkdir(parents=True, exist_ok=True)
            model_err_dir = err_dir / model
            model_err_dir.mkdir(parents=True, exist_ok=True)

            out_path = model_dir / f"{case_id}{out_name_suffix}.png"
            if out_path.exists() and os.getenv("SKIP_EXISTING", "1").strip().lower() not in {"0", "false", "no", "n"}:
                print(f"[{i}/{total}] skip {case_id}{out_name_suffix} ({model})", flush=True)
                break

            last_err = None
            if not case_counted:
                requested_count += 1
                case_counted = True
            for attempt in range(1, max_retries + 2):
                try:
                    print(
                        f"[{i}/{total}] request {case_id} ({model}) prompt_chars={len(prompt)} "
                        f"timeout_s={os.getenv('OPENAI_TIMEOUT_SECONDS', '?')}",
                        flush=True,
                    )
                    img_bytes = generate_image(prompt, model=model)
                    out_path.write_bytes(img_bytes)
                    print(f"[{i}/{total}] ok   {case_id} ({model}) -> {out_path}", flush=True)
                    last_err = None
                    break
                except Exception as e:
                    last_err = e
                    print(
                        f"[{i}/{total}] fail {case_id} ({model}) attempt {attempt}/{max_retries+1}: {repr(e)}",
                        flush=True,
                    )
                    if attempt <= max_retries:
                        time.sleep(retry_sleep_s)
                        continue
                    break

            if last_err is None:
                break

            # Save error detail for later alignment/debug (does not include API keys).
            try:
                (model_err_dir / f"{case_id}{out_name_suffix}.txt").write_text(
                    repr(last_err), encoding="utf-8"
                )
            except Exception:
                pass

            # Try next model in fallback list.
            continue
        else:
            fail_count += 1
            # Optionally fail at end if FAIL_FAST.
            if os.getenv("FAIL_FAST", "0").strip().lower() in {"1", "true", "yes", "y"}:
                raise RuntimeError(f"All image models failed for {case_id}")

    if fail_count:
        if os.getenv("STRICT", "0").strip().lower() in {"1", "true", "yes", "y"}:
            raise RuntimeError(f"Image generation finished with failures: {fail_count}/{total}")
        print(f"Finished with failures: {fail_count}/{total} (see {err_dir})", flush=True)


if __name__ == "__main__":
    main()


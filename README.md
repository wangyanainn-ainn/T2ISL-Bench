## Text2SceneLeakageBench (T2ISL-Bench)

This repository supports **automated image generation** and **multimodal evaluation** for **Text2SceneLeakageBench**:

Intermediate splits used to build the merge are **not** included; see the `provenance` field inside each merged JSON.

### Paper model setup (reference)

| Role | Models (set `IMAGE_MODEL` / gateway env per run) |
|------|---------------------------------------------------|
| **Image generation** | **Nano-banana-2**, **GPT‑image‑1.5**, **WAN 2.6** image, **Doubao / Seedream** (Volcengine Ark) |
| **VLM evaluation** (SSP/SLR) | **Gemini 3 Pro** (`AUDIT_MODEL`, default `gemini-3-pro`) |
| **OCR** (TAA — text in image) | **[PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)** (local, no cloud LLM). Stage 1 in `mm_auto_eval` calls `PaddleOCR(use_angle_cls=True, lang="ch")`, then matches reading-order boxes to `target_text` with light anchor heuristics. |

Exact **image/VLM model id strings** depend on your API gateway; see `.env.example`.

### OCR install (for `mm_auto_eval`)

PaddleOCR is **not** listed in `requirements.txt` (install footprint varies by platform). For evaluation:

```bash
pip install paddleocr
```

Follow [PaddleOCR install docs](https://www.paddleocr.ai/latest/en/installation.html) if you need a specific PaddlePaddle CPU/GPU wheel.

If `paddleocr` fails on **`Polygon` / `lanms`** build tools (common on Windows), install the stub package plus Shapely, then retry:

```bash
pip install -e polygon_shim
```

### Validate cases vs seeds

```bash
python run_pipeline.py
```

Optional: write a root copy for tools that expect `benchmark_cases.json`:

```bash
python run_pipeline.py --export
```

### Environment variables

Create a `.env` (not committed) based on `.env.example`.

- **API_BASE_URL**: OpenAI-compatible gateway for chat / many image routes.
- **OPENAI_API_KEY** / **IMAGE_API_KEY** / **VLLM_API_KEY** / **ARK_API_KEY**: as required by each backbone (Doubao uses **Ark** when configured).
- **IMAGE_MODEL** or **IMAGE_MODELS**: which generator to run (or fallback order).

### Image generation

Default case file is the merged Text2Scene JSON. Override with `BENCHMARK_CASES_PATH` if needed.

```bash
python run_image_generation.py
```

Each case must have `case_id` and `prompt`, or `prompt_zh` / `prompt_en` (language chosen with `TEXT2SCENE_PROMPT_LANG`, default `zh`).

### Multimodal OCR + VLM evaluation

Pipeline order in `mm_auto_eval.main`:

1. **OCR (TAA)** — **PaddleOCR** on disk image paths; produces `taa` and related fields.
2. **VLM** — **Gemini 3 Pro** (default) for SSP/SLR via OpenAI-compatible multimodal chat.

Prepare `mm_eval_samples.jsonl` (or set `MM_EVAL_INPUT`), then:

```bash
python -m mm_auto_eval.main
```

**VLM judge:** defaults to **Gemini 3 Pro** (`AUDIT_MODEL=gemini-3-pro`; synonym `MM_EVAL_MODEL`). The endpoint must support **vision + text** in the same OpenAI-compatible chat API used by `mm_auto_eval`.

Defaults: input `mm_eval_samples.jsonl`, output `outputs/results.jsonl`. See `.env.example` for timeouts and other flags. **OCR quality** can be tuned with `OCR_MIN_SCORE`.

### Security / release checklist

- Never commit `.env`. Use `.env.example` only as a template.
- If this repo (or any copy-paste) ever included real API keys, **rotate those keys** at the provider.
- Run `git status` before `git push` and confirm no secrets or `outputs/` are staged.

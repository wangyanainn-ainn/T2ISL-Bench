from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Dict, Optional

from mm_auto_eval.ocr_eval import evaluate_taa
from mm_auto_eval.utils import load_done_ids, read_samples, write_replace_by_id
from mm_auto_eval.vlm_eval import call_ssp_slr


def compute_clr(taa: Any, ssp: Any, slr: Any) -> Optional[int]:
    if taa != 1:
        return None
    # semantic uncertain -> clr must be null
    if ssp == "uncertain" or slr == "uncertain" or ssp is None or slr is None:
        return None
    if ssp == 0 or slr == 1:
        return 1
    if ssp == 1 and slr == 0:
        return 0
    return None


def _as_tri(v: Any) -> Any:
    if v in (0, 1, "uncertain"):
        return v
    if isinstance(v, bool):
        return 1 if v else 0
    if isinstance(v, (int, float)) and v in (0, 1):
        return int(v)
    if isinstance(v, str):
        t = v.strip().lower()
        if t in {"0", "no", "false"}:
            return 0
        if t in {"1", "yes", "true"}:
            return 1
        if t in {"uncertain", "unknown", "maybe"}:
            return "uncertain"
    return "uncertain"


def main() -> None:
    input_path = Path(os.getenv("MM_EVAL_INPUT", "mm_eval_samples.jsonl"))
    out_path = Path(os.getenv("MM_EVAL_OUTPUT", "outputs/results.jsonl"))
    out_path.parent.mkdir(parents=True, exist_ok=True)

    skip_done = (os.getenv("SKIP_EXISTING") or "1").strip().lower() in {"1", "true", "yes", "y"}
    min_ocr_score = float(os.getenv("OCR_MIN_SCORE", "0.5"))
    sleep_s = float(os.getenv("MM_EVAL_SLEEP_SECONDS", "0") or "0")
    max_samples_raw = (os.getenv("MM_EVAL_MAX_SAMPLES") or "").strip()
    max_samples = int(max_samples_raw) if max_samples_raw else None

    samples = read_samples(input_path)
    done = load_done_ids(out_path) if skip_done else set()

    total = len(samples)
    processed = 0
    for i, s in enumerate(samples, start=1):
        if max_samples is not None and processed >= max_samples:
            break
        if skip_done and s.id in done:
            print(f"[{i}/{total}] skip {s.id}", flush=True)
            continue
        # If the image hasn't been generated yet, skip it for now (do NOT write an "uncertain" record),
        # so we can evaluate it later after generation completes.
        if not Path(s.image_path).exists():
            print(f"[{i}/{total}] skip_missing_image {s.id}", flush=True)
            continue

        stage1 = evaluate_taa(
            image_path=Path(s.image_path),
            target_text=s.target_text,
            text_anchor=s.text_anchor,
            min_score=min_ocr_score,
        )
        taa = stage1.get("taa")

        # Stage 2 now runs for ALL samples (not gated by TAA)
        ssp: Any
        slr: Any
        ssp_reason: Any
        slr_reason: Any
        subject_identity: Any
        scene_cues: Any
        raw_stage2: Any
        try:
            stage2, raw = call_ssp_slr(
                image_path=s.image_path,
                subject_description=s.subject_description,
                text_anchor=s.text_anchor,
                target_text=s.target_text,
                relation=s.relation,
                prompt=s.prompt,
            )
            raw_stage2 = raw
            ssp = _as_tri(stage2.get("ssp"))
            slr = _as_tri(stage2.get("slr"))
            ssp_reason = stage2.get("ssp_reason") or ""
            slr_reason = stage2.get("slr_reason") or ""
            subject_identity = stage2.get("subject_identity") or ""
            scene_cues = stage2.get("scene_cues") or ""
        except Exception as e:
            ssp = "uncertain"
            slr = "uncertain"
            ssp_reason = f"stage2_failed: {repr(e)}"
            slr_reason = f"stage2_failed: {repr(e)}"
            subject_identity = ""
            scene_cues = ""
            raw_stage2 = ""

        clr = compute_clr(taa, ssp, slr)

        rec: Dict[str, Any] = {
            "id": s.id,
            "image_path": s.image_path,
            "subject_description": s.subject_description,
            "text_anchor": s.text_anchor,
            "target_text": s.target_text,
            "relation": s.relation,
            "prompt": s.prompt,
            "taa": stage1.get("taa"),
            "taa_reason": stage1.get("taa_reason"),
            "visible_text": stage1.get("visible_text"),
            "text_location": stage1.get("text_location"),
            "ocr_raw": stage1.get("ocr_raw"),
            "ssp": ssp,
            "ssp_reason": ssp_reason,
            "slr": slr,
            "slr_reason": slr_reason,
            "subject_identity": subject_identity,
            "scene_cues": scene_cues,
            "clr": clr,
            "raw_model_output_stage2": raw_stage2,
        }

        write_replace_by_id(out_path, rec)
        print(f"[{i}/{total}] done {s.id} taa={rec['taa']} clr={rec['clr']}", flush=True)
        processed += 1
        if sleep_s > 0:
            time.sleep(sleep_s)

    print(f"[mm_auto_eval] wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()


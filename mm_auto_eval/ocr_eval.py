from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .utils import normalize_text


@dataclass
class OcrBox:
    # (x1,y1) (x2,y2) (x3,y3) (x4,y4) polygon
    poly: List[List[float]]
    text: str
    score: float

    def bbox(self) -> Tuple[float, float, float, float]:
        xs = [p[0] for p in self.poly]
        ys = [p[1] for p in self.poly]
        return min(xs), min(ys), max(xs), max(ys)

    def area(self) -> float:
        x0, y0, x1, y1 = self.bbox()
        return max(0.0, x1 - x0) * max(0.0, y1 - y0)

    def center(self) -> Tuple[float, float]:
        x0, y0, x1, y1 = self.bbox()
        return (x0 + x1) / 2.0, (y0 + y1) / 2.0


def _parse_anchor_hint(text_anchor: str) -> Dict[str, bool]:
    t = (text_anchor or "").lower()
    # Very lightweight directional hints (Chinese + English)
    return {
        "top": any(k in t for k in ["上", "顶部", "top", "upper"]),
        "bottom": any(k in t for k in ["下", "底部", "bottom", "lower"]),
        "left": any(k in t for k in ["左", "left"]),
        "right": any(k in t for k in ["右", "right"]),
        "center": any(k in t for k in ["中", "中央", "center", "middle"]),
        "front": any(k in t for k in ["正面", "front"]),
    }


def _anchor_score(hints: Dict[str, bool], cx: float, cy: float) -> float:
    # Coordinate system: y increases downward.
    # Map to a soft score in [0,1] based on how well the centroid matches the hinted region.
    score = 0.5
    if hints.get("top"):
        score += 0.2 if cy < 0.4 else -0.1
    if hints.get("bottom"):
        score += 0.2 if cy > 0.6 else -0.1
    if hints.get("left"):
        score += 0.15 if cx < 0.4 else -0.05
    if hints.get("right"):
        score += 0.15 if cx > 0.6 else -0.05
    if hints.get("center") or hints.get("front"):
        score += 0.15 if 0.35 <= cx <= 0.65 and 0.35 <= cy <= 0.65 else -0.05
    return max(0.0, min(1.0, score))


def _load_paddleocr():
    # Optional dependency
    from paddleocr import PaddleOCR  # type: ignore

    # Chinese+English: use ch (covers en) with angle cls
    return PaddleOCR(use_angle_cls=True, lang="ch")


def run_ocr(image_path: Path) -> Tuple[List[OcrBox], Any, Optional[str]]:
    """
    Returns:
      - list of parsed OcrBox
      - ocr_raw: raw paddleocr output (for debug)
      - error_reason: if OCR backend not available or failed
    """
    try:
        ocr = _load_paddleocr()
    except Exception as e:
        return [], [], f"ocr_backend_unavailable: {repr(e)}"

    try:
        raw = ocr.ocr(str(image_path), cls=True)
    except Exception as e:
        return [], [], f"ocr_failed: {repr(e)}"

    boxes: List[OcrBox] = []
    # PaddleOCR output: [ [ [poly, (text, score)], ... ] ] (or similar)
    lines = raw[0] if isinstance(raw, list) and raw else []
    for item in lines or []:
        try:
            poly = item[0]
            text, score = item[1]
            if not text:
                continue
            poly2 = [[float(p[0]), float(p[1])] for p in poly]
            boxes.append(OcrBox(poly=poly2, text=str(text), score=float(score)))
        except Exception:
            continue
    return boxes, raw, None


def evaluate_taa(
    image_path: Path,
    target_text: str,
    text_anchor: str,
    min_score: float = 0.5,
) -> Dict[str, Any]:
    """
    Stage 1:
    - Prefer OCR to detect text.
    - Use lightweight heuristics for anchor/main-label:
      pick candidate boxes matching target_text, prefer the largest box and boxes matching anchor hints.
    """
    boxes, ocr_raw, err = run_ocr(image_path)
    if err:
        return {
            "taa": "uncertain",
            "taa_reason": err,
            "visible_text": "",
            "text_location": "",
            "ocr_raw": ocr_raw,
        }

    filtered = [b for b in boxes if b.score >= min_score and b.text.strip()]
    if not filtered:
        return {
            "taa": 0,
            "taa_reason": "ocr_no_text_detected",
            "visible_text": "",
            "text_location": "",
            "ocr_raw": ocr_raw,
        }

    norm_target = normalize_text(target_text)
    matches: List[OcrBox] = []
    for b in filtered:
        if normalize_text(b.text) == norm_target:
            matches.append(b)

    visible_text_all = " | ".join([b.text for b in filtered[:50]])

    if not matches:
        # target text not found
        return {
            "taa": 0,
            "taa_reason": "target_text_not_found_by_ocr",
            "visible_text": visible_text_all,
            "text_location": "",
            "ocr_raw": ocr_raw,
        }

    # Choose best match: prefer larger area and anchor hint
    hints = _parse_anchor_hint(text_anchor)
    best = None
    best_score = -1.0
    for b in matches:
        x0, y0, x1, y1 = b.bbox()
        cx, cy = b.center()
        # normalize center to [0,1] using image bbox extents inferred from all boxes
        # fallback: use relative position among boxes
        all_x0 = min(bb.bbox()[0] for bb in filtered)
        all_y0 = min(bb.bbox()[1] for bb in filtered)
        all_x1 = max(bb.bbox()[2] for bb in filtered)
        all_y1 = max(bb.bbox()[3] for bb in filtered)
        w = max(1.0, all_x1 - all_x0)
        h = max(1.0, all_y1 - all_y0)
        rcx = (cx - all_x0) / w
        rcy = (cy - all_y0) / h
        a_score = _anchor_score(hints, rcx, rcy) if any(hints.values()) else 0.5
        score = (b.area() ** 0.5) * (0.5 + a_score)
        if score > best_score:
            best = b
            best_score = score

    assert best is not None
    x0, y0, x1, y1 = best.bbox()
    text_location = f"bbox=({x0:.1f},{y0:.1f})-({x1:.1f},{y1:.1f})"

    # Main-label heuristic: best match should be among top-2 largest text areas.
    sorted_by_area = sorted(filtered, key=lambda b: b.area(), reverse=True)
    is_mainish = best in sorted_by_area[:2]

    if not is_mainish:
        taa = "uncertain"
        taa_reason = "target_text_found_but_not_main_label_by_area_heuristic"
    else:
        taa = 1
        taa_reason = "target_text_found_and_main_label_heuristic_ok"

    return {
        "taa": taa,
        "taa_reason": taa_reason,
        "visible_text": best.text,
        "text_location": text_location,
        "ocr_raw": ocr_raw,
    }


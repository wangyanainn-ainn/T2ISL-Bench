# Stub for PaddleOCR when lanms-neo cannot be compiled (no C++ build tools).
# merge_quadrangle_n9 is used in NMS; identity fallback so inference still runs.
import numpy as np


def merge_quadrangle_n9(boxes, nms_thresh):
    if len(boxes) == 0:
        return np.array([])
    return np.array(boxes, dtype=np.float32)

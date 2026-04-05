import cv2
import numpy as np
from typing import Dict, List, Tuple
from constants import ROI_SIZE, SSIM_THRESHOLD, DIFF_PIXEL_THRESHOLD, DIFF_PIXEL_PERCENT, EDGE_DIFF_PERCENT


CLASSIFIER_OPTIONS: List[Tuple[str, str]] = [
    ("legacy_ssim_edge", "Legacy SSIM+Diff+Edge"),
    ("ssim_diff_gate", "SSIM+Diff Gate"),
    ("mad", "MAD (mean abs diff)"),
    ("gaussian_mad", "Gaussian MAD"),
    ("block_mad", "Block MAD voting"),
    ("percentile_mad", "Percentile MAD P75"),
    ("max_block_mad", "Max Block MAD"),
    ("histogram_intersection", "Histogram Intersection"),
    ("variance_ratio", "Variance Ratio"),
    ("hybrid_mad_hist", "Hybrid MAD+Hist"),
    ("roi_ensemble", "ROI Ensemble (recommended)"),
]

DEFAULT_CLASSIFIER = "roi_ensemble"

# Tuned thresholds for this GUI pipeline (perspective warp + CLAHE + blur).
MAD_THRESHOLD = 22.0
GAUSSIAN_MAD_THRESHOLD = 23.0
BLOCK_MAD_THRESHOLD = 15.0
BLOCK_VOTE_THRESHOLD = 0.65
P75_THRESHOLD = 30.0
MAX_BLOCK_THRESHOLD = 36.0
HIST_INTERSECTION_THRESHOLD = 0.84
VAR_RATIO_THRESHOLD = 1.45
ENSEMBLE_INNER_RATIO = 0.70
ENSEMBLE_INNER_AREA_PCT_DEFAULT = ENSEMBLE_INNER_RATIO * ENSEMBLE_INNER_RATIO * 100.0

_ensemble_inner_area_pct = float(ENSEMBLE_INNER_AREA_PCT_DEFAULT)


def preprocess_roi(roi):
    roi_u8 = np.asarray(roi, dtype=np.uint8)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    roi_eq = clahe.apply(roi_u8)
    roi_blur = cv2.GaussianBlur(roi_eq, (5, 5), 0)
    return roi_blur


def classifier_label(method_key: str) -> str:
    label_map = {key: label for key, label in CLASSIFIER_OPTIONS}
    return label_map.get(method_key, label_map[DEFAULT_CLASSIFIER])


def normalize_classifier_key(method_key: str) -> str:
    valid = {key for key, _ in CLASSIFIER_OPTIONS}
    return method_key if method_key in valid else DEFAULT_CLASSIFIER


def set_ensemble_inner_area_pct(area_pct: float) -> float:
    global _ensemble_inner_area_pct
    try:
        value = float(area_pct)
    except Exception:
        value = float(ENSEMBLE_INNER_AREA_PCT_DEFAULT)
    _ensemble_inner_area_pct = float(np.clip(value, 4.0, 100.0))
    return _ensemble_inner_area_pct


def get_ensemble_inner_area_pct() -> float:
    return float(_ensemble_inner_area_pct)


def _ensemble_inner_side_ratio() -> float:
    return float(np.sqrt(np.clip(get_ensemble_inner_area_pct() / 100.0, 0.04, 1.0)))


def _build_gaussian_kernel(height: int = ROI_SIZE,
                           width: int = ROI_SIZE) -> np.ndarray:
    sigma_x = width / 3.0
    sigma_y = height / 3.0
    ax = np.arange(width, dtype=np.float32) - (width / 2.0) + 0.5
    ay = np.arange(height, dtype=np.float32) - (height / 2.0) + 0.5
    xx, yy = np.meshgrid(ax, ay)
    kernel = np.exp(-((xx * xx) / (2.0 * sigma_x * sigma_x)
                      + (yy * yy) / (2.0 * sigma_y * sigma_y)))
    kernel = kernel / np.sum(kernel)
    # Keep weighted MAD on the same order of magnitude as plain MAD.
    return kernel * float(height * width)


_GAUSS_KERNEL = _build_gaussian_kernel()


def _center_crop_ratio(roi: np.ndarray,
                       ratio: float = ENSEMBLE_INNER_RATIO) -> np.ndarray:
    h, w = roi.shape[:2]
    ratio = float(np.clip(ratio, 0.20, 1.0))
    inner_h = max(4, int(round(h * ratio)))
    inner_w = max(4, int(round(w * ratio)))
    y0 = max(0, (h - inner_h) // 2)
    x0 = max(0, (w - inner_w) // 2)
    y1 = min(h, y0 + inner_h)
    x1 = min(w, x0 + inner_w)
    return roi[y0:y1, x0:x1]


def _compute_similarity_metrics(ref_pre: np.ndarray, test_pre: np.ndarray) -> Tuple[float, float, float]:
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2
    ref_f = ref_pre.astype(np.float32)
    test_f = test_pre.astype(np.float32)
    mu1 = cv2.GaussianBlur(ref_f, (7, 7), 1.5)
    mu2 = cv2.GaussianBlur(test_f, (7, 7), 1.5)
    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.GaussianBlur(ref_f * ref_f, (7, 7), 1.5) - mu1_sq
    sigma2_sq = cv2.GaussianBlur(test_f * test_f, (7, 7), 1.5) - mu2_sq
    sigma12 = cv2.GaussianBlur(ref_f * test_f, (7, 7), 1.5) - mu1_mu2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )
    ssim_map = np.clip(ssim_map, 0, 1)
    mean_ssim = float(np.mean(ssim_map))

    diff_map = cv2.absdiff(test_pre, ref_pre)
    diff_map = cv2.GaussianBlur(diff_map, (3, 3), 0)
    thresh_val = max(DIFF_PIXEL_THRESHOLD, min(80.0, np.mean(diff_map) + np.std(diff_map) * 0.7))
    _, thresh = cv2.threshold(diff_map, thresh_val, 255, cv2.THRESH_BINARY)
    thresh = cv2.morphologyEx(
        thresh,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
    )
    thresh = cv2.morphologyEx(
        thresh,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
    )
    area = float(max(1, thresh.shape[0] * thresh.shape[1]))
    diff_ratio = int(np.count_nonzero(thresh)) / area

    edge_ref = cv2.Canny(ref_pre, 50, 150)
    edge_test = cv2.Canny(test_pre, 50, 150)
    edge_diff = cv2.bitwise_xor(edge_ref, edge_test)
    edge_diff = cv2.morphologyEx(
        edge_diff,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
    )
    edge_ratio = float(np.count_nonzero(edge_diff)) / area

    return mean_ssim, diff_ratio, edge_ratio


def _block_stats(diff_map: np.ndarray, block_size: int = 8) -> Tuple[float, float]:
    h, w = diff_map.shape[:2]
    n_blocks_y = max(1, h // block_size)
    n_blocks_x = max(1, w // block_size)
    total_blocks = n_blocks_y * n_blocks_x
    occupied_blocks = 0
    max_block_mad = 0.0

    for by in range(n_blocks_y):
        for bx in range(n_blocks_x):
            y0 = by * block_size
            x0 = bx * block_size
            block = diff_map[y0:y0 + block_size, x0:x0 + block_size]
            block_mad = float(np.mean(block))
            if block_mad > BLOCK_MAD_THRESHOLD:
                occupied_blocks += 1
            if block_mad > max_block_mad:
                max_block_mad = block_mad

    vote_ratio = occupied_blocks / float(total_blocks)
    return vote_ratio, max_block_mad


def _histogram_intersection(ref_pre: np.ndarray, test_pre: np.ndarray, n_bins: int = 16) -> float:
    ref_hist = cv2.calcHist([ref_pre], [0], None, [n_bins], [0, 256])
    test_hist = cv2.calcHist([test_pre], [0], None, [n_bins], [0, 256])

    ref_hist = ref_hist / max(float(np.sum(ref_hist)), 1.0)
    test_hist = test_hist / max(float(np.sum(test_hist)), 1.0)
    return float(np.sum(np.minimum(ref_hist, test_hist)))


def _variance_ratio(ref_pre: np.ndarray, test_pre: np.ndarray) -> float:
    ref_var = max(float(np.var(ref_pre.astype(np.float32))), 1.0)
    test_var = max(float(np.var(test_pre.astype(np.float32))), 1.0)
    return test_var / ref_var


def _to_u8_for_display(img: np.ndarray) -> np.ndarray:
    arr = np.asarray(img)
    if arr.dtype == np.uint8:
        return arr.copy()

    arr_f = arr.astype(np.float32)
    finite_mask = np.isfinite(arr_f)
    if not np.any(finite_mask):
        return np.zeros(arr_f.shape, dtype=np.uint8)

    valid = arr_f[finite_mask]
    lo = float(np.min(valid))
    hi = float(np.max(valid))
    if abs(hi - lo) < 1e-6:
        return np.zeros(arr_f.shape, dtype=np.uint8)
    norm = (arr_f - lo) / (hi - lo)
    return np.clip(norm * 255.0, 0.0, 255.0).astype(np.uint8)


def roi_debug_steps(ref_roi: np.ndarray, test_roi: np.ndarray,
                    method_key: str = DEFAULT_CLASSIFIER) -> List[Tuple[str, np.ndarray]]:
    ref_raw = _to_u8_for_display(ref_roi)
    test_raw = _to_u8_for_display(test_roi)
    ref_pre = preprocess_roi(ref_raw)
    test_pre = preprocess_roi(test_raw)

    steps: List[Tuple[str, np.ndarray]] = [
        ("ref_raw", ref_raw),
        ("test_raw", test_raw),
        ("ref_pre", ref_pre),
        ("test_pre", test_pre),
        ("diff_pre", cv2.absdiff(test_pre, ref_pre)),
    ]

    return steps


def _norm_above(value: float, threshold: float, band: float) -> float:
    return float(np.clip((value - threshold) / max(band, 1e-6), -1.0, 1.0))


def _norm_below(value: float, threshold: float, band: float) -> float:
    return float(np.clip((threshold - value) / max(band, 1e-6), -1.0, 1.0))


def _confidence_from_margin(margin: float) -> float:
    return float(np.clip(abs(margin), 0.0, 1.0))


def compute_slot_diff(ref_roi, test_roi):
    ref_pre = preprocess_roi(ref_roi)
    test_pre = preprocess_roi(test_roi)
    mean_ssim, diff_ratio, edge_ratio = _compute_similarity_metrics(ref_pre, test_pre)

    occupied = ((mean_ssim < SSIM_THRESHOLD and diff_ratio > DIFF_PIXEL_PERCENT)
                or edge_ratio > EDGE_DIFF_PERCENT)
    return mean_ssim, diff_ratio, edge_ratio, occupied


def classify_roi(ref_roi: np.ndarray, test_roi: np.ndarray,
                 method_key: str = DEFAULT_CLASSIFIER) -> Dict[str, object]:
    method_key = normalize_classifier_key(method_key)
    method_name = classifier_label(method_key)

    ref_pre = preprocess_roi(ref_roi)
    test_pre = preprocess_roi(test_roi)
    ref_eval = ref_pre
    test_eval = test_pre
    ensemble_inner_ratio = _ensemble_inner_side_ratio()
    if method_key == "roi_ensemble":
        ref_eval = _center_crop_ratio(ref_pre, ensemble_inner_ratio)
        test_eval = _center_crop_ratio(test_pre, ensemble_inner_ratio)

    mean_ssim, diff_ratio, edge_ratio = _compute_similarity_metrics(ref_eval, test_eval)

    diff_map = cv2.absdiff(test_eval, ref_eval).astype(np.float32)
    mad = float(np.mean(diff_map))
    gauss_kernel = _GAUSS_KERNEL
    if gauss_kernel.shape != diff_map.shape:
        gauss_kernel = _build_gaussian_kernel(diff_map.shape[0], diff_map.shape[1])
    diff_area = float(max(1, diff_map.shape[0] * diff_map.shape[1]))
    gaussian_mad = float(np.sum(diff_map * gauss_kernel) / diff_area)
    block_vote_ratio, max_block_mad = _block_stats(diff_map)
    p75_mad = float(np.percentile(diff_map, 75))
    hist_inter = _histogram_intersection(ref_eval, test_eval)
    var_ratio = _variance_ratio(ref_eval, test_eval)

    metrics = {
        "mean_ssim": mean_ssim,
        "diff_ratio": diff_ratio,
        "edge_ratio": edge_ratio,
        "mad": mad,
        "gaussian_mad": gaussian_mad,
        "block_vote_ratio": block_vote_ratio,
        "p75_mad": p75_mad,
        "max_block_mad": max_block_mad,
        "hist_intersection": hist_inter,
        "variance_ratio": var_ratio,
        "ensemble_inner_ratio": float(ensemble_inner_ratio),
        "ensemble_inner_area_pct": float(get_ensemble_inner_area_pct()),
    }

    if method_key == "legacy_ssim_edge":
        occupied = ((mean_ssim < SSIM_THRESHOLD and diff_ratio > DIFF_PIXEL_PERCENT)
                    or edge_ratio > EDGE_DIFF_PERCENT)
        confidence = min(1.0, max(
            0.0,
            (DIFF_PIXEL_PERCENT * 4.0 - diff_ratio * 3.5),
            (EDGE_DIFF_PERCENT * 4.0 - edge_ratio * 3.5),
            ((SSIM_THRESHOLD - mean_ssim) * 2.0),
        ))
        metric_name = "legacy_score"
        metric_value = float(confidence) * 100.0

    elif method_key == "ssim_diff_gate":
        occupied = (mean_ssim < SSIM_THRESHOLD and diff_ratio > DIFF_PIXEL_PERCENT)
        if occupied:
            gate_strength = max(
                _norm_below(mean_ssim, SSIM_THRESHOLD, 0.15),
                _norm_above(diff_ratio, DIFF_PIXEL_PERCENT, 0.10),
            )
            confidence = _confidence_from_margin(gate_strength)
        else:
            release_strength = max(
                _norm_above(mean_ssim, SSIM_THRESHOLD, 0.15),
                _norm_below(diff_ratio, DIFF_PIXEL_PERCENT, 0.10),
            )
            confidence = _confidence_from_margin(release_strength)
        metric_name = "ssim_diff_gate"
        metric_value = (SSIM_THRESHOLD - mean_ssim) * 100.0 + (diff_ratio - DIFF_PIXEL_PERCENT) * 100.0

    elif method_key == "mad":
        margin = _norm_above(mad, MAD_THRESHOLD, 6.0)
        occupied = margin > 0.0
        confidence = _confidence_from_margin(margin)
        metric_name = "mad"
        metric_value = mad

    elif method_key == "gaussian_mad":
        margin = _norm_above(gaussian_mad, GAUSSIAN_MAD_THRESHOLD, 6.0)
        occupied = margin > 0.0
        confidence = _confidence_from_margin(margin)
        metric_name = "gaussian_mad"
        metric_value = gaussian_mad

    elif method_key == "block_mad":
        margin = _norm_above(block_vote_ratio, BLOCK_VOTE_THRESHOLD, 0.28)
        occupied = margin > 0.0 and diff_ratio > DIFF_PIXEL_PERCENT
        confidence = _confidence_from_margin(margin)
        metric_name = "block_vote_ratio"
        metric_value = block_vote_ratio * 100.0

    elif method_key == "percentile_mad":
        margin = _norm_above(p75_mad, P75_THRESHOLD, 10.0)
        occupied = margin > 0.0
        confidence = _confidence_from_margin(margin)
        metric_name = "p75_mad"
        metric_value = p75_mad

    elif method_key == "max_block_mad":
        margin = _norm_above(max_block_mad, MAX_BLOCK_THRESHOLD, 12.0)
        occupied = margin > 0.0
        confidence = _confidence_from_margin(margin)
        metric_name = "max_block_mad"
        metric_value = max_block_mad

    elif method_key == "histogram_intersection":
        margin = _norm_below(hist_inter, HIST_INTERSECTION_THRESHOLD, 0.18)
        occupied = margin > 0.0
        confidence = _confidence_from_margin(margin)
        metric_name = "hist_intersection"
        metric_value = hist_inter

    elif method_key == "variance_ratio":
        margin = _norm_above(var_ratio, VAR_RATIO_THRESHOLD, 0.70)
        occupied = margin > 0.0
        confidence = _confidence_from_margin(margin)
        metric_name = "variance_ratio"
        metric_value = var_ratio

    elif method_key == "hybrid_mad_hist":
        score = (
            0.35 * _norm_above(mad, MAD_THRESHOLD, 6.0)
            + 0.25 * _norm_above(p75_mad, P75_THRESHOLD, 10.0)
            + 0.20 * _norm_above(max_block_mad, MAX_BLOCK_THRESHOLD, 12.0)
            + 0.20 * _norm_below(hist_inter, HIST_INTERSECTION_THRESHOLD, 0.18)
        )
        occupied = score > 0.00
        confidence = float(np.clip(abs(score), 0.0, 1.0))
        metric_name = "hybrid_score"
        metric_value = score * 100.0

    else:  # roi_ensemble
        score = (
            0.24 * _norm_above(mad, MAD_THRESHOLD, 6.0)
            + 0.14 * _norm_above(gaussian_mad, GAUSSIAN_MAD_THRESHOLD, 6.0)
            + 0.06 * _norm_above(block_vote_ratio, BLOCK_VOTE_THRESHOLD, 0.28)
            + 0.20 * _norm_above(p75_mad, P75_THRESHOLD, 10.0)
            + 0.14 * _norm_above(max_block_mad, MAX_BLOCK_THRESHOLD, 12.0)
            + 0.12 * _norm_below(hist_inter, HIST_INTERSECTION_THRESHOLD, 0.18)
            + 0.06 * _norm_above(var_ratio, VAR_RATIO_THRESHOLD, 0.70)
            + 0.04 * _norm_below(mean_ssim, SSIM_THRESHOLD, 0.15)
        )
        occupied = score > 0.00
        confidence = float(np.clip(abs(score), 0.0, 1.0))
        metric_name = "ensemble_score"
        metric_value = score * 100.0

    return {
        "method_key": method_key,
        "method_name": method_name,
        "occupied": bool(occupied),
        "confidence": float(np.clip(confidence, 0.0, 1.0)),
        "metric_name": metric_name,
        "metric_value": float(metric_value),
        "metrics": metrics,
    }


def warp_roi(gray, s):
    src = np.array(s.pts, dtype=np.float32)
    dst = np.array([[0, 0], [ROI_SIZE - 1, 0],
                    [ROI_SIZE - 1, ROI_SIZE - 1], [0, ROI_SIZE - 1]],
                   dtype=np.float32)
    try:
        M = cv2.getPerspectiveTransform(src, dst)
        return cv2.warpPerspective(gray, M, (ROI_SIZE, ROI_SIZE))
    except Exception:
        return None


def detect_image_lines(gray):
    detected_lines = []
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5))
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=80,
                            minLineLength=gray.shape[0] // 6,
                            maxLineGap=20)
    if lines is None:
        return detected_lines
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
        if angle > 75 or angle < 15:
            detected_lines.append((x1, y1, x2, y2))
    return detected_lines


def nearest_detected_line(ix, iy, detected_lines, threshold):
    best_i = -1
    best_d = threshold + 1
    p = np.array([ix, iy], dtype=np.float32)
    for i, (x1, y1, x2, y2) in enumerate(detected_lines):
        a = np.array([x1, y1], dtype=np.float32)
        b = np.array([x2, y2], dtype=np.float32)
        v = b - a
        w = p - a
        v_len2 = np.dot(v, v)
        if v_len2 < 1e-6:
            d = float(np.linalg.norm(w))
        else:
            t = np.clip(np.dot(w, v) / v_len2, 0.0, 1.0)
            proj = a + t * v
            d = float(np.linalg.norm(p - proj))
        if d < best_d:
            best_d = d
            best_i = i
    return best_i


def project_to_line(ix, iy, line):
    x1, y1, x2, y2 = line
    p = np.array([ix, iy], dtype=np.float32)
    a = np.array([x1, y1], dtype=np.float32)
    b = np.array([x2, y2], dtype=np.float32)
    v = b - a
    v_len2 = np.dot(v, v)
    if v_len2 < 1e-6:
        proj = a
    else:
        t = np.clip(np.dot(p - a, v) / v_len2, 0.0, 1.0)
        proj = a + t * v
    return int(round(proj[0])), int(round(proj[1]))


def snap_to_detected_line(ix, iy, detected_lines, threshold):
    if not detected_lines:
        return ix, iy
    line_idx = nearest_detected_line(ix, iy, detected_lines, threshold)
    if line_idx < 0:
        return ix, iy
    return project_to_line(ix, iy, detected_lines[line_idx])

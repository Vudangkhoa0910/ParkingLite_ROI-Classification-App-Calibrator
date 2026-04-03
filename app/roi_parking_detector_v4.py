#!/usr/bin/env python3
"""
ParkingLite ROI Parking Detector v4.0
=====================================
Ultra-lightweight parking slot detection via reference frame comparison.
Designed for ESP32-CAM: integer math, 32x32 ROI, <15KB RAM.

Usage:
    python3 roi_parking_detector_v4.py \
        --empty ../parking_empty.png \
        --test ../parking_with_car.png \
        --output ../output/roi_v4/

Author: ParkingLite Research Team
"""

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Tuple, Optional

import cv2
import numpy as np

# ─── Constants (mirroring ESP32 firmware) ───────────────────────────────
ROI_SIZE = 32                  # 32x32 pixel ROI (same as firmware)
EDGE_SOBEL_THRESH = 30         # Sobel magnitude threshold
MAD_THRESH_DEFAULT = 12.0      # Mean Absolute Difference threshold
BLOCK_MAD_THRESH = 15.0        # Block-level MAD threshold
BLOCK_VOTE_RATIO = 0.4         # 40% blocks must differ
EDGE_RATIO_THRESH = 1.5        # Edge density ratio threshold
HIST_INTERSECT_THRESH = 0.75   # Histogram intersection threshold
VARIANCE_RATIO_THRESH = 1.8    # Variance ratio threshold
COMBINED_THRESH = 0.5          # Combined score threshold


# ─── Data Structures ───────────────────────────────────────────────────
@dataclass
class ROIRect:
    """Parking slot ROI rectangle."""
    x: int
    y: int
    w: int
    h: int
    label: str = ""

@dataclass
class ClassifyResult:
    """Classification result for a single slot."""
    slot_idx: int
    label: str
    prediction: int           # 0=empty, 1=occupied
    confidence: float         # 0.0 - 1.0
    raw_metric: float         # method-specific raw value
    method: str = ""

@dataclass
class SlotResults:
    """All classification results for one slot."""
    slot_idx: int
    label: str
    ground_truth: int         # -1=unknown, 0=empty, 1=occupied
    methods: dict = field(default_factory=dict)
    best_prediction: int = -1
    best_confidence: float = 0.0


# ─── Image Alignment ───────────────────────────────────────────────────
def align_images(ref_gray: np.ndarray, test_gray: np.ndarray,
                 max_features: int = 5000,
                 match_ratio: float = 0.7) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Align test image to reference using ORB + homography.
    Returns: (aligned_test, homography_matrix, num_inliers)
    """
    h_ref, w_ref = ref_gray.shape[:2]

    # ORB feature detection
    orb = cv2.ORB_create(nfeatures=max_features)
    kp1, des1 = orb.detectAndCompute(ref_gray, None)
    kp2, des2 = orb.detectAndCompute(test_gray, None)

    if des1 is None or des2 is None or len(kp1) < 10 or len(kp2) < 10:
        print("[WARN] Not enough features, using resize-only alignment")
        aligned = cv2.resize(test_gray, (w_ref, h_ref), interpolation=cv2.INTER_LINEAR)
        return aligned, np.eye(3), 0

    # Brute-force matching with ratio test
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    raw_matches = bf.knnMatch(des2, des1, k=2)

    good_matches = []
    for m_pair in raw_matches:
        if len(m_pair) == 2:
            m, n = m_pair
            if m.distance < match_ratio * n.distance:
                good_matches.append(m)

    print(f"  Features: ref={len(kp1)}, test={len(kp2)}, good_matches={len(good_matches)}")

    if len(good_matches) < 10:
        print("[WARN] Too few matches, using resize-only alignment")
        aligned = cv2.resize(test_gray, (w_ref, h_ref), interpolation=cv2.INTER_LINEAR)
        return aligned, np.eye(3), 0

    # Compute homography
    pts_test = np.float32([kp2[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    pts_ref = np.float32([kp1[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(pts_test, pts_ref, cv2.RANSAC, 5.0)
    n_inliers = int(mask.sum()) if mask is not None else 0
    print(f"  Homography inliers: {n_inliers}/{len(good_matches)}")

    if H is None:
        print("[WARN] Homography failed, using resize-only alignment")
        aligned = cv2.resize(test_gray, (w_ref, h_ref), interpolation=cv2.INTER_LINEAR)
        return aligned, np.eye(3), 0

    # Warp test image to reference coordinate system
    aligned = cv2.warpPerspective(test_gray, H, (w_ref, h_ref),
                                  flags=cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_REPLICATE)

    # ECC refinement for sub-pixel accuracy
    try:
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 1e-4)
        _, warp_matrix = cv2.findTransformECC(
            ref_gray, aligned, np.eye(2, 3, dtype=np.float32),
            cv2.MOTION_AFFINE, criteria,
            inputMask=None, gaussFiltSize=5
        )
        aligned = cv2.warpAffine(aligned, warp_matrix, (w_ref, h_ref),
                                 flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
                                 borderMode=cv2.BORDER_REPLICATE)
        print("  ECC refinement: OK")
    except cv2.error:
        print("  ECC refinement: skipped (convergence failed)")

    return aligned, H, n_inliers


# ─── Slot Detection ────────────────────────────────────────────────────
def detect_parking_lines(gray: np.ndarray) -> Tuple[List[int], List[int]]:
    """
    Detect vertical parking lines using Hough transform.
    Returns: (vertical_x_positions, horizontal_y_positions)
    """
    h, w = gray.shape[:2]

    # Edge detection
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    # Morphological close to connect broken lines
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5))
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    # Hough lines
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=80,
                            minLineLength=h // 6, maxLineGap=20)

    if lines is None:
        return [], []

    vertical_xs = []
    horizontal_ys = []

    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)

        # Vertical lines (within 15 degrees of vertical)
        if angle > 75:
            mid_x = (x1 + x2) // 2
            vertical_xs.append(mid_x)

        # Horizontal lines (within 15 degrees of horizontal)
        elif angle < 15:
            mid_y = (y1 + y2) // 2
            horizontal_ys.append(mid_y)

    return vertical_xs, horizontal_ys


def cluster_positions(positions: List[int], min_gap: int = 30) -> List[int]:
    """Cluster nearby positions and return centroids."""
    if not positions:
        return []

    sorted_pos = sorted(positions)
    clusters = [[sorted_pos[0]]]

    for p in sorted_pos[1:]:
        if p - clusters[-1][-1] < min_gap:
            clusters[-1].append(p)
        else:
            clusters.append([p])

    return [int(np.mean(c)) for c in clusters]


def auto_detect_slots(gray: np.ndarray) -> List[ROIRect]:
    """
    Automatically detect parking slot ROIs from line structure.
    Returns list of ROIRect for each detected slot.
    Uses median-width filtering to reject false line detections.
    """
    h, w = gray.shape[:2]
    print("\n[AUTO-DETECT] Analyzing parking structure...")

    vertical_xs, horizontal_ys = detect_parking_lines(gray)

    # Cluster detected lines
    v_lines = cluster_positions(vertical_xs, min_gap=w // 25)
    h_lines = cluster_positions(horizontal_ys, min_gap=h // 10)

    print(f"  Raw vertical lines ({len(v_lines)}): {v_lines}")
    print(f"  Raw horizontal lines ({len(h_lines)}): {h_lines}")

    # ─── Fix: Filter false vertical lines using median width ───
    if len(v_lines) >= 3:
        gaps = [v_lines[i+1] - v_lines[i] for i in range(len(v_lines)-1)]
        median_gap = float(np.median(gaps))
        print(f"  Slot widths: {gaps}, median={median_gap:.0f}")

        # Rebuild lines: reject first/last line if it creates an outlier gap
        filtered_lines = list(v_lines)

        # Check first gap: if > 1.15× median, the first line is likely false
        # (arrow markings, image edge, etc.)
        # Replace it with an estimated position based on median width
        if len(filtered_lines) >= 2:
            first_gap = filtered_lines[1] - filtered_lines[0]
            if first_gap > median_gap * 1.15:
                estimated_start = int(filtered_lines[1] - median_gap)
                print(f"  [FIX] First line x={filtered_lines[0]} creates wide gap "
                      f"({first_gap:.0f} > {median_gap*1.15:.0f}), "
                      f"adjusted to x={estimated_start}")
                filtered_lines[0] = max(estimated_start, 5)

        # Check last gap similarly
        if len(filtered_lines) >= 2:
            last_gap = filtered_lines[-1] - filtered_lines[-2]
            if last_gap > median_gap * 1.15:
                estimated_end = int(filtered_lines[-2] + median_gap)
                print(f"  [FIX] Last line x={filtered_lines[-1]} creates wide gap "
                      f"({last_gap:.0f}), adjusted to x={estimated_end}")
                filtered_lines[-1] = min(estimated_end, w - 5)

        v_lines = filtered_lines
        print(f"  Filtered vertical lines ({len(v_lines)}): {v_lines}")

    # Find the main horizontal divider (should be near middle)
    h_mid = h // 2
    divider_y = h_mid
    if h_lines:
        divider_y = min(h_lines, key=lambda y: abs(y - h_mid))
    print(f"  Horizontal divider at y={divider_y}")

    # Generate slot ROIs from vertical lines
    # Order: ALL top row first (T0..T7), then ALL bottom row (B0..B7)
    slots = []
    top_slots = []
    bot_slots = []

    if len(v_lines) >= 2:
        for i in range(len(v_lines) - 1):
            x_left = v_lines[i]
            x_right = v_lines[i + 1]
            slot_w = x_right - x_left

            if slot_w < w // 25 or slot_w > w // 3:
                continue

            # Inset margins to avoid white line pixels
            margin_x = max(int(slot_w * 0.12), 4)
            margin_y_top = max(int(divider_y * 0.06), 5)

            # Top row slot (between top of image and divider)
            top_y = margin_y_top
            top_h = divider_y - top_y - margin_y_top
            if top_h > 20:
                top_slots.append(ROIRect(
                    x=x_left + margin_x,
                    y=top_y,
                    w=slot_w - 2 * margin_x,
                    h=top_h,
                    label=f"T{len(top_slots)}"
                ))

            # Bottom row slot (between divider and bottom of image)
            bot_y = divider_y + margin_y_top
            bot_h = min(h - bot_y - margin_y_top,
                        divider_y - 2 * margin_y_top)
            if bot_h > 20:
                bot_slots.append(ROIRect(
                    x=x_left + margin_x,
                    y=bot_y,
                    w=slot_w - 2 * margin_x,
                    h=bot_h,
                    label=f"B{len(bot_slots)}"
                ))

    # Combine: top row first, then bottom row
    slots = top_slots + bot_slots

    print(f"  Detected {len(top_slots)} top + {len(bot_slots)} bottom "
          f"= {len(slots)} total slot ROIs")

    # Fallback: if detection fails, use grid-based approach
    if len(slots) < 4:
        print("  [FALLBACK] Using grid-based slot detection...")
        slots = grid_based_slots(w, h, divider_y)

    return slots


def grid_based_slots(img_w: int, img_h: int, divider_y: int,
                     n_cols: int = 8, margin_pct: float = 0.05) -> List[ROIRect]:
    """
    Fallback: generate slot ROIs using uniform grid.
    """
    margin_x = int(img_w * margin_pct)
    margin_y = int(img_h * margin_pct)

    usable_w = img_w - 2 * margin_x
    slot_w = usable_w // n_cols
    inset = max(int(slot_w * 0.08), 2)

    slots = []

    for col in range(n_cols):
        x = margin_x + col * slot_w + inset

        # Top row
        top_y = margin_y
        top_h = divider_y - top_y - inset
        if top_h > 20:
            slots.append(ROIRect(x=x, y=top_y, w=slot_w - 2 * inset,
                                 h=top_h, label=f"T{col}"))

        # Bottom row
        bot_y = divider_y + inset
        bot_h = min(img_h - bot_y - margin_y, divider_y - 2 * inset)
        if bot_h > 20:
            slots.append(ROIRect(x=x, y=bot_y, w=slot_w - 2 * inset,
                                 h=bot_h, label=f"B{col}"))

    return slots


# ─── ROI Extraction (ESP32-compatible bilinear interpolation) ──────────
def extract_roi_bilinear(gray: np.ndarray, roi: ROIRect,
                         target_size: int = ROI_SIZE) -> np.ndarray:
    """
    Extract and resize ROI to target_size×target_size using bilinear interpolation.
    Mimics ESP32 fixed-point 16.16 bilinear interpolation.
    """
    h, w = gray.shape[:2]

    # Clamp ROI to image bounds
    x1 = max(0, roi.x)
    y1 = max(0, roi.y)
    x2 = min(w, roi.x + roi.w)
    y2 = min(h, roi.y + roi.h)

    crop = gray[y1:y2, x1:x2]

    if crop.size == 0:
        return np.zeros((target_size, target_size), dtype=np.uint8)

    # Resize to target_size × target_size
    resized = cv2.resize(crop, (target_size, target_size),
                         interpolation=cv2.INTER_LINEAR)
    return resized


# ─── Classification Methods (all ESP32-portable) ──────────────────────

def classify_mad(ref_roi: np.ndarray, cur_roi: np.ndarray,
                 threshold: float = MAD_THRESH_DEFAULT) -> ClassifyResult:
    """
    Method 1: Mean Absolute Difference.
    ESP32: pure integer, O(1024) for 32×32.
    """
    diff = np.abs(ref_roi.astype(np.int16) - cur_roi.astype(np.int16))
    mad = float(np.mean(diff))
    pred = 1 if mad > threshold else 0
    conf = min(abs(mad - threshold) / max(threshold, 1), 1.0)
    return ClassifyResult(
        slot_idx=0, label="", prediction=pred,
        confidence=conf, raw_metric=mad, method="mad"
    )


def classify_block_mad(ref_roi: np.ndarray, cur_roi: np.ndarray,
                       block_size: int = 8,
                       block_thresh: float = BLOCK_MAD_THRESH,
                       vote_ratio: float = BLOCK_VOTE_RATIO) -> ClassifyResult:
    """
    Method 2: Block-based MAD with voting.
    Divide ROI into blocks, each block votes occupied/empty.
    More robust to partial occlusion and shadows.
    ESP32: O(1024), same as MAD but with local decisions.
    """
    n_blocks = ROI_SIZE // block_size
    n_total = n_blocks * n_blocks
    n_occupied = 0
    block_scores = []

    for by in range(n_blocks):
        for bx in range(n_blocks):
            y1 = by * block_size
            x1 = bx * block_size
            ref_block = ref_roi[y1:y1+block_size, x1:x1+block_size]
            cur_block = cur_roi[y1:y1+block_size, x1:x1+block_size]
            block_mad = float(np.mean(np.abs(
                ref_block.astype(np.int16) - cur_block.astype(np.int16)
            )))
            block_scores.append(block_mad)
            if block_mad > block_thresh:
                n_occupied += 1

    ratio = n_occupied / n_total
    pred = 1 if ratio > vote_ratio else 0
    conf = min(abs(ratio - vote_ratio) / max(vote_ratio, 0.01), 1.0)
    return ClassifyResult(
        slot_idx=0, label="", prediction=pred,
        confidence=conf, raw_metric=ratio, method="block_mad"
    )


def classify_edge_ratio(ref_roi: np.ndarray, cur_roi: np.ndarray,
                        threshold: float = EDGE_RATIO_THRESH) -> ClassifyResult:
    """
    Method 3: Edge density ratio.
    Compare edge counts between reference and current.
    ESP32: integer Sobel, O(1024).
    """
    def edge_count(img, thresh=EDGE_SOBEL_THRESH):
        # Integer Sobel (Manhattan norm, same as ESP32)
        gx = cv2.Sobel(img, cv2.CV_16S, 1, 0, ksize=3)
        gy = cv2.Sobel(img, cv2.CV_16S, 0, 1, ksize=3)
        mag = np.abs(gx) + np.abs(gy)  # Manhattan approximation
        return int(np.sum(mag > thresh * 4))  # Scale for 16-bit Sobel

    ref_edges = max(edge_count(ref_roi), 1)
    cur_edges = max(edge_count(cur_roi), 1)
    ratio = cur_edges / ref_edges

    pred = 1 if ratio > threshold else 0
    conf = min(abs(ratio - threshold) / max(threshold, 0.01), 1.0)
    return ClassifyResult(
        slot_idx=0, label="", prediction=pred,
        confidence=conf, raw_metric=ratio, method="edge_ratio"
    )


def classify_histogram(ref_roi: np.ndarray, cur_roi: np.ndarray,
                       n_bins: int = 16,
                       threshold: float = HIST_INTERSECT_THRESH) -> ClassifyResult:
    """
    Method 4: Histogram intersection.
    ESP32: 16-bin histogram = 32 bytes, O(1024 + 16).
    """
    ref_hist = cv2.calcHist([ref_roi], [0], None, [n_bins], [0, 256])
    cur_hist = cv2.calcHist([cur_roi], [0], None, [n_bins], [0, 256])

    # Normalize
    ref_hist = ref_hist / max(ref_hist.sum(), 1)
    cur_hist = cur_hist / max(cur_hist.sum(), 1)

    # Intersection
    intersection = float(np.sum(np.minimum(ref_hist, cur_hist)))

    pred = 1 if intersection < threshold else 0
    conf = min(abs(intersection - threshold) / max(threshold, 0.01), 1.0)
    return ClassifyResult(
        slot_idx=0, label="", prediction=pred,
        confidence=conf, raw_metric=intersection, method="histogram"
    )


def classify_variance_ratio(ref_roi: np.ndarray, cur_roi: np.ndarray,
                            threshold: float = VARIANCE_RATIO_THRESH) -> ClassifyResult:
    """
    Method 5: Variance ratio.
    Cars introduce texture → higher variance than empty concrete.
    ESP32: O(1024) one-pass sum + sum_of_squares.
    """
    ref_var = max(float(np.var(ref_roi.astype(np.float32))), 1.0)
    cur_var = max(float(np.var(cur_roi.astype(np.float32))), 1.0)
    ratio = cur_var / ref_var

    pred = 1 if ratio > threshold else 0
    conf = min(abs(ratio - threshold) / max(threshold, 0.01), 1.0)
    return ClassifyResult(
        slot_idx=0, label="", prediction=pred,
        confidence=conf, raw_metric=ratio, method="variance_ratio"
    )


def classify_gaussian_mad(ref_roi: np.ndarray, cur_roi: np.ndarray,
                          threshold: float = MAD_THRESH_DEFAULT) -> ClassifyResult:
    """
    Method 6: Gaussian-weighted MAD.
    Center pixels weighted more than edges — robust to alignment errors.
    ESP32: precompute weight table (1024 bytes), O(1024).
    """
    size = ref_roi.shape[0]
    # Create 2D Gaussian weight kernel (center-focused)
    sigma = size / 3.0  # ~10.7 for 32×32
    ax = np.arange(size) - size / 2.0 + 0.5
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    kernel = kernel / kernel.sum() * (size * size)  # Normalize to same scale as MAD

    diff = np.abs(ref_roi.astype(np.float32) - cur_roi.astype(np.float32))
    weighted_mad = float(np.sum(diff * kernel) / (size * size))

    pred = 1 if weighted_mad > threshold else 0
    conf = min(abs(weighted_mad - threshold) / max(threshold, 1), 1.0)
    return ClassifyResult(
        slot_idx=0, label="", prediction=pred,
        confidence=conf, raw_metric=weighted_mad, method="gaussian_mad"
    )


def classify_local_aligned_mad(ref_gray: np.ndarray, cur_gray: np.ndarray,
                               roi: ROIRect,
                               search_radius: int = 15,
                               threshold: float = MAD_THRESH_DEFAULT) -> ClassifyResult:
    """
    Method 7: Local-aligned MAD.
    For each slot, search within ±search_radius to find best local alignment,
    then compute MAD at the best-aligned position.
    Handles cases where global homography is slightly off at edges.
    ESP32: not needed (camera is fixed), but valuable for demo with handheld photos.
    """
    h, w = ref_gray.shape[:2]

    # Extract reference ROI (fixed position)
    ref_roi = extract_roi_bilinear(ref_gray, roi)

    best_mad = float('inf')
    best_dx, best_dy = 0, 0

    # Search around the original position
    for dy in range(-search_radius, search_radius + 1, 3):
        for dx in range(-search_radius, search_radius + 1, 3):
            shifted_roi = ROIRect(
                x=max(0, roi.x + dx),
                y=max(0, roi.y + dy),
                w=roi.w, h=roi.h, label=roi.label
            )
            # Bounds check
            if shifted_roi.x + shifted_roi.w > w or shifted_roi.y + shifted_roi.h > h:
                continue

            cur_roi = extract_roi_bilinear(cur_gray, shifted_roi)
            mad = float(np.mean(np.abs(
                ref_roi.astype(np.int16) - cur_roi.astype(np.int16)
            )))

            if mad < best_mad:
                # We want the MINIMUM MAD for alignment (best match = lowest diff)
                # But for empty slots, low MAD is correct
                # For occupied slots, even best-aligned MAD should be high
                best_mad = mad
                best_dx, best_dy = dx, dy

    # Also compute MAD at original position for comparison
    original_roi = extract_roi_bilinear(cur_gray, roi)
    original_mad = float(np.mean(np.abs(
        ref_roi.astype(np.int16) - original_roi.astype(np.int16)
    )))

    # Use the MAXIMUM of (original, best-aligned) to detect change
    # If car is present: both original and best-aligned MAD will be high
    # If empty: best-aligned MAD will be low (better alignment = lower diff)
    # Key insight: use best-aligned for empty slots, original for occupied
    # Simple heuristic: if best alignment barely improves, use original
    improvement_ratio = (original_mad - best_mad) / max(original_mad, 1)
    if improvement_ratio > 0.5:
        # Large improvement = misalignment was the issue, use best-aligned
        mad = best_mad
    else:
        # Small improvement = genuine content difference
        mad = original_mad

    pred = 1 if mad > threshold else 0
    conf = min(abs(mad - threshold) / max(threshold, 1), 1.0)
    return ClassifyResult(
        slot_idx=0, label="", prediction=pred,
        confidence=conf, raw_metric=mad, method="local_aligned_mad"
    )


def classify_percentile_mad(ref_roi: np.ndarray, cur_roi: np.ndarray,
                            percentile: float = 75.0,
                            threshold: float = 15.0) -> ClassifyResult:
    """
    Method 9: Percentile-based MAD.
    Instead of mean diff, uses the Pth percentile of pixel differences.
    Robust when car only partially covers ROI (edge slots).
    ESP32: sort 1024 values (~5ms) or use histogram-based percentile (~1ms).
    """
    diff = np.abs(ref_roi.astype(np.int16) - cur_roi.astype(np.int16)).ravel()
    pval = float(np.percentile(diff, percentile))

    pred = 1 if pval > threshold else 0
    conf = min(abs(pval - threshold) / max(threshold, 1), 1.0)
    return ClassifyResult(
        slot_idx=0, label="", prediction=pred,
        confidence=conf, raw_metric=pval, method="percentile_mad"
    )


def classify_max_block(ref_roi: np.ndarray, cur_roi: np.ndarray,
                       block_size: int = 8,
                       threshold: float = 20.0) -> ClassifyResult:
    """
    Method 10: Maximum block MAD.
    Divide ROI into blocks, use the MAX block's MAD.
    Detects car even if it only covers one region of the ROI.
    ESP32: O(1024), same as regular block MAD.
    """
    n_blocks = ROI_SIZE // block_size
    max_block_mad = 0.0

    for by in range(n_blocks):
        for bx in range(n_blocks):
            y1 = by * block_size
            x1 = bx * block_size
            ref_block = ref_roi[y1:y1+block_size, x1:x1+block_size]
            cur_block = cur_roi[y1:y1+block_size, x1:x1+block_size]
            block_mad = float(np.mean(np.abs(
                ref_block.astype(np.int16) - cur_block.astype(np.int16)
            )))
            max_block_mad = max(max_block_mad, block_mad)

    pred = 1 if max_block_mad > threshold else 0
    conf = min(abs(max_block_mad - threshold) / max(threshold, 1), 1.0)
    return ClassifyResult(
        slot_idx=0, label="", prediction=pred,
        confidence=conf, raw_metric=max_block_mad, method="max_block"
    )


def classify_dual_align_mad(ref_roi: np.ndarray,
                            homo_roi: np.ndarray,
                            resize_roi: np.ndarray,
                            threshold: float = MAD_THRESH_DEFAULT) -> ClassifyResult:
    """
    Method 8: Dual-alignment MAD.
    Compares slot using BOTH homography-aligned AND simple-resize-aligned images.
    Takes the MAXIMUM MAD — if either alignment shows a car, it's occupied.
    Solves edge-region alignment distortion problem.
    ESP32: not needed (fixed camera), but critical for handheld-photo demos.
    """
    mad_homo = float(np.mean(np.abs(
        ref_roi.astype(np.int16) - homo_roi.astype(np.int16)
    )))
    mad_resize = float(np.mean(np.abs(
        ref_roi.astype(np.int16) - resize_roi.astype(np.int16)
    )))

    # Use maximum — if either sees a car, classify as occupied
    mad = max(mad_homo, mad_resize)

    pred = 1 if mad > threshold else 0
    conf = min(abs(mad - threshold) / max(threshold, 1), 1.0)
    return ClassifyResult(
        slot_idx=0, label="", prediction=pred,
        confidence=conf, raw_metric=mad,
        method="dual_align_mad"
    )


def classify_combined(ref_roi: np.ndarray, cur_roi: np.ndarray,
                      threshold: float = COMBINED_THRESH) -> ClassifyResult:
    """
    Method 6: Combined weighted score.
    Fuses MAD + Block MAD + Edge Ratio for robust decision.
    """
    r_mad = classify_mad(ref_roi, cur_roi)
    r_block = classify_block_mad(ref_roi, cur_roi)
    r_edge = classify_edge_ratio(ref_roi, cur_roi)
    r_hist = classify_histogram(ref_roi, cur_roi)

    r_gauss = classify_gaussian_mad(ref_roi, cur_roi)
    r_pct = classify_percentile_mad(ref_roi, cur_roi)
    r_maxblk = classify_max_block(ref_roi, cur_roi)

    # Weighted vote: emphasize robust methods
    components = {
        'mad':            (r_mad,    0.15),
        'gaussian_mad':   (r_gauss,  0.15),
        'block_mad':      (r_block,  0.10),
        'percentile_mad': (r_pct,    0.20),
        'max_block':      (r_maxblk, 0.20),
        'edge_ratio':     (r_edge,   0.10),
        'histogram':      (r_hist,   0.10),
    }
    score = sum(w * r.prediction * (0.5 + 0.5 * r.confidence)
                for r, w in components.values())

    pred = 1 if score > threshold else 0
    conf = min(abs(score - threshold) / max(threshold, 0.01), 1.0)
    return ClassifyResult(
        slot_idx=0, label="", prediction=pred,
        confidence=conf, raw_metric=score, method="combined"
    )


# ─── Threshold Optimization ───────────────────────────────────────────
def optimize_thresholds(ref_gray: np.ndarray, test_gray: np.ndarray,
                        slots: List[ROIRect],
                        ground_truth: List[int]) -> dict:
    """
    Grid search to find optimal thresholds for each method.
    Returns dict of {method_name: best_threshold}.
    """
    print("\n[THRESHOLD OPTIMIZATION] Running grid search...")

    # Extract all ROIs
    ref_rois = [extract_roi_bilinear(ref_gray, s) for s in slots]
    cur_rois = [extract_roi_bilinear(test_gray, s) for s in slots]

    # Compute raw metrics for each method
    methods_metrics = {
        'mad': [],
        'gaussian_mad': [],
        'block_mad': [],
        'edge_ratio': [],
        'histogram': [],
        'variance_ratio': [],
    }

    for i in range(len(slots)):
        if ground_truth[i] < 0:
            continue
        ref_r, cur_r = ref_rois[i], cur_rois[i]

        # MAD
        diff = np.abs(ref_r.astype(np.int16) - cur_r.astype(np.int16))
        methods_metrics['mad'].append((float(np.mean(diff)), ground_truth[i]))

        # Gaussian MAD
        r = classify_gaussian_mad(ref_r, cur_r)
        methods_metrics['gaussian_mad'].append((r.raw_metric, ground_truth[i]))

        # Block MAD
        r = classify_block_mad(ref_r, cur_r)
        methods_metrics['block_mad'].append((r.raw_metric, ground_truth[i]))

        # Edge ratio
        r = classify_edge_ratio(ref_r, cur_r)
        methods_metrics['edge_ratio'].append((r.raw_metric, ground_truth[i]))

        # Histogram
        r = classify_histogram(ref_r, cur_r)
        methods_metrics['histogram'].append((r.raw_metric, ground_truth[i]))

        # Variance ratio
        r = classify_variance_ratio(ref_r, cur_r)
        methods_metrics['variance_ratio'].append((r.raw_metric, ground_truth[i]))

    # Grid search for each method
    best_thresholds = {}

    for method_name, metrics_gt in methods_metrics.items():
        if not metrics_gt:
            continue

        values = [m[0] for m in metrics_gt]
        labels = [m[1] for m in metrics_gt]

        # For histogram: lower intersection = more different (inverted)
        inverted = method_name == 'histogram'

        v_min, v_max = min(values), max(values)
        best_acc = 0
        best_t = (v_min + v_max) / 2

        for t in np.linspace(v_min - 0.1, v_max + 0.1, 200):
            correct = 0
            for val, gt in zip(values, labels):
                if inverted:
                    pred = 1 if val < t else 0
                else:
                    pred = 1 if val > t else 0
                if pred == gt:
                    correct += 1
            acc = correct / len(labels)
            if acc > best_acc:
                best_acc = acc
                best_t = t

        best_thresholds[method_name] = {
            'threshold': round(best_t, 4),
            'accuracy': round(best_acc, 4),
            'metric_range': [round(v_min, 4), round(v_max, 4)],
        }
        print(f"  {method_name}: threshold={best_t:.4f}, accuracy={best_acc:.1%}, "
              f"range=[{v_min:.2f}, {v_max:.2f}]")

    return best_thresholds


# ─── Main Classification Pipeline ─────────────────────────────────────
def classify_all_slots(ref_gray: np.ndarray, test_gray: np.ndarray,
                       slots: List[ROIRect],
                       ground_truth: Optional[List[int]] = None,
                       custom_thresholds: Optional[dict] = None,
                       resize_gray: Optional[np.ndarray] = None
                       ) -> List[SlotResults]:
    """
    Classify all slots using all methods.
    resize_gray: simple-resized test image (no homography) for dual-alignment.
    """
    results = []

    # Standard ROI-based methods
    roi_methods = {
        'mad': classify_mad,
        'gaussian_mad': classify_gaussian_mad,
        'block_mad': classify_block_mad,
        'percentile_mad': classify_percentile_mad,
        'max_block': classify_max_block,
        'edge_ratio': classify_edge_ratio,
        'histogram': classify_histogram,
        'variance_ratio': classify_variance_ratio,
        'combined': classify_combined,
    }

    for idx, slot in enumerate(slots):
        ref_roi = extract_roi_bilinear(ref_gray, slot)
        cur_roi = extract_roi_bilinear(test_gray, slot)

        gt = ground_truth[idx] if ground_truth and idx < len(ground_truth) else -1
        sr = SlotResults(slot_idx=idx, label=slot.label, ground_truth=gt)

        # Standard methods (use pre-extracted ROIs)
        for method_name, fn in roi_methods.items():
            r = fn(ref_roi, cur_roi)
            r.slot_idx = idx
            r.label = slot.label
            sr.methods[method_name] = r

        # Local-aligned MAD (needs full image access for search)
        r = classify_local_aligned_mad(ref_gray, test_gray, slot)
        r.slot_idx = idx
        r.label = slot.label
        sr.methods['local_aligned_mad'] = r

        # Dual-alignment MAD (uses both homography and resize)
        if resize_gray is not None:
            resize_roi = extract_roi_bilinear(resize_gray, slot)
            r = classify_dual_align_mad(ref_roi, cur_roi, resize_roi)
            r.slot_idx = idx
            r.label = slot.label
            sr.methods['dual_align_mad'] = r

        # Best prediction = combined (fuses all methods)
        best = sr.methods.get('combined', sr.methods.get('mad'))
        sr.best_prediction = best.prediction
        sr.best_confidence = best.confidence

        results.append(sr)

    return results


# ─── Visualization ─────────────────────────────────────────────────────
def visualize_results(ref_color: np.ndarray, test_color: np.ndarray,
                      aligned_color: np.ndarray,
                      slots: List[ROIRect], results: List[SlotResults],
                      output_dir: str):
    """Generate comprehensive visualization images."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    os.makedirs(output_dir, exist_ok=True)

    # ─── 1. Slot overlay on aligned test image ───
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))

    # Reference with ROIs
    axes[0].imshow(ref_color, cmap='gray' if ref_color.ndim == 2 else None)
    axes[0].set_title('Reference (Empty)', fontsize=14, fontweight='bold')
    for i, slot in enumerate(slots):
        color = '#00FF00'  # green for empty
        axes[0].add_patch(Rectangle(
            (slot.x, slot.y), slot.w, slot.h,
            linewidth=2, edgecolor=color, facecolor=color, alpha=0.2))
        axes[0].text(slot.x + slot.w // 2, slot.y + slot.h // 2,
                     slot.label, ha='center', va='center',
                     fontsize=8, fontweight='bold', color='white',
                     bbox=dict(boxstyle='round,pad=0.2', facecolor='green', alpha=0.7))
    axes[0].axis('off')

    # Test with classification results
    axes[1].imshow(aligned_color, cmap='gray' if aligned_color.ndim == 2 else None)
    axes[1].set_title('Detection Results', fontsize=14, fontweight='bold')
    for i, (slot, res) in enumerate(zip(slots, results)):
        if res.best_prediction == 1:
            color = '#FF0000'
            status = 'OCC'
        else:
            color = '#00FF00'
            status = 'FREE'

        axes[1].add_patch(Rectangle(
            (slot.x, slot.y), slot.w, slot.h,
            linewidth=2, edgecolor=color, facecolor=color, alpha=0.25))

        # Correctness indicator
        if res.ground_truth >= 0:
            correct = res.best_prediction == res.ground_truth
            indicator = "✓" if correct else "✗"
        else:
            indicator = ""

        axes[1].text(slot.x + slot.w // 2, slot.y + slot.h // 2,
                     f"{status}\n{res.best_confidence:.0%}\n{indicator}",
                     ha='center', va='center', fontsize=7, fontweight='bold',
                     color='white',
                     bbox=dict(boxstyle='round,pad=0.2',
                               facecolor='red' if res.best_prediction else 'green',
                               alpha=0.8))
    axes[1].axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'slot_detection_result.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    # ─── 2. ROI comparison grid ───
    n_slots = len(slots)
    if n_slots > 0:
        ref_gray = cv2.cvtColor(ref_color, cv2.COLOR_BGR2GRAY) if ref_color.ndim == 3 else ref_color
        test_gray = cv2.cvtColor(aligned_color, cv2.COLOR_BGR2GRAY) if aligned_color.ndim == 3 else aligned_color

        n_cols = min(n_slots, 8)
        n_rows_display = (n_slots + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows_display * 3, n_cols,
                                 figsize=(n_cols * 2.2, n_rows_display * 6))
        if n_rows_display * 3 == 1:
            axes = np.array([axes])
        if n_cols == 1:
            axes = axes.reshape(-1, 1)

        for i in range(n_slots):
            col = i % n_cols
            row_group = i // n_cols

            ref_roi = extract_roi_bilinear(ref_gray, slots[i])
            cur_roi = extract_roi_bilinear(test_gray, slots[i])
            diff_img = np.abs(ref_roi.astype(np.int16) - cur_roi.astype(np.int16)).astype(np.uint8)

            r0 = row_group * 3
            # Reference ROI
            axes[r0, col].imshow(ref_roi, cmap='gray', vmin=0, vmax=255)
            axes[r0, col].set_title(f'{slots[i].label} ref', fontsize=8)
            axes[r0, col].axis('off')

            # Current ROI
            axes[r0 + 1, col].imshow(cur_roi, cmap='gray', vmin=0, vmax=255)
            res = results[i]
            color = 'red' if res.best_prediction else 'green'
            axes[r0 + 1, col].set_title(
                f'{"OCC" if res.best_prediction else "FREE"} ({res.best_confidence:.0%})',
                fontsize=8, color=color, fontweight='bold')
            axes[r0 + 1, col].axis('off')

            # Diff heatmap
            axes[r0 + 2, col].imshow(diff_img, cmap='hot', vmin=0, vmax=80)
            mad = float(np.mean(diff_img))
            axes[r0 + 2, col].set_title(f'Δ={mad:.1f}', fontsize=8)
            axes[r0 + 2, col].axis('off')

        # Hide unused axes
        for i in range(n_slots, n_rows_display * n_cols):
            col = i % n_cols
            row_group = i // n_cols
            r0 = row_group * 3
            for dr in range(3):
                if r0 + dr < axes.shape[0] and col < axes.shape[1]:
                    axes[r0 + dr, col].axis('off')

        plt.suptitle('ROI Comparison: Reference | Current | Difference',
                     fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'roi_comparison_grid.png'),
                    dpi=150, bbox_inches='tight')
        plt.close()

    # ─── 3. Method comparison bar chart ───
    methods = ['mad', 'gaussian_mad', 'block_mad', 'edge_ratio',
               'histogram', 'variance_ratio', 'local_aligned_mad',
               'dual_align_mad', 'combined']
    if results:
        methods = [m for m in methods if m in results[0].methods]

    if any(r.ground_truth >= 0 for r in results):
        method_acc = {}
        for m in methods:
            correct = sum(1 for r in results
                          if r.ground_truth >= 0 and
                          r.methods[m].prediction == r.ground_truth)
            total = sum(1 for r in results if r.ground_truth >= 0)
            method_acc[m] = correct / max(total, 1)

        fig, ax = plt.subplots(figsize=(10, 5))
        colors = ['#2196F3', '#1565C0', '#4CAF50', '#FF9800', '#9C27B0',
                  '#F44336', '#FF5722', '#009688']
        bars = ax.bar(method_acc.keys(), method_acc.values(),
                      color=colors[:len(method_acc)])
        ax.set_ylabel('Accuracy', fontsize=12)
        ax.set_title('Method Comparison — Real Image Accuracy', fontsize=14,
                     fontweight='bold')
        ax.set_ylim(0, 1.05)
        ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)

        for bar, acc in zip(bars, method_acc.values()):
            ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.02,
                    f'{acc:.1%}', ha='center', va='bottom', fontweight='bold')

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'method_comparison.png'),
                    dpi=150, bbox_inches='tight')
        plt.close()

    # ─── 4. Difference heatmap (full image) ───
    ref_g = cv2.cvtColor(ref_color, cv2.COLOR_BGR2GRAY) if ref_color.ndim == 3 else ref_color
    test_g = cv2.cvtColor(aligned_color, cv2.COLOR_BGR2GRAY) if aligned_color.ndim == 3 else aligned_color
    diff_full = np.abs(ref_g.astype(np.int16) - test_g.astype(np.int16)).astype(np.uint8)

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    axes[0].imshow(ref_g, cmap='gray')
    axes[0].set_title('Reference (Empty)', fontsize=12)
    axes[0].axis('off')

    axes[1].imshow(test_g, cmap='gray')
    axes[1].set_title('Test (Aligned)', fontsize=12)
    axes[1].axis('off')

    axes[2].imshow(diff_full, cmap='hot', vmin=0, vmax=80)
    axes[2].set_title('Pixel Difference Heatmap', fontsize=12)
    axes[2].axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'diff_heatmap_full.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\n[VIS] Saved visualizations to {output_dir}/")


# ─── ESP32 Performance Simulation ──────────────────────────────────────
def esp32_benchmark(ref_gray: np.ndarray, test_gray: np.ndarray,
                    slots: List[ROIRect]) -> dict:
    """
    Simulate ESP32 processing constraints and measure performance.
    Uses integer-only operations where possible.
    """
    print("\n[ESP32 BENCHMARK] Simulating embedded constraints...")

    n_slots = len(slots)
    total_ops = 0
    total_bytes = 0

    # ROI extraction timing
    t0 = time.perf_counter()
    ref_rois = []
    cur_rois = []
    for s in slots:
        ref_rois.append(extract_roi_bilinear(ref_gray, s))
        cur_rois.append(extract_roi_bilinear(test_gray, s))
    t_extract = (time.perf_counter() - t0) * 1000

    # MAD classification timing (integer-only simulation)
    t0 = time.perf_counter()
    bitmap = 0
    for i in range(n_slots):
        ref_r = ref_rois[i]
        cur_r = cur_rois[i]

        # Integer MAD (same as ESP32 would compute)
        total_diff = 0
        for y in range(ROI_SIZE):
            for x in range(ROI_SIZE):
                d = int(cur_r[y, x]) - int(ref_r[y, x])
                total_diff += abs(d)
        mad_x10 = (total_diff * 10) // (ROI_SIZE * ROI_SIZE)
        total_ops += ROI_SIZE * ROI_SIZE  # 1024 ops per slot

        if mad_x10 > int(MAD_THRESH_DEFAULT * 10):
            bitmap |= (1 << i)

    t_classify = (time.perf_counter() - t0) * 1000

    # Memory estimation
    cal_mem = n_slots * ROI_SIZE * ROI_SIZE  # Reference frames
    runtime_mem = ROI_SIZE * ROI_SIZE  # 1 ROI buffer
    total_mem = cal_mem + runtime_mem

    stats = {
        'n_slots': n_slots,
        'roi_size': ROI_SIZE,
        'extraction_ms': round(t_extract, 2),
        'classification_ms': round(t_classify, 2),
        'total_ms': round(t_extract + t_classify, 2),
        'ops_per_frame': total_ops,
        'calibration_bytes': cal_mem,
        'runtime_ram_bytes': runtime_mem,
        'total_memory_bytes': total_mem,
        'bitmap_result': f'0x{bitmap:04X}',
        'bitmap_binary': f'{bitmap:0{n_slots}b}',
        'estimated_esp32_ms': round((t_extract + t_classify) * 0.3, 1),
    }

    print(f"  Slots: {n_slots}")
    print(f"  ROI extraction: {t_extract:.2f} ms")
    print(f"  Classification: {t_classify:.2f} ms")
    print(f"  Total: {t_extract + t_classify:.2f} ms")
    print(f"  Calibration memory: {cal_mem:,} bytes ({cal_mem/1024:.1f} KB)")
    print(f"  Runtime RAM: {runtime_mem:,} bytes")
    print(f"  Bitmap: {stats['bitmap_binary']} (0x{bitmap:04X})")

    return stats


# ─── Results Summary ───────────────────────────────────────────────────
def print_results_table(results: List[SlotResults]):
    """Print formatted results table."""
    methods = ['mad', 'gaussian_mad', 'block_mad', 'edge_ratio',
               'histogram', 'variance_ratio', 'local_aligned_mad',
               'dual_align_mad', 'combined']

    # Filter to only methods present in results
    if results:
        methods = [m for m in methods if m in results[0].methods]

    print("\n" + "=" * 150)
    print("CLASSIFICATION RESULTS")
    print("=" * 150)

    # Header
    header = f"{'Slot':<6} {'GT':<4}"
    for m in methods:
        header += f" {m:<14}"
    header += f" {'Correct':<8}"
    print(header)
    print("-" * 150)

    # Per-slot results
    method_correct = {m: 0 for m in methods}
    method_total = {m: 0 for m in methods}

    for r in results:
        gt_str = "OCC" if r.ground_truth == 1 else ("FREE" if r.ground_truth == 0 else "?")
        line = f"{r.label:<6} {gt_str:<4}"

        correct_count = 0
        for m in methods:
            mr = r.methods[m]
            pred_str = "OCC" if mr.prediction == 1 else "FREE"
            if r.ground_truth >= 0:
                ok = mr.prediction == r.ground_truth
                method_correct[m] += int(ok)
                method_total[m] += 1
                if ok:
                    correct_count += 1
                mark = "✓" if ok else "✗"
            else:
                mark = " "
            line += f" {pred_str}({mr.confidence:.0%}){mark:<3}"

        if r.ground_truth >= 0:
            line += f" {correct_count}/{len(methods)}"
        print(line)

    # Summary
    print("-" * 150)
    total_with_gt = sum(1 for r in results if r.ground_truth >= 0)
    if total_with_gt > 0:
        summary = f"{'ACC':<6} {'':4}"
        for m in methods:
            if method_total[m] > 0:
                acc = method_correct[m] / method_total[m]
                summary += f" {acc:<14.1%}"
            else:
                summary += f" {'N/A':<14}"
        print(summary)
    print("=" * 150)


def save_results_json(results: List[SlotResults], slots: List[ROIRect],
                      stats: dict, thresholds: dict,
                      output_dir: str):
    """Save all results to JSON for paper/analysis."""
    output = {
        'metadata': {
            'version': 'v4.0',
            'roi_size': ROI_SIZE,
            'n_slots': len(slots),
            'methods': ['mad', 'gaussian_mad', 'block_mad', 'edge_ratio',
                        'histogram', 'variance_ratio', 'local_aligned_mad',
                        'dual_align_mad', 'combined'],
        },
        'slots': [],
        'optimized_thresholds': thresholds,
        'esp32_benchmark': stats,
        'accuracy_summary': {},
    }

    methods = output['metadata']['methods']

    for r in results:
        slot_data = {
            'idx': r.slot_idx,
            'label': r.label,
            'ground_truth': r.ground_truth,
            'best_prediction': r.best_prediction,
            'best_confidence': round(r.best_confidence, 4),
        }
        for m in methods:
            mr = r.methods[m]
            slot_data[m] = {
                'prediction': mr.prediction,
                'confidence': round(mr.confidence, 4),
                'raw_metric': round(mr.raw_metric, 4),
            }
        output['slots'].append(slot_data)

    # Accuracy summary
    for m in methods:
        correct = sum(1 for r in results
                      if r.ground_truth >= 0 and
                      r.methods[m].prediction == r.ground_truth)
        total = sum(1 for r in results if r.ground_truth >= 0)
        output['accuracy_summary'][m] = {
            'correct': correct,
            'total': total,
            'accuracy': round(correct / max(total, 1), 4),
        }

    filepath = os.path.join(output_dir, 'results_v4.json')
    with open(filepath, 'w') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n[SAVE] Results saved to {filepath}")


# ─── Main ──────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description='ParkingLite ROI Parking Detector v4.0')
    parser.add_argument('--empty', '-e', required=True,
                        help='Path to reference (empty) parking lot image')
    parser.add_argument('--test', '-t', required=True,
                        help='Path to test (with cars) parking lot image')
    parser.add_argument('--output', '-o', default='../output/roi_v4',
                        help='Output directory for results')
    parser.add_argument('--ground-truth', '-g', default=None,
                        help='Comma-separated ground truth: 1=occupied, 0=empty '
                             '(top row L→R, then bottom row L→R)')
    parser.add_argument('--no-optimize', action='store_true',
                        help='Skip threshold optimization')
    parser.add_argument('--no-vis', action='store_true',
                        help='Skip visualization generation')
    args = parser.parse_args()

    output_dir = args.output
    os.makedirs(output_dir, exist_ok=True)

    # ─── Load images ───
    print("=" * 60)
    print("ParkingLite ROI Parking Detector v4.0")
    print("=" * 60)

    ref_bgr = cv2.imread(args.empty, cv2.IMREAD_COLOR)
    test_bgr = cv2.imread(args.test, cv2.IMREAD_COLOR)

    if ref_bgr is None:
        print(f"[ERROR] Cannot load reference image: {args.empty}")
        sys.exit(1)
    if test_bgr is None:
        print(f"[ERROR] Cannot load test image: {args.test}")
        sys.exit(1)

    print(f"\n[LOAD] Reference: {ref_bgr.shape[1]}x{ref_bgr.shape[0]}")
    print(f"[LOAD] Test:      {test_bgr.shape[1]}x{test_bgr.shape[0]}")

    ref_gray = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2GRAY)
    test_gray = cv2.cvtColor(test_bgr, cv2.COLOR_BGR2GRAY)

    # ─── Align images ───
    print("\n[ALIGN] Aligning test image to reference coordinate system...")
    aligned_gray, H, n_inliers = align_images(ref_gray, test_gray)

    # Also align color image for visualization
    h_ref, w_ref = ref_gray.shape[:2]
    if H is not None and not np.allclose(H, np.eye(3)):
        aligned_bgr = cv2.warpPerspective(test_bgr, H, (w_ref, h_ref),
                                           flags=cv2.INTER_LINEAR,
                                           borderMode=cv2.BORDER_REPLICATE)
    else:
        aligned_bgr = cv2.resize(test_bgr, (w_ref, h_ref))

    # ─── Also create simple-resize alignment (no homography) ───
    # This is more robust at image edges where homography distorts
    resize_gray = cv2.resize(test_gray, (w_ref, h_ref), interpolation=cv2.INTER_LINEAR)
    resize_bgr = cv2.resize(test_bgr, (w_ref, h_ref), interpolation=cv2.INTER_LINEAR)

    # ─── Detect parking slots ───
    slots = auto_detect_slots(ref_gray)

    if not slots:
        print("[ERROR] No parking slots detected!")
        sys.exit(1)

    # ─── Parse ground truth ───
    ground_truth = []
    if args.ground_truth:
        ground_truth = [int(x.strip()) for x in args.ground_truth.split(',')]
        # Pad if needed
        while len(ground_truth) < len(slots):
            ground_truth.append(-1)
        print(f"\n[GT] Ground truth: {ground_truth[:len(slots)]}")
    else:
        ground_truth = [-1] * len(slots)

    # ─── Threshold optimization ───
    thresholds = {}
    if not args.no_optimize and any(gt >= 0 for gt in ground_truth):
        thresholds = optimize_thresholds(ref_gray, aligned_gray, slots, ground_truth)

    # ─── Classify all slots ───
    print("\n[CLASSIFY] Running all methods on all slots...")
    results = classify_all_slots(ref_gray, aligned_gray, slots,
                                 ground_truth, thresholds,
                                 resize_gray=resize_gray)

    # ─── Print results ───
    print_results_table(results)

    # ─── ESP32 benchmark ───
    stats = esp32_benchmark(ref_gray, aligned_gray, slots)

    # ─── Visualization ───
    if not args.no_vis:
        print("\n[VIS] Generating visualizations...")
        visualize_results(ref_bgr, test_bgr, aligned_bgr,
                          slots, results, output_dir)

    # ─── Save JSON ───
    save_results_json(results, slots, stats, thresholds, output_dir)

    # ─── Final summary ───
    total_with_gt = sum(1 for r in results if r.ground_truth >= 0)
    if total_with_gt > 0:
        all_m = [m for m in ['mad', 'gaussian_mad', 'block_mad', 'edge_ratio',
                 'histogram', 'variance_ratio', 'local_aligned_mad',
                 'dual_align_mad', 'combined']
                 if results and m in results[0].methods]
        best_method = max(
            all_m,
            key=lambda m: sum(1 for r in results
                              if r.ground_truth >= 0 and
                              r.methods[m].prediction == r.ground_truth)
        )
        best_acc = sum(1 for r in results
                       if r.ground_truth >= 0 and
                       r.methods[best_method].prediction == r.ground_truth) / total_with_gt

        print(f"\n{'=' * 60}")
        print(f"BEST METHOD: {best_method} — Accuracy: {best_acc:.1%}")
        print(f"{'=' * 60}")

    return results


if __name__ == '__main__':
    main()

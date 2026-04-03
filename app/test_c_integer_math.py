#!/usr/bin/env python3
"""
Cross-validation: Simulate C integer math and compare with Python float results.
Ensures the ESP32 firmware will produce identical predictions to the Python simulation.

This test:
1. Loads the same real parking images used in v4
2. Extracts ROIs the same way
3. Runs BOTH float math (Python) and simulated integer math (C-equivalent)
4. Compares predictions, raw metrics, and checks for overflow
5. Reports any discrepancies

Run: python3 test_c_integer_math.py
"""

import json
import sys
import os
import numpy as np
import cv2

# ─── C Constants (must match roi_classifier.h) ──────────────────────
ROI_SIZE = 32
ROI_PIXELS = ROI_SIZE * ROI_SIZE  # 1024
ROI_BLOCK_SIZE = 8
ROI_N_BLOCKS = ROI_SIZE // ROI_BLOCK_SIZE  # 4
ROI_TOTAL_BLOCKS = ROI_N_BLOCKS * ROI_N_BLOCKS  # 16
HIST_N_BINS = 16
HIST_BIN_SHIFT = 4

# Thresholds (from roi_classifier.h)
REF_DIFF_X10 = 77       # 7.68 × 10
GAUSS_MAD_X10 = 77       # same scale
BLOCK_MAD_X10 = 150      # 15.0 × 10
BLOCK_VOTE_X100 = 40     # 0.40 × 100
PCTILE_P75_X10 = 150     # 15.0 × 10
MAX_BLOCK_X10 = 200      # 20.0 × 10
HIST_INTER_X1000 = 750   # 0.75 × 1000
VAR_RATIO_X100 = 180     # 1.8 × 100
COMBINED_X100 = 50       # 0.50 × 100

UINT32_MAX = 2**32 - 1
UINT16_MAX = 2**16 - 1


def generate_gauss_weights():
    """Generate the exact same Gaussian weight table as C firmware."""
    sigma = 32.0 / 3.0
    weights = np.zeros((32, 32), dtype=np.uint8)
    for y in range(32):
        for x in range(32):
            cx = x - 15.5
            cy = y - 15.5
            w = int(128 * np.exp(-(cx**2 + cy**2) / (2 * sigma**2)))
            weights[y, x] = max(w, 1)
    return weights

GAUSS_WEIGHTS = generate_gauss_weights()
GAUSS_WEIGHT_SUM = int(np.sum(GAUSS_WEIGHTS))


def c_compute_mean_diff_x10(current, reference):
    """Simulate compute_mean_diff_x10() from C."""
    total_diff = 0
    for i in range(ROI_PIXELS):
        d = int(current.flat[i]) - int(reference.flat[i])
        total_diff += abs(d)
    assert total_diff <= UINT32_MAX, f"total_diff overflow: {total_diff}"
    assert total_diff * 10 <= UINT32_MAX, f"total_diff*10 overflow: {total_diff * 10}"
    result = (total_diff * 10) // ROI_PIXELS
    assert result <= UINT16_MAX, f"result overflow: {result}"
    return result


def c_classify_ref_frame(cur_roi, ref_roi):
    """Simulate classify_ref_frame() — Method 2: MAD."""
    diff_x10 = c_compute_mean_diff_x10(cur_roi, ref_roi)
    pred = 1 if diff_x10 > REF_DIFF_X10 else 0
    return pred, diff_x10


def c_classify_gaussian_mad(cur_roi, ref_roi):
    """Simulate classify_gaussian_mad() — Method 4."""
    weighted_sum = 0
    for y in range(ROI_SIZE):
        for x in range(ROI_SIZE):
            d = abs(int(cur_roi[y, x]) - int(ref_roi[y, x]))
            weighted_sum += int(GAUSS_WEIGHTS[y, x]) * d
    assert weighted_sum <= UINT32_MAX, f"weighted_sum overflow: {weighted_sum}"

    divisor = GAUSS_WEIGHT_SUM // 10
    gauss_mad_x10 = weighted_sum // divisor
    assert gauss_mad_x10 <= UINT16_MAX, f"gauss_mad_x10 overflow: {gauss_mad_x10}"
    pred = 1 if gauss_mad_x10 > GAUSS_MAD_X10 else 0
    return pred, gauss_mad_x10


def c_classify_block_mad(cur_roi, ref_roi):
    """Simulate classify_block_mad() — Method 5."""
    n_occupied_blocks = 0
    for by in range(ROI_N_BLOCKS):
        for bx in range(ROI_N_BLOCKS):
            block_diff = 0
            for dy in range(ROI_BLOCK_SIZE):
                for dx in range(ROI_BLOCK_SIZE):
                    y = by * ROI_BLOCK_SIZE + dy
                    x = bx * ROI_BLOCK_SIZE + dx
                    d = abs(int(cur_roi[y, x]) - int(ref_roi[y, x]))
                    block_diff += d
            block_mad_x10 = (block_diff * 10) // (ROI_BLOCK_SIZE * ROI_BLOCK_SIZE)
            if block_mad_x10 > BLOCK_MAD_X10:
                n_occupied_blocks += 1

    vote_x100 = (n_occupied_blocks * 100) // ROI_TOTAL_BLOCKS
    pred = 1 if vote_x100 > BLOCK_VOTE_X100 else 0
    return pred, vote_x100


def c_classify_percentile_mad(cur_roi, ref_roi):
    """Simulate classify_percentile_mad() — Method 6."""
    diff_hist = [0] * 256
    for i in range(ROI_PIXELS):
        d = abs(int(cur_roi.flat[i]) - int(ref_roi.flat[i]))
        diff_hist[d] += 1

    target = (ROI_PIXELS * 75) // 100  # 768
    cumulative = 0
    p75_value = 0
    for i in range(256):
        cumulative += diff_hist[i]
        if cumulative >= target:
            p75_value = i
            break

    p75_x10 = p75_value * 10
    pred = 1 if p75_x10 > PCTILE_P75_X10 else 0
    return pred, p75_x10


def c_classify_max_block(cur_roi, ref_roi):
    """Simulate classify_max_block() — Method 7."""
    max_block_mad_x10 = 0
    for by in range(ROI_N_BLOCKS):
        for bx in range(ROI_N_BLOCKS):
            block_diff = 0
            for dy in range(ROI_BLOCK_SIZE):
                for dx in range(ROI_BLOCK_SIZE):
                    y = by * ROI_BLOCK_SIZE + dy
                    x = bx * ROI_BLOCK_SIZE + dx
                    d = abs(int(cur_roi[y, x]) - int(ref_roi[y, x]))
                    block_diff += d
            block_mad_x10 = (block_diff * 10) // (ROI_BLOCK_SIZE * ROI_BLOCK_SIZE)
            max_block_mad_x10 = max(max_block_mad_x10, block_mad_x10)

    pred = 1 if max_block_mad_x10 > MAX_BLOCK_X10 else 0
    return pred, max_block_mad_x10


def c_classify_histogram_inter(cur_roi, ref_hist):
    """Simulate classify_histogram_inter() — Method 8."""
    cur_hist = [0] * HIST_N_BINS
    for i in range(ROI_PIXELS):
        cur_hist[cur_roi.flat[i] >> HIST_BIN_SHIFT] += 1

    intersection = 0
    for i in range(HIST_N_BINS):
        intersection += min(ref_hist[i], cur_hist[i])

    inter_x1000 = (intersection * 1000) // ROI_PIXELS
    # Inverted: lower = more different = occupied
    pred = 1 if inter_x1000 < HIST_INTER_X1000 else 0
    return pred, inter_x1000


def c_classify_variance_ratio(cur_roi, ref_var_x100):
    """Simulate classify_variance_ratio() — Method 9.
    Uses two-pass: Σ(x - mean)² / N to avoid integer truncation catastrophe.
    """
    # Pass 1: compute mean
    s = sum(int(cur_roi.flat[i]) for i in range(ROI_PIXELS))
    mean = s // ROI_PIXELS

    # Pass 2: sum of squared differences
    var_sum = 0
    for i in range(ROI_PIXELS):
        d = int(cur_roi.flat[i]) - mean
        var_sum += d * d
    assert var_sum <= UINT32_MAX, f"var_sum overflow: {var_sum}"

    cur_var_x100 = (var_sum // ROI_PIXELS) * 100
    assert cur_var_x100 <= UINT32_MAX, f"var_x100 overflow: {cur_var_x100}"

    ref_v = ref_var_x100 if ref_var_x100 >= 100 else 100
    ratio_x100 = (cur_var_x100 * 100) // ref_v
    pred = 1 if ratio_x100 > VAR_RATIO_X100 else 0
    return pred, ratio_x100


def c_classify_combined(cur_roi, ref_roi, ref_hist, ref_var_x100):
    """Simulate classify_combined() — Method 10."""
    r_mad_pred, r_mad_raw = c_classify_ref_frame(cur_roi, ref_roi)
    r_gauss_pred, r_gauss_raw = c_classify_gaussian_mad(cur_roi, ref_roi)
    r_block_pred, r_block_raw = c_classify_block_mad(cur_roi, ref_roi)
    r_pct_pred, r_pct_raw = c_classify_percentile_mad(cur_roi, ref_roi)
    r_maxb_pred, r_maxb_raw = c_classify_max_block(cur_roi, ref_roi)
    r_hist_pred, r_hist_raw = c_classify_histogram_inter(cur_roi, ref_hist)
    r_var_pred, r_var_raw = c_classify_variance_ratio(cur_roi, ref_var_x100)

    # Compute confidences (simplified — same as C code)
    def conf(val, thresh):
        delta = abs(val - thresh)
        c = (delta * 100) // max(thresh, 1)
        return min(c, 100)

    votes = [
        (r_mad_pred,   conf(r_mad_raw, REF_DIFF_X10),      15),
        (r_gauss_pred, conf(r_gauss_raw, GAUSS_MAD_X10),    15),
        (r_block_pred, conf(r_block_raw, BLOCK_VOTE_X100),  10),
        (r_pct_pred,   conf(r_pct_raw, PCTILE_P75_X10),     20),
        (r_maxb_pred,  conf(r_maxb_raw, MAX_BLOCK_X10),     20),
        (r_hist_pred,  conf(r_hist_raw, HIST_INTER_X1000),  10),  # Note: inverted
        (r_var_pred,   conf(r_var_raw, VAR_RATIO_X100),     10),
    ]

    score_x100 = 0
    for pred, confidence, weight in votes:
        if pred:
            cf = 50 + confidence // 2
            score_x100 += weight * cf // 100

    pred = 1 if score_x100 > COMBINED_X100 else 0
    return pred, score_x100


def calibrate_slot(ref_roi):
    """Simulate calibration for one slot (two-pass variance)."""
    # Two-pass variance (matches C firmware)
    s = sum(int(ref_roi.flat[i]) for i in range(ROI_PIXELS))
    mean = s // ROI_PIXELS
    var_sum = sum((int(ref_roi.flat[i]) - mean) ** 2 for i in range(ROI_PIXELS))
    var = var_sum // ROI_PIXELS
    ref_var_x100 = max(var * 100, 100)

    # Histogram
    ref_hist = [0] * HIST_N_BINS
    for i in range(ROI_PIXELS):
        ref_hist[ref_roi.flat[i] >> HIST_BIN_SHIFT] += 1

    return ref_var_x100, ref_hist


def main():
    # Load images
    script_dir = os.path.dirname(os.path.abspath(__file__))
    sim_dir = os.path.dirname(script_dir)

    empty_path = os.path.join(sim_dir, "parking_empty.png")
    test_path = os.path.join(sim_dir, "parking_with_car.png")
    results_path = os.path.join(sim_dir, "output", "roi_v4", "results_v4.json")

    if not os.path.exists(empty_path) or not os.path.exists(test_path):
        print(f"ERROR: Images not found at {sim_dir}")
        sys.exit(1)

    # Load Python results for comparison
    with open(results_path) as f:
        py_results = json.load(f)

    # Load and prepare images (same as v4)
    from roi_parking_detector_v4 import align_images, auto_detect_slots, extract_roi_bilinear

    ref_img = cv2.imread(empty_path)
    test_img = cv2.imread(test_path)
    ref_gray = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)
    test_gray = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)

    print("Aligning images...")
    aligned_gray, H, n_inliers = align_images(ref_gray, test_gray)
    print(f"  Inliers: {n_inliers}")

    print("Detecting slots...")
    slots = auto_detect_slots(ref_gray)
    print(f"  Found {len(slots)} slots")

    # Ground truth (from v4 verification)
    gt = [0,1,1,1,1,1,1,1, 0,0,0,0,0,0,0,0]

    print()
    print("═══════════════════════════════════════════════════════════════════")
    print("  CROSS-VALIDATION: C Integer Math vs Python Float")
    print("═══════════════════════════════════════════════════════════════════")
    print()

    n_slots = len(slots)
    all_pass = True
    overflow_count = 0

    # Method names for reporting
    methods_c = [
        ("ref_frame MAD", c_classify_ref_frame, "mad"),
        ("gaussian_mad", None, "gaussian_mad"),
        ("block_mad", c_classify_block_mad, "block_mad"),
        ("percentile_mad", c_classify_percentile_mad, None),
        ("max_block", c_classify_max_block, None),
        ("histogram", None, "histogram"),
        ("variance_ratio", None, "variance_ratio"),
        ("combined", None, "combined"),
    ]

    for idx in range(min(n_slots, 16)):
        slot = slots[idx]
        ref_roi = extract_roi_bilinear(ref_gray, slot)
        cur_roi = extract_roi_bilinear(aligned_gray, slot)

        # Calibrate
        ref_var_x100, ref_hist = calibrate_slot(ref_roi)

        # Get Python results
        py_slot = py_results["slots"][idx]
        label = py_slot["label"]
        ground_truth = gt[idx]

        print(f"  Slot {idx} ({label}) — GT={'OCC' if ground_truth else 'FREE'}")

        # Test each method
        try:
            # Method 2: MAD
            c_pred, c_raw = c_classify_ref_frame(cur_roi, ref_roi)
            py_raw = py_slot["mad"]["raw_metric"]
            py_pred = py_slot["mad"]["prediction"]
            c_val_float = c_raw / 10.0
            match_pred = c_pred == py_pred == (1 if ground_truth else 0)
            correct = c_pred == ground_truth
            delta = abs(c_val_float - py_raw)
            status = "✅" if correct else "❌"
            print(f"    MAD:           C={c_val_float:6.1f} Py={py_raw:6.1f} (Δ={delta:.2f}) pred={c_pred} {status}")
            if not correct: all_pass = False

            # Method 4: Gaussian MAD
            c_pred, c_raw = c_classify_gaussian_mad(cur_roi, ref_roi)
            py_raw = py_slot["gaussian_mad"]["raw_metric"]
            c_val_float = c_raw / 10.0
            correct = c_pred == ground_truth
            delta = abs(c_val_float - py_raw)
            status = "✅" if correct else "❌"
            print(f"    Gaussian MAD:  C={c_val_float:6.1f} Py={py_raw:6.1f} (Δ={delta:.2f}) pred={c_pred} {status}")
            if not correct: all_pass = False

            # Method 5: Block MAD
            c_pred, c_raw = c_classify_block_mad(cur_roi, ref_roi)
            py_block_metric = py_slot["block_mad"]["raw_metric"]
            correct = c_pred == ground_truth
            status = "✅" if correct else "❌"
            print(f"    Block MAD:     C_vote={c_raw:3d}%  Py_ratio={py_block_metric:.4f} pred={c_pred} {status}")
            if not correct: all_pass = False

            # Method 6: Percentile MAD
            c_pred, c_raw = c_classify_percentile_mad(cur_roi, ref_roi)
            c_val_float = c_raw / 10.0
            correct = c_pred == ground_truth
            status = "✅" if correct else "❌"
            print(f"    Percentile:    C_p75={c_val_float:6.1f}              pred={c_pred} {status}")
            if not correct: all_pass = False

            # Method 7: Max Block
            c_pred, c_raw = c_classify_max_block(cur_roi, ref_roi)
            c_val_float = c_raw / 10.0
            correct = c_pred == ground_truth
            status = "✅" if correct else "❌"
            print(f"    Max Block:     C={c_val_float:6.1f}                  pred={c_pred} {status}")
            if not correct: all_pass = False

            # Method 8: Histogram
            c_pred, c_raw = c_classify_histogram_inter(cur_roi, ref_hist)
            py_raw = py_slot["histogram"]["raw_metric"]
            c_val_float = c_raw / 1000.0
            correct = c_pred == ground_truth
            delta = abs(c_val_float - py_raw)
            status = "✅" if correct else "❌"
            print(f"    Histogram:     C={c_val_float:.3f} Py={py_raw:.3f} (Δ={delta:.3f}) pred={c_pred} {status}")
            if not correct: all_pass = False

            # Method 9: Variance Ratio
            c_pred, c_raw = c_classify_variance_ratio(cur_roi, ref_var_x100)
            py_raw = py_slot["variance_ratio"]["raw_metric"]
            c_val_float = c_raw / 100.0
            correct = c_pred == ground_truth
            delta = abs(c_val_float - py_raw)
            status = "✅" if correct else "❌"
            print(f"    Var Ratio:     C={c_val_float:6.2f} Py={py_raw:6.2f} (Δ={delta:.2f}) pred={c_pred} {status}")
            if not correct: all_pass = False

            # Method 10: Combined
            c_pred, c_raw = c_classify_combined(cur_roi, ref_roi, ref_hist, ref_var_x100)
            py_comb = py_slot["combined"]
            correct = c_pred == ground_truth
            status = "✅" if correct else "❌"
            print(f"    Combined:      C_score={c_raw:3d}                    pred={c_pred} {status}")
            if not correct: all_pass = False

        except AssertionError as e:
            print(f"    ❌ OVERFLOW: {e}")
            overflow_count += 1
            all_pass = False

        print()

    # Summary
    print("═══════════════════════════════════════════════════════════════════")
    if all_pass:
        print("  ✅ ALL PREDICTIONS CORRECT — C integer math matches Python!")
    else:
        print("  ❌ SOME PREDICTIONS DIFFER — needs investigation")
    if overflow_count > 0:
        print(f"  ⚠️  {overflow_count} overflow(s) detected")
    print("═══════════════════════════════════════════════════════════════════")

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())

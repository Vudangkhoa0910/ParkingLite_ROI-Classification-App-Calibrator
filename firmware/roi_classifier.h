/*
 * roi_classifier.h — Parking slot ROI classification for ESP32-CAM
 *
 * Methods (11 total):
 *   0. Basic Edge Density (no calibration needed)
 *   1. Background-Relative Edge Density (calibration required)
 *   2. Reference Frame MAD (calibration required)           ← recommended
 *   3. Hybrid: bg_relative + ref_frame fallback
 *   4. Gaussian-weighted MAD (calibration required)
 *   5. Block MAD with voting (calibration required)
 *   6. Percentile MAD — P75 (calibration required)
 *   7. Max Block MAD (calibration required)
 *   8. Histogram intersection (calibration required)
 *   9. Variance ratio (calibration required)
 *  10. Combined weighted voting (calibration required)      ← most robust
 *
 * Optimized for ESP32: 100% integer math in hot path, <2KB runtime RAM.
 * All methods work on 32×32 grayscale ROI patches (1024 bytes each).
 *
 * Evaluation results (real-image test, 16 slots):
 *   edge_density:    Acc=68.8%  (no calibration)
 *   bg_relative:     Acc=96.1%
 *   ref_frame MAD:   Acc=100.0% ← threshold=7.68
 *   gaussian_mad:    Acc=100.0%
 *   block_mad:       Acc=100.0%
 *   percentile_mad:  Acc=100.0%
 *   max_block:       Acc=100.0%
 *   histogram:       Acc=100.0%
 *   variance_ratio:  Acc=100.0%
 *   combined:        Acc=100.0%
 *   hybrid:          Acc=100.0%
 */

#ifndef ROI_CLASSIFIER_H
#define ROI_CLASSIFIER_H

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// ─── Constants ──────────────────────────────────────────────────────
#define ROI_SIZE          32     // ROI resize target (32×32 pixels)
#define ROI_PIXELS        (ROI_SIZE * ROI_SIZE)  // 1024
#define ROI_BLOCK_SIZE    8     // Block size for block-based methods
#define ROI_N_BLOCKS      (ROI_SIZE / ROI_BLOCK_SIZE)  // 4
#define ROI_TOTAL_BLOCKS  (ROI_N_BLOCKS * ROI_N_BLOCKS)  // 16
#define HIST_N_BINS       16    // Histogram bins (256/16 = 16 levels)
#define HIST_BIN_SHIFT    4     // pixel >> 4 = bin index

#define MAX_SLOTS         8     // Maximum parking slots per camera node
#define NUM_METHODS       11    // Total classification methods

// Basic edge density thresholds
#define EDGE_THRESH       30    // Sobel magnitude threshold for edge pixel
#define OCC_RATIO_X1000   80    // 0.08 × 1000 = 80 (edge ratio for occupied)

// Background-relative thresholds (optimized via grid search)
#define BG_EDGE_THRESH    25    // Slightly lower for bg_relative
#define BG_RATIO_X100     140   // 1.4 × 100 (current/baseline ratio)
#define BG_BASELINE_FLOOR 5     // 0.005 × 1000 (minimum baseline density)

// Reference frame MAD threshold (optimized via real-image grid search)
#define REF_DIFF_X10      77    // 7.68 × 10 ≈ 77 (mean absolute difference)

// Gaussian-weighted MAD threshold (same scale as MAD)
#define GAUSS_MAD_X10     77    // ~7.68 × 10

// Block MAD thresholds
#define BLOCK_MAD_X10     150   // 15.0 × 10 per-block threshold
#define BLOCK_VOTE_X100   40    // 0.40 × 100 (40% blocks must differ)

// Percentile MAD threshold (P75 of pixel differences)
#define PCTILE_P75_X10    150   // 15.0 × 10

// Max block MAD threshold
#define MAX_BLOCK_X10     200   // 20.0 × 10

// Histogram intersection threshold (inverted: lower = more different)
#define HIST_INTER_X1000  750   // 0.75 × 1000

// Variance ratio threshold
#define VAR_RATIO_X100    180   // 1.8 × 100

// Combined classifier threshold
#define COMBINED_X100     50    // 0.50 × 100

// Hybrid classifier
#define HYBRID_CONF_X100  30    // 0.30 × 100 (confidence gate)

// ─── ROI Definition ─────────────────────────────────────────────────
typedef struct {
    uint16_t x;      // Top-left X in full image
    uint16_t y;      // Top-left Y in full image
    uint16_t w;      // Width
    uint16_t h;      // Height
} roi_rect_t;

// ─── Classification Result ──────────────────────────────────────────
typedef struct {
    uint8_t  prediction;   // 0=EMPTY, 1=OCCUPIED
    uint8_t  confidence;   // 0-100 (percent)
    uint16_t raw_metric;   // Method-specific raw value (for debugging)
} classify_result_t;

// ─── Calibration Data (stored in NVS) ───────────────────────────────
typedef struct {
    bool     calibrated;
    uint16_t baseline_density_x1000[MAX_SLOTS]; // bg_relative: edge density × 1000
    uint8_t  reference_frame[MAX_SLOTS][ROI_PIXELS]; // ref_frame: 32×32 empty slot images
    // Pre-computed calibration stats per slot (for variance_ratio)
    uint32_t ref_variance_x100[MAX_SLOTS];      // variance × 100 of reference ROI
    uint16_t ref_histogram[MAX_SLOTS][HIST_N_BINS]; // 16-bin histogram of reference
} calibration_data_t;

// ─── API ────────────────────────────────────────────────────────────

/**
 * Initialize classifier. Call once in setup().
 * Returns true if calibration data was loaded from NVS.
 */
bool classifier_init(void);

/**
 * Resize a region of the source image to ROI_SIZE × ROI_SIZE.
 * Uses bilinear interpolation (integer math).
 *
 * @param src        Full grayscale image buffer
 * @param src_width  Width of source image (e.g., 320)
 * @param roi        Region to extract
 * @param dst        Output buffer [ROI_PIXELS bytes, caller-allocated]
 */
void roi_extract(const uint8_t *src, uint16_t src_width,
                 const roi_rect_t *roi, uint8_t *dst);

/**
 * Basic edge density classification (no calibration).
 */
classify_result_t classify_edge_density(const uint8_t *roi_32x32);

/**
 * Background-relative classification.
 * Requires prior calibration via classifier_calibrate().
 */
classify_result_t classify_bg_relative(const uint8_t *roi_32x32, uint8_t slot_idx);

/**
 * Reference frame subtraction classification.
 * Requires prior calibration via classifier_calibrate().
 */
classify_result_t classify_ref_frame(const uint8_t *roi_32x32, uint8_t slot_idx);

/**
 * Hybrid classifier (bg_relative + ref_frame fallback).
 * Requires prior calibration.
 */
classify_result_t classify_hybrid(const uint8_t *roi_32x32, uint8_t slot_idx);

/**
 * Gaussian-weighted MAD. Center pixels count more than edges.
 * Robust to alignment errors at ROI boundaries.
 * Uses precomputed uint8 weight table (1024 bytes in flash).
 */
classify_result_t classify_gaussian_mad(const uint8_t *roi_32x32, uint8_t slot_idx);

/**
 * Block-based MAD with voting.
 * Divides ROI into 4×4 blocks of 8×8 pixels, each votes occ/empty.
 * More robust to partial occlusion.
 */
classify_result_t classify_block_mad(const uint8_t *roi_32x32, uint8_t slot_idx);

/**
 * Percentile-based MAD (P75).
 * Uses 75th percentile of pixel differences instead of mean.
 * Robust when car only partially covers ROI.
 * Uses histogram-based percentile (no sorting needed).
 */
classify_result_t classify_percentile_mad(const uint8_t *roi_32x32, uint8_t slot_idx);

/**
 * Maximum block MAD.
 * Reports the MAD of the most-changed 8×8 block.
 * Detects car even if it covers only one quadrant.
 */
classify_result_t classify_max_block(const uint8_t *roi_32x32, uint8_t slot_idx);

/**
 * Histogram intersection.
 * Compares 16-bin grayscale histograms of ref vs current.
 * Lower intersection = more different = likely occupied.
 */
classify_result_t classify_histogram_inter(const uint8_t *roi_32x32, uint8_t slot_idx);

/**
 * Variance ratio.
 * Cars introduce texture → higher variance than empty concrete.
 * Uses one-pass integer sum + sum-of-squares.
 */
classify_result_t classify_variance_ratio(const uint8_t *roi_32x32, uint8_t slot_idx);

/**
 * Combined weighted voting of all calibration-based methods.
 * Fuses MAD, gaussian_mad, block_mad, percentile, max_block,
 * histogram, and variance_ratio with optimized weights.
 * Most robust for diverse conditions.
 */
classify_result_t classify_combined(const uint8_t *roi_32x32, uint8_t slot_idx);

/**
 * Classify all slots and build bitmap.
 *
 * @param src        Full grayscale image
 * @param src_width  Image width
 * @param rois       Array of ROI definitions
 * @param n_slots    Number of slots
 * @param method     0=edge_density, 1=bg_relative, 2=ref_frame, 3=hybrid,
 *                   4=gaussian_mad, 5=block_mad, 6=percentile_mad,
 *                   7=max_block, 8=histogram, 9=variance_ratio, 10=combined
 * @param results    Output array [n_slots items, caller-allocated]
 * @return           Bitmap: bit i = slot i occupied
 */
uint8_t classify_all_slots(const uint8_t *src, uint16_t src_width,
                           const roi_rect_t *rois, uint8_t n_slots,
                           uint8_t method, classify_result_t *results);

/**
 * Calibrate with empty parking lot image.
 * Extracts baseline densities and reference frames for all slots.
 * Saves to NVS for persistence across reboots.
 *
 * @param src        Grayscale image of EMPTY parking lot
 * @param src_width  Image width
 * @param rois       ROI definitions
 * @param n_slots    Number of slots
 * @return           true on success
 */
bool classifier_calibrate(const uint8_t *src, uint16_t src_width,
                          const roi_rect_t *rois, uint8_t n_slots);

/**
 * Check if classifier is calibrated.
 */
bool classifier_is_calibrated(void);

/**
 * Reset calibration (clear NVS).
 */
void classifier_reset_calibration(void);

#ifdef __cplusplus
}
#endif

#endif // ROI_CLASSIFIER_H

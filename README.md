# ParkingLite ROI Classification & Calibrator

> Smart Parking system using image-based ROI classification on ESP32-CAM microcontroller.

**Research project** — NCKH 2025-2026, Phenikaa University, Vietnam.

---

## Overview

ParkingLite is a **low-cost smart parking solution** (~200,000 VND/slot) that monitors 8 parking slots using a single ESP32-CAM. Instead of deep learning (GPU required), it uses **pixel comparison (MAD algorithm)** running entirely on integer math — achieving **100% accuracy** in 3.2ms.

### Two Research Contributions

| | Contribution A | Contribution B |
|---|---|---|
| **Name** | ROI Classification on MCU | Confidence-Aware Adaptive Protocol |
| **Problem** | Detect vehicles without GPU | When to send, how much data |
| **Method** | Compare pixels (MAD) with reference image | Classifier confidence controls protocol |
| **Result** | **100% accuracy**, 3.2ms, integer math | **99.9% bandwidth saving** vs MQTT |

### System Architecture

```
ESP32-CAM                    Gateway              Cloud/App
+--------------+  ESP-NOW   +----------+  MQTT  +----------+
| Camera OV2640|  5-13 B    | ESP32/RPi|        | Dashboard|
| ROI Classify |----------->| Aggregate|------->| Analytics|
| Adaptive TX  |  per event | Offline  |        |          |
| 100% integer |            |          |        |          |
+--------------+            +----------+        +----------+
  ~61ms active               Offline-first        Optional
```

---

## Repository Structure

```
ROI/
+-- app/                          # Main applications
|   +-- roi_calibration_tool.py   # GUI Calibrator (tkinter)
|   +-- roi_parking_detector_v4.py# ROI classification engine (11 methods)
|   +-- test_c_integer_math.py    # C/Python cross-validation
|
+-- firmware/                     # ESP32-CAM firmware (C/Arduino)
|   +-- roi_classifier.h/c       # 11 classification methods
|   +-- adaptive_tx.h/c          # Adaptive protocol state machine
|   +-- sensor_cam_main.ino      # Main loop integration
|   +-- camera_config.h          # OV2640 pin mapping
|
+-- simulation/                   # Protocol simulation
|   +-- adaptive_protocol_sim.py  # 24h simulation, 3 strategies
|   +-- esp32_pipeline_demo.py    # Full pipeline visualization
|
+-- images/samples/               # Sample parking lot images
|   +-- parking_empty.png         # Reference (empty lot)
|   +-- parking_with_car.png      # Test (with vehicles)
|   +-- parking_3a_ref.png        # Real outdoor reference
|   +-- parking_3b_test.jpg       # Real outdoor test
|
+-- configs/                      # Example ROI configurations
|   +-- roi_config_*.json         # Polygon coordinates
|   +-- roi_config_*.h            # C code for firmware
|   +-- roi_config_*.py           # Python import
|
+-- docs/                         # Documentation
+-- README.md
+-- requirements.txt
```

---

## Quick Start

### 1. Install Dependencies

```bash
pip install numpy opencv-python Pillow
```

### 2. Launch the Calibrator GUI

```bash
cd app
python3 roi_calibration_tool.py --image ../images/samples/parking_empty.png
```

### 3. Calibrate ROIs (one-time per parking lot)

| Step | Action |
|------|--------|
| 1 | Open reference image (empty parking lot) |
| 2 | Click **[Auto Grid]** button |
| 3 | Click 4 corners of the parking area (TL -> TR -> BR -> BL) |
| 4 | Adjust grid: Rows/Cols spinboxes in right panel |
| 5 | Click **[Auto Number]** to label slots (T0..Tn, B0..Bn) |
| 6 | Switch to **[Edit/Move]** to fine-tune individual corners |
| 7 | Click **[Save Config]** to export JSON + C + Python |

### 4. Classify Vehicles (repeat anytime)

| Step | Action |
|------|--------|
| 1 | Click **[Quick: Load Test & Classify]** |
| 2 | Select image with parked cars |
| 3 | Results appear instantly on canvas |
| 4 | Red = Occupied, Green = Free |

### 5. Toggle Views

- **[View: Reference]** — Show empty lot with slot overlay
- **[View: Test]** — Show test image with classification results

---

## ROI Calibrator Features

### Three Modes

| Mode | Button | Description |
|------|--------|-------------|
| **Draw** | `[Draw Slot]` | Click 4 corners to create one slot (any quadrilateral) |
| **Edit** | `[Edit/Move]` | Drag corners to reshape, drag body to move |
| **Grid** | `[Auto Grid]` | Click 4 area corners, auto-generate perspective grid |

### Perspective Support

The tool supports **angled camera views** where parking slots appear as trapezoids (near = large, far = small). Each slot is a free-form quadrilateral with 4 draggable corners. ROI extraction uses `cv2.getPerspectiveTransform` for accurate warping to 32x32.

### Export Formats

When you save, **3 files** are generated:

1. **JSON** — Full polygon coordinates (4 corners per slot)
2. **C header** — `roi_rect_t` array for ESP32 firmware
3. **Python** — Direct import for simulation scripts

---

## Classification Algorithm (MAD)

```
MAD = (1/N) x Sum|current[i] - reference[i]|    N = 1024 (32x32)

Integer math: mad_x10 = sum * 10 / 1024
Threshold: 7.68 (mad_x10 > 77 -> OCCUPIED)
```

### Results on Real Data (16 slots)

| Group | MAD Range | Mean |
|-------|-----------|------|
| FREE | 3.8 - 7.5 | 5.2 |
| OCCUPIED | 36.6 - 52.6 | 44.7 |
| **Safety margin** | **29.1 units** | |

**10 out of 11 methods achieve 100% accuracy.**

### 11 Classification Methods

| # | Method | Accuracy |
|---|--------|----------|
| 0 | Edge Density | 68.8% |
| 1 | BG Relative | 96.1% |
| 2 | **MAD** | **100%** |
| 3 | Hybrid | 100% |
| 4 | Gaussian MAD | 100% |
| 5 | Block MAD | 100% |
| 6 | Percentile P75 | 100% |
| 7 | Max Block | 100% |
| 8 | Histogram Intersection | 100% |
| 9 | Variance Ratio | 100% |
| 10 | **Combined Voting** | **100%** |

---

## Adaptive Protocol

State machine with 4 states, driven by classifier confidence:

| State | Scan Interval | Condition |
|-------|:---:|---|
| IDLE | 30s | conf >= 30% (all stable) |
| ACTIVE | 5s | 15% <= conf < 30% |
| WATCHING | 2s | conf < 15% (borderline) |
| BURST | 0.5s | bitmap changed, confirm 3x |

### 24h Simulation Results

| Metric | MQTT/JSON | **Adaptive (Ours)** | Savings |
|--------|:---------:|:---:|:---:|
| TX Bytes | 3,110 KB | **2.4 KB** | 99.9% |
| Camera Scans | 17,280 | **6,823** | 60.5% |
| Duty Cycle | 1.16% | **0.46%** | 60.3% |

---

## Hardware

| Device | Role | Cost |
|--------|------|------|
| ESP32-CAM AI-Thinker | Sensor node (camera + classify + transmit) | ~120K VND |
| ESP32-S3 / Raspberry Pi | Gateway (aggregate + offline DB) | ~200-800K VND |

**Total cost per slot: ~200,000 VND** (10x cheaper than sensor-based solutions)

---

## Team

- **Vu Dang Khoa** (Team Lead) — K16 CNTT, Phenikaa University
- Tieu Cong Tuan
- Trinh Van Toan
- Phan Vu Hoai Nam

**Advisor:** Dr. Pham Ngoc Hung

**Project type:** Student Research (NCKH) 2025-2026, Phenikaa University

---

## License

This project is part of an academic research initiative. For educational and research purposes.

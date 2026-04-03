/*
 * sensor_cam_main.ino — ESP32-CAM Smart Parking Sensor Node v2.0
 *
 * Hardware: ESP32-CAM AI-Thinker (ESP32 + OV2640 + 4MB PSRAM)
 * Protocol: LiteComm v3.3 gradient mesh over ESP-NOW
 * Classification: 11 methods, default=combined (most robust)
 *
 * Workflow:
 *   1. Capture grayscale image (320×240)
 *   2. Extract ROIs (parking slot regions, 32×32 each)
 *   3. Classify each ROI → occupied/empty bitmap
 *   4. Transmit bitmap via LiteComm DATA frame (ESP-NOW)
 *
 * Serial commands:
 *   "CAL"        — Calibrate with current (empty) image
 *   "RESET"      — Reset calibration
 *   "STATUS"     — Print current slot states
 *   "METHOD X"   — Set classification method (0-10)
 *   "INTERVAL X" — Set scan interval (ms)
 *   "ROI X Y W H I" — Set ROI for slot I
 *   "FLASH X"    — Flash LED (0=off, 1=on, 2=momentary)
 *
 * Methods (real-image evaluation):
 *   0  edge_density     Acc=68.8%  (no calibration needed)
 *   1  bg_relative      Acc=96.1%
 *   2  ref_frame MAD    Acc=100.0% ← recommended simple
 *   3  hybrid           Acc=100.0%
 *   4  gaussian_mad     Acc=100.0%
 *   5  block_mad        Acc=100.0%
 *   6  percentile_mad   Acc=100.0%
 *   7  max_block        Acc=100.0%
 *   8  histogram        Acc=100.0%
 *   9  variance_ratio   Acc=100.0%
 *  10  combined         Acc=100.0% ← default (most robust)
 */

#include <Arduino.h>
#include "esp_camera.h"
#include "esp_log.h"
#include "nvs_flash.h"

#include "camera_config.h"
#include "roi_classifier.h"
// LiteComm: copy litecomm.h + litecomm.c from Firmware/litecomm/
// #include "litecomm.h"

static const char *TAG = "SENSOR_CAM";

// ─── Configuration ──────────────────────────────────────────────────
#define NODE_ID           0x01    // Unique ID for this node (change per device)
#define SCAN_INTERVAL_MS  5000    // Camera scan every 5 seconds
#define DEFAULT_METHOD    10      // 0=edge, 2=ref_mad, 10=combined (best)

// ─── ROI Configuration ─────────────────────────────────────────────
// Default: 5×2 grid for 10 parking slots in 320×240 frame
// Adjust these based on your camera angle and parking lot layout!
#define N_SLOTS   8   // Max 8 slots per node (LiteComm bitmap limit)

static roi_rect_t slot_rois[MAX_SLOTS] = {
    // Row 1 (top): slots 0-3
    { .x =  10, .y =  30, .w = 60, .h = 80 },  // Slot 0
    { .x =  80, .y =  30, .w = 60, .h = 80 },  // Slot 1
    { .x = 150, .y =  30, .w = 60, .h = 80 },  // Slot 2
    { .x = 220, .y =  30, .w = 60, .h = 80 },  // Slot 3
    // Row 2 (bottom): slots 4-7
    { .x =  10, .y = 130, .w = 60, .h = 80 },  // Slot 4
    { .x =  80, .y = 130, .w = 60, .h = 80 },  // Slot 5
    { .x = 150, .y = 130, .w = 60, .h = 80 },  // Slot 6
    { .x = 220, .y = 130, .w = 60, .h = 80 },  // Slot 7
};

// ─── State ──────────────────────────────────────────────────────────
static uint8_t  current_bitmap = 0;
static uint8_t  last_bitmap = 0xFF;  // Force first send
static uint8_t  classify_method = DEFAULT_METHOD;
static uint32_t scan_interval = SCAN_INTERVAL_MS;
static uint32_t last_scan_time = 0;
static uint32_t frame_count = 0;
static classify_result_t slot_results[MAX_SLOTS];

// ─── Function declarations ──────────────────────────────────────────
static bool init_camera(void);
static void process_frame(void);
static void send_litecomm_data(uint8_t bitmap);
static void handle_serial_command(void);
static void print_status(void);
static void led_flash(uint8_t mode);


// ═════════════════════════════════════════════════════════════════════
// Setup
// ═════════════════════════════════════════════════════════════════════

void setup() {
    Serial.begin(115200);
    Serial.println("\n=================================");
    Serial.println("  Smart Parking Sensor Node");
    Serial.println("  ESP32-CAM + LiteComm v3.3");
    Serial.println("=================================\n");
    
    // Initialize NVS (for calibration storage)
    esp_err_t ret = nvs_flash_init();
    if (ret == ESP_ERR_NVS_NO_FREE_PAGES || ret == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        nvs_flash_erase();
        nvs_flash_init();
    }
    
    // Initialize camera
    if (!init_camera()) {
        Serial.println("[FATAL] Camera init failed!");
        while (1) delay(1000);
    }
    Serial.println("[OK] Camera initialized (QVGA 320x240 grayscale)");
    
    // Initialize classifier (loads calibration from NVS if available)
    bool has_cal = classifier_init();
    if (has_cal) {
        Serial.println("[OK] Calibration data loaded from NVS");
        Serial.printf("     Method %d ready with calibrated thresholds\n", classify_method);
    } else {
        Serial.println("[WARN] No calibration. Using method 0 (edge_density)");
        Serial.println("       Send 'CAL' to calibrate with empty parking lot");
        if (classify_method > 0) {
            Serial.println("       Methods 1-10 require calibration -> falling back to 0");
        }
    }
    
    // Initialize LiteComm (ESP-NOW mesh)
    // TODO: Uncomment when litecomm.h is included
    // litecomm_config_t lc_cfg = {
    //     .node_id = NODE_ID,
    //     .is_gateway = false,
    //     .on_data_rx = NULL,
    //     .on_cmd_rx = NULL,
    // };
    // litecomm_init(&lc_cfg);
    
    // LED flash to indicate ready
    pinMode(FLASH_LED_PIN, OUTPUT);
    led_flash(2);  // Momentary flash
    
    Serial.printf("\n[READY] Node 0x%02X | Method=%d | Interval=%ums | Slots=%d\n\n",
                  NODE_ID, classify_method, scan_interval, N_SLOTS);
}


// ═════════════════════════════════════════════════════════════════════
// Main Loop
// ═════════════════════════════════════════════════════════════════════

void loop() {
    uint32_t now = millis();
    
    // LiteComm tick (beacon, routing maintenance)
    // litecomm_tick(now);
    
    // Check serial commands
    if (Serial.available()) {
        handle_serial_command();
    }
    
    // Periodic scan
    if (now - last_scan_time >= scan_interval) {
        last_scan_time = now;
        process_frame();
    }
    
    delay(10);
}


// ═════════════════════════════════════════════════════════════════════
// Camera Init
// ═════════════════════════════════════════════════════════════════════

static bool init_camera(void) {
    camera_config_t config = get_camera_config();
    
    esp_err_t err = esp_camera_init(&config);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "Camera init failed: 0x%x", err);
        return false;
    }
    
    // Optimize sensor settings for parking detection
    sensor_t *s = esp_camera_sensor_get();
    if (s) {
        s->set_brightness(s, 0);      // Default brightness
        s->set_contrast(s, 1);        // Slightly increased contrast
        s->set_saturation(s, 0);      // N/A for grayscale
        s->set_gainceiling(s, GAINCEILING_8X);  // Allow gain for low light
        s->set_whitebal(s, 1);        // Auto white balance
        s->set_gain_ctrl(s, 1);       // Auto gain
        s->set_exposure_ctrl(s, 1);   // Auto exposure
        s->set_aec2(s, 1);            // AEC DSP
        s->set_ae_level(s, 0);        // AE level
    }
    
    // Warm-up: take and discard first few frames
    for (int i = 0; i < 5; i++) {
        camera_fb_t *fb = esp_camera_fb_get();
        if (fb) esp_camera_fb_return(fb);
        delay(100);
    }
    
    return true;
}


// ═════════════════════════════════════════════════════════════════════
// Frame Processing & Classification
// ═════════════════════════════════════════════════════════════════════

static void process_frame(void) {
    uint32_t t0 = millis();
    
    // Capture frame
    camera_fb_t *fb = esp_camera_fb_get();
    if (!fb) {
        ESP_LOGE(TAG, "Frame capture failed");
        return;
    }
    
    // Verify format
    if (fb->format != PIXFORMAT_GRAYSCALE) {
        ESP_LOGE(TAG, "Unexpected format: %d (expected GRAYSCALE)", fb->format);
        esp_camera_fb_return(fb);
        return;
    }
    
    frame_count++;
    
    // Determine effective method (fallback if not calibrated)
    uint8_t effective_method = classify_method;
    if (effective_method > 0 && !classifier_is_calibrated()) {
        effective_method = 0;  // Fall back to edge_density
    }
    
    // Classify all slots
    current_bitmap = classify_all_slots(
        fb->buf, fb->width,
        slot_rois, N_SLOTS,
        effective_method, slot_results
    );
    
    uint32_t classify_ms = millis() - t0;
    
    // Print results
    Serial.printf("[%6u] Bitmap=0b", frame_count);
    for (int i = N_SLOTS - 1; i >= 0; i--) {
        Serial.print((current_bitmap >> i) & 1);
    }
    Serial.printf(" (0x%02X) | %ums | M%d", current_bitmap, classify_ms, effective_method);
    
    // Slot details
    uint8_t n_occupied = 0;
    for (int i = 0; i < N_SLOTS; i++) {
        if (slot_results[i].prediction) n_occupied++;
    }
    Serial.printf(" | %d/%d occupied\n", n_occupied, N_SLOTS);
    
    // Send via LiteComm if bitmap changed
    if (current_bitmap != last_bitmap) {
        Serial.printf("         → CHANGE detected! Old=0x%02X New=0x%02X\n",
                      last_bitmap, current_bitmap);
        send_litecomm_data(current_bitmap);
        last_bitmap = current_bitmap;
    }
    
    // Return frame buffer
    esp_camera_fb_return(fb);
}

static void send_litecomm_data(uint8_t bitmap) {
    // TODO: Uncomment when litecomm.h is included
    // litecomm_send_data(bitmap, 0, LC_PRI_HIGH);
    
    // For now, print what would be sent
    Serial.printf("         [TX] LiteComm DATA: node=0x%02X bitmap=0x%02X\n",
                  NODE_ID, bitmap);
}


// ═════════════════════════════════════════════════════════════════════
// Serial Command Handler
// ═════════════════════════════════════════════════════════════════════

static void handle_serial_command(void) {
    String line = Serial.readStringUntil('\n');
    line.trim();
    line.toUpperCase();
    
    if (line == "CAL") {
        Serial.println("\n[CAL] Calibrating with current frame (ensure lot is EMPTY)...");
        led_flash(1);  // LED on during calibration
        
        camera_fb_t *fb = esp_camera_fb_get();
        if (fb && fb->format == PIXFORMAT_GRAYSCALE) {
            bool ok = classifier_calibrate(fb->buf, fb->width, slot_rois, N_SLOTS);
            if (ok) {
                Serial.println("[CAL] Success! Calibration saved to NVS.");
                Serial.printf("[CAL] Recommend method 2 (ref_frame) or 10 (combined).\n");
            } else {
                Serial.println("[CAL] FAILED. Check NVS storage.");
            }
            esp_camera_fb_return(fb);
        } else {
            Serial.println("[CAL] Frame capture failed!");
        }
        
        led_flash(0);  // LED off
    }
    else if (line == "RESET") {
        classifier_reset_calibration();
        Serial.println("[RESET] Calibration cleared. Falling back to method 0.");
    }
    else if (line == "STATUS") {
        print_status();
    }
    else if (line.startsWith("METHOD ")) {
        int m = line.substring(7).toInt();
        if (m >= 0 && m <= 10) {
            classify_method = m;
            Serial.printf("[CONFIG] Method set to %d\n", m);
            if (m > 0 && !classifier_is_calibrated()) {
                Serial.println("         WARNING: Method requires calibration. Send 'CAL' first.");
            }
        } else {
            Serial.println("[ERROR] Method must be 0-10");
        }
    }
    else if (line.startsWith("INTERVAL ")) {
        int ms = line.substring(9).toInt();
        if (ms >= 1000 && ms <= 60000) {
            scan_interval = ms;
            Serial.printf("[CONFIG] Scan interval set to %u ms\n", scan_interval);
        } else {
            Serial.println("[ERROR] Interval must be 1000-60000 ms");
        }
    }
    else if (line.startsWith("ROI ")) {
        // Parse: ROI X Y W H I
        int x, y, w, h, idx;
        if (sscanf(line.c_str(), "ROI %d %d %d %d %d", &x, &y, &w, &h, &idx) == 5) {
            if (idx >= 0 && idx < MAX_SLOTS && x >= 0 && y >= 0 && w > 0 && h > 0) {
                slot_rois[idx].x = x;
                slot_rois[idx].y = y;
                slot_rois[idx].w = w;
                slot_rois[idx].h = h;
                Serial.printf("[ROI] Slot %d: (%d,%d,%d,%d)\n", idx, x, y, w, h);
            } else {
                Serial.println("[ERROR] Invalid ROI parameters");
            }
        } else {
            Serial.println("[ERROR] Usage: ROI X Y W H INDEX");
        }
    }
    else if (line.startsWith("FLASH ")) {
        int mode = line.substring(6).toInt();
        led_flash(mode);
    }
    else if (line.length() > 0) {
        Serial.println("Commands: CAL | RESET | STATUS | METHOD 0-10 | INTERVAL ms | ROI x y w h idx | FLASH 0/1/2");
    }
}

// Method name lookup table
static const char *method_names[] = {
    "edge_density",   // 0
    "bg_relative",    // 1
    "ref_frame_mad",  // 2
    "hybrid",         // 3
    "gaussian_mad",   // 4
    "block_mad",      // 5
    "percentile_mad", // 6
    "max_block",      // 7
    "histogram",      // 8
    "variance_ratio", // 9
    "combined",       // 10
};

static void print_status(void) {
    const char *mname = (classify_method < NUM_METHODS) ?
                        method_names[classify_method] : "unknown";

    Serial.println("\n╔══════════════════════════════════════════════╗");
    Serial.println("║    Smart Parking Sensor v2.0 — Status        ║");
    Serial.println("╠══════════════════════════════════════════════╣");
    Serial.printf( "║  Node ID:        0x%02X                        ║\n", NODE_ID);
    Serial.printf( "║  Frame count:    %-8u                    ║\n", frame_count);
    Serial.printf( "║  Method:         %2d (%s)%*s║\n",
                   classify_method, mname,
                   (int)(16 - strlen(mname)), "");
    Serial.printf( "║  Calibrated:     %-3s                         ║\n",
        classifier_is_calibrated() ? "YES" : "NO");
    Serial.printf( "║  Scan interval:  %-5u ms                    ║\n", scan_interval);
    Serial.printf( "║  Active slots:   %-2d                          ║\n", N_SLOTS);
    Serial.printf( "║  Current bitmap: 0x%02X                        ║\n", current_bitmap);
    Serial.println("╠══════════════════════════════════════════════╣");

    uint8_t n_occ = 0;
    for (int i = 0; i < N_SLOTS; i++) {
        const char *state = slot_results[i].prediction ? "OCC " : "FREE";
        Serial.printf("║  Slot %d: %s  conf=%3d%%  raw=%5d       ║\n",
                      i, state, slot_results[i].confidence, slot_results[i].raw_metric);
        if (slot_results[i].prediction) n_occ++;
    }

    Serial.println("╠══════════════════════════════════════════════╣");
    Serial.printf( "║  Occupancy: %d/%d slots (%.0f%%)                ║\n",
                   n_occ, N_SLOTS, (float)n_occ / N_SLOTS * 100);
    Serial.printf( "║  Free heap:  %u bytes                  ║\n", esp_get_free_heap_size());
    Serial.printf( "║  PSRAM free: %u bytes                 ║\n", ESP.getFreePsram());
    Serial.printf( "║  Cal data:   %u bytes                 ║\n",
                   (unsigned)sizeof(calibration_data_t));
    Serial.println("╚══════════════════════════════════════════════╝\n");
}

static void led_flash(uint8_t mode) {
    switch (mode) {
        case 0: digitalWrite(FLASH_LED_PIN, LOW); break;
        case 1: digitalWrite(FLASH_LED_PIN, HIGH); break;
        case 2: // Momentary
            digitalWrite(FLASH_LED_PIN, HIGH);
            delay(200);
            digitalWrite(FLASH_LED_PIN, LOW);
            break;
    }
}

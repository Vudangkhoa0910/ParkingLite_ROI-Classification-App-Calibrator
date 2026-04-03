/*
 * camera_config.h — ESP32-CAM AI-Thinker OV2640 pin configuration
 * 
 * Board: ESP32-CAM (AI-Thinker module)
 * Camera: OV2640
 * Resolution: QVGA 320×240 (grayscale for classification)
 * Framework: Arduino + esp32-camera
 */

#ifndef CAMERA_CONFIG_H
#define CAMERA_CONFIG_H

#include "esp_camera.h"

// ─── AI-Thinker ESP32-CAM Pin Map ───────────────────────────────────
#define PWDN_GPIO_NUM     32
#define RESET_GPIO_NUM    -1
#define XCLK_GPIO_NUM      0
#define SIOD_GPIO_NUM     26
#define SIOC_GPIO_NUM     27

#define Y9_GPIO_NUM       35
#define Y8_GPIO_NUM       34
#define Y7_GPIO_NUM       39
#define Y6_GPIO_NUM       36
#define Y5_GPIO_NUM       21
#define Y4_GPIO_NUM       19
#define Y3_GPIO_NUM       18
#define Y2_GPIO_NUM        5

#define VSYNC_GPIO_NUM    25
#define HREF_GPIO_NUM     23
#define PCLK_GPIO_NUM     22

// ─── LED Flash ──────────────────────────────────────────────────────
#define FLASH_LED_PIN      4   // Built-in flash LED

// ─── Camera Configuration ───────────────────────────────────────────

static camera_config_t get_camera_config(void) {
    camera_config_t config;
    config.ledc_channel = LEDC_CHANNEL_0;
    config.ledc_timer   = LEDC_TIMER_0;
    config.pin_d0       = Y2_GPIO_NUM;
    config.pin_d1       = Y3_GPIO_NUM;
    config.pin_d2       = Y4_GPIO_NUM;
    config.pin_d3       = Y5_GPIO_NUM;
    config.pin_d4       = Y6_GPIO_NUM;
    config.pin_d5       = Y7_GPIO_NUM;
    config.pin_d6       = Y8_GPIO_NUM;
    config.pin_d7       = Y9_GPIO_NUM;
    config.pin_xclk     = XCLK_GPIO_NUM;
    config.pin_pclk     = PCLK_GPIO_NUM;
    config.pin_vsync    = VSYNC_GPIO_NUM;
    config.pin_href     = HREF_GPIO_NUM;
    config.pin_sccb_sda = SIOD_GPIO_NUM;
    config.pin_sccb_scl = SIOC_GPIO_NUM;
    config.pin_pwdn     = PWDN_GPIO_NUM;
    config.pin_reset    = RESET_GPIO_NUM;
    config.xclk_freq_hz = 20000000;       // 20 MHz XCLK
    config.pixel_format = PIXFORMAT_GRAYSCALE; // 8-bit grayscale for ROI classification
    config.frame_size   = FRAMESIZE_QVGA; // 320×240
    config.jpeg_quality = 12;             // Not used for grayscale, but set anyway
    config.fb_count     = 1;              // Single frame buffer (save RAM)
    config.fb_location  = CAMERA_FB_IN_PSRAM; // Use PSRAM for frame buffer
    config.grab_mode    = CAMERA_GRAB_LATEST;
    return config;
}

// ─── Resolution Constants ───────────────────────────────────────────
#define CAM_WIDTH   320
#define CAM_HEIGHT  240

#endif // CAMERA_CONFIG_H

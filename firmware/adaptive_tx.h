/**
 * @file adaptive_tx.h
 * @brief Confidence-Aware Adaptive Scan & Transmit Controller
 *
 * Cross-layer protocol that bridges ROI classification and communication:
 *   - ROI classifier produces per-slot confidence (0-100%)
 *   - This module uses min(confidence) to adapt scan interval & transmission
 *
 * Design principles:
 *   1. SEPARATE from classifier — takes (bitmap, confidences[]) as input
 *   2. SEPARATE from transport — outputs raw frame bytes
 *   3. 100% integer math (consistent with ROI classifier philosophy)
 *   4. Zero heap allocation (static state, embedded-friendly)
 *
 * Three core mechanisms:
 *   A. Adaptive Scan Interval — confidence drives scan frequency
 *   B. BURST Confirmation    — verify changes before reporting
 *   C. Tiered Transmission   — send minimal data, only when needed
 *
 * Frame types (all include CRC-8):
 *   HEARTBEAT  5 bytes — periodic keep-alive
 *   EVENT      8 bytes — confirmed state change
 *   STATUS    13 bytes — detailed per-slot MAD values
 *
 * @version 1.0.0
 * @date 2026-03
 */

#ifndef ADAPTIVE_TX_H
#define ADAPTIVE_TX_H

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * CONSTANTS — Scan Interval Tiers
 *
 * Confidence drives which tier is active:
 *   conf_min < CONF_LOW  → WATCHING (scan fast, slot near threshold)
 *   conf_min < CONF_HIGH → ACTIVE  (normal operation)
 *   conf_min ≥ CONF_HIGH → IDLE    (all stable, scan slowly)
 *   bitmap changed       → BURST   (confirm change rapidly)
 * ========================================================================== */

#define ATX_T_BURST_MS       500     /* Scan interval during BURST confirmation  */
#define ATX_T_WATCH_MS       2000    /* Some slot near threshold                 */
#define ATX_T_ACTIVE_MS      5000    /* Normal operation                         */
#define ATX_T_IDLE_MS        30000   /* All slots highly confident               */

/* ---- Confidence thresholds (0-100 scale) ---- */
#define ATX_CONF_LOW         15      /* Below → WATCHING mode                    */
#define ATX_CONF_HIGH        30      /* Above → IDLE mode                        */

/* ---- BURST confirmation ---- */
#define ATX_BURST_CONFIRM    3       /* Consecutive matching scans to confirm    */
#define ATX_BURST_MAX_TRIES  10      /* Max scans in BURST before forced exit    */

/* ---- Heartbeat & Status intervals ---- */
#define ATX_HEARTBEAT_MS     300000  /* Keep-alive every 5 minutes               */
#define ATX_STATUS_MS        900000  /* Detailed report every 15 minutes         */

/* ---- Frame size limits ---- */
#define ATX_MAX_SLOTS        8       /* Matches ROI classifier MAX_SLOTS         */
#define ATX_FRAME_MAX_SIZE   16      /* Maximum frame buffer size                */

/* ============================================================================
 * FRAME CONTROL BYTE
 *
 * Bits [7:6] — Frame Type
 *   00 = HEARTBEAT (keep-alive)
 *   01 = EVENT     (confirmed state change)
 *   10 = STATUS    (detailed per-slot data)
 *   11 = CMD       (gateway → node command)
 *
 * Bits [5:4] — Node State (tells gateway our current mode)
 *   00 = IDLE
 *   01 = ACTIVE
 *   10 = WATCHING
 *   11 = BURST
 *
 * Bit  [3]   — ACK requested
 * Bit  [2]   — Calibrated flag (1 = ROI calibrated)
 * Bits [1:0] — Reserved
 * ========================================================================== */

#define ATX_FT_HEARTBEAT    0x00
#define ATX_FT_EVENT        0x40
#define ATX_FT_STATUS       0x80
#define ATX_FT_CMD          0xC0
#define ATX_FT_MASK         0xC0

#define ATX_ST_IDLE         0x00
#define ATX_ST_ACTIVE       0x10
#define ATX_ST_WATCHING     0x20
#define ATX_ST_BURST        0x30
#define ATX_ST_MASK         0x30

#define ATX_FLAG_ACK_REQ    0x08
#define ATX_FLAG_CALIBRATED 0x04

/* ============================================================================
 * WIRE-FORMAT FRAME STRUCTURES
 *
 * All frames:  [CTRL][NODE_ID][SEQ][...payload...][CRC-8]
 * CRC-8/MAXIM over all bytes except CRC itself.
 * ========================================================================== */

/**
 * @brief HEARTBEAT frame — periodic keep-alive (5 bytes)
 *
 * Sent every ATX_HEARTBEAT_MS. Proves node is alive and reports
 * current bitmap. Gateway can detect node failure if missing.
 *
 *  [0] ctrl     — ATX_FT_HEARTBEAT | state | flags
 *  [1] node_id  — source node ID
 *  [2] seq      — rolling sequence number
 *  [3] bitmap   — current 8-slot occupancy bitmap
 *  [4] crc8     — CRC-8/MAXIM over bytes 0..3
 */
typedef struct __attribute__((packed)) {
    uint8_t ctrl;
    uint8_t node_id;
    uint8_t seq;
    uint8_t bitmap;
    uint8_t crc8;
} atx_heartbeat_t;                  /* 5 bytes */

/**
 * @brief EVENT frame — confirmed state change (8 bytes)
 *
 * Sent ONLY when bitmap change is confirmed by BURST verification.
 * Carries before/after bitmaps + minimum confidence + current interval.
 *
 *  [0] ctrl       — ATX_FT_EVENT | state | flags
 *  [1] node_id    — source node ID
 *  [2] seq        — rolling sequence number
 *  [3] new_bitmap — new occupancy bitmap (after change)
 *  [4] old_bitmap — previous bitmap (before change)
 *  [5] min_conf   — minimum slot confidence (0-100)
 *  [6] interval_s — current scan interval in seconds (0-255)
 *  [7] crc8       — CRC-8/MAXIM over bytes 0..6
 */
typedef struct __attribute__((packed)) {
    uint8_t ctrl;
    uint8_t node_id;
    uint8_t seq;
    uint8_t new_bitmap;
    uint8_t old_bitmap;
    uint8_t min_conf;
    uint8_t interval_s;
    uint8_t crc8;
} atx_event_t;                      /* 8 bytes */

/**
 * @brief STATUS frame — detailed per-slot report (13 bytes for 8 slots)
 *
 * Sent periodically (every ATX_STATUS_MS) or on gateway request.
 * Contains per-slot raw_metric values for remote analysis.
 *
 *  [0]    ctrl     — ATX_FT_STATUS | state | flags
 *  [1]    node_id  — source node ID
 *  [2]    seq      — rolling sequence number
 *  [3]    bitmap   — current occupancy bitmap
 *  [4..11] mad[8]  — per-slot MAD × 2 (uint8, range 0-127.5, resolution 0.5)
 *  [12]   crc8     — CRC-8/MAXIM over bytes 0..11
 */
typedef struct __attribute__((packed)) {
    uint8_t ctrl;
    uint8_t node_id;
    uint8_t seq;
    uint8_t bitmap;
    uint8_t mad[ATX_MAX_SLOTS];     /* MAD × 2, capped at 255 */
    uint8_t crc8;
} atx_status_t;                     /* 13 bytes */

/* ============================================================================
 * NODE STATE MACHINE
 * ========================================================================== */

typedef enum {
    ATX_STATE_IDLE     = 0,   /* All slots confident → scan slowly        */
    ATX_STATE_ACTIVE   = 1,   /* Normal operation → scan at default rate   */
    ATX_STATE_WATCHING = 2,   /* Some slot near threshold → scan faster    */
    ATX_STATE_BURST    = 3,   /* Change detected → rapid confirm scans     */
} atx_state_t;

/* ============================================================================
 * TRANSMISSION DECISION OUTPUT
 * ========================================================================== */

typedef enum {
    ATX_TX_NONE      = 0,   /* No transmission needed                     */
    ATX_TX_HEARTBEAT = 1,   /* Send HEARTBEAT (keep-alive)                */
    ATX_TX_EVENT     = 2,   /* Send EVENT (confirmed change)              */
    ATX_TX_STATUS    = 3,   /* Send STATUS (detailed report)              */
} atx_tx_decision_t;

/* ============================================================================
 * MODULE STATE (opaque to caller)
 * ========================================================================== */

typedef struct {
    /* Configuration */
    uint8_t  node_id;
    uint8_t  n_slots;
    bool     calibrated;

    /* Current state */
    atx_state_t state;
    uint8_t  current_bitmap;
    uint8_t  prev_bitmap;
    uint8_t  min_confidence;
    uint8_t  confidences[ATX_MAX_SLOTS];
    uint16_t raw_metrics[ATX_MAX_SLOTS];

    /* BURST confirmation */
    uint8_t  pending_bitmap;        /* bitmap being confirmed            */
    uint8_t  old_bitmap;            /* bitmap BEFORE the change (for EVENT) */
    uint8_t  burst_agree_count;     /* consecutive matching scans        */
    uint8_t  burst_total_scans;     /* total scans in this BURST period  */

    /* Timing */
    uint32_t scan_interval_ms;      /* current adaptive scan interval    */
    uint32_t last_tx_ms;            /* last transmission timestamp       */
    uint32_t last_status_ms;        /* last STATUS frame timestamp       */
    uint32_t state_enter_ms;        /* when current state was entered    */

    /* Sequence & counters */
    uint8_t  seq;                   /* rolling frame sequence number     */
    uint32_t total_scans;           /* lifetime scan counter             */
    uint32_t total_tx_bytes;        /* lifetime TX byte counter          */
    uint16_t events_sent;           /* lifetime EVENT counter            */
    uint16_t heartbeats_sent;       /* lifetime HEARTBEAT counter        */

    /* Pending transmission */
    atx_tx_decision_t pending_tx;   /* what to send next                 */
} atx_state_data_t;

/* ============================================================================
 * PUBLIC API
 *
 * Usage flow in main loop:
 *   1. atx_init(node_id, n_slots)         — call once in setup()
 *   2. interval = atx_get_scan_interval() — how long to sleep
 *   3. [capture + classify]               — ROI classifier does its work
 *   4. atx_update(bitmap, confs, raws)    — feed results to protocol
 *   5. decision = atx_get_tx_decision()   — should we transmit?
 *   6. if yes: atx_build_frame(buf, len)  — get frame bytes to send
 *   7. [send via ESP-NOW]                 — transport layer sends
 *   8. goto 2
 * ========================================================================== */

/**
 * @brief Initialize the adaptive controller.
 * @param node_id  Unique node identifier (0x01-0xFE)
 * @param n_slots  Number of parking slots (1-8)
 */
void atx_init(uint8_t node_id, uint8_t n_slots);

/**
 * @brief Set calibration status (from ROI classifier).
 * @param calibrated  true if ROI classifier has calibration data
 */
void atx_set_calibrated(bool calibrated);

/**
 * @brief Get the recommended scan interval based on current confidence.
 * @return Scan interval in milliseconds
 */
uint32_t atx_get_scan_interval(void);

/**
 * @brief Feed classification results to the adaptive controller.
 *
 * Call this after each scan. The controller updates its state machine,
 * decides if the bitmap has changed, and determines if transmission is needed.
 *
 * @param bitmap       8-bit occupancy bitmap from classifier
 * @param confidences  Per-slot confidence array (0-100), length = n_slots
 * @param raw_metrics  Per-slot raw metric values (MAD×10 etc.), length = n_slots
 * @param now_ms       Current timestamp (millis())
 */
void atx_update(uint8_t bitmap, const uint8_t *confidences,
                const uint16_t *raw_metrics, uint32_t now_ms);

/**
 * @brief Check what transmission is needed (if any).
 * @return ATX_TX_NONE, ATX_TX_HEARTBEAT, ATX_TX_EVENT, or ATX_TX_STATUS
 */
atx_tx_decision_t atx_get_tx_decision(void);

/**
 * @brief Build the frame bytes for the pending transmission.
 *
 * Call this only when atx_get_tx_decision() != ATX_TX_NONE.
 * After calling, the pending TX is cleared.
 *
 * @param buf  Output buffer (must be at least ATX_FRAME_MAX_SIZE bytes)
 * @param len  Output: actual frame length written
 * @return true on success, false if no pending TX
 */
bool atx_build_frame(uint8_t *buf, uint8_t *len);

/**
 * @brief Get current node state.
 */
atx_state_t atx_get_state(void);

/**
 * @brief Get state name string (for debug/display).
 */
const char *atx_state_name(atx_state_t state);

/**
 * @brief Get internal state data (for monitoring/debug).
 */
const atx_state_data_t *atx_get_state_data(void);

#ifdef __cplusplus
}
#endif

#endif /* ADAPTIVE_TX_H */

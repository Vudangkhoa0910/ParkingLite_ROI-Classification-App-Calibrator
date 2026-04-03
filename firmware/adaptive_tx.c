/**
 * @file adaptive_tx.c
 * @brief Confidence-Aware Adaptive Scan & Transmit Controller — Implementation
 *
 * State machine:
 *
 *   ┌──────────┐  conf≥HIGH  ┌──────────┐  conf<HIGH  ┌──────────┐
 *   │   IDLE   │◄────────────│  ACTIVE   │────────────►│ WATCHING │
 *   │ scan 30s │  conf≥HIGH  │  scan 5s  │  conf<LOW   │  scan 2s │
 *   └────┬─────┘             └─────┬─────┘             └────┬─────┘
 *        │ bitmap≠prev             │ bitmap≠prev             │ bitmap≠prev
 *        ▼                         ▼                         ▼
 *   ┌────────────────────────────────────────────────────────────┐
 *   │                        BURST                               │
 *   │  scan 0.5s, count consecutive matches                     │
 *   │  if agree_count ≥ 3 → CONFIRM → send EVENT → return      │
 *   │  if total_scans ≥ 10 → timeout → accept anyway           │
 *   └────────────────────────────────────────────────────────────┘
 *
 * 100% integer math. Zero heap allocation. < 200 bytes RAM.
 *
 * @version 1.0.0
 */

#include "adaptive_tx.h"
#include <string.h>

/* ============================================================================
 * CRC-8/MAXIM (same as LiteComm — reusable)
 * Polynomial: 0x31, Init: 0x00, Reflected I/O
 * ========================================================================== */

static const uint8_t crc8_table[256] = {
    0x00,0x5E,0xBC,0xE2,0x61,0x3F,0xDD,0x83,0xC2,0x9C,0x7E,0x20,0xA3,0xFD,0x1F,0x41,
    0x9D,0xC3,0x21,0x7F,0xFC,0xA2,0x40,0x1E,0x5F,0x01,0xE3,0xBD,0x3E,0x60,0x82,0xDC,
    0x23,0x7D,0x9F,0xC1,0x42,0x1C,0xFE,0xA0,0xE1,0xBF,0x5D,0x03,0x80,0xDE,0x3C,0x62,
    0xBE,0xE0,0x02,0x5C,0xDF,0x81,0x63,0x3D,0x7C,0x22,0xC0,0x9E,0x1D,0x43,0xA1,0xFF,
    0x46,0x18,0xFA,0xA4,0x27,0x79,0x9B,0xC5,0x84,0xDA,0x38,0x66,0xE5,0xBB,0x59,0x07,
    0xDB,0x85,0x67,0x39,0xBA,0xE4,0x06,0x58,0x19,0x47,0xA5,0xFB,0x78,0x26,0xC4,0x9A,
    0x65,0x3B,0xD9,0x87,0x04,0x5A,0xB8,0xE6,0xA7,0xF9,0x1B,0x45,0xC6,0x98,0x7A,0x24,
    0xF8,0xA6,0x44,0x1A,0x99,0xC7,0x25,0x7B,0x3A,0x64,0x86,0xD8,0x5B,0x05,0xE7,0xB9,
    0x8C,0xD2,0x30,0x6E,0xED,0xB3,0x51,0x0F,0x4E,0x10,0xF2,0xAC,0x2F,0x71,0x93,0xCD,
    0x11,0x4F,0xAD,0xF3,0x70,0x2E,0xCC,0x92,0xD3,0x8D,0x6F,0x31,0xB2,0xEC,0x0E,0x50,
    0xAF,0xF1,0x13,0x4D,0xCE,0x90,0x72,0x2C,0x6D,0x33,0xD1,0x8F,0x0C,0x52,0xB0,0xEE,
    0x32,0x6C,0x8E,0xD0,0x53,0x0D,0xEF,0xB1,0xF0,0xAE,0x4C,0x12,0x91,0xCF,0x2D,0x73,
    0xCA,0x94,0x76,0x28,0xAB,0xF5,0x17,0x49,0x08,0x56,0xB4,0xEA,0x69,0x37,0xD5,0x8B,
    0x57,0x09,0xEB,0xB5,0x36,0x68,0x8A,0xD4,0x95,0xCB,0x29,0x77,0xF4,0xAA,0x48,0x16,
    0xE9,0xB7,0x55,0x0B,0x88,0xD6,0x34,0x6A,0x2B,0x75,0x97,0xC9,0x4A,0x14,0xF6,0xA8,
    0x74,0x2A,0xC8,0x96,0x15,0x4B,0xA9,0xF7,0xB6,0xE8,0x0A,0x54,0xD7,0x89,0x6B,0x35,
};

static uint8_t atx_crc8(const uint8_t *data, uint8_t len) {
    uint8_t crc = 0x00;
    for (uint8_t i = 0; i < len; i++) {
        crc = crc8_table[crc ^ data[i]];
    }
    return crc;
}

/* ============================================================================
 * STATIC STATE (single instance — typical embedded pattern)
 * ========================================================================== */

static atx_state_data_t s_atx;

/* ============================================================================
 * HELPER: Build CTRL byte from current state
 * ========================================================================== */

static uint8_t build_ctrl(uint8_t frame_type) {
    uint8_t state_bits;
    switch (s_atx.state) {
        case ATX_STATE_IDLE:     state_bits = ATX_ST_IDLE;     break;
        case ATX_STATE_ACTIVE:   state_bits = ATX_ST_ACTIVE;   break;
        case ATX_STATE_WATCHING: state_bits = ATX_ST_WATCHING; break;
        case ATX_STATE_BURST:    state_bits = ATX_ST_BURST;    break;
        default:                 state_bits = ATX_ST_ACTIVE;   break;
    }
    uint8_t flags = s_atx.calibrated ? ATX_FLAG_CALIBRATED : 0;
    /* EVENT frames request ACK */
    if (frame_type == ATX_FT_EVENT) flags |= ATX_FLAG_ACK_REQ;
    return frame_type | state_bits | flags;
}

/* ============================================================================
 * HELPER: Compute minimum confidence across active slots
 * ========================================================================== */

/**
 * @brief Compute the 2nd-lowest confidence (robust to 1 noisy slot).
 *
 * Using min(confidence) makes the system overly sensitive to a single
 * noisy slot that fluctuates near the threshold. Using the 2nd-lowest
 * value provides robustness: 1 noisy slot won't keep the entire system
 * in WATCHING mode permanently.
 */
static uint8_t compute_min_confidence(const uint8_t *confs, uint8_t n) {
    uint8_t lo1 = 100, lo2 = 100;  /* lowest and 2nd-lowest */
    for (uint8_t i = 0; i < n; i++) {
        if (confs[i] < lo1) {
            lo2 = lo1;
            lo1 = confs[i];
        } else if (confs[i] < lo2) {
            lo2 = confs[i];
        }
    }
    return (n <= 1) ? lo1 : lo2;  /* fall back to min for single-slot */
}

/* ============================================================================
 * HELPER: Determine target state from confidence level
 *
 *   conf_min < CONF_LOW  → WATCHING  (something is near the threshold)
 *   conf_min < CONF_HIGH → ACTIVE    (normal, moderate confidence)
 *   conf_min ≥ CONF_HIGH → IDLE      (everything is stable)
 * ========================================================================== */

static atx_state_t confidence_to_state(uint8_t min_conf) {
    if (min_conf < ATX_CONF_LOW)  return ATX_STATE_WATCHING;
    if (min_conf < ATX_CONF_HIGH) return ATX_STATE_ACTIVE;
    return ATX_STATE_IDLE;
}

/* ============================================================================
 * HELPER: State → scan interval mapping
 * ========================================================================== */

static uint32_t state_to_interval(atx_state_t state) {
    switch (state) {
        case ATX_STATE_IDLE:     return ATX_T_IDLE_MS;
        case ATX_STATE_ACTIVE:   return ATX_T_ACTIVE_MS;
        case ATX_STATE_WATCHING: return ATX_T_WATCH_MS;
        case ATX_STATE_BURST:    return ATX_T_BURST_MS;
        default:                 return ATX_T_ACTIVE_MS;
    }
}

/* ============================================================================
 * PUBLIC API
 * ========================================================================== */

void atx_init(uint8_t node_id, uint8_t n_slots) {
    memset(&s_atx, 0, sizeof(s_atx));
    s_atx.node_id = node_id;
    s_atx.n_slots = (n_slots > ATX_MAX_SLOTS) ? ATX_MAX_SLOTS : n_slots;
    s_atx.state = ATX_STATE_ACTIVE;
    s_atx.scan_interval_ms = ATX_T_ACTIVE_MS;
    s_atx.prev_bitmap = 0xFF;   /* force first EVENT on first real scan */
    s_atx.pending_tx = ATX_TX_NONE;
}

void atx_set_calibrated(bool calibrated) {
    s_atx.calibrated = calibrated;
}

uint32_t atx_get_scan_interval(void) {
    return s_atx.scan_interval_ms;
}

atx_state_t atx_get_state(void) {
    return s_atx.state;
}

const char *atx_state_name(atx_state_t state) {
    switch (state) {
        case ATX_STATE_IDLE:     return "IDLE";
        case ATX_STATE_ACTIVE:   return "ACTIVE";
        case ATX_STATE_WATCHING: return "WATCHING";
        case ATX_STATE_BURST:    return "BURST";
        default:                 return "UNKNOWN";
    }
}

const atx_state_data_t *atx_get_state_data(void) {
    return &s_atx;
}

/* ============================================================================
 * CORE: State machine update
 *
 * Called after each ROI classification pass. Decides:
 *   1. Has the bitmap changed? → enter BURST
 *   2. If in BURST, is the change confirmed? → send EVENT
 *   3. Update state based on confidence → adjust scan interval
 *   4. Is it time for HEARTBEAT or STATUS? → queue it
 * ========================================================================== */

void atx_update(uint8_t bitmap, const uint8_t *confidences,
                const uint16_t *raw_metrics, uint32_t now_ms) {
    s_atx.total_scans++;
    s_atx.current_bitmap = bitmap;

    /* Store per-slot data */
    for (uint8_t i = 0; i < s_atx.n_slots; i++) {
        s_atx.confidences[i] = confidences[i];
        s_atx.raw_metrics[i] = raw_metrics[i];
    }
    s_atx.min_confidence = compute_min_confidence(confidences, s_atx.n_slots);

    /* Initialize timing on first call */
    if (s_atx.state_enter_ms == 0) {
        s_atx.state_enter_ms = now_ms;
        s_atx.last_tx_ms     = now_ms;
        s_atx.last_status_ms = now_ms;
    }

    s_atx.pending_tx = ATX_TX_NONE;  /* reset each cycle */

    /* ── BURST STATE: rapid-confirm a detected change ──────────────── */
    if (s_atx.state == ATX_STATE_BURST) {
        s_atx.burst_total_scans++;

        if (bitmap == s_atx.pending_bitmap) {
            /* Matches the pending change — increment agreement */
            s_atx.burst_agree_count++;
        } else if (bitmap == s_atx.prev_bitmap) {
            /* Reverted to old bitmap — false alarm, cancel BURST */
            s_atx.burst_agree_count = 0;
            s_atx.burst_total_scans = 0;
            atx_state_t target = confidence_to_state(s_atx.min_confidence);
            s_atx.state = target;
            s_atx.scan_interval_ms = state_to_interval(target);
            s_atx.state_enter_ms = now_ms;
            return;
        } else {
            /* Different bitmap from both old and pending — restart BURST */
            s_atx.pending_bitmap = bitmap;
            s_atx.burst_agree_count = 1;
            /* Don't reset total_scans — count toward timeout */
        }

        /* Check confirmation or timeout */
        if (s_atx.burst_agree_count >= ATX_BURST_CONFIRM ||
            s_atx.burst_total_scans >= ATX_BURST_MAX_TRIES) {

            /* ── CONFIRMED: commit the change ───────────────────── */
            s_atx.prev_bitmap = s_atx.current_bitmap;
            s_atx.pending_tx = ATX_TX_EVENT;

            /* Exit BURST → re-evaluate state from confidence */
            atx_state_t target = confidence_to_state(s_atx.min_confidence);
            s_atx.state = target;
            s_atx.scan_interval_ms = state_to_interval(target);
            s_atx.state_enter_ms = now_ms;
            s_atx.last_tx_ms = now_ms;
        }
        /* else: stay in BURST, keep scanning at 500ms */
        return;
    }

    /* ── NORMAL STATES (IDLE / ACTIVE / WATCHING) ──────────────────── */

    /* Check for bitmap change → enter BURST */
    if (bitmap != s_atx.prev_bitmap) {
        s_atx.state = ATX_STATE_BURST;
        s_atx.scan_interval_ms = ATX_T_BURST_MS;
        s_atx.pending_bitmap = bitmap;
        s_atx.old_bitmap = s_atx.prev_bitmap;  /* save for EVENT frame */
        s_atx.burst_agree_count = 1;   /* this scan is the first match */
        s_atx.burst_total_scans = 1;
        s_atx.state_enter_ms = now_ms;
        return;
    }

    /* No change — update state from confidence */
    atx_state_t target = confidence_to_state(s_atx.min_confidence);
    if (target != s_atx.state) {
        s_atx.state = target;
        s_atx.scan_interval_ms = state_to_interval(target);
        s_atx.state_enter_ms = now_ms;
    }

    /* ── Periodic transmissions ────────────────────────────────────── */

    /* STATUS report (every 15 min) — most informative, lower priority */
    if ((now_ms - s_atx.last_status_ms) >= ATX_STATUS_MS) {
        s_atx.pending_tx = ATX_TX_STATUS;
        s_atx.last_status_ms = now_ms;
        s_atx.last_tx_ms = now_ms;
        return;
    }

    /* HEARTBEAT (every 5 min) — proves we're alive */
    if ((now_ms - s_atx.last_tx_ms) >= ATX_HEARTBEAT_MS) {
        s_atx.pending_tx = ATX_TX_HEARTBEAT;
        s_atx.last_tx_ms = now_ms;
    }
}

/* ============================================================================
 * QUERY: What to send?
 * ========================================================================== */

atx_tx_decision_t atx_get_tx_decision(void) {
    return s_atx.pending_tx;
}

/* ============================================================================
 * BUILD FRAME: Pack bytes for ESP-NOW transmission
 * ========================================================================== */

bool atx_build_frame(uint8_t *buf, uint8_t *len) {
    if (s_atx.pending_tx == ATX_TX_NONE) return false;

    switch (s_atx.pending_tx) {

    case ATX_TX_HEARTBEAT: {
        atx_heartbeat_t *f = (atx_heartbeat_t *)buf;
        f->ctrl    = build_ctrl(ATX_FT_HEARTBEAT);
        f->node_id = s_atx.node_id;
        f->seq     = s_atx.seq++;
        f->bitmap  = s_atx.current_bitmap;
        f->crc8    = atx_crc8(buf, sizeof(atx_heartbeat_t) - 1);
        *len = sizeof(atx_heartbeat_t);   /* 5 bytes */
        s_atx.heartbeats_sent++;
        s_atx.total_tx_bytes += *len;
        break;
    }

    case ATX_TX_EVENT: {
        atx_event_t *f = (atx_event_t *)buf;
        f->ctrl       = build_ctrl(ATX_FT_EVENT);
        f->node_id    = s_atx.node_id;
        f->seq        = s_atx.seq++;
        f->new_bitmap = s_atx.current_bitmap;
        f->old_bitmap = s_atx.old_bitmap;   /* saved when entering BURST */
        f->min_conf   = s_atx.min_confidence;
        f->interval_s = (uint8_t)(s_atx.scan_interval_ms / 1000);
        if (s_atx.scan_interval_ms < 1000) f->interval_s = 0;
        f->crc8       = atx_crc8(buf, sizeof(atx_event_t) - 1);
        *len = sizeof(atx_event_t);       /* 8 bytes */
        s_atx.events_sent++;
        s_atx.total_tx_bytes += *len;
        break;
    }

    case ATX_TX_STATUS: {
        atx_status_t *f = (atx_status_t *)buf;
        f->ctrl    = build_ctrl(ATX_FT_STATUS);
        f->node_id = s_atx.node_id;
        f->seq     = s_atx.seq++;
        f->bitmap  = s_atx.current_bitmap;
        for (uint8_t i = 0; i < ATX_MAX_SLOTS; i++) {
            if (i < s_atx.n_slots) {
                /* raw_metric is MAD×10; encode as MAD×2 in 1 byte (range 0-127.5) */
                uint16_t mad_x2 = s_atx.raw_metrics[i] / 5;  /* ×10 / 5 = ×2 */
                f->mad[i] = (mad_x2 > 255) ? 255 : (uint8_t)mad_x2;
            } else {
                f->mad[i] = 0;
            }
        }
        f->crc8 = atx_crc8(buf, sizeof(atx_status_t) - 1);
        *len = sizeof(atx_status_t);      /* 13 bytes */
        s_atx.total_tx_bytes += *len;
        break;
    }

    default:
        return false;
    }

    s_atx.pending_tx = ATX_TX_NONE;
    return true;
}

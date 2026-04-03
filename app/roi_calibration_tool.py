#!/usr/bin/env python3
"""
ParkingLite ROI Calibrator — Full GUI Application.

Visual tool to define parking slot ROIs on any camera image.
Supports perspective quadrilaterals (angled camera views).

Usage:
    python3 roi_calibration_tool.py
    python3 roi_calibration_tool.py --image path/to/image.jpg
"""

import sys
import os
import json
import copy
import math
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog
from typing import List, Optional, Tuple
from dataclasses import dataclass
from PIL import Image, ImageTk, ImageDraw, ImageFont
import numpy as np
import cv2

ROI_SIZE = 32

# ══════════════════════════════════════════════════════════
# Data model
# ══════════════════════════════════════════════════════════

@dataclass
class Slot:
    pts: List[List[int]]   # 4 corners [[x,y], ...]
    label: str = ""
    slot_idx: int = -1
    prediction: int = -1   # -1=unknown, 0=free, 1=occupied
    confidence: float = 0.0
    mad_value: float = 0.0

    @property
    def center(self):
        return (int(np.mean([p[0] for p in self.pts])),
                int(np.mean([p[1] for p in self.pts])))

    @property
    def area(self):
        return float(cv2.contourArea(np.array(self.pts, dtype=np.float32)))

    def contains(self, px, py):
        pts = np.array(self.pts, dtype=np.float32).reshape(-1, 1, 2)
        return cv2.pointPolygonTest(pts, (float(px), float(py)), False) >= 0

    def nearest_corner(self, px, py, threshold=15):
        best_i, best_d = -1, threshold + 1
        for i, (cx, cy) in enumerate(self.pts):
            d = math.hypot(px - cx, py - cy)
            if d < best_d:
                best_i, best_d = i, d
        return best_i

    def bounding_rect(self):
        xs = [p[0] for p in self.pts]
        ys = [p[1] for p in self.pts]
        return (min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys))

    def to_dict(self):
        return {'pts': [list(p) for p in self.pts],
                'label': self.label, 'slot_idx': self.slot_idx}

    @staticmethod
    def from_dict(d):
        if 'pts' in d:
            return Slot(pts=[list(p) for p in d['pts']],
                        label=d.get('label', ''),
                        slot_idx=d.get('slot_idx', -1))
        x, y, w, h = d['x'], d['y'], d['w'], d['h']
        return Slot(pts=[[x,y],[x+w,y],[x+w,y+h],[x,y+h]],
                    label=d.get('label',''), slot_idx=d.get('slot_idx',-1))


# ══════════════════════════════════════════════════════════
# Main Application
# ══════════════════════════════════════════════════════════

class App(tk.Tk):
    def __init__(self, image_path=None):
        super().__init__()
        self.title("ParkingLite ROI Calibrator")
        self.geometry("1400x850")
        self.configure(bg='#1e1e2e')

        # State
        self.img_cv = None        # OpenCV BGR (reference/main)
        self.img_pil = None       # PIL RGB (reference/main)
        self._display_pil = None  # Currently displayed image (ref or test)
        self._display_cv = None
        self.img_path = None
        self.ref_path = None
        self.test_path = None
        self.slots: List[Slot] = []
        self.selected: int = -1
        self.undo_stack = []
        self.zoom = 1.0
        self.pan = [0, 0]

        # Interaction
        self.mode = 'edit'   # 'draw', 'edit', 'grid'
        self.draw_pts = []
        self.dragging_corner = -1
        self.dragging_slot = False
        self.drag_start = (0, 0)
        self.drag_slot_offsets = []

        # Grid params
        self.grid_pts = []
        self.grid_rows = tk.IntVar(value=2)
        self.grid_cols = tk.IntVar(value=8)

        self._build_ui()
        self._bind_keys()

        if image_path and os.path.exists(image_path):
            self._load_image(image_path)

    # ──────────────────────────────────────────
    # UI Layout
    # ──────────────────────────────────────────

    def _build_ui(self):
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('TFrame', background='#1e1e2e')
        style.configure('TLabel', background='#1e1e2e', foreground='#cdd6f4',
                        font=('Segoe UI', 10))
        style.configure('TButton', font=('Segoe UI', 9), padding=4)
        style.configure('Header.TLabel', font=('Segoe UI', 11, 'bold'),
                        foreground='#89b4fa')
        style.configure('Mode.TButton', font=('Segoe UI', 10, 'bold'), padding=6)
        style.configure('Action.TButton', font=('Segoe UI', 10), padding=5)
        style.configure('Accent.TButton', font=('Segoe UI', 10, 'bold'), padding=6)

        # ── Top toolbar ──
        top = ttk.Frame(self, style='TFrame')
        top.pack(fill=tk.X, padx=5, pady=(5, 0))

        ttk.Button(top, text="Open Image", command=self._open_image,
                   style='Action.TButton').pack(side=tk.LEFT, padx=2)
        ttk.Button(top, text="Load Config", command=self._open_config,
                   style='Action.TButton').pack(side=tk.LEFT, padx=2)

        ttk.Separator(top, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y,
                                                      padx=8, pady=2)

        # Mode buttons
        self.btn_draw = ttk.Button(top, text="Draw Slot",
                                    command=lambda: self._set_mode('draw'),
                                    style='Mode.TButton')
        self.btn_draw.pack(side=tk.LEFT, padx=2)
        self.btn_edit = ttk.Button(top, text="Edit / Move",
                                    command=lambda: self._set_mode('edit'),
                                    style='Mode.TButton')
        self.btn_edit.pack(side=tk.LEFT, padx=2)
        self.btn_grid = ttk.Button(top, text="Auto Grid",
                                    command=lambda: self._set_mode('grid'),
                                    style='Mode.TButton')
        self.btn_grid.pack(side=tk.LEFT, padx=2)

        ttk.Separator(top, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y,
                                                      padx=8, pady=2)

        ttk.Button(top, text="Zoom +", command=self._zoom_in,
                   style='TButton').pack(side=tk.LEFT, padx=1)
        ttk.Button(top, text="Zoom -", command=self._zoom_out,
                   style='TButton').pack(side=tk.LEFT, padx=1)
        ttk.Button(top, text="Fit", command=self._zoom_fit,
                   style='TButton').pack(side=tk.LEFT, padx=1)

        ttk.Separator(top, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y,
                                                      padx=8, pady=2)

        ttk.Button(top, text="Undo", command=self._undo,
                   style='TButton').pack(side=tk.LEFT, padx=2)

        # ── Main area: Canvas + Right panel ──
        main = ttk.Frame(self, style='TFrame')
        main.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Canvas (left)
        canvas_frame = ttk.Frame(main, style='TFrame')
        canvas_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(canvas_frame, bg='#11111b', highlightthickness=0,
                                cursor='crosshair')
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # Right panel
        right = ttk.Frame(main, style='TFrame', width=300)
        right.pack(side=tk.RIGHT, fill=tk.Y, padx=(5, 0))
        right.pack_propagate(False)

        self._build_right_panel(right)

        # ── Status bar ──
        self.status_var = tk.StringVar(value="Open an image to start.")
        status = ttk.Label(self, textvariable=self.status_var,
                           font=('Consolas', 9), foreground='#a6adc8')
        status.pack(fill=tk.X, padx=5, pady=(0, 3))

    def _build_right_panel(self, parent):
        # ── Actions ──
        ttk.Label(parent, text="ACTIONS", style='Header.TLabel').pack(
            anchor=tk.W, pady=(5, 3))

        btn_frame = ttk.Frame(parent, style='TFrame')
        btn_frame.pack(fill=tk.X, pady=2)

        ttk.Button(btn_frame, text="Auto Number (N)",
                   command=self._auto_number, style='Action.TButton').pack(
                       fill=tk.X, pady=1)
        ttk.Button(btn_frame, text="Delete Selected (Del)",
                   command=self._delete_selected, style='Action.TButton').pack(
                       fill=tk.X, pady=1)
        ttk.Button(btn_frame, text="Duplicate (D)",
                   command=self._duplicate, style='Action.TButton').pack(
                       fill=tk.X, pady=1)
        ttk.Button(btn_frame, text="Clear All",
                   command=self._clear_all, style='Action.TButton').pack(
                       fill=tk.X, pady=1)

        ttk.Separator(parent, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=8)

        # ── Grid settings ──
        ttk.Label(parent, text="GRID SETTINGS", style='Header.TLabel').pack(
            anchor=tk.W, pady=(0, 3))
        grid_f = ttk.Frame(parent, style='TFrame')
        grid_f.pack(fill=tk.X, pady=2)

        ttk.Label(grid_f, text="Rows:").pack(side=tk.LEFT)
        ttk.Spinbox(grid_f, from_=1, to=8, width=4,
                     textvariable=self.grid_rows).pack(side=tk.LEFT, padx=4)
        ttk.Label(grid_f, text="Cols:").pack(side=tk.LEFT, padx=(8, 0))
        ttk.Spinbox(grid_f, from_=1, to=16, width=4,
                     textvariable=self.grid_cols).pack(side=tk.LEFT, padx=4)

        ttk.Separator(parent, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=8)

        # ── Classification ──
        ttk.Label(parent, text="CLASSIFY", style='Header.TLabel').pack(
            anchor=tk.W, pady=(0, 3))

        ref_f = ttk.Frame(parent, style='TFrame')
        ref_f.pack(fill=tk.X, pady=1)
        ttk.Button(ref_f, text="Set Reference Image",
                   command=self._set_ref).pack(fill=tk.X)

        test_f = ttk.Frame(parent, style='TFrame')
        test_f.pack(fill=tk.X, pady=1)
        ttk.Button(test_f, text="Set Test Image",
                   command=self._set_test).pack(fill=tk.X)

        self.ref_label = ttk.Label(parent, text="Ref: (none)",
                                    font=('Consolas', 8))
        self.ref_label.pack(anchor=tk.W)
        self.test_label = ttk.Label(parent, text="Test: (none)",
                                     font=('Consolas', 8))
        self.test_label.pack(anchor=tk.W)

        ttk.Button(parent, text="Run MAD Classification",
                   command=self._classify, style='Accent.TButton').pack(
                       fill=tk.X, pady=2)

        # Quick workflow: load test + classify + show in one click
        ttk.Button(parent, text="Quick: Load Test & Classify",
                   command=self._quick_classify, style='Accent.TButton').pack(
                       fill=tk.X, pady=2)

        # Toggle view buttons
        view_f = ttk.Frame(parent, style='TFrame')
        view_f.pack(fill=tk.X, pady=2)
        ttk.Button(view_f, text="View: Reference",
                   command=self._show_ref_image).pack(side=tk.LEFT, expand=True,
                                                       fill=tk.X, padx=(0, 2))
        ttk.Button(view_f, text="View: Test",
                   command=self._show_test_image).pack(side=tk.LEFT, expand=True,
                                                        fill=tk.X, padx=(2, 0))

        ttk.Separator(parent, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=8)

        # ── Save ──
        ttk.Label(parent, text="EXPORT", style='Header.TLabel').pack(
            anchor=tk.W, pady=(0, 3))
        ttk.Button(parent, text="Save Config (JSON + C + Python)",
                   command=self._save, style='Accent.TButton').pack(
                       fill=tk.X, pady=2)

        ttk.Separator(parent, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=8)

        # ── Slot list ──
        ttk.Label(parent, text="SLOT LIST", style='Header.TLabel').pack(
            anchor=tk.W, pady=(0, 3))

        list_frame = ttk.Frame(parent, style='TFrame')
        list_frame.pack(fill=tk.BOTH, expand=True)

        self.slot_list = tk.Listbox(
            list_frame, bg='#181825', fg='#cdd6f4', selectbackground='#45475a',
            font=('Consolas', 9), borderwidth=0, highlightthickness=0)
        scrollbar = ttk.Scrollbar(list_frame, command=self.slot_list.yview)
        self.slot_list.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.slot_list.pack(fill=tk.BOTH, expand=True)
        self.slot_list.bind('<<ListboxSelect>>', self._on_list_select)

    # ──────────────────────────────────────────
    # Key bindings
    # ──────────────────────────────────────────

    def _bind_keys(self):
        self.bind('<Delete>', lambda e: self._delete_selected())
        self.bind('<BackSpace>', lambda e: self._delete_selected())
        self.bind('<Key-d>', lambda e: self._duplicate())
        self.bind('<Key-n>', lambda e: self._auto_number())
        self.bind('<Key-s>', lambda e: self._save())
        self.bind('<Key-r>', lambda e: self._classify())
        self.bind('<Key-u>', lambda e: self._undo())
        self.bind('<Key-1>', lambda e: self._set_mode('draw'))
        self.bind('<Key-2>', lambda e: self._set_mode('edit'))
        self.bind('<Key-3>', lambda e: self._set_mode('grid'))
        self.bind('<Escape>', lambda e: self._cancel())
        self.bind('<plus>', lambda e: self._zoom_in())
        self.bind('<equal>', lambda e: self._zoom_in())
        self.bind('<minus>', lambda e: self._zoom_out())
        self.bind('<Key-f>', lambda e: self._zoom_fit())

        # Canvas mouse
        self.canvas.bind('<Button-1>', self._on_click)
        self.canvas.bind('<B1-Motion>', self._on_drag)
        self.canvas.bind('<ButtonRelease-1>', self._on_release)
        self.canvas.bind('<Button-3>', self._on_right_click)
        self.canvas.bind('<Motion>', self._on_motion)
        self.canvas.bind('<MouseWheel>', self._on_scroll)
        # Linux scroll
        self.canvas.bind('<Button-4>', lambda e: self._zoom_in())
        self.canvas.bind('<Button-5>', lambda e: self._zoom_out())

    # ──────────────────────────────────────────
    # Image loading
    # ──────────────────────────────────────────

    def _open_image(self):
        path = filedialog.askopenfilename(
            title="Open Parking Lot Image",
            filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp *.tiff"),
                       ("All", "*.*")])
        if path:
            self._load_image(path)

    def _load_image(self, path, is_test=False):
        img = cv2.imread(path)
        if img is None:
            messagebox.showerror("Error", f"Cannot load: {path}")
            return
        if is_test:
            # Load as test image and show it, keep slots + ref
            self.test_path = path
            self.test_label.configure(text=f"Test: {os.path.basename(path)}")
            self._display_cv = img
            self._display_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            self.status_var.set(f"Showing TEST image: {os.path.basename(path)} — "
                                f"Click 'Run MAD Classification' or press R")
        else:
            # Load as main/reference image
            self.img_cv = img
            self.img_path = path
            self.img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            self._display_cv = img
            self._display_pil = self.img_pil
            # Auto-set as reference
            self.ref_path = path
            self.ref_label.configure(text=f"Ref: {os.path.basename(path)}")
            h, w = img.shape[:2]
            self.status_var.set(
                f"Image: {w}x{h} — {os.path.basename(path)} "
                f"(auto-set as Reference)")
        self._zoom_fit()
        self._redraw()

    def _open_config(self):
        path = filedialog.askopenfilename(
            title="Load ROI Config",
            filetypes=[("JSON", "*.json"), ("All", "*.*")])
        if not path:
            return
        try:
            with open(path) as f:
                data = json.load(f)
            self.slots = [Slot.from_dict(d) for d in
                          data.get('rois', data.get('slots', []))]
            self.selected = -1
            self._refresh_list()
            self._redraw()
            self.status_var.set(f"Loaded {len(self.slots)} slots from {os.path.basename(path)}")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    # ──────────────────────────────────────────
    # Coordinate transforms
    # ──────────────────────────────────────────

    def _img2canvas(self, x, y):
        return x * self.zoom + self.pan[0], y * self.zoom + self.pan[1]

    def _canvas2img(self, cx, cy):
        return (cx - self.pan[0]) / self.zoom, (cy - self.pan[1]) / self.zoom

    # ──────────────────────────────────────────
    # Drawing
    # ──────────────────────────────────────────

    def _redraw(self):
        self.canvas.delete('all')
        display = getattr(self, '_display_pil', self.img_pil)
        if display is None:
            return

        # Render image at zoom
        w = int(display.width * self.zoom)
        h = int(display.height * self.zoom)
        resized = display.resize((w, h), Image.LANCZOS)
        self._tk_img = ImageTk.PhotoImage(resized)
        self.canvas.create_image(self.pan[0], self.pan[1], anchor=tk.NW,
                                  image=self._tk_img)

        # Draw slots
        for i, s in enumerate(self.slots):
            self._draw_slot(i, s)

        # Draw in-progress points
        if self.mode == 'draw' and self.draw_pts:
            self._draw_progress(self.draw_pts, '#FF6600', "Slot")
        if self.mode == 'grid' and self.grid_pts:
            labels = ['TL', 'TR', 'BR', 'BL']
            self._draw_progress(self.grid_pts, '#FF0066', "Grid", labels)

    def _draw_slot(self, idx, s: Slot):
        is_sel = (idx == self.selected)
        spts = [self._img2canvas(p[0], p[1]) for p in s.pts]

        # Fill color
        if s.prediction == 1:
            fill = '#cc3333'
            outline = '#ff4444'
        elif s.prediction == 0:
            fill = '#33aa33'
            outline = '#44ff44'
        elif is_sel:
            fill = '#2266cc'
            outline = '#44aaff'
        else:
            fill = '#cc8800'
            outline = '#ffaa00'

        # Polygon fill (semi-transparent via stipple)
        flat = [coord for pt in spts for coord in pt]
        self.canvas.create_polygon(flat, fill=fill, outline='',
                                    stipple='gray25', tags=f'slot_{idx}')
        # Border
        lw = 3 if is_sel else 2
        self.canvas.create_polygon(flat, fill='', outline=outline,
                                    width=lw, tags=f'slot_{idx}')

        # Corner handles
        for j, (cx, cy) in enumerate(spts):
            r = 6 if is_sel else 4
            color = '#ff4444' if is_sel else '#ffcc00'
            self.canvas.create_oval(cx - r, cy - r, cx + r, cy + r,
                                     fill=color, outline='white', width=1,
                                     tags=f'corner_{idx}_{j}')

        # Label at center
        ccx, ccy = s.center
        scx, scy = self._img2canvas(ccx, ccy)
        label = s.label or f'#{idx}'
        if s.prediction >= 0:
            status = 'OCC' if s.prediction == 1 else 'FREE'
            label += f'\n{s.mad_value:.1f} ({status})'

        self.canvas.create_text(scx, scy, text=label, fill='white',
                                 font=('Consolas', 9, 'bold'),
                                 tags=f'label_{idx}')

    def _draw_progress(self, pts, color, prefix, labels=None):
        """Draw in-progress corner clicks."""
        for j, p in enumerate(pts):
            cx, cy = self._img2canvas(p[0], p[1])
            self.canvas.create_oval(cx - 8, cy - 8, cx + 8, cy + 8,
                                     fill=color, outline='white', width=2)
            lbl = labels[j] if labels and j < len(labels) else f'{j+1}'
            self.canvas.create_text(cx + 14, cy - 10, text=lbl,
                                     fill=color, font=('Segoe UI', 10, 'bold'))
            if j > 0:
                px, py = self._img2canvas(pts[j-1][0], pts[j-1][1])
                self.canvas.create_line(px, py, cx, cy, fill=color,
                                         width=2, dash=(6, 3))

        need = 4
        placed = len(pts)
        self.status_var.set(
            f"{prefix}: {placed}/{need} corners placed — "
            f"{'Click next corner' if placed < need else 'Complete!'} "
            f"| Right-click to undo last point")

    # ──────────────────────────────────────────
    # Mouse handlers
    # ──────────────────────────────────────────

    def _on_click(self, event):
        if self.img_cv is None:
            return
        ix, iy = self._canvas2img(event.x, event.y)
        ix, iy = int(ix), int(iy)

        if self.mode == 'draw':
            self.draw_pts.append([ix, iy])
            if len(self.draw_pts) == 4:
                self._snapshot()
                slot = Slot(pts=self.draw_pts.copy(),
                            label=f'S{len(self.slots)}',
                            slot_idx=len(self.slots))
                self.slots.append(slot)
                self.selected = len(self.slots) - 1
                self.draw_pts.clear()
                self._refresh_list()
                self.status_var.set(
                    f"Created slot #{len(self.slots)-1} — "
                    f"Click 4 more corners for next slot, or switch to Edit mode")
            self._redraw()

        elif self.mode == 'edit':
            # Check corners of selected slot first
            if self.selected >= 0:
                s = self.slots[self.selected]
                ci = s.nearest_corner(ix, iy, int(15 / self.zoom))
                if ci >= 0:
                    self._snapshot()
                    self.dragging_corner = ci
                    return

            # Check all slots
            for i in range(len(self.slots) - 1, -1, -1):
                s = self.slots[i]
                ci = s.nearest_corner(ix, iy, int(15 / self.zoom))
                if ci >= 0:
                    self.selected = i
                    self._snapshot()
                    self.dragging_corner = ci
                    self._refresh_list()
                    self._redraw()
                    return
                if s.contains(ix, iy):
                    self.selected = i
                    self._snapshot()
                    self.dragging_slot = True
                    self.drag_start = (ix, iy)
                    self._refresh_list()
                    self._redraw()
                    return

            self.selected = -1
            self._refresh_list()
            self._redraw()

        elif self.mode == 'grid':
            if len(self.grid_pts) < 4:
                self.grid_pts.append([ix, iy])
                if len(self.grid_pts) == 4:
                    self._generate_grid()
            self._redraw()

    def _on_drag(self, event):
        if self.img_cv is None or self.mode != 'edit':
            return
        ix, iy = self._canvas2img(event.x, event.y)
        h, w = self.img_cv.shape[:2]
        ix = max(0, min(w - 1, int(ix)))
        iy = max(0, min(h - 1, int(iy)))

        if self.dragging_corner >= 0 and self.selected >= 0:
            self.slots[self.selected].pts[self.dragging_corner] = [ix, iy]
            self._redraw()
        elif self.dragging_slot and self.selected >= 0:
            s = self.slots[self.selected]
            dx = ix - self.drag_start[0]
            dy = iy - self.drag_start[1]
            for p in s.pts:
                p[0] = max(0, min(w - 1, p[0] + dx))
                p[1] = max(0, min(h - 1, p[1] + dy))
            self.drag_start = (ix, iy)
            self._redraw()

    def _on_release(self, event):
        if self.dragging_corner >= 0 or self.dragging_slot:
            self._refresh_list()
        self.dragging_corner = -1
        self.dragging_slot = False

    def _on_right_click(self, event):
        if self.mode == 'draw' and self.draw_pts:
            self.draw_pts.pop()
            self._redraw()
        elif self.mode == 'grid' and self.grid_pts:
            self.grid_pts.pop()
            self._redraw()
        elif self.mode == 'edit':
            self.selected = -1
            self._refresh_list()
            self._redraw()

    def _on_motion(self, event):
        if self.img_cv is None:
            return
        ix, iy = self._canvas2img(event.x, event.y)
        h, w = self.img_cv.shape[:2]
        if 0 <= ix < w and 0 <= iy < h:
            if self.mode == 'edit' and self.selected < 0:
                self.status_var.set(
                    f"Edit mode — Click a slot to select | "
                    f"Pos: ({int(ix)}, {int(iy)})")

    def _on_scroll(self, event):
        if event.delta > 0:
            self._zoom_in()
        else:
            self._zoom_out()

    def _on_list_select(self, event):
        sel = self.slot_list.curselection()
        if sel:
            self.selected = sel[0]
            self._redraw()

    # ──────────────────────────────────────────
    # Mode switching
    # ──────────────────────────────────────────

    def _set_mode(self, mode):
        self.mode = mode
        self.draw_pts.clear()
        self.grid_pts.clear()
        self.dragging_corner = -1
        self.dragging_slot = False

        messages = {
            'draw': "DRAW MODE — Click 4 corners (clockwise or counter-clockwise) to create a slot",
            'edit': "EDIT MODE — Click to select, drag corners to reshape, drag body to move",
            'grid': f"GRID MODE — Click 4 corners of parking area (TL → TR → BR → BL), "
                    f"then auto-generate {self.grid_rows.get()}x{self.grid_cols.get()} grid",
        }
        self.status_var.set(messages.get(mode, ''))
        self._redraw()

    def _cancel(self):
        self.draw_pts.clear()
        self.grid_pts.clear()
        self.selected = -1
        self._redraw()

    # ──────────────────────────────────────────
    # Zoom
    # ──────────────────────────────────────────

    def _zoom_in(self):
        self.zoom = min(5.0, self.zoom * 1.2)
        self._redraw()

    def _zoom_out(self):
        self.zoom = max(0.05, self.zoom / 1.2)
        self._redraw()

    def _zoom_fit(self):
        if self.img_cv is None:
            return
        self.canvas.update_idletasks()
        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        h, w = self.img_cv.shape[:2]
        self.zoom = min(cw / max(w, 1), ch / max(h, 1)) * 0.95
        self.pan = [0, 0]
        self._redraw()

    # ──────────────────────────────────────────
    # Slot operations
    # ──────────────────────────────────────────

    def _snapshot(self):
        self.undo_stack.append(copy.deepcopy(self.slots))
        if len(self.undo_stack) > 50:
            self.undo_stack.pop(0)

    def _undo(self):
        if self.undo_stack:
            self.slots = self.undo_stack.pop()
            self.selected = -1
            self._refresh_list()
            self._redraw()
            self.status_var.set("Undo")

    def _delete_selected(self):
        if self.selected >= 0 and self.selected < len(self.slots):
            self._snapshot()
            label = self.slots[self.selected].label
            del self.slots[self.selected]
            self.selected = -1
            self._refresh_list()
            self._redraw()
            self.status_var.set(f"Deleted slot {label}")

    def _duplicate(self):
        if self.selected >= 0:
            self._snapshot()
            s = self.slots[self.selected]
            new_pts = [[p[0] + 30, p[1] + 30] for p in s.pts]
            dup = Slot(pts=new_pts, label=f'S{len(self.slots)}',
                       slot_idx=len(self.slots))
            self.slots.append(dup)
            self.selected = len(self.slots) - 1
            self._refresh_list()
            self._redraw()

    def _clear_all(self):
        if self.slots and messagebox.askyesno("Clear All",
                "Delete all slots? This cannot be undone."):
            self._snapshot()
            self.slots.clear()
            self.selected = -1
            self._refresh_list()
            self._redraw()

    def _auto_number(self):
        if not self.slots:
            return
        self._snapshot()
        centers = [(s, s.center[1], s.center[0]) for s in self.slots]
        ys = [cy for _, cy, _ in centers]
        y_mid = (min(ys) + max(ys)) / 2

        top = sorted([(s, cx) for s, cy, cx in centers if cy < y_mid],
                     key=lambda t: t[1])
        bot = sorted([(s, cx) for s, cy, cx in centers if cy >= y_mid],
                     key=lambda t: t[1])

        for i, (s, _) in enumerate(top):
            s.label = f'T{i}'
            s.slot_idx = i
        for i, (s, _) in enumerate(bot):
            s.label = f'B{i}'
            s.slot_idx = len(top) + i

        self._refresh_list()
        self._redraw()
        self.status_var.set(f"Numbered: {len(top)} top + {len(bot)} bottom = {len(self.slots)}")

    # ──────────────────────────────────────────
    # Grid generation
    # ──────────────────────────────────────────

    def _generate_grid(self):
        if len(self.grid_pts) != 4:
            return
        self._snapshot()
        tl, tr, br, bl = [np.array(p, dtype=np.float64) for p in self.grid_pts]
        rows = self.grid_rows.get()
        cols = self.grid_cols.get()

        self.slots.clear()
        for r in range(rows):
            t0, t1 = r / rows, (r + 1) / rows
            row_tl = tl + (bl - tl) * t0
            row_tr = tr + (br - tr) * t0
            row_bl = tl + (bl - tl) * t1
            row_br = tr + (br - tr) * t1

            for c in range(cols):
                s0, s1 = c / cols, (c + 1) / cols
                p0 = row_tl + (row_tr - row_tl) * s0
                p1 = row_tl + (row_tr - row_tl) * s1
                p2 = row_bl + (row_br - row_bl) * s1
                p3 = row_bl + (row_br - row_bl) * s0

                pts = [[int(p0[0]), int(p0[1])], [int(p1[0]), int(p1[1])],
                       [int(p2[0]), int(p2[1])], [int(p3[0]), int(p3[1])]]
                prefix = 'T' if r == 0 else 'B' if r == 1 else f'R{r}'
                self.slots.append(Slot(pts=pts, label=f'{prefix}{c}',
                                        slot_idx=r * cols + c))

        self.grid_pts.clear()
        self.mode = 'edit'
        self._refresh_list()
        self._redraw()
        self.status_var.set(
            f"Generated {rows}x{cols} = {len(self.slots)} perspective slots — "
            f"Switch to Edit mode to adjust individual corners")

    # ──────────────────────────────────────────
    # Classification
    # ──────────────────────────────────────────

    def _set_ref(self):
        path = filedialog.askopenfilename(
            title="Select Reference Image (empty lot)",
            filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp"), ("All", "*.*")])
        if path:
            self.ref_path = path
            self.ref_label.configure(text=f"Ref: {os.path.basename(path)}")

    def _set_test(self):
        path = filedialog.askopenfilename(
            title="Select Test Image (with cars)",
            filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp"), ("All", "*.*")])
        if path:
            self._load_image(path, is_test=True)

    def _classify(self):
        if not self.ref_path or not self.test_path:
            messagebox.showinfo("Classification",
                "Please set both Reference and Test images first.")
            return
        if not self.slots:
            messagebox.showinfo("Classification", "No slots defined.")
            return

        ref = cv2.imread(self.ref_path, cv2.IMREAD_GRAYSCALE)
        test = cv2.imread(self.test_path, cv2.IMREAD_GRAYSCALE)
        if ref is None or test is None:
            messagebox.showerror("Error", "Cannot load ref/test images")
            return

        results = []
        for s in self.slots:
            ref_roi = self._warp_roi(ref, s)
            test_roi = self._warp_roi(test, s)
            if ref_roi is None or test_roi is None:
                s.prediction = -1
                continue

            diff_sum = int(np.sum(np.abs(test_roi.astype(np.int32) -
                                          ref_roi.astype(np.int32))))
            mad_x10 = diff_sum * 10 // (ROI_SIZE * ROI_SIZE)
            s.mad_value = mad_x10 / 10.0
            s.prediction = 1 if mad_x10 > 77 else 0
            s.confidence = min(100, abs(mad_x10 - 77) * 100 // 77) / 100.0
            status = 'OCC' if s.prediction else 'FREE'
            results.append(f"{s.label}: MAD={s.mad_value:.1f} → {status} "
                          f"(conf={s.confidence*100:.0f}%)")

        self._refresh_list()
        self._redraw()

        occ = sum(1 for s in self.slots if s.prediction == 1)
        free = sum(1 for s in self.slots if s.prediction == 0)
        self.status_var.set(
            f"Classification done: {occ} occupied, {free} free "
            f"(threshold τ=7.68)")

        # Show results in popup
        msg = f"Results ({len(self.slots)} slots):\n\n"
        msg += '\n'.join(results)
        msg += f"\n\nSummary: {occ} OCC / {free} FREE"
        messagebox.showinfo("MAD Classification Results", msg)

    def _warp_roi(self, gray, s):
        src = np.array(s.pts, dtype=np.float32)
        dst = np.array([[0, 0], [ROI_SIZE-1, 0],
                        [ROI_SIZE-1, ROI_SIZE-1], [0, ROI_SIZE-1]],
                       dtype=np.float32)
        try:
            M = cv2.getPerspectiveTransform(src, dst)
            return cv2.warpPerspective(gray, M, (ROI_SIZE, ROI_SIZE))
        except Exception:
            return None

    def _quick_classify(self):
        """One-click: pick test image → show it → classify → display results."""
        if not self.ref_path:
            messagebox.showinfo("Quick Classify",
                "Open a reference image (empty lot) first, "
                "then use this button to load a test image and classify.")
            return
        if not self.slots:
            messagebox.showinfo("Quick Classify", "No slots defined yet.")
            return

        path = filedialog.askopenfilename(
            title="Select Test Image (with cars) — will auto-classify",
            filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp"), ("All", "*.*")])
        if not path:
            return

        # Load test image and display it
        self._load_image(path, is_test=True)
        # Run classification immediately
        self._classify()

    def _show_ref_image(self):
        """Switch canvas to show the reference image."""
        if self.img_pil is None:
            return
        self._display_pil = self.img_pil
        self._redraw()
        self.status_var.set(f"Showing REFERENCE image: {os.path.basename(self.ref_path or '')}")

    def _show_test_image(self):
        """Switch canvas to show the test image."""
        if not self.test_path:
            messagebox.showinfo("View Test", "No test image loaded yet.")
            return
        test_cv = cv2.imread(self.test_path)
        if test_cv is not None:
            self._display_pil = Image.fromarray(cv2.cvtColor(test_cv, cv2.COLOR_BGR2RGB))
            self._redraw()
            self.status_var.set(f"Showing TEST image: {os.path.basename(self.test_path)}")

    # ──────────────────────────────────────────
    # Save / Export
    # ──────────────────────────────────────────

    def _save(self):
        if not self.slots:
            messagebox.showinfo("Save", "No slots to save.")
            return

        base = os.path.splitext(os.path.basename(self.img_path or 'untitled'))[0]
        default_dir = os.path.dirname(os.path.abspath(self.img_path or '.'))

        path = filedialog.asksaveasfilename(
            title="Save ROI Config",
            initialdir=default_dir,
            initialfile=f'roi_config_{base}.json',
            filetypes=[("JSON", "*.json")],
            defaultextension='.json')
        if not path:
            return

        base_out = os.path.splitext(path)[0]

        # JSON
        data = {
            'image': os.path.basename(self.img_path or ''),
            'image_size': [self.img_cv.shape[1], self.img_cv.shape[0]]
                          if self.img_cv is not None else [0, 0],
            'roi_size': ROI_SIZE,
            'n_slots': len(self.slots),
            'rois': [s.to_dict() for s in self.slots]
        }
        with open(path, 'w') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        # C header
        cpath = base_out + '.h'
        with open(cpath, 'w') as f:
            img_name = os.path.basename(self.img_path or 'unknown')
            w = self.img_cv.shape[1] if self.img_cv is not None else 0
            h = self.img_cv.shape[0] if self.img_cv is not None else 0
            f.write(f'// Auto-generated ROI config — {img_name}\n')
            f.write(f'// Image: {w}x{h}, Slots: {len(self.slots)}\n\n')
            f.write(f'#define N_SLOTS {len(self.slots)}\n\n')
            f.write('static roi_rect_t slot_rois[N_SLOTS] = {\n')
            for s in self.slots:
                r = s.bounding_rect()
                lbl = f' // {s.label}' if s.label else ''
                f.write(f'    {{ .x = {r[0]:4d}, .y = {r[1]:4d}, '
                        f'.w = {r[2]:4d}, .h = {r[3]:4d} }},{lbl}\n')
            f.write('};\n\n')
            f.write('/* Perspective polygon corners:\n')
            for s in self.slots:
                f.write(f'   {s.label}: {s.pts}\n')
            f.write('*/\n')

        # Python
        pypath = base_out + '.py'
        with open(pypath, 'w') as f:
            f.write(f'# Auto-generated ROI config\n\n')
            f.write('SLOT_POLYGONS = [\n')
            for s in self.slots:
                f.write(f'    {{"label": "{s.label}", "pts": {s.pts}}},\n')
            f.write(']\n')

        self.status_var.set(
            f"Saved: {os.path.basename(path)}, "
            f"{os.path.basename(cpath)}, {os.path.basename(pypath)}")
        messagebox.showinfo("Saved",
            f"Config saved:\n\n"
            f"  JSON: {os.path.basename(path)}\n"
            f"  C:       {os.path.basename(cpath)}\n"
            f"  Python: {os.path.basename(pypath)}\n\n"
            f"({len(self.slots)} slots)")

    # ──────────────────────────────────────────
    # Slot list refresh
    # ──────────────────────────────────────────

    def _refresh_list(self):
        self.slot_list.delete(0, tk.END)
        for i, s in enumerate(self.slots):
            line = f"{'>' if i == self.selected else ' '} "
            line += f"{s.label or f'#{i}':5s}"
            line += f"  area={s.area:6.0f}"
            if s.prediction == 1:
                line += f"  OCC  MAD={s.mad_value:.1f}"
            elif s.prediction == 0:
                line += f"  FREE MAD={s.mad_value:.1f}"
            self.slot_list.insert(tk.END, line)

        if 0 <= self.selected < len(self.slots):
            self.slot_list.selection_set(self.selected)
            self.slot_list.see(self.selected)


# ══════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════

def main():
    import argparse
    parser = argparse.ArgumentParser(description='ParkingLite ROI Calibrator')
    parser.add_argument('--image', '-i', help='Image to open on start')
    args, _ = parser.parse_known_args()

    # Also accept positional argument
    if not args.image and len(sys.argv) > 1 and not sys.argv[1].startswith('-'):
        args.image = sys.argv[1]

    app = App(image_path=args.image)
    app.mainloop()


if __name__ == '__main__':
    main()

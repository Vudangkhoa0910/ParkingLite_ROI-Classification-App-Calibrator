import copy
import math
import os
import json
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog
from typing import List, Tuple
from PIL import Image, ImageTk
import numpy as np
import cv2

from constants import ROI_SIZE, LINE_FOCUS_THRESHOLD, SSIM_THRESHOLD, DIFF_PIXEL_PERCENT, EDGE_DIFF_PERCENT
from models import Slot
import config_io
import classification
import grid_helpers


def build_ui(app):
    style = ttk.Style()
    style.theme_use('clam')
    style.configure('TFrame', background='#1e1e2e')
    style.configure('TLabel', background='#1e1e2e', foreground='#cdd6f4',
                    font=('Segoe UI', 10))
    style.configure('TButton', font=('Segoe UI', 9), padding=4)
    style.configure('Header.TLabel', font=('Segoe UI', 11, 'bold'),
                    foreground='#89b4fa')
    style.configure('Mode.TButton', font=('Segoe UI', 10, 'bold'), padding=6,
                    background='#2e2e3f', foreground='#cdd6f4')
    style.configure('ActiveMode.TButton', font=('Segoe UI', 10, 'bold'), padding=6,
                    background='#89b4fa', foreground='#1e1e2e')
    style.map('ActiveMode.TButton', background=[('active', '#7fa2ee')])
    style.configure('Action.TButton', font=('Segoe UI', 10), padding=5)
    style.configure('Accent.TButton', font=('Segoe UI', 10, 'bold'), padding=6)

    top = ttk.Frame(app, style='TFrame')
    top.pack(fill=tk.X, padx=5, pady=(5, 0))

    ttk.Button(top, text="Open Image", command=app._open_image,
               style='Action.TButton').pack(side=tk.LEFT, padx=2)
    ttk.Button(top, text="Load Config", command=app._open_config,
               style='Action.TButton').pack(side=tk.LEFT, padx=2)

    ttk.Separator(top, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y,
                                                  padx=8, pady=2)

    app.btn_draw = ttk.Button(top, text="Draw Slot",
                                command=lambda: app._set_mode('draw'),
                                style='Mode.TButton')
    app.btn_draw.pack(side=tk.LEFT, padx=2)
    app.btn_edit = ttk.Button(top, text="Edit / Move",
                                command=lambda: app._set_mode('edit'),
                                style='Mode.TButton')
    app.btn_edit.pack(side=tk.LEFT, padx=2)
    app.btn_select = ttk.Button(top, text="Select Grid",
                                  command=lambda: app._set_mode('select'),
                                  style='Mode.TButton')
    app.btn_select.pack(side=tk.LEFT, padx=2)
    app.btn_grid = ttk.Button(top, text="Auto Grid",
                                command=lambda: app._set_mode('grid'),
                                style='Mode.TButton')
    app.btn_grid.pack(side=tk.LEFT, padx=2)

    ttk.Separator(top, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y,
                                                  padx=8, pady=2)

    ttk.Button(top, text="Zoom +", command=app._zoom_in,
               style='TButton').pack(side=tk.LEFT, padx=1)
    ttk.Button(top, text="Zoom -", command=app._zoom_out,
               style='TButton').pack(side=tk.LEFT, padx=1)
    ttk.Button(top, text="Fit", command=app._zoom_fit,
               style='TButton').pack(side=tk.LEFT, padx=1)

    ttk.Separator(top, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y,
                                                  padx=8, pady=2)

    ttk.Button(top, text="Undo", command=app._undo,
               style='TButton').pack(side=tk.LEFT, padx=2)

    main = ttk.Frame(app, style='TFrame')
    main.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

    canvas_frame = ttk.Frame(main, style='TFrame')
    canvas_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    app.canvas = tk.Canvas(canvas_frame, bg='#11111b', highlightthickness=0,
                            cursor='crosshair')
    app.canvas.pack(fill=tk.BOTH, expand=True)

    right = ttk.Frame(main, style='TFrame', width=300)
    right.pack(side=tk.RIGHT, fill=tk.Y, padx=(5, 0))
    right.pack_propagate(False)

    build_right_panel(app, right)

    app.status_var = tk.StringVar(value="Open an image to start.")
    status = ttk.Label(app, textvariable=app.status_var,
                       font=('Consolas', 9), foreground='#a6adc8')
    status.pack(fill=tk.X, padx=5, pady=(0, 3))


def build_right_panel(app, parent):
    ttk.Label(parent, text="ACTIONS", style='Header.TLabel').pack(
        anchor=tk.W, pady=(5, 3))

    btn_frame = ttk.Frame(parent, style='TFrame')
    btn_frame.pack(fill=tk.X, pady=2)

    ttk.Button(btn_frame, text="Auto Number (N)",
               command=app._auto_number, style='Action.TButton').pack(
                   fill=tk.X, pady=1)
    ttk.Button(btn_frame, text="Delete Selected (Del)",
               command=app._delete_selected, style='Action.TButton').pack(
                   fill=tk.X, pady=1)
    ttk.Button(btn_frame, text="Duplicate (D)",
               command=app._duplicate, style='Action.TButton').pack(
                   fill=tk.X, pady=1)
    ttk.Button(btn_frame, text="Clear All",
               command=app._clear_all, style='Action.TButton').pack(
                   fill=tk.X, pady=1)

    ttk.Separator(parent, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=8)

    ttk.Label(parent, text="GRID SETTINGS", style='Header.TLabel').pack(
        anchor=tk.W, pady=(0, 3))
    grid_f = ttk.Frame(parent, style='TFrame')
    grid_f.pack(fill=tk.X, pady=2)

    ttk.Label(grid_f, text="Rows:").pack(side=tk.LEFT)
    ttk.Spinbox(grid_f, from_=1, to=8, width=4,
                 textvariable=app.grid_rows).pack(side=tk.LEFT, padx=4)
    ttk.Label(grid_f, text="Cols:").pack(side=tk.LEFT, padx=(8, 0))
    ttk.Spinbox(grid_f, from_=1, to=16, width=4,
                 textvariable=app.grid_cols).pack(side=tk.LEFT, padx=4)

    ttk.Separator(parent, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=8)

    ttk.Label(parent, text="CLASSIFY", style='Header.TLabel').pack(
        anchor=tk.W, pady=(0, 3))

    ref_f = ttk.Frame(parent, style='TFrame')
    ref_f.pack(fill=tk.X, pady=1)
    ttk.Button(ref_f, text="Set Reference Image",
               command=app._set_ref).pack(fill=tk.X)

    test_f = ttk.Frame(parent, style='TFrame')
    test_f.pack(fill=tk.X, pady=1)
    ttk.Button(test_f, text="Set Test Image",
               command=app._set_test).pack(fill=tk.X)

    app.ref_label = ttk.Label(parent, text="Ref: (none)",
                                font=('Consolas', 8))
    app.ref_label.pack(anchor=tk.W)
    app.test_label = ttk.Label(parent, text="Test: (none)",
                                 font=('Consolas', 8))
    app.test_label.pack(anchor=tk.W)

    ttk.Button(parent, text="Run MAD Classification",
               command=app._classify, style='Accent.TButton').pack(
                   fill=tk.X, pady=2)

    ttk.Button(parent, text="Quick: Load Test & Classify",
               command=app._quick_classify, style='Accent.TButton').pack(
                   fill=tk.X, pady=2)

    view_f = ttk.Frame(parent, style='TFrame')
    view_f.pack(fill=tk.X, pady=2)
    ttk.Button(view_f, text="View: Reference",
               command=app._show_ref_image).pack(side=tk.LEFT, expand=True,
                                                   fill=tk.X, padx=(0, 2))
    ttk.Button(view_f, text="View: Test",
               command=app._show_test_image).pack(side=tk.LEFT, expand=True,
                                                    fill=tk.X, padx=(2, 0))

    ttk.Separator(parent, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=8)

    ttk.Label(parent, text="EXPORT", style='Header.TLabel').pack(
        anchor=tk.W, pady=(0, 3))
    ttk.Button(parent, text="Save Config (JSON + C + Python)",
               command=app._save, style='Accent.TButton').pack(
                   fill=tk.X, pady=2)

    ttk.Separator(parent, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=8)

    ttk.Label(parent, text="SLOT LIST", style='Header.TLabel').pack(
        anchor=tk.W, pady=(0, 3))

    list_frame = ttk.Frame(parent, style='TFrame')
    list_frame.pack(fill=tk.BOTH, expand=True)

    app.slot_list = tk.Listbox(
        list_frame, bg='#181825', fg='#cdd6f4', selectbackground='#45475a',
        font=('Consolas', 9), borderwidth=0, highlightthickness=0)
    scrollbar = ttk.Scrollbar(list_frame, command=app.slot_list.yview)
    app.slot_list.configure(yscrollcommand=scrollbar.set)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    app.slot_list.pack(fill=tk.BOTH, expand=True)
    app.slot_list.bind('<<ListboxSelect>>', app._on_list_select)


def bind_keys(app):
    app.bind('<Delete>', lambda e: app._delete_selected())
    app.bind('<BackSpace>', lambda e: app._delete_selected())
    app.bind('<Control-d>', lambda e: app._duplicate())
    app.bind('<Control-z>', lambda e: app._undo())
    app.bind('<Control-Z>', lambda e: app._undo())
    app.bind('<Key-D>', lambda e: app._set_mode('draw'))
    app.bind('<Key-d>', lambda e: app._set_mode('draw'))
    app.bind('<Key-E>', lambda e: app._set_mode('edit'))
    app.bind('<Key-e>', lambda e: app._set_mode('edit'))
    app.bind('<Key-n>', lambda e: app._auto_number())
    app.bind('<Key-s>', lambda e: app._save())
    app.bind('<Key-r>', lambda e: app._classify())
    app.bind('<Key-u>', lambda e: app._undo())
    app.bind('<Key-1>', lambda e: app._set_mode('draw'))
    app.bind('<Key-2>', lambda e: app._set_mode('edit'))
    app.bind('<Key-3>', lambda e: app._set_mode('grid'))
    app.bind('<Key-4>', lambda e: app._set_mode('select'))
    app.bind('<Escape>', lambda e: app._cancel())
    app.bind('<plus>', lambda e: app._zoom_in())
    app.bind('<equal>', lambda e: app._zoom_in())
    app.bind('<minus>', lambda e: app._zoom_out())
    app.bind('<Key-f>', lambda e: app._zoom_fit())

    app.canvas.bind('<Button-1>', app._on_click)
    app.canvas.bind('<B1-Motion>', app._on_drag)
    app.canvas.bind('<ButtonRelease-1>', app._on_release)
    app.canvas.bind('<Button-3>', app._on_right_click)
    app.canvas.bind('<Motion>', app._on_motion)
    app.canvas.bind('<MouseWheel>', app._on_scroll)
    app.canvas.bind('<Button-4>', lambda e: app._zoom_in())
    app.canvas.bind('<Button-5>', lambda e: app._zoom_out())


def _open_image(app):
    path = filedialog.askopenfilename(
        title="Open Parking Lot Image",
        filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp *.tiff"),
                   ("All", "*.*")])
    if path:
        _load_image(app, path)


def _load_image(app, path, is_test=False):
    img = cv2.imread(path)
    if img is None:
        messagebox.showerror("Error", f"Cannot load: {path}")
        return False
    if is_test:
        app.test_path = path
        app.test_label.configure(text=f"Test: {os.path.basename(path)}")
        if app.ref_cv is not None and app.ref_path != path:
            if app.ref_cv.shape[:2] != img.shape[:2]:
                img = cv2.resize(img, (app.ref_cv.shape[1], app.ref_cv.shape[0]),
                                 interpolation=cv2.INTER_LINEAR)
                app.status_var.set(
                    f"Resized TEST image to match Reference size {app.ref_cv.shape[1]}x{app.ref_cv.shape[0]}")
        app.test_cv = img
        app.test_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        app._display_cv = app.test_cv
        app._display_pil = app.test_pil
        if not app.status_var.get().startswith("Resized TEST"):
            app.status_var.set(f"Showing TEST image: {os.path.basename(path)} — "
                                f"Click 'Run MAD Classification' or press R")
    else:
        app.img_cv = img
        app.img_path = path
        app.img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        app.ref_cv = img
        app.ref_pil = app.img_pil
        app._display_cv = img
        app._display_pil = app.img_pil
        app._update_detected_lines()
        app.ref_path = path
        app.ref_label.configure(text=f"Ref: {os.path.basename(path)}")
        app.slots.clear()
        if app._load_auto_config(path):
            app.status_var.set(
                f"Loaded cached slot config for {os.path.basename(path)}")
        else:
            h, w = img.shape[:2]
            app.status_var.set(
                f"Image: {w}x{h} — {os.path.basename(path)} "
                f"(auto-set as Reference)")
    app._zoom_fit()
    app._redraw()
    return True


def _open_config(app):
    path = filedialog.askopenfilename(
        title="Load ROI Config",
        filetypes=[("JSON", "*.json"), ("All", "*.*")])
    if not path:
        return
    try:
        data = config_io.load_config_file(path)
        app.slots, app.grid_groups, app.active_grid_group, app.grid_bounds = \
            config_io.load_from_config_data(data)
        app.selected = -1
        app._refresh_list()
        app._redraw()
        app.status_var.set(f"Loaded {len(app.slots)} slots from {os.path.basename(path)}")
        app._save_auto_config()
    except Exception as e:
        messagebox.showerror("Error", str(e))


def _load_auto_config(app, image_path=None):
    config_path = config_io.auto_config_path(image_path, app.ref_path, app.img_path)
    if not config_path:
        return False
    data = config_io.load_auto_config(config_path)
    if not data:
        return False
    app.slots, app.grid_groups, app.active_grid_group, app.grid_bounds = \
        config_io.load_from_config_data(data)
    app.selected = -1
    app._refresh_list()
    app._redraw()
    app.status_var.set(f"Loaded auto config from {os.path.basename(config_path)}")
    return True


def _save_auto_config(app):
    config_path = config_io.auto_config_path(None, app.ref_path, app.img_path)
    config_io.save_auto_config(config_path, app.img_cv, app.ref_path,
                               app.img_path, app.slots, app.grid_groups,
                               app.active_grid_group)


def compute_slot_diff(ref_roi, test_roi):
    return classification.compute_slot_diff(ref_roi, test_roi)


def img2canvas(app, x, y):
    return x * app.zoom + app.pan[0], y * app.zoom + app.pan[1]


def canvas2img(app, cx, cy):
    return (cx - app.pan[0]) / app.zoom, (cy - app.pan[1]) / app.zoom


def redraw(app):
    app.canvas.delete('all')
    display = getattr(app, '_display_pil', app.img_pil)
    if display is None:
        return

    w = int(display.width * app.zoom)
    h = int(display.height * app.zoom)
    resized = display.resize((w, h), Image.LANCZOS)
    app._tk_img = ImageTk.PhotoImage(resized)
    app.canvas.create_image(app.pan[0], app.pan[1], anchor=tk.NW,
                             image=app._tk_img)

    for i, s in enumerate(app.slots):
        draw_slot(app, i, s)

    if app.preview_slot is not None:
        draw_preview_slot(app, app.preview_slot)

    if app.grid_groups:
        draw_grid_bounds(app)

    if app.hovered_edge >= 0 and app.hovered_slot >= 0:
        draw_hover_edge(app, app.slots[app.hovered_slot], app.hovered_edge)
    elif app.hovered_detected_line >= 0:
        draw_hover_detected_line(app, app.hovered_detected_line)

    if app.hovered_grid_edge >= 0:
        draw_hover_grid_edge(app, app.hovered_grid_edge)

    if app.mode == 'draw' and app.draw_preview_pt is not None:
        draw_snap_preview(app, app.draw_preview_pt)

    if app.mode == 'draw' and app.draw_pts:
        app.draw_progress(app.draw_pts, '#FF6600', "Slot")
    if app.mode == 'grid' and app.grid_pts:
        labels = ['TL', 'TR', 'BR', 'BL']
        draw_progress(app, app.grid_pts, '#FF0066', "Grid", labels)
    if app.mode == 'select' and app.select_drag_box is not None:
        (x0, y0), (x1, y1) = app.select_drag_box
        a = img2canvas(app, x0, y0)
        b = img2canvas(app, x1, y0)
        c = img2canvas(app, x1, y1)
        d = img2canvas(app, x0, y1)
        app.canvas.create_polygon([a[0], a[1], b[0], b[1], c[0], c[1], d[0], d[1]],
                                  outline='#ffffff', fill='', width=2,
                                  dash=(6, 4), tags='select_box')
        app.canvas.create_text((a[0] + c[0]) / 2, a[1] - 14,
                                text='Select Grid', fill='#ffffff',
                                font=('Segoe UI', 10, 'bold'), tags='select_box')


def draw_slot(app, idx, s: Slot):
    is_sel = (idx == app.selected)
    spts = [img2canvas(app, p[0], p[1]) for p in s.pts]

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

    flat = [coord for pt in spts for coord in pt]
    app.canvas.create_polygon(flat, fill=fill, outline='',
                                stipple='gray25', tags=f'slot_{idx}')
    lw = 3 if is_sel else 2
    app.canvas.create_polygon(flat, fill='', outline=outline,
                                width=lw, tags=f'slot_{idx}')

    for j, (cx, cy) in enumerate(spts):
        r = 6 if is_sel else 4
        color = '#ff4444' if is_sel else '#ffcc00'
        app.canvas.create_oval(cx - r, cy - r, cx + r, cy + r,
                                 fill=color, outline='white', width=1,
                                 tags=f'corner_{idx}_{j}')

    ccx, ccy = s.center
    scx, scy = img2canvas(app, ccx, ccy)
    label = s.label or f'#{idx}'
    if s.prediction >= 0:
        status = 'OCC' if s.prediction == 1 else 'FREE'
        label += f'\n{s.mad_value:.1f} ({status})'

    app.canvas.create_text(scx, scy, text=label, fill='white',
                             font=('Consolas', 9, 'bold'),
                             tags=f'label_{idx}')


def draw_preview_slot(app, s: Slot):
    spts = [img2canvas(app, p[0], p[1]) for p in s.pts]
    flat = [coord for pt in spts for coord in pt]
    app.canvas.create_polygon(flat, fill='', outline='#00ff88',
                               width=3, dash=(6, 4), tags='preview_slot')


def draw_snap_preview(app, pt):
    cx, cy = img2canvas(app, pt[0], pt[1])
    app.canvas.create_oval(cx - 6, cy - 6, cx + 6, cy + 6,
                            outline='#00ffcc', width=2, dash=(4, 2),
                            fill='', tags='snap_preview')
    app.canvas.create_text(cx + 14, cy - 10, text='LINE',
                            fill='#00ffcc', font=('Segoe UI', 9, 'bold'),
                            tags='snap_preview')


def draw_hover_edge(app, slot: Slot, edge_i: int):
    a = img2canvas(app, *slot.pts[edge_i])
    b = img2canvas(app, *slot.pts[(edge_i + 1) % 4])
    app.canvas.create_line(a[0], a[1], b[0], b[1],
                            fill='#ffff00', width=5,
                            dash=(6, 4), capstyle=tk.ROUND,
                            tags='hover_edge')


def draw_hover_detected_line(app, line_idx: int):
    x1, y1, x2, y2 = app.detected_lines[line_idx]
    a = img2canvas(app, x1, y1)
    b = img2canvas(app, x2, y2)
    app.canvas.create_line(a[0], a[1], b[0], b[1],
                            fill='#00ffcc', width=4,
                            dash=(4, 2), capstyle=tk.ROUND,
                            tags='hover_detected_line')


def draw_hover_grid_edge(app, edge_i: int, bounds=None):
    if bounds is None or len(bounds) != 4:
        return
    a = img2canvas(app, *bounds[edge_i])
    b = img2canvas(app, *bounds[(edge_i + 1) % 4])
    app.canvas.create_line(a[0], a[1], b[0], b[1],
                            fill='#88ff88', width=5,
                            dash=(4, 2), capstyle=tk.ROUND,
                            tags='hover_grid_edge')


def _nearest_grid_edge_in_bounds(app, px: float, py: float, bounds, threshold: int = 12) -> int:
    return grid_helpers.nearest_grid_edge_in_bounds(px, py, bounds, threshold)


def _nearest_grid_corner_in_bounds(app, px: float, py: float, bounds, threshold: int = 12) -> int:
    return grid_helpers.nearest_grid_corner_in_bounds(px, py, bounds, threshold)


def _point_in_polygon(app, px: float, py: float, polygon: List[List[int]]) -> bool:
    return grid_helpers.point_in_polygon(px, py, polygon)


def _find_nearest_grid_handle(app, px: float, py: float, threshold: int = 12):
    return grid_helpers.find_nearest_grid_handle(px, py, app.grid_groups, threshold)


def draw_grid_bounds(app):
    if not app.grid_groups:
        return
    for group_idx, group in enumerate(app.grid_groups):
        bounds = group['bounds']
        if len(bounds) != 4:
            continue
        pts = [img2canvas(app, x, y) for x, y in bounds]
        flat = [coord for pt in pts for coord in pt]
        is_active = group_idx == app.active_grid_group
        is_hovered = group_idx == app.hovered_grid_group
        outline = '#00ff88' if is_active or is_hovered else '#88ccff'
        width = 4 if is_active else 3
        app.canvas.create_polygon(flat, fill='', outline=outline,
                                   width=width, dash=(6, 4), tags='grid_bounds')
        if is_active or is_hovered:
            for idx, (cx, cy) in enumerate(pts):
                r = 6 if idx == app.hovered_grid_corner else 5
                app.canvas.create_oval(cx - r, cy - r, cx + r, cy + r,
                                        fill=outline, outline='white', width=2,
                                        tags=f'grid_corner_{idx}')
                label = ['TL', 'TR', 'BR', 'BL'][idx]
                app.canvas.create_text(cx + 12, cy - 12, text=label,
                                        fill=outline, font=('Segoe UI', 9, 'bold'),
                                        tags='grid_bounds')
            if is_hovered and app.hovered_grid_edge >= 0:
                draw_hover_grid_edge(app, app.hovered_grid_edge, bounds)


def regenerate_grid_for_group(app, group_idx: int):
    if group_idx < 0 or group_idx >= len(app.grid_groups):
        return
    grid_helpers.regenerate_grid_for_group(group_idx, app.grid_groups, app.slots)
    app.grid_bounds = app.grid_groups[group_idx]['bounds']


def make_adjacent_slot(app, slot: Slot, edge_i: int,
                       ix: int, iy: int) -> Optional[Slot]:
    if app.drag_edge_start is None or not slot.pts:
        return None
    pts = np.array(slot.pts, dtype=np.float32)
    a = pts[edge_i]
    b = pts[(edge_i + 1) % 4]
    edge_vec = b - a
    if np.linalg.norm(edge_vec) < 1e-3:
        return None

    start = np.array(app.drag_edge_start, dtype=np.float32)
    delta = np.array([ix, iy], dtype=np.float32) - start
    if np.linalg.norm(delta) < 8:
        return None

    a_off = a + delta
    b_off = b + delta
    new_pts = [
        [int(round(a_off[0])), int(round(a_off[1]))],
        [int(round(b_off[0])), int(round(b_off[1]))],
        [int(round(b[0])), int(round(b[1]))],
        [int(round(a[0])), int(round(a[1]))]
    ]
    return Slot(pts=new_pts, label=f'S{len(app.slots)}',
                slot_idx=len(app.slots))


def draw_progress(app, pts, color, prefix, labels=None):
    for j, p in enumerate(pts):
        cx, cy = img2canvas(app, p[0], p[1])
        app.canvas.create_oval(cx - 8, cy - 8, cx + 8, cy + 8,
                                 fill=color, outline='white', width=2)
        lbl = labels[j] if labels and j < len(labels) else f'{j+1}'
        app.canvas.create_text(cx + 14, cy - 10, text=lbl,
                                 fill=color, font=('Segoe UI', 10, 'bold'))
        if j > 0:
            px, py = img2canvas(app, pts[j-1][0], pts[j-1][1])
            app.canvas.create_line(px, py, cx, cy, fill=color,
                                     width=2, dash=(6, 3))

    need = 4
    placed = len(pts)
    app.status_var.set(
        f"{prefix}: {placed}/{need} corners placed — "
        f"{'Click next corner' if placed < need else 'Complete!'} "
        f"| Right-click to undo last point")


def on_click(app, event):
    if app.img_cv is None:
        return
    ix, iy = canvas2img(app, event.x, event.y)
    ix, iy = int(ix), int(iy)
    ix, iy = classification.snap_to_detected_line(ix, iy, app.detected_lines, LINE_FOCUS_THRESHOLD)

    if app.mode == 'draw':
        app._snapshot()
        app.draw_pts.append([ix, iy])
        if len(app.draw_pts) == 4:
            slot = Slot(pts=app.draw_pts.copy(),
                        label=f'S{len(app.slots)}',
                        slot_idx=len(app.slots))
            app.slots.append(slot)
            app.selected = len(app.slots) - 1
            app.draw_pts.clear()
            app._refresh_list()
            app._save_auto_config()
            app.status_var.set(
                f"Created slot #{len(app.slots)-1} — "
                f"Click 4 more corners for next slot, or switch to Edit mode")
        app._redraw()

    elif app.mode == 'edit':
        if app.grid_groups:
            group_idx, gci, gei = _find_nearest_grid_handle(app, ix, iy, int(15 / app.zoom))
            if group_idx >= 0 and gci >= 0:
                app._snapshot()
                app.dragging_grid_group = group_idx
                app.dragging_grid_corner = gci
                app.active_grid_group = group_idx
                app.grid_bounds = app.grid_groups[group_idx]['bounds']
                app.status_var.set("Drag grid corner to reshape this auto grid.")
                return
            if group_idx >= 0 and gei >= 0:
                app._snapshot()
                app.dragging_grid_group = group_idx
                app.dragging_grid_edge = gei
                app.active_grid_group = group_idx
                app.grid_bounds_orig = copy.deepcopy(app.grid_groups[group_idx]['bounds'])
                app.grid_bounds = copy.deepcopy(app.grid_groups[group_idx]['bounds'])
                app.drag_start = (ix, iy)
                app.status_var.set("Drag grid edge to reshape this auto grid.")
                return
        if app.selected >= 0:
            s = app.slots[app.selected]
            ci = s.nearest_corner(ix, iy, int(15 / app.zoom))
            if ci >= 0:
                app._snapshot()
                app.dragging_corner = ci
                return
            edge_i = s.nearest_edge(ix, iy, int(15 / app.zoom))
            if edge_i >= 0:
                app._snapshot()
                app.dragging_edge = edge_i
                app.drag_edge_start = s.closest_point_on_edge(ix, iy, edge_i)
                app.drag_slot_start = copy.deepcopy(s.pts)
                app.preview_slot = None
                app.status_var.set(
                    "Drag selected edge outward to create an adjacent slot. Release to insert.")
                return

        for i in range(len(app.slots) - 1, -1, -1):
            s = app.slots[i]
            ci = s.nearest_corner(ix, iy, int(15 / app.zoom))
            if ci >= 0:
                app.selected = i
                app._snapshot()
                app.dragging_corner = ci
                app._refresh_list()
                app._redraw()
                return
            if s.contains(ix, iy):
                app.selected = i
                app._snapshot()
                app.dragging_slot = True
                app.drag_start = (ix, iy)
                app._refresh_list()
                app._redraw()
                return

        app.selected = -1
        app._refresh_list()
        app._redraw()

    elif app.mode == 'grid':
        if len(app.grid_pts) < 4:
            app.grid_pts.append([ix, iy])
            if len(app.grid_pts) == 4:
                app._generate_grid()
        app._redraw()
    elif app.mode == 'select':
        if app.grid_groups:
            group_idx, gci, gei = _find_nearest_grid_handle(app, ix, iy, int(15 / app.zoom))
            if group_idx >= 0 and gci >= 0:
                app._snapshot()
                app.dragging_grid_group = group_idx
                app.dragging_grid_corner = gci
                app.active_grid_group = group_idx
                app.hovered_grid_group = group_idx
                app.grid_bounds = app.grid_groups[group_idx]['bounds']
                app.select_drag_start = None
                app.select_drag_box = None
                app.status_var.set("Drag selected grid corner to reshape this auto grid.")
                return
            if group_idx >= 0 and gei >= 0:
                app._snapshot()
                app.dragging_grid_group = group_idx
                app.dragging_grid_edge = gei
                app.active_grid_group = group_idx
                app.hovered_grid_group = group_idx
                app.grid_bounds_orig = copy.deepcopy(app.grid_groups[group_idx]['bounds'])
                app.grid_bounds = copy.deepcopy(app.grid_groups[group_idx]['bounds'])
                app.drag_start = (ix, iy)
                app.select_drag_start = None
                app.select_drag_box = None
                app.status_var.set("Drag selected grid edge to reshape this auto grid.")
                return
            if group_idx >= 0:
                app.active_grid_group = group_idx
                app.hovered_grid_group = group_idx
                app.grid_bounds = app.grid_groups[group_idx]['bounds']
                app.selected = -1
                app.select_drag_start = None
                app.select_drag_box = None
                app._refresh_list()
                app._redraw()
                app._save_auto_config()
                app.status_var.set(f"Selected grid #{group_idx + 1}")
                return
        app.select_drag_start = (ix, iy)
        app.select_drag_box = None
        app.active_grid_group = -1
        app.grid_bounds = []
        app.selected = -1
        app._refresh_list()
        app._redraw()
        app.status_var.set("Drag to select a grid group, or click its border/corner.")


def on_drag(app, event):
    if app.img_cv is None or app.mode not in ('edit', 'select'):
        return
    ix, iy = canvas2img(app, event.x, event.y)
    h, w = app.img_cv.shape[:2]
    ix = max(0, min(w - 1, int(ix)))
    iy = max(0, min(h - 1, int(iy)))
    ix, iy = classification.snap_to_detected_line(ix, iy, app.detected_lines, max(6, int(12 / app.zoom)))

    if app.dragging_grid_corner >= 0 and app.dragging_grid_group >= 0:
        group = app.grid_groups[app.dragging_grid_group]
        group['bounds'][app.dragging_grid_corner] = [ix, iy]
        app.grid_bounds = group['bounds']
        app._regenerate_grid_for_group(app.dragging_grid_group)
        app._refresh_list()
        app._redraw()
        return

    if app.mode == 'select' and app.select_drag_start is not None and app.dragging_grid_corner < 0 and app.dragging_grid_edge < 0:
        app.select_drag_box = [app.select_drag_start, (ix, iy)]
        app._redraw()
        return

    if app.dragging_grid_edge >= 0 and app.dragging_grid_group >= 0:
        group = app.grid_groups[app.dragging_grid_group]
        dx = ix - app.drag_start[0]
        dy = iy - app.drag_start[1]
        new_bounds = []
        for i, pt in enumerate(app.grid_bounds_orig):
            if i == app.dragging_grid_edge or i == (app.dragging_grid_edge + 1) % 4:
                new_pt = np.array(pt, dtype=np.float32) + np.array([dx, dy], dtype=np.float32)
                new_bounds.append([int(round(new_pt[0])), int(round(new_pt[1]))])
            else:
                new_bounds.append(list(pt))
        group['bounds'] = new_bounds
        app.grid_bounds = new_bounds
        app._regenerate_grid_for_group(app.dragging_grid_group)
        app._refresh_list()
        app._redraw()
        return

    if app.dragging_edge >= 0 and app.selected >= 0:
        app.preview_slot = app._make_adjacent_slot(
            app.slots[app.selected], app.dragging_edge, ix, iy)
        app._redraw()
        return

    if app.dragging_corner >= 0 and app.selected >= 0:
        app.slots[app.selected].pts[app.dragging_corner] = [ix, iy]
        app._redraw()
    elif app.dragging_slot and app.selected >= 0:
        s = app.slots[app.selected]
        dx = ix - app.drag_start[0]
        dy = iy - app.drag_start[1]
        for p in s.pts:
            p[0] = max(0, min(w - 1, p[0] + dx))
            p[1] = max(0, min(h - 1, p[1] + dy))
        app.drag_start = (ix, iy)
        app._redraw()


def on_release(app, event):
    if app.dragging_edge >= 0:
        if app.preview_slot is not None:
            app.slots.append(app.preview_slot)
            app.selected = len(app.slots) - 1
            app._refresh_list()
            app.status_var.set("Added adjacent slot from edge drag.")
        app.preview_slot = None
    if app.mode == 'select' and app.select_drag_start is not None and app.select_drag_box is not None:
        x0, y0 = app.select_drag_start
        x1, y1 = app.select_drag_box[1]
        x_min, x_max = sorted((x0, x1))
        y_min, y_max = sorted((y0, y1))
        selected_index = -1
        for group_idx, group in enumerate(app.grid_groups):
            if any(x_min <= px <= x_max and y_min <= py <= y_max
                   for px, py in group['bounds']):
                selected_index = group_idx
                break
        if selected_index >= 0:
            app.active_grid_group = selected_index
            app.hovered_grid_group = selected_index
            app.grid_bounds = app.grid_groups[selected_index]['bounds']
            app.status_var.set(f"Selected grid #{selected_index + 1}")
        else:
            app.status_var.set("No grid group found inside selection.")
        app.select_drag_start = None
        app.select_drag_box = None
        app._refresh_list()
        app._redraw()
    if (app.dragging_corner >= 0 or app.dragging_slot or
            app.dragging_grid_corner >= 0 or app.dragging_grid_edge >= 0):
        app._refresh_list()
        app._save_auto_config()
    app.dragging_corner = -1
    app.dragging_slot = False
    app.dragging_edge = -1
    app.dragging_grid_corner = -1
    app.dragging_grid_edge = -1
    app.dragging_grid_group = -1


def on_right_click(app, event):
    if app.mode == 'draw' and app.draw_pts:
        app.draw_pts.pop()
        app._redraw()
    elif app.mode == 'grid' and app.grid_pts:
        app.grid_pts.pop()
        app._redraw()
    elif app.mode == 'edit':
        app.selected = -1
        app._refresh_list()
        app._redraw()


def on_motion(app, event):
    if app.img_cv is None:
        return
    ix, iy = canvas2img(app, event.x, event.y)
    h, w = app.img_cv.shape[:2]
    if 0 <= ix < w and 0 <= iy < h:
        if app.mode in ('edit', 'draw', 'select'):
            best_slot = -1
            best_edge = -1
            threshold = LINE_FOCUS_THRESHOLD
            best_dist = threshold + 1
            app.draw_preview_pt = None

            if app.mode in ('edit', 'select'):
                if app.grid_groups:
                    group_idx, gci, gei = _find_nearest_grid_handle(app, ix, iy, int(12 / app.zoom))
                    if group_idx >= 0 and gci >= 0:
                        app.hovered_grid_group = group_idx
                        app.hovered_grid_corner = gci
                        app.hovered_grid_edge = -1
                    elif group_idx >= 0 and gei >= 0:
                        app.hovered_grid_group = group_idx
                        app.hovered_grid_corner = -1
                        app.hovered_grid_edge = gei
                    else:
                        app.hovered_grid_group = -1
                        app.hovered_grid_corner = -1
                        app.hovered_grid_edge = -1
                else:
                    app.hovered_grid_group = -1
                    app.hovered_grid_corner = -1
                    app.hovered_grid_edge = -1

            if app.mode == 'edit':
                if app.selected >= 0:
                    edge_i = app.slots[app.selected].nearest_edge(ix, iy, threshold)
                    if edge_i >= 0:
                        best_slot = app.selected
                        best_edge = edge_i
                        best_dist = 0

                if best_slot < 0:
                    for idx, slot in enumerate(app.slots):
                        edge_i = slot.nearest_edge(ix, iy, threshold)
                        if edge_i >= 0:
                            a = np.array(slot.pts[edge_i], dtype=np.float32)
                            b = np.array(slot.pts[(edge_i + 1) % 4], dtype=np.float32)
                            dist = Slot._point_segment_distance(np.array([ix, iy], dtype=np.float32), a, b)
                            if dist < best_dist:
                                best_dist = dist
                                best_slot = idx
                                best_edge = edge_i
            else:
                best_slot = -1
                best_edge = -1

            prev_detected_line = app.hovered_detected_line
            app.hovered_detected_line = -1
            if app.detected_lines:
                line_idx = classification.nearest_detected_line(ix, iy, app.detected_lines, threshold)
                if line_idx >= 0:
                    app.hovered_detected_line = line_idx
                    if app.mode == 'draw':
                        snapped = classification.project_to_line(ix, iy, app.detected_lines[line_idx])
                        app.draw_preview_pt = snapped

            if (best_edge != app.hovered_edge or best_slot != app.hovered_slot or
                    app.hovered_detected_line != prev_detected_line or
                    app.hovered_grid_group != app._prev_hovered_grid_group or
                    app.hovered_grid_corner != app._prev_hovered_grid_corner or
                    app.hovered_grid_edge != app._prev_hovered_grid_edge):
                app.hovered_edge = best_edge
                app.hovered_slot = best_slot
                app._prev_hovered_grid_group = app.hovered_grid_group
                app._prev_hovered_grid_corner = app.hovered_grid_corner
                app._prev_hovered_grid_edge = app.hovered_grid_edge
                app._redraw()

            if best_edge >= 0:
                app.canvas.config(cursor='hand2')
                app.status_var.set(
                    "Hover edge selected — click and drag to create adjacent slot")
            elif app.hovered_grid_corner >= 0:
                app.canvas.config(cursor='hand2')
                app.status_var.set(
                    "Hover grid corner — click and drag to reshape the full auto grid")
            elif app.hovered_grid_edge >= 0:
                app.canvas.config(cursor='hand2')
                app.status_var.set(
                    "Hover grid edge — click and drag to reshape the full auto grid")
            elif app.hovered_detected_line >= 0:
                app.canvas.config(cursor='hand2')
                app.status_var.set(
                    "Hover image line — edge focus activated")
            else:
                app.canvas.config(cursor='crosshair')
                app.status_var.set(
                    f"{app.mode.title()} mode — Pos: ({int(ix)}, {int(iy)})")
        else:
            if (app.hovered_edge != -1 or app.hovered_slot != -1 or
                    app.hovered_detected_line != -1 or app.hovered_grid_corner != -1 or
                    app.hovered_grid_edge != -1):
                app.hovered_edge = -1
                app.hovered_slot = -1
                app.hovered_detected_line = -1
                app.hovered_grid_corner = -1
                app.hovered_grid_edge = -1
                app._prev_hovered_grid_corner = -1
                app._prev_hovered_grid_edge = -1
                app._redraw()
            app.canvas.config(cursor='crosshair')


def on_scroll(app, event):
    if event.delta > 0:
        app._zoom_in()
    else:
        app._zoom_out()


def on_list_select(app, event):
    sel = app.slot_list.curselection()
    if sel:
        app.selected = sel[0]
        app._redraw()


def set_mode(app, mode):
    app.mode = mode
    app.draw_pts.clear()
    app.grid_pts.clear()
    app.dragging_corner = -1
    app.dragging_slot = False
    app.dragging_edge = -1
    app.dragging_grid_corner = -1
    app.dragging_grid_edge = -1
    app.dragging_grid_group = -1
    app.hovered_edge = -1
    app.hovered_grid_corner = -1
    app.hovered_grid_edge = -1
    app.hovered_grid_group = -1
    app.draw_preview_pt = None
    app.select_drag_start = None
    app.select_drag_box = None
    app.canvas.config(cursor='crosshair')
    app._update_mode_button_styles()

    messages = {
        'draw': "DRAW MODE — Click 4 corners (clockwise or counter-clockwise) to create a slot",
        'edit': "EDIT MODE — Click to select, drag corners to reshape, drag body to move",
        'select': "SELECT GRID MODE — Click an existing grid to select it, then drag corners or edges",
        'grid': f"GRID MODE — Click 4 corners of parking area (TL → TR → BR → BL), "
                f"then auto-generate {app.grid_rows.get()}x{app.grid_cols.get()} grid",
    }
    app.status_var.set(messages.get(mode, ''))
    app._redraw()


def update_mode_button_styles(app):
    app.btn_draw.configure(style='ActiveMode.TButton' if app.mode == 'draw' else 'Mode.TButton')
    app.btn_edit.configure(style='ActiveMode.TButton' if app.mode == 'edit' else 'Mode.TButton')
    app.btn_select.configure(style='ActiveMode.TButton' if app.mode == 'select' else 'Mode.TButton')
    app.btn_grid.configure(style='ActiveMode.TButton' if app.mode == 'grid' else 'Mode.TButton')


def cancel(app):
    app.draw_pts.clear()
    app.grid_pts.clear()
    app.selected = -1
    app.hovered_edge = -1
    app.select_drag_start = None
    app.select_drag_box = None
    app.canvas.config(cursor='crosshair')
    app._redraw()


def zoom_in(app):
    app.zoom = min(5.0, app.zoom * 1.2)
    app._redraw()


def zoom_out(app):
    app.zoom = max(0.05, app.zoom / 1.2)
    app._redraw()


def zoom_fit(app):
    if app.img_cv is None:
        return
    app.canvas.update_idletasks()
    cw = app.canvas.winfo_width()
    ch = app.canvas.winfo_height()
    h, w = app.img_cv.shape[:2]
    app.zoom = min(cw / max(w, 1), ch / max(h, 1)) * 0.95
    app.pan = [0, 0]
    app._redraw()


def snapshot(app):
    app.undo_stack.append({
        'slots': copy.deepcopy(app.slots),
        'draw_pts': copy.deepcopy(app.draw_pts),
        'grid_bounds': copy.deepcopy(app.grid_bounds),
        'grid_groups': copy.deepcopy(app.grid_groups),
        'active_grid_group': app.active_grid_group
    })
    if len(app.undo_stack) > 50:
        app.undo_stack.pop(0)


def undo(app):
    if app.undo_stack:
        state = app.undo_stack.pop()
        app.slots = state['slots']
        app.draw_pts = state['draw_pts']
        app.grid_bounds = state.get('grid_bounds', [])
        app.grid_groups = state.get('grid_groups', [])
        app.active_grid_group = state.get('active_grid_group', -1)
        app.selected = -1
        app._refresh_list()
        app._redraw()
        app.status_var.set("Undo")


def delete_selected(app):
    if app.mode == 'select' and app.active_grid_group >= 0 and app.active_grid_group < len(app.grid_groups):
        app._snapshot()
        group = app.grid_groups.pop(app.active_grid_group)
        start = group.get('start', 0)
        count = group.get('count', 0)
        end = min(start + count, len(app.slots))
        if start < len(app.slots):
            del app.slots[start:end]
        for g in app.grid_groups[app.active_grid_group:]:
            g['start'] = max(0, g.get('start', 0) - count)
        app.active_grid_group = -1
        app.grid_bounds = []
        app.selected = -1
        app._smart_number_slots()
        app._refresh_list()
        app._save_auto_config()
        app._redraw()
        app.status_var.set(f"Deleted selected grid group and its {count} slots")
        return

    if app.selected >= 0 and app.selected < len(app.slots):
        app._snapshot()
        label = app.slots[app.selected].label
        del app.slots[app.selected]
        app.selected = -1
        app._refresh_list()
        app._save_auto_config()
        app._redraw()
        app.status_var.set(f"Deleted slot {label}")


def duplicate(app):
    if app.selected >= 0:
        app._snapshot()
        s = app.slots[app.selected]
        new_pts = [[p[0] + 30, p[1] + 30] for p in s.pts]
        dup = Slot(pts=new_pts, label=f'S{len(app.slots)}',
                   slot_idx=len(app.slots))
        app.slots.append(dup)
        app.selected = len(app.slots) - 1
        app._refresh_list()
        app._save_auto_config()
        app._redraw()


def clear_all(app):
    if app.slots and messagebox.askyesno("Clear All",
            "Delete all slots? This cannot be undone."):
        app._snapshot()
        app.slots.clear()
        app.selected = -1
        app._refresh_list()
        app._save_auto_config()
        app._redraw()


def auto_number(app):
    if not app.slots:
        return
    app._snapshot()
    app._smart_number_slots()


def smart_number_slots(app):
    if not app.slots:
        return
    slots_sorted = sorted(app.slots, key=lambda s: (s.center[1], s.center[0]))
    for i, s in enumerate(slots_sorted):
        s.label = f'S{i+1}'
        s.slot_idx = i
    app._refresh_list()
    app._save_auto_config()
    app._redraw()
    app.status_var.set(f"Numbered {len(app.slots)} slots by spatial order")


def generate_grid(app):
    if len(app.grid_pts) != 4:
        return
    app._snapshot()
    tl, tr, br, bl = [np.array(p, dtype=np.float64) for p in app.grid_pts]
    rows = app.grid_rows.get()
    cols = app.grid_cols.get()
    start_idx = len(app.slots)

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
            app.slots.append(Slot(pts=pts, label='', slot_idx=len(app.slots)))

    group_bounds = [list(app.grid_pts[0]), list(app.grid_pts[1]),
                    list(app.grid_pts[2]), list(app.grid_pts[3])]
    app.grid_groups.append({
        'bounds': group_bounds,
        'rows': rows,
        'cols': cols,
        'start': start_idx,
        'count': rows * cols
    })
    app.active_grid_group = len(app.grid_groups) - 1
    app.grid_bounds = group_bounds
    app.grid_pts.clear()
    app.mode = 'edit'
    app._smart_number_slots()
    app._refresh_list()
    app._save_auto_config()
    app._redraw()
    app.status_var.set(
        f"Appended {rows}x{cols} = {rows * cols} slots — "
        f"Switch to Edit mode to adjust individual corners")


def update_detected_lines(app):
    if app.ref_cv is None:
        app.detected_lines = []
        return
    gray = cv2.cvtColor(app.ref_cv, cv2.COLOR_BGR2GRAY)
    app.detected_lines = classification.detect_image_lines(gray)


def nearest_detected_line(app, px: float, py: float,
                           threshold: int = LINE_FOCUS_THRESHOLD) -> int:
    return classification.nearest_detected_line(px, py, app.detected_lines, threshold)


def project_to_line(app, px: float, py: float, line_idx: int) -> Tuple[int, int]:
    return classification.project_to_line(px, py, app.detected_lines[line_idx])


def snap_to_detected_line(app, ix: int, iy: int, threshold: int = LINE_FOCUS_THRESHOLD) -> Tuple[int, int]:
    return classification.snap_to_detected_line(ix, iy, app.detected_lines, threshold)


def set_ref(app):
    path = filedialog.askopenfilename(
        title="Select Reference Image (empty lot)",
        filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp"), ("All", "*.*")])
    if path:
        ref_img = cv2.imread(path)
        if ref_img is None:
            messagebox.showerror("Error", f"Cannot load reference image: {path}")
            return
        app.ref_path = path
        app.ref_cv = ref_img
        app.ref_pil = Image.fromarray(cv2.cvtColor(ref_img, cv2.COLOR_BGR2RGB))
        app.ref_label.configure(text=f"Ref: {os.path.basename(path)}")
        app._update_detected_lines()
        app.status_var.set(
            f"Reference image set: {os.path.basename(path)}")

        if app.test_cv is not None and app.test_cv.shape[:2] != ref_img.shape[:2]:
            app.test_cv = cv2.resize(app.test_cv,
                                      (ref_img.shape[1], ref_img.shape[0]),
                                      interpolation=cv2.INTER_LINEAR)
            app.test_pil = Image.fromarray(cv2.cvtColor(app.test_cv, cv2.COLOR_BGR2RGB))
            if app._display_cv is app.test_cv or app._display_pil is app.test_pil:
                app._display_cv = app.test_cv
                app._display_pil = app.test_pil
            app.status_var.set(
                f"Resized TEST image to match new Reference size {ref_img.shape[1]}x{ref_img.shape[0]}")


def set_test(app):
    path = filedialog.askopenfilename(
        title="Select Test Image (with cars)",
        filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp"), ("All", "*.*")])
    if path:
        _load_image(app, path, is_test=True)


def classify(app):
    if not app.ref_path or not app.test_path:
        messagebox.showinfo("Classification",
            "Please set both Reference and Test images first.")
        return
    if not app.slots:
        messagebox.showinfo("Classification", "No slots defined.")
        return

    ref = cv2.imread(app.ref_path, cv2.IMREAD_GRAYSCALE)
    test = cv2.imread(app.test_path, cv2.IMREAD_GRAYSCALE)
    if ref is None or test is None:
        messagebox.showerror("Error", "Cannot load ref/test images")
        return
    try:
        if ref.shape[:2] != test.shape[:2]:
            test = cv2.resize(test, (ref.shape[1], ref.shape[0]),
                               interpolation=cv2.INTER_LINEAR)
            app.test_cv = test
            app.test_pil = Image.fromarray(cv2.cvtColor(test, cv2.COLOR_BGR2RGB))
            app.status_var.set(
                f"Resized TEST image to match Reference size {ref.shape[1]}x{ref.shape[0]} for classification")
            if app._display_cv is not None and app.test_path:
                app._display_cv = app.test_cv
                app._display_pil = app.test_pil
    except Exception as e:
        messagebox.showerror("Error", f"Failed resizing test image: {e}")
        return
    if ref.shape[:2] != test.shape[:2]:
        test = cv2.resize(test, (ref.shape[1], ref.shape[0]),
                           interpolation=cv2.INTER_LINEAR)
        app.test_cv = test
        app.test_pil = Image.fromarray(cv2.cvtColor(test, cv2.COLOR_BGR2RGB))
        app.status_var.set(
            f"Resized TEST image to match Reference size {ref.shape[1]}x{ref.shape[0]} for classification")
        if app._display_cv is not None and app.test_path:
            app._display_cv = app.test_cv
            app._display_pil = app.test_pil

    results = []
    for s in app.slots:
        ref_roi = warp_roi(app, ref, s)
        test_roi = warp_roi(app, test, s)
        if ref_roi is None or test_roi is None:
            s.prediction = -1
            continue

        mean_ssim, diff_ratio, edge_ratio, occupied = compute_slot_diff(ref_roi, test_roi)
        s.mad_value = round((1.0 - mean_ssim) * 100.0, 1)
        confidence_score = min(1.0, max(
            0.0,
            (DIFF_PIXEL_PERCENT * 4.0 - diff_ratio * 3.5),
            (EDGE_DIFF_PERCENT * 4.0 - edge_ratio * 3.5),
            ((SSIM_THRESHOLD - mean_ssim) * 2.0)
        ))
        s.confidence = round(confidence_score, 2)
        s.prediction = 1 if occupied and s.confidence >= 0.55 else 0
        status = 'OCC' if s.prediction else 'FREE'
        results.append(f"{s.label}: SSIM={mean_ssim:.3f} → {status} "
                      f"(conf={s.confidence*100:.0f}% / {diff_ratio*100:.1f}% diff / {edge_ratio*100:.1f}% edge)")

    app._refresh_list()
    app._redraw()

    occ = sum(1 for s in app.slots if s.prediction == 1)
    free = sum(1 for s in app.slots if s.prediction == 0)
    app.status_var.set(
        f"Classification done: {occ} occupied, {free} free "
        f"(SSIM threshold τ={SSIM_THRESHOLD:.2f})")

    msg = f"Results ({len(app.slots)} slots):\n\n"
    msg += '\n'.join(results)
    msg += f"\n\nSummary: {occ} OCC / {free} FREE"
    messagebox.showinfo("MAD Classification Results", msg)


def warp_roi(app, gray, s):
    src = np.array(s.pts, dtype=np.float32)
    dst = np.array([[0, 0], [ROI_SIZE-1, 0],
                    [ROI_SIZE-1, ROI_SIZE-1], [0, ROI_SIZE-1]],
                   dtype=np.float32)
    try:
        M = cv2.getPerspectiveTransform(src, dst)
        return cv2.warpPerspective(gray, M, (ROI_SIZE, ROI_SIZE))
    except Exception:
        return None


def quick_classify(app):
    if not app.ref_path:
        messagebox.showinfo("Quick Classify",
            "Open a reference image (empty lot) first, "
            "then use this button to load a test image and classify.")
        return
    if not app.slots:
        messagebox.showinfo("Quick Classify", "No slots defined yet.")
        return

    path = filedialog.askopenfilename(
        title="Select Test Image (with cars) — will auto-classify",
        filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp"), ("All", "*.*")])
    if not path:
        return

    if _load_image(app, path, is_test=True):
        try:
            app._classify()
        except Exception as e:
            messagebox.showerror("Quick Classify", f"Classification failed: {e}")
            app.status_var.set("Quick classification failed.")


def show_ref_image(app):
    if app.ref_pil is None:
        return
    app._display_pil = app.ref_pil
    app._display_cv = app.ref_cv
    app._redraw()
    app.status_var.set(f"Showing REFERENCE image: {os.path.basename(app.ref_path or '')}")


def show_test_image(app):
    if not app.test_path:
        messagebox.showinfo("View Test", "No test image loaded yet.")
        return
    if app.test_pil is not None:
        app._display_pil = app.test_pil
        app._display_cv = app.test_cv
        app._redraw()
        app.status_var.set(f"Showing TEST image: {os.path.basename(app.test_path)}")
    else:
        test_cv = cv2.imread(app.test_path)
        if test_cv is not None:
            app._display_pil = Image.fromarray(cv2.cvtColor(test_cv, cv2.COLOR_BGR2RGB))
            app._display_cv = test_cv
            app._redraw()
            app.status_var.set(f"Showing TEST image: {os.path.basename(app.test_path)}")


def save(app):
    if not app.slots:
        messagebox.showinfo("Save", "No slots to save.")
        return

    base = os.path.splitext(os.path.basename(app.img_path or 'untitled'))[0]
    default_dir = os.path.dirname(os.path.abspath(app.img_path or '.'))

    path = filedialog.asksaveasfilename(
        title="Save ROI Config",
        initialdir=default_dir,
        initialfile=f'roi_config_{base}.json',
        filetypes=[("JSON", "*.json")],
        defaultextension='.json')
    if not path:
        return

    base_out = os.path.splitext(path)[0]
    data = {
        'image': os.path.basename(app.img_path or ''),
        'image_size': [app.img_cv.shape[1], app.img_cv.shape[0]]
                      if app.img_cv is not None else [0, 0],
        'roi_size': ROI_SIZE,
        'n_slots': len(app.slots),
        'rois': [s.to_dict() for s in app.slots],
        'grid_groups': [copy.deepcopy(g) for g in app.grid_groups],
        'active_grid_group': app.active_grid_group
    }
    with open(path, 'w') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    app._save_auto_config()

    cpath = base_out + '.h'
    with open(cpath, 'w') as f:
        img_name = os.path.basename(app.img_path or 'unknown')
        w = app.img_cv.shape[1] if app.img_cv is not None else 0
        h = app.img_cv.shape[0] if app.img_cv is not None else 0
        f.write(f'// Auto-generated ROI config — {img_name}\n')
        f.write(f'// Image: {w}x{h}, Slots: {len(app.slots)}\n\n')
        f.write(f'#define N_SLOTS {len(app.slots)}\n\n')
        f.write('static roi_rect_t slot_rois[N_SLOTS] = {\n')
        for s in app.slots:
            r = s.bounding_rect()
            lbl = f' // {s.label}' if s.label else ''
            f.write(f'    {{ .x = {r[0]:4d}, .y = {r[1]:4d}, '
                    f'.w = {r[2]:4d}, .h = {r[3]:4d} }},{lbl}\n')
        f.write('};\n\n')
        f.write('/* Perspective polygon corners:\n')
        for s in app.slots:
            f.write(f'   {s.label}: {s.pts}\n')
        f.write('*/\n')

    pypath = base_out + '.py'
    with open(pypath, 'w') as f:
        f.write(f'# Auto-generated ROI config\n\n')
        f.write('SLOT_POLYGONS = [\n')
        for s in app.slots:
            f.write(f'    {{"label": "{s.label}", "pts": {s.pts}}},\n')
        f.write(']\n')

    app.status_var.set(
        f"Saved: {os.path.basename(path)}, "
        f"{os.path.basename(cpath)}, {os.path.basename(pypath)}")
    messagebox.showinfo("Saved",
        f"Config saved:\n\n"
        f"  JSON: {os.path.basename(path)}\n"
        f"  C:       {os.path.basename(cpath)}\n"
        f"  Python: {os.path.basename(pypath)}\n\n"
        f"({len(app.slots)} slots)")


def refresh_list(app):
    app.slot_list.delete(0, tk.END)
    for i, s in enumerate(app.slots):
        line = f"{'>' if i == app.selected else ' '} "
        line += f"{s.label or f'#{i}':5s}"
        line += f"  area={s.area:6.0f}"
        if s.prediction == 1:
            line += f"  OCC  MAD={s.mad_value:.1f}"
        elif s.prediction == 0:
            line += f"  FREE MAD={s.mad_value:.1f}"
        app.slot_list.insert(tk.END, line)

    if 0 <= app.selected < len(app.slots):
        app.slot_list.selection_set(app.selected)
        app.slot_list.see(app.selected)

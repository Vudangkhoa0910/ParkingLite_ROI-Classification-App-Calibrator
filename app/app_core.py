import copy
import json
import os
import tkinter as tk
from tkinter import filedialog, messagebox
from typing import Optional, Tuple

import cv2
import numpy as np
from PIL import Image

from constants import ROI_SIZE
from models import Slot
import config_io
import classification
import grid_helpers


class AppLogic:
    def _bind_keys(self):
        self.bind('<Delete>', lambda e: self._delete_selected())
        self.bind('<BackSpace>', lambda e: self._delete_selected())
        self.bind('<Control-d>', lambda e: self._duplicate())
        self.bind('<Control-z>', lambda e: self._undo())
        self.bind('<Control-Z>', lambda e: self._undo())
        self.bind('<Control-y>', lambda e: self._redo())
        self.bind('<Control-Y>', lambda e: self._redo())
        self.bind('<Control-Shift-z>', lambda e: self._redo())
        self.bind('<Control-Shift-Z>', lambda e: self._redo())
        self.bind('<Key-D>', lambda e: self._set_mode('draw'))
        self.bind('<Key-d>', lambda e: self._set_mode('draw'))
        self.bind('<Key-E>', lambda e: self._set_mode('edit'))
        self.bind('<Key-e>', lambda e: self._set_mode('edit'))
        self.bind('<Key-G>', lambda e: self._set_mode('grid'))
        self.bind('<Key-g>', lambda e: self._set_mode('grid'))
        self.bind('<Key-T>', lambda e: self._set_mode('tile'))
        self.bind('<Key-t>', lambda e: self._set_mode('tile'))
        self.bind('<Key-Q>', lambda e: self._set_mode('select'))
        self.bind('<Key-q>', lambda e: self._set_mode('select'))
        self.bind('<Key-n>', lambda e: self._auto_number())
        self.bind('<Key-s>', lambda e: self._save())
        self.bind('<Key-r>', lambda e: self._classify())
        self.bind('<Key-u>', lambda e: self._undo())
        self.bind('<Escape>', lambda e: self._cancel())
        self.bind('<plus>', lambda e: self._zoom_in())
        self.bind('<equal>', lambda e: self._zoom_in())
        self.bind('<minus>', lambda e: self._zoom_out())
        self.bind('<Key-f>', lambda e: self._zoom_fit())

        self.canvas.bind('<Button-1>', self._on_click)
        self.canvas.bind('<B1-Motion>', self._on_drag)
        self.canvas.bind('<ButtonRelease-1>', self._on_release)
        self.canvas.bind('<Button-3>', self._on_right_click)
        self.canvas.bind('<Motion>', self._on_motion)
        self.canvas.bind('<MouseWheel>', self._on_scroll)
        self.canvas.bind('<Button-4>', lambda e: self._zoom_in())
        self.canvas.bind('<Button-5>', lambda e: self._zoom_out())

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
            return False

        if is_test:
            self.test_path = path
            self.test_label.configure(text=f"Test: {os.path.basename(path)}")
            if self.ref_cv is not None and self.ref_path != path:
                if self.ref_cv.shape[:2] != img.shape[:2]:
                    img = cv2.resize(img, (self.ref_cv.shape[1], self.ref_cv.shape[0]),
                                     interpolation=cv2.INTER_LINEAR)
                    self.status_var.set(
                        f"Resized TEST image to match Reference size {self.ref_cv.shape[1]}x{self.ref_cv.shape[0]}")
            self.test_cv = img
            self.test_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            self._display_cv = self.test_cv
            self._display_pil = self.test_pil
            if not self.status_var.get().startswith("Resized TEST"):
                self.status_var.set(f"Showing TEST image: {os.path.basename(path)} — "
                                    f"Click 'Run ROI Classification' or press R")
        else:
            self.img_cv = img
            self.img_path = path
            self.img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            self.ref_cv = img
            self.ref_pil = self.img_pil
            self._display_cv = img
            self._display_pil = self.img_pil
            self._update_detected_lines()
            self.ref_path = path
            self.ref_label.configure(text=f"Ref: {os.path.basename(path)}")
            self.slots.clear()
            self.undo_stack.clear()
            self.redo_stack.clear()
            if self._load_auto_config(path):
                self.status_var.set(
                    f"Loaded cached slot config for {os.path.basename(path)}")
            else:
                h, w = img.shape[:2]
                self.status_var.set(
                    f"Image: {w}x{h} — {os.path.basename(path)} "
                    f"(auto-set as Reference)")

        self._zoom_fit()
        self._redraw()
        return True

    def _open_config(self):
        path = filedialog.askopenfilename(
            title="Load ROI Config",
            filetypes=[("JSON", "*.json"), ("All", "*.*")])
        if not path:
            return
        try:
            data = config_io.load_config_file(path)
            self.slots, self.grid_groups, self.active_grid_group, self.grid_bounds = \
                config_io.load_from_config_data(data)
            self.undo_stack, self.redo_stack = config_io.load_history_stacks(data)
            self.selected = -1
            self._refresh_list()
            self._redraw()
            self.status_var.set(f"Loaded {len(self.slots)} slots from {os.path.basename(path)}")
            self._save_auto_config()
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def _auto_config_path(self, image_path=None):
        return config_io.auto_config_path(image_path, self.ref_path, self.img_path)

    def _load_auto_config(self, image_path=None):
        config_path = self._auto_config_path(image_path)
        if not config_path:
            return False
        data = config_io.load_auto_config(config_path)
        if not data:
            return False
        self.slots, self.grid_groups, self.active_grid_group, self.grid_bounds = \
            config_io.load_from_config_data(data)
        self.undo_stack, self.redo_stack = config_io.load_history_stacks(data)
        self.selected = -1
        self._refresh_list()
        self._redraw()
        self.status_var.set(f"Loaded auto config from {os.path.basename(config_path)}")
        return True

    def _save_auto_config(self):
        config_path = self._auto_config_path(self.img_path)
        config_io.save_auto_config(config_path, self.img_cv, self.ref_path,
                                   self.img_path, self.slots, self.grid_groups,
                                   self.active_grid_group,
                                   undo_stack=self.undo_stack,
                                   redo_stack=self.redo_stack)

    def _compute_slot_diff(self, ref_roi, test_roi):
        return classification.compute_slot_diff(ref_roi, test_roi)

    def _selected_classifier_key(self):
        if hasattr(self, 'classify_method_var') and hasattr(self, 'classifier_label_to_key'):
            label = self.classify_method_var.get()
            return self.classifier_label_to_key.get(label, classification.DEFAULT_CLASSIFIER)
        return classification.DEFAULT_CLASSIFIER

    def _on_classifier_changed(self, _event=None):
        method_key = self._selected_classifier_key()
        self.status_var.set(f"Classification method: {classification.classifier_label(method_key)}")

    def _img2canvas(self, x, y):
        return x * self.zoom + self.pan[0], y * self.zoom + self.pan[1]

    def _canvas2img(self, cx, cy):
        return (cx - self.pan[0]) / self.zoom, (cy - self.pan[1]) / self.zoom

    def _set_mode(self, mode):
        self.mode = mode
        self.draw_pts.clear()
        self.grid_pts.clear()
        self.dragging_corner = -1
        self.dragging_slot = False
        self.dragging_edge = -1
        self.dragging_grid_corner = -1
        self.dragging_grid_edge = -1
        self.dragging_grid_group = -1
        self.hovered_edge = -1
        self.hovered_grid_corner = -1
        self.hovered_grid_edge = -1
        self.hovered_grid_group = -1
        self.draw_preview_pt = None
        self.select_drag_start = None
        self.select_drag_box = None
        self.canvas.config(cursor='crosshair')
        self._update_mode_button_styles()

        self.tile_preview_slots.clear()
        self.tile_box = None
        self.tile_drag_start = None
        self.dragging_tile = False
        self.tile_rows = max(1, int(self.tile_rows_var.get()))
        self.tile_cols = max(1, int(self.tile_cols_var.get()))
        self.tile_group_start = -1
        self.tile_base_count = 0
        self.tile_u_min = 0
        self.tile_u_max = 0
        self.tile_v_min = 0
        self.tile_v_max = 0

        messages = {
            'draw': "DRAW MODE — Click 4 corners (clockwise or counter-clockwise) to create a slot",
            'edit': "EDIT MODE — Click to select, drag corners to reshape, drag body to move",
            'select': "SELECT GRID MODE — Click an existing grid to select it, then drag corners or edges",
            'grid': f"GRID MODE — Click 4 corners of parking area (TL → TR → BR → BL), "
                    f"then auto-generate {self.grid_rows.get()}x{self.grid_cols.get()} grid",
            'tile': f"TILE MODE — Click 4 corners for template box (split by {self.tile_rows_var.get()}x{self.tile_cols_var.get()}), "
                f"then drag inside to replicate",
        }
        self.status_var.set(messages.get(mode, ''))
        if hasattr(self, '_update_mode_settings_panel'):
            self._update_mode_settings_panel()
        self._redraw()

    def _update_mode_button_styles(self):
        self.btn_draw.configure(style='ActiveMode.TButton' if self.mode == 'draw' else 'Mode.TButton')
        self.btn_edit.configure(style='ActiveMode.TButton' if self.mode == 'edit' else 'Mode.TButton')
        self.btn_select.configure(style='ActiveMode.TButton' if self.mode == 'select' else 'Mode.TButton')
        self.btn_grid.configure(style='ActiveMode.TButton' if self.mode == 'grid' else 'Mode.TButton')
        self.btn_tile.configure(style='ActiveMode.TButton' if self.mode == 'tile' else 'Mode.TButton')

    def _cancel(self):
        self.draw_pts.clear()
        self.grid_pts.clear()
        self.selected = -1
        self.hovered_edge = -1
        self.select_drag_start = None
        self.select_drag_box = None
        self.tile_preview_slots.clear()
        self.tile_drag_start = None
        self.dragging_tile = False
        if self.mode == 'tile':
            self.tile_box = None
        self.canvas.config(cursor='crosshair')
        self._redraw()

    def _on_close(self):
        try:
            self._save_auto_config()
        finally:
            self.destroy()

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

    def _capture_state(self):
        return {
            'slots': copy.deepcopy(self.slots),
            'draw_pts': copy.deepcopy(self.draw_pts),
            'grid_bounds': copy.deepcopy(self.grid_bounds),
            'grid_groups': copy.deepcopy(self.grid_groups),
            'active_grid_group': self.active_grid_group
        }

    def _restore_state(self, state):
        self.slots = copy.deepcopy(state.get('slots', []))
        self.draw_pts = copy.deepcopy(state.get('draw_pts', []))
        self.grid_groups = copy.deepcopy(state.get('grid_groups', []))
        self.active_grid_group = state.get('active_grid_group', -1)
        self.grid_bounds = copy.deepcopy(state.get('grid_bounds', []))
        if (self.active_grid_group >= 0 and self.active_grid_group < len(self.grid_groups)
                and not self.grid_bounds):
            self.grid_bounds = copy.deepcopy(self.grid_groups[self.active_grid_group].get('bounds', []))
        self.selected = -1

    def _snapshot(self):
        self.undo_stack.append(self._capture_state())
        self.redo_stack.clear()
        if len(self.undo_stack) > 100:
            self.undo_stack.pop(0)

    def _undo(self):
        if self.undo_stack:
            self.redo_stack.append(self._capture_state())
            if len(self.redo_stack) > 100:
                self.redo_stack.pop(0)
            state = self.undo_stack.pop()
            self._restore_state(state)
            self._refresh_list()
            self._redraw()
            self._save_auto_config()
            self.status_var.set("Undo")

    def _redo(self):
        if self.redo_stack:
            self.undo_stack.append(self._capture_state())
            if len(self.undo_stack) > 100:
                self.undo_stack.pop(0)
            state = self.redo_stack.pop()
            self._restore_state(state)
            self._refresh_list()
            self._redraw()
            self._save_auto_config()
            self.status_var.set("Redo")

    def _clear_history(self):
        if not self.undo_stack and not self.redo_stack:
            self.status_var.set("History is already empty")
            return
        if messagebox.askyesno("Clear History", "Clear Undo/Redo history?"):
            self.undo_stack.clear()
            self.redo_stack.clear()
            self._save_auto_config()
            self.status_var.set("Undo/Redo history cleared")

    def _show_history_info(self):
        config_path = self._auto_config_path() or "(no image)"
        messagebox.showinfo(
            "History Info",
            f"Undo entries: {len(self.undo_stack)}\n"
            f"Redo entries: {len(self.redo_stack)}\n"
            f"Auto JSON: {config_path}"
        )

    def _delete_selected(self):
        if self.mode == 'select' and self.active_grid_group >= 0 and self.active_grid_group < len(self.grid_groups):
            self._snapshot()
            group = self.grid_groups.pop(self.active_grid_group)
            start = group.get('start', 0)
            count = group.get('count', 0)
            end = min(start + count, len(self.slots))
            if start < len(self.slots):
                del self.slots[start:end]
            for g in self.grid_groups[self.active_grid_group:]:
                g['start'] = max(0, g.get('start', 0) - count)
            self.active_grid_group = -1
            self.grid_bounds = []
            self.selected = -1
            self._smart_number_slots()
            self._refresh_list()
            self._save_auto_config()
            self._redraw()
            self.status_var.set(f"Deleted selected grid group and its {count} slots")
            return

        if self.selected >= 0 and self.selected < len(self.slots):
            self._snapshot()
            label = self.slots[self.selected].label
            del self.slots[self.selected]
            self.selected = -1
            self._refresh_list()
            self._save_auto_config()
            self._redraw()
            self.status_var.set(f"Deleted slot {label}")

    def _save_auto_json(self):
        self._save_auto_config()
        path = self._auto_config_path(self.img_path)
        if path:
            messagebox.showinfo("Saved Auto JSON",
                                f"Auto config saved to:\n{path}")
            self.status_var.set(f"Auto JSON saved: {os.path.basename(path)}")
        else:
            messagebox.showwarning("Save Auto JSON", "No image loaded, cannot save auto JSON.")
            self.status_var.set("Cannot save auto JSON without an image")

    def _show_auto_config_path(self):
        path = self._auto_config_path(self.img_path)
        if path:
            exists = os.path.exists(path)
            messagebox.showinfo("Auto JSON Path",
                                f"Auto config file:\n{path}\n\nExists: {exists}")
            self.status_var.set(f"Auto JSON path: {os.path.basename(path)}")
        else:
            messagebox.showwarning("Auto JSON Path", "No image loaded, no auto config path available.")
            self.status_var.set("No auto config path available")

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
            self._save_auto_config()
            self._redraw()

    def _clear_all(self):
        if self.slots and messagebox.askyesno("Clear All",
                "Delete all slots? This cannot be undone."):
            self._snapshot()
            self.slots.clear()
            self.selected = -1
            self._refresh_list()
            self._save_auto_config()
            self._redraw()

    def _auto_number(self):
        if not self.slots:
            return
        self._snapshot()
        self._smart_number_slots()

    def _smart_number_slots(self):
        if not self.slots:
            return
        slots_sorted = sorted(self.slots, key=lambda s: (s.center[1], s.center[0]))
        for i, s in enumerate(slots_sorted):
            s.label = f'S{i+1}'
            s.slot_idx = i
        self._refresh_list()
        self._save_auto_config()
        self._redraw()
        self.status_var.set(f"Numbered {len(self.slots)} slots by spatial order")

    def _generate_grid(self):
        if len(self.grid_pts) != 4:
            return
        self._snapshot()
        tl, tr, br, bl = [np.array(p, dtype=np.float64) for p in self.grid_pts]
        rows = self.grid_rows.get()
        cols = self.grid_cols.get()
        start_idx = len(self.slots)

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
                self.slots.append(Slot(pts=pts, label='', slot_idx=len(self.slots)))

        group_bounds = [list(self.grid_pts[0]), list(self.grid_pts[1]),
                        list(self.grid_pts[2]), list(self.grid_pts[3])]
        self.grid_groups.append({
            'bounds': group_bounds,
            'rows': rows,
            'cols': cols,
            'start': start_idx,
            'count': rows * cols
        })
        self.active_grid_group = len(self.grid_groups) - 1
        self.grid_bounds = group_bounds
        self.grid_pts.clear()
        self.mode = 'edit'
        self._smart_number_slots()
        self._refresh_list()
        self._save_auto_config()
        self._redraw()
        self.status_var.set(
            f"Appended {rows}x{cols} = {rows * cols} slots — "
            f"Switch to Edit mode to adjust individual corners")

    def _update_detected_lines(self):
        if self.ref_cv is None:
            self.detected_lines = []
            return
        gray = cv2.cvtColor(self.ref_cv, cv2.COLOR_BGR2GRAY)
        self.detected_lines = classification.detect_image_lines(gray)

    def _nearest_detected_line(self, px: float, py: float,
                               threshold: int = 12) -> int:
        return classification.nearest_detected_line(px, py, self.detected_lines, threshold)

    def _project_to_line(self, px: float, py: float, line_idx: int) -> Tuple[int, int]:
        return classification.project_to_line(px, py, self.detected_lines[line_idx])

    def _snap_to_detected_line(self, ix: int, iy: int,
                               threshold: int = 12) -> Tuple[int, int]:
        return classification.snap_to_detected_line(ix, iy, self.detected_lines, threshold)

    def _find_nearest_grid_handle(self, px: float, py: float, threshold: int = 12):
        return grid_helpers.find_nearest_grid_handle(px, py, self.grid_groups, threshold)

    def _regenerate_grid_for_group(self, group_idx: int):
        grid_helpers.regenerate_grid_for_group(group_idx, self.grid_groups, self.slots)
        self.grid_bounds = self.grid_groups[group_idx]['bounds']

    def _normalize_quad_points(self, pts):
        arr = np.array(pts, dtype=np.float32)
        if arr.shape != (4, 2):
            return arr
        center = np.mean(arr, axis=0)
        angles = np.arctan2(arr[:, 1] - center[1], arr[:, 0] - center[0])
        arr = arr[np.argsort(angles)]
        start = int(np.argmin(arr[:, 0] + arr[:, 1]))
        arr = np.roll(arr, -start, axis=0)
        oriented_area = cv2.contourArea(arr.reshape(-1, 1, 2), oriented=True)
        if oriented_area < 0:
            arr = np.array([arr[0], arr[3], arr[2], arr[1]], dtype=np.float32)
        return arr

    def _build_tile_cells(self, base_pts: np.ndarray, rows: int, cols: int,
                          start_idx: int, u_shift: float = 0.0,
                          v_shift: float = 0.0):
        p0, p1, p2, p3 = [np.array(p, dtype=np.float32) for p in base_pts]

        def bilinear(u: float, v: float):
            top = p0 + (p1 - p0) * u
            bottom = p3 + (p2 - p3) * u
            return top + (bottom - top) * v

        cells = []
        for r in range(rows):
            v0 = (r / rows) + v_shift
            v1 = ((r + 1) / rows) + v_shift
            for c in range(cols):
                u0 = (c / cols) + u_shift
                u1 = ((c + 1) / cols) + u_shift
                q0 = bilinear(u0, v0)
                q1 = bilinear(u1, v0)
                q2 = bilinear(u1, v1)
                q3 = bilinear(u0, v1)
                pts = [[int(round(q0[0])), int(round(q0[1]))],
                       [int(round(q1[0])), int(round(q1[1]))],
                       [int(round(q2[0])), int(round(q2[1]))],
                       [int(round(q3[0])), int(round(q3[1]))]]
                slot_idx = start_idx + len(cells)
                cells.append(Slot(pts=pts, label=f'S{slot_idx}', slot_idx=slot_idx))
        return cells

    def _tile_bilinear_point(self, base_pts: np.ndarray, u: float, v: float):
        p0, p1, p2, p3 = [np.array(p, dtype=np.float32) for p in base_pts]
        top = p0 + (p1 - p0) * u
        bottom = p3 + (p2 - p3) * u
        return top + (bottom - top) * v

    def _compute_tile_block_steps(self, ix: int, iy: int):
        if self.tile_box is None or self.tile_drag_start is None:
            return 0, 0

        base_pts = np.array(self.tile_box.pts, dtype=np.float32)
        width_top = base_pts[1] - base_pts[0]
        width_bottom = base_pts[2] - base_pts[3]
        height_left = base_pts[3] - base_pts[0]
        height_right = base_pts[2] - base_pts[1]
        width_axis = (width_top + width_bottom) * 0.5
        height_axis = (height_left + height_right) * 0.5
        width_len = float(np.linalg.norm(width_axis))
        height_len = float(np.linalg.norm(height_axis))
        if width_len < 1e-3 or height_len < 1e-3:
            return 0, 0

        drag = np.array([ix, iy], dtype=np.float32) - np.array(self.tile_drag_start, dtype=np.float32)
        width_unit = width_axis / width_len
        height_unit = height_axis / height_len
        proj_w = float(np.dot(drag, width_unit))
        proj_h = float(np.dot(drag, height_unit))

        step_u = int(np.sign(proj_w) * int(abs(proj_w) / width_len + 0.5))
        step_v = int(np.sign(proj_h) * int(abs(proj_h) / height_len + 0.5))
        return step_u, step_v

    def _build_tile_group_slots(self, start_idx: int, u_min: int, u_max: int,
                                v_min: int, v_max: int):
        if self.tile_box is None:
            return [], [], 0, 0

        base_pts = np.array(self.tile_box.pts, dtype=np.float32)
        rows = max(1, int(getattr(self, 'tile_rows', self.tile_rows_var.get())))
        cols = max(1, int(getattr(self, 'tile_cols', self.tile_cols_var.get())))

        all_slots = []
        for v_shift in range(v_min, v_max + 1):
            for u_shift in range(u_min, u_max + 1):
                all_slots.extend(self._build_tile_cells(
                    base_pts, rows, cols,
                    start_idx=start_idx + len(all_slots),
                    u_shift=float(u_shift),
                    v_shift=float(v_shift),
                ))

        rows_total = (v_max - v_min + 1) * rows
        cols_total = (u_max - u_min + 1) * cols

        tl = self._tile_bilinear_point(base_pts, float(u_min), float(v_min))
        tr = self._tile_bilinear_point(base_pts, float(u_max + 1), float(v_min))
        br = self._tile_bilinear_point(base_pts, float(u_max + 1), float(v_max + 1))
        bl = self._tile_bilinear_point(base_pts, float(u_min), float(v_max + 1))
        bounds = [
            [int(round(tl[0])), int(round(tl[1]))],
            [int(round(tr[0])), int(round(tr[1]))],
            [int(round(br[0])), int(round(br[1]))],
            [int(round(bl[0])), int(round(bl[1]))],
        ]

        return all_slots, bounds, rows_total, cols_total

    def _apply_tile_group(self):
        if self.tile_group_start < 0 or self.tile_base_count <= 0 or self.tile_box is None:
            return 0

        start = self.tile_group_start
        base_count = self.tile_base_count
        u_min = int(self.tile_u_min)
        u_max = int(self.tile_u_max)
        v_min = int(self.tile_v_min)
        v_max = int(self.tile_v_max)

        all_slots, bounds, rows_total, cols_total = self._build_tile_group_slots(
            start_idx=start,
            u_min=u_min,
            u_max=u_max,
            v_min=v_min,
            v_max=v_max,
        )
        if not all_slots:
            return 0

        self.slots[start:start + base_count] = all_slots
        delta = len(all_slots) - base_count

        for g in self.grid_groups:
            if g.get('start', 0) > start:
                g['start'] = g.get('start', 0) + delta

        self.grid_groups.append({
            'bounds': bounds,
            'rows': rows_total,
            'cols': cols_total,
            'start': start,
            'count': len(all_slots),
        })
        self.active_grid_group = len(self.grid_groups) - 1
        self.grid_bounds = bounds
        self.selected = start if len(all_slots) > 0 else -1
        return max(0, delta)

    def _build_tile_clone_slots(self, ix: int, iy: int):
        if self.tile_box is None or self.tile_drag_start is None:
            return []

        base_pts = np.array(self.tile_box.pts, dtype=np.float32)
        rows = max(1, int(getattr(self, 'tile_rows', self.tile_rows_var.get())))
        cols = max(1, int(getattr(self, 'tile_cols', self.tile_cols_var.get())))

        step_u, step_v = self._compute_tile_block_steps(ix, iy)
        clones = []
        if step_u == 0 and step_v == 0:
            self.tile_u_min = 0
            self.tile_u_max = 0
            self.tile_v_min = 0
            self.tile_v_max = 0
            return []

        self.tile_u_min = min(0, step_u)
        self.tile_u_max = max(0, step_u)
        self.tile_v_min = min(0, step_v)
        self.tile_v_max = max(0, step_v)

        for v_shift in range(self.tile_v_min, self.tile_v_max + 1):
            for u_shift in range(self.tile_u_min, self.tile_u_max + 1):
                if u_shift == 0 and v_shift == 0:
                    continue
                clones.extend(self._build_tile_cells(
                    base_pts, rows, cols,
                    start_idx=len(self.slots) + len(clones),
                    u_shift=float(u_shift),
                    v_shift=float(v_shift),
                ))

        return clones

    def _make_adjacent_slot(self, slot: Slot, edge_i: int,
                            ix: int, iy: int) -> Optional[Slot]:
        if self.drag_edge_start is None or not slot.pts:
            return None
        pts = np.array(slot.pts, dtype=np.float32)
        a = pts[edge_i]
        b = pts[(edge_i + 1) % 4]
        edge_vec = b - a
        if np.linalg.norm(edge_vec) < 1e-3:
            return None

        start = np.array(self.drag_edge_start, dtype=np.float32)
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
        return Slot(pts=new_pts, label=f'S{len(self.slots)}',
                    slot_idx=len(self.slots))

    def _edge_interior_direction(self, slot: Slot, edge_i: int) -> np.ndarray:
        pts = np.array(slot.pts, dtype=np.float32)
        centroid = np.mean(pts, axis=0)
        a = pts[edge_i]
        b = pts[(edge_i + 1) % 4]
        mid = (a + b) / 2.0
        return centroid - mid

    def _set_ref(self):
        path = filedialog.askopenfilename(
            title="Select Reference Image (empty lot)",
            filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp"), ("All", "*.*")])
        if path:
            ref_img = cv2.imread(path)
            if ref_img is None:
                messagebox.showerror("Error", f"Cannot load reference image: {path}")
                return
            self.ref_path = path
            self.ref_cv = ref_img
            self.ref_pil = Image.fromarray(cv2.cvtColor(ref_img, cv2.COLOR_BGR2RGB))
            self.ref_label.configure(text=f"Ref: {os.path.basename(path)}")
            self._update_detected_lines()
            self.status_var.set(
                f"Reference image set: {os.path.basename(path)}")

            if self.test_cv is not None and self.test_cv.shape[:2] != ref_img.shape[:2]:
                self.test_cv = cv2.resize(self.test_cv,
                                          (ref_img.shape[1], ref_img.shape[0]),
                                          interpolation=cv2.INTER_LINEAR)
                self.test_pil = Image.fromarray(cv2.cvtColor(self.test_cv, cv2.COLOR_BGR2RGB))
                if self._display_cv is self.test_cv or self._display_pil is self.test_pil:
                    self._display_cv = self.test_cv
                    self._display_pil = self.test_pil
                self.status_var.set(
                    f"Resized TEST image to match new Reference size {ref_img.shape[1]}x{ref_img.shape[0]}")

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

        if ref.shape[:2] != test.shape[:2]:
            test = cv2.resize(test, (ref.shape[1], ref.shape[0]),
                               interpolation=cv2.INTER_LINEAR)
            self.test_cv = test
            self.test_pil = Image.fromarray(cv2.cvtColor(test, cv2.COLOR_BGR2RGB))
            self.status_var.set(
                f"Resized TEST image to match Reference size {ref.shape[1]}x{ref.shape[0]} for classification")
            if self._display_cv is not None and self.test_path:
                self._display_cv = self.test_cv
                self._display_pil = self.test_pil

        method_key = self._selected_classifier_key()
        method_name = classification.classifier_label(method_key)

        results = []
        for s in self.slots:
            ref_roi = self._warp_roi(ref, s)
            test_roi = self._warp_roi(test, s)
            if ref_roi is None or test_roi is None:
                s.prediction = -1
                continue

            clf = classification.classify_roi(ref_roi, test_roi, method_key)
            s.mad_value = round(float(clf['metric_value']), 1)
            s.confidence = round(float(clf['confidence']), 2)
            s.prediction = 1 if clf['occupied'] else 0
            status = 'OCC' if s.prediction else 'FREE'
            m = clf['metrics']
            results.append(
                f"{s.label}: {status} "
                f"({clf['metric_name']}={float(clf['metric_value']):.3f}, "
                f"conf={s.confidence*100:.0f}%, "
                f"ssim={float(m['mean_ssim']):.3f}, "
                f"diff={float(m['diff_ratio'])*100:.1f}%, "
                f"edge={float(m['edge_ratio'])*100:.1f}%)"
            )

        self._refresh_list()
        self._redraw()

        occ = sum(1 for s in self.slots if s.prediction == 1)
        free = sum(1 for s in self.slots if s.prediction == 0)
        self.status_var.set(
            f"Classification done [{method_name}]: {occ} occupied, {free} free")

        msg = f"Results ({len(self.slots)} slots):\n\n"
        msg += '\n'.join(results)
        msg += f"\n\nSummary: {occ} OCC / {free} FREE"
        messagebox.showinfo(f"ROI Classification Results — {method_name}", msg)

    def _warp_roi(self, gray, s: Slot):
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

        if self._load_image(path, is_test=True):
            try:
                self._classify()
            except Exception as e:
                messagebox.showerror("Quick Classify", f"Classification failed: {e}")
                self.status_var.set("Quick classification failed.")

    def _show_ref_image(self):
        if self.ref_pil is None:
            return
        self._display_pil = self.ref_pil
        self._display_cv = self.ref_cv
        self._redraw()
        self.status_var.set(f"Showing REFERENCE image: {os.path.basename(self.ref_path or '')}")

    def _show_test_image(self):
        if not self.test_path:
            messagebox.showinfo("View Test", "No test image loaded yet.")
            return
        if self.test_pil is not None:
            self._display_pil = self.test_pil
            self._display_cv = self.test_cv
            self._redraw()
            self.status_var.set(f"Showing TEST image: {os.path.basename(self.test_path)}")
            return
        test_cv = cv2.imread(self.test_path)
        if test_cv is not None:
            self._display_pil = Image.fromarray(cv2.cvtColor(test_cv, cv2.COLOR_BGR2RGB))
            self._display_cv = test_cv
            self._redraw()
            self.status_var.set(f"Showing TEST image: {os.path.basename(self.test_path)}")

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
        data = {
            'image': os.path.basename(self.img_path or ''),
            'image_size': [self.img_cv.shape[1], self.img_cv.shape[0]]
                          if self.img_cv is not None else [0, 0],
            'roi_size': ROI_SIZE,
            'n_slots': len(self.slots),
            'rois': [s.to_dict() for s in self.slots],
            'grid_groups': [copy.deepcopy(g) for g in self.grid_groups],
            'active_grid_group': self.active_grid_group
        }
        with open(path, 'w') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        self._save_auto_config()

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

    def _refresh_list(self):
        self.slot_list.delete(0, tk.END)
        for i, s in enumerate(self.slots):
            line = f"{'>' if i == self.selected else ' '} "
            line += f"{s.label or f'#{i}':5s}"
            line += f"  area={s.area:6.0f}"
            if s.prediction == 1:
                line += f"  OCC  SCORE={s.mad_value:.1f}"
            elif s.prediction == 0:
                line += f"  FREE SCORE={s.mad_value:.1f}"
            self.slot_list.insert(tk.END, line)

        if 0 <= self.selected < len(self.slots):
            self.slot_list.selection_set(self.selected)
            self.slot_list.see(self.selected)

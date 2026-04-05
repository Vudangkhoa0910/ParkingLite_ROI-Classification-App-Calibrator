#!/usr/bin/env python3
"""
ParkingLite ROI Calibrator — Full GUI Application.

Visual tool to define parking slot ROIs on any camera image.
Supports perspective quadrilaterals (angled camera views).

Usage:
    python3 roi_calibration_tool.py
    python3 roi_calibration_tool.py --image path/to/image.jpg
"""

import argparse
import os
import sys
import tkinter as tk

from app_core import AppLogic
from app_drawing import DrawingMixin
from app_layout import LayoutMixin
from app_mouse import MouseMixin


class App(tk.Tk, LayoutMixin, DrawingMixin, MouseMixin, AppLogic):
    def __init__(self, image_path=None):
        super().__init__()
        self.title("ParkingLite ROI Calibrator")
        self.geometry("1400x850")
        self.configure(bg='#1e1e2e')

        self.img_cv = None
        self.img_pil = None
        self.ref_cv = None
        self.ref_pil = None
        self.test_cv = None
        self.test_pil = None
        self._display_pil = None
        self._display_cv = None
        self.img_path = None
        self.ref_path = None
        self.test_path = None

        self.slots = []
        self.detected_lines = []
        self.hovered_detected_line = -1
        self.selected = -1
        self.undo_stack = []
        self.redo_stack = []
        self.zoom = 1.0
        self.pan = [0, 0]

        self.mode = 'edit'
        self.draw_pts = []
        self.dragging_corner = -1
        self.dragging_slot = False
        self.dragging_edge = -1
        self.dragging_grid_corner = -1
        self.dragging_grid_edge = -1
        self.grid_bounds_orig = []
        self.drag_edge_start = (0, 0)
        self.drag_slot_start = []
        self.preview_slot = None
        self.hovered_slot = -1
        self.hovered_edge = -1
        self.hovered_grid_corner = -1
        self.hovered_grid_edge = -1
        self._prev_hovered_grid_corner = -1
        self._prev_hovered_grid_edge = -1
        self.drag_start = (0, 0)
        self.drag_slot_offsets = []
        self.draw_preview_pt = None

        self.grid_pts = []
        self.grid_bounds = []
        self.grid_groups = []
        self.active_grid_group = -1
        self.dragging_grid_group = -1
        self.hovered_grid_group = -1
        self._prev_hovered_grid_group = -1
        self.grid_rows = tk.IntVar(value=2)
        self.grid_cols = tk.IntVar(value=8)
        self.tile_rows_var = tk.IntVar(value=1)
        self.tile_cols_var = tk.IntVar(value=1)
        self.select_drag_start = None
        self.select_drag_box = None

        self.tile_box = None
        self.tile_drag_start = None
        self.dragging_tile = False
        self.tile_preview_slots = []
        self.tile_rows = 1
        self.tile_cols = 1
        self.tile_group_start = -1
        self.tile_base_count = 0
        self.tile_u_min = 0
        self.tile_u_max = 0
        self.tile_v_min = 0
        self.tile_v_max = 0

        self._build_ui()
        self._bind_keys()
        self._update_mode_button_styles()
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        if image_path and os.path.exists(image_path):
            self._load_image(image_path)


def main():
    parser = argparse.ArgumentParser(description='ParkingLite ROI Calibrator')
    parser.add_argument('--image', '-i', help='Image to open on start')
    args, _ = parser.parse_known_args()

    if not args.image and len(sys.argv) > 1 and not sys.argv[1].startswith('-'):
        args.image = sys.argv[1]

    app = App(image_path=args.image)
    app.mainloop()


if __name__ == '__main__':
    main()

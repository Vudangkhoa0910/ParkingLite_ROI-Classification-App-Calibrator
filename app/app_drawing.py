import numpy as np
import tkinter as tk
from typing import List
from PIL import Image, ImageTk

class DrawingMixin:
    def _redraw(self):
        self.canvas.delete('all')
        display = getattr(self, '_display_pil', self.img_pil)
        if display is None:
            return

        w = int(display.width * self.zoom)
        h = int(display.height * self.zoom)
        resized = display.resize((w, h), Image.LANCZOS)
        self._tk_img = ImageTk.PhotoImage(resized)
        self.canvas.create_image(self.pan[0], self.pan[1], anchor=tk.NW,
                                  image=self._tk_img)

        for i, s in enumerate(self.slots):
            self._draw_slot(i, s)

        if self.preview_slot is not None:
            self._draw_preview_slot(self.preview_slot)

        if self.grid_groups:
            self._draw_grid_bounds()

        if self.hovered_edge >= 0 and self.hovered_slot >= 0:
            self._draw_hover_edge(self.slots[self.hovered_slot], self.hovered_edge)
        elif self.hovered_detected_line >= 0:
            self._draw_hover_detected_line(self.hovered_detected_line)

        if self.hovered_grid_edge >= 0:
            self._draw_hover_grid_edge(self.hovered_grid_edge)

        if self.mode == 'draw' and self.draw_preview_pt is not None:
            self._draw_snap_preview(self.draw_preview_pt)

        if self.mode == 'draw' and self.draw_pts:
            self._draw_progress(self.draw_pts, '#FF6600', "Slot")
        if self.mode == 'tile' and self.draw_pts:
            self._draw_progress(self.draw_pts, '#FF00AA', "Tile")
        if self.mode == 'grid' and self.grid_pts:
            labels = ['TL', 'TR', 'BR', 'BL']
            self._draw_progress(self.grid_pts, '#FF0066', "Grid", labels)
        if self.mode == 'tile' and getattr(self, 'tile_preview_slots', None):
            for slot in self.tile_preview_slots:
                self._draw_preview_slot(slot)
        if self.mode == 'select' and self.select_drag_box is not None:
            (x0, y0), (x1, y1) = self.select_drag_box
            a = self._img2canvas(x0, y0)
            b = self._img2canvas(x1, y0)
            c = self._img2canvas(x1, y1)
            d = self._img2canvas(x0, y1)
            self.canvas.create_polygon([a[0], a[1], b[0], b[1], c[0], c[1], d[0], d[1]],
                                       outline='#ffffff', fill='', width=2,
                                       dash=(6, 4), tags='select_box')
            self.canvas.create_text((a[0] + c[0]) / 2, a[1] - 14,
                                    text='Select Grid', fill='#ffffff',
                                    font=('Segoe UI', 10, 'bold'), tags='select_box')

    def _draw_slot(self, idx, s):
        is_sel = (idx == self.selected)
        spts = [self._img2canvas(p[0], p[1]) for p in s.pts]

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
        self.canvas.create_polygon(flat, fill=fill, outline='',
                                   stipple='gray25', tags=f'slot_{idx}')
        lw = 3 if is_sel else 2
        self.canvas.create_polygon(flat, fill='', outline=outline,
                                   width=lw, tags=f'slot_{idx}')

        for j, (cx, cy) in enumerate(spts):
            r = 6 if is_sel else 4
            color = '#ff4444' if is_sel else '#ffcc00'
            self.canvas.create_oval(cx - r, cy - r, cx + r, cy + r,
                                     fill=color, outline='white', width=1,
                                     tags=f'corner_{idx}_{j}')

        ccx, ccy = s.center
        scx, scy = self._img2canvas(ccx, ccy)
        label = s.label or f'#{idx}'
        if s.prediction >= 0:
            status = 'OCC' if s.prediction == 1 else 'FREE'
            label += f'\n{s.mad_value:.1f} ({status})'

        self.canvas.create_text(scx, scy, text=label, fill='white',
                                 font=('Consolas', 9, 'bold'),
                                 tags=f'label_{idx}')

    def _draw_preview_slot(self, s):
        spts = [self._img2canvas(p[0], p[1]) for p in s.pts]
        flat = [coord for pt in spts for coord in pt]
        self.canvas.create_polygon(flat, fill='', outline='#00ff88',
                                   width=3, dash=(6, 4), tags='preview_slot')

    def _draw_snap_preview(self, pt):
        cx, cy = self._img2canvas(pt[0], pt[1])
        self.canvas.create_oval(cx - 6, cy - 6, cx + 6, cy + 6,
                                outline='#00ffcc', width=2, dash=(4, 2),
                                fill='', tags='snap_preview')
        self.canvas.create_text(cx + 14, cy - 10, text='LINE',
                                fill='#00ffcc', font=('Segoe UI', 9, 'bold'),
                                tags='snap_preview')

    def _draw_hover_edge(self, slot, edge_i):
        a = self._img2canvas(*slot.pts[edge_i])
        b = self._img2canvas(*slot.pts[(edge_i + 1) % 4])
        self.canvas.create_line(a[0], a[1], b[0], b[1],
                                fill='#ffff00', width=5,
                                dash=(6, 4), capstyle=self.tk.ROUND,
                                tags='hover_edge')

    def _draw_hover_detected_line(self, line_idx):
        x1, y1, x2, y2 = self.detected_lines[line_idx]
        a = self._img2canvas(x1, y1)
        b = self._img2canvas(x2, y2)
        self.canvas.create_line(a[0], a[1], b[0], b[1],
                                fill='#00ffcc', width=4,
                                dash=(4, 2), capstyle=self.tk.ROUND,
                                tags='hover_detected_line')

    def _draw_hover_grid_edge(self, edge_i, bounds=None):
        if bounds is None or len(bounds) != 4:
            return
        a = self._img2canvas(*bounds[edge_i])
        b = self._img2canvas(*bounds[(edge_i + 1) % 4])
        self.canvas.create_line(a[0], a[1], b[0], b[1],
                                fill='#88ff88', width=5,
                                dash=(4, 2), capstyle=self.tk.ROUND,
                                tags='hover_grid_edge')

    def _draw_progress(self, pts, color, prefix, labels=None):
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

    def _draw_grid_bounds(self):
        if not self.grid_groups:
            return
        for group_idx, group in enumerate(self.grid_groups):
            bounds = group['bounds']
            if len(bounds) != 4:
                continue
            pts = [self._img2canvas(x, y) for x, y in bounds]
            flat = [coord for pt in pts for coord in pt]
            is_active = group_idx == self.active_grid_group
            is_hovered = group_idx == self.hovered_grid_group
            outline = '#00ff88' if is_active or is_hovered else '#88ccff'
            width = 4 if is_active else 3
            self.canvas.create_polygon(flat, fill='', outline=outline,
                                       width=width, dash=(6, 4), tags='grid_bounds')
            if is_active or is_hovered:
                for idx, (cx, cy) in enumerate(pts):
                    r = 6 if idx == self.hovered_grid_corner else 5
                    self.canvas.create_oval(cx - r, cy - r, cx + r, cy + r,
                                            fill=outline, outline='white', width=2,
                                            tags=f'grid_corner_{idx}')
                    label = ['TL', 'TR', 'BR', 'BL'][idx]
                    self.canvas.create_text(cx + 12, cy - 12, text=label,
                                            fill=outline, font=('Segoe UI', 9, 'bold'),
                                            tags='grid_bounds')
                if is_hovered and self.hovered_grid_edge >= 0:
                    self._draw_hover_grid_edge(self.hovered_grid_edge, bounds)

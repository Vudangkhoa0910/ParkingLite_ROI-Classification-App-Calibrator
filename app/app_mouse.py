import copy
import numpy as np
from models import Slot
from constants import LINE_FOCUS_THRESHOLD
import grid_helpers

class MouseMixin:
    def _find_grid_group_for_slot(self, slot_idx: int) -> int:
        for gi, group in enumerate(getattr(self, 'grid_groups', [])):
            start = int(group.get('start', 0))
            count = int(group.get('count', 0))
            if start <= slot_idx < (start + count):
                return gi
        return -1

    def _collect_linked_corners(self, slot_idx: int, corner_idx: int,
                                match_threshold: float = 2.5):
        if slot_idx < 0 or slot_idx >= len(self.slots):
            return []
        if corner_idx < 0 or corner_idx >= 4:
            return []

        group_idx = self._find_grid_group_for_slot(slot_idx)
        if group_idx < 0:
            return []

        group = self.grid_groups[group_idx]
        start = max(0, int(group.get('start', 0)))
        count = max(0, int(group.get('count', 0)))
        end = min(len(self.slots), start + count)

        anchor = np.array(self.slots[slot_idx].pts[corner_idx], dtype=np.float32)
        linked = []
        for si in range(start, end):
            for ci, p in enumerate(self.slots[si].pts):
                pt = np.array(p, dtype=np.float32)
                if float(np.linalg.norm(pt - anchor)) <= float(match_threshold):
                    linked.append((si, ci, float(p[0]), float(p[1])))
        return linked

    def _on_click(self, event):
        if self.img_cv is None:
            return
        ix, iy = self._canvas2img(event.x, event.y)
        ix, iy = int(ix), int(iy)
        if not (self.mode == 'tile' and self.tile_box is not None):
            ix, iy = self._snap_to_detected_line(ix, iy, LINE_FOCUS_THRESHOLD)

        if self.mode == 'draw':
            self._snapshot()
            self.draw_pts.append([ix, iy])
            if len(self.draw_pts) == 4:
                slot = Slot(pts=self.draw_pts.copy(),
                            label=f'S{len(self.slots)}',
                            slot_idx=len(self.slots))
                self.slots.append(slot)
                self.selected = len(self.slots) - 1
                self.draw_pts.clear()
                self._refresh_list()
                self._save_auto_config()
                self.status_var.set(
                    f"Created slot #{len(self.slots)-1} — "
                    f"Click 4 more corners for next slot, or switch to Edit mode")
            self._redraw()
            return

        if self.mode == 'tile':
            if self.tile_box is None:
                self._snapshot()
                self.draw_pts.append([ix, iy])
                if len(self.draw_pts) == 4:
                    self.tile_rows = max(1, int(self.tile_rows_var.get()))
                    self.tile_cols = max(1, int(self.tile_cols_var.get()))
                    normalized = self._normalize_quad_points(self.draw_pts)
                    self.tile_box = Slot(pts=normalized.astype(int).tolist(),
                                         label='TILE_TEMPLATE',
                                         slot_idx=-1)
                    self.tile_group_start = len(self.slots)
                    self.tile_u_min = 0
                    self.tile_u_max = 0
                    self.tile_v_min = 0
                    self.tile_v_max = 0
                    base_slots = self._build_tile_cells(
                        np.array(self.tile_box.pts, dtype=np.float32),
                        self.tile_rows,
                        self.tile_cols,
                        start_idx=len(self.slots),
                    )
                    self.slots.extend(base_slots)
                    self.tile_base_count = len(base_slots)
                    self.selected = len(self.slots) - 1 if base_slots else -1
                    self.draw_pts.clear()
                    self._refresh_list()
                    self._save_auto_config()
                    self.status_var.set(
                        f"Tile base created ({self.tile_rows}x{self.tile_cols}). "
                        "Drag inside to clone more boxes.")
                self._redraw()
                return
            if self.tile_box.contains(ix, iy):
                self._snapshot()
                self.dragging_tile = True
                self.tile_drag_start = (ix, iy)
                self.tile_preview_slots = []
                self.status_var.set("Drag to replicate this tile block.")
                return
            # Click outside tile: return to Edit mode and apply click for immediate selection/edit.
            self.tile_box = None
            self.draw_pts.clear()
            self.tile_preview_slots = []
            self.dragging_tile = False
            self.tile_drag_start = None
            self._set_mode('edit')
            self.status_var.set("Exited Tile mode. Switched to Edit mode.")
            self._on_click(event)
            return

        if self.mode == 'edit':
            if self.grid_groups:
                group_idx, gci, gei = self._find_nearest_grid_handle(
                    ix, iy, int(15 / self.zoom))
                if group_idx >= 0 and gci >= 0:
                    self._snapshot()
                    self.dragging_grid_group = group_idx
                    self.dragging_grid_corner = gci
                    self.active_grid_group = group_idx
                    self.grid_bounds = self.grid_groups[group_idx]['bounds']
                    self.status_var.set("Drag grid corner to reshape this auto grid.")
                    return
            if self.selected >= 0:
                s = self.slots[self.selected]
                ci = s.nearest_corner(ix, iy, int(15 / self.zoom))
                if ci >= 0:
                    self._snapshot()
                    self.dragging_corner = ci
                    self.drag_corner_anchor_start = [float(s.pts[ci][0]), float(s.pts[ci][1])]
                    self.drag_linked_corners = self._collect_linked_corners(self.selected, ci)
                    if len(self.drag_linked_corners) <= 1:
                        self.drag_linked_corners = []
                    return
                edge_i = s.nearest_edge(ix, iy, int(15 / self.zoom))
                if edge_i >= 0:
                    self._snapshot()
                    self.dragging_edge = edge_i
                    self.drag_edge_start = s.closest_point_on_edge(ix, iy, edge_i)
                    self.drag_slot_start = copy.deepcopy(s.pts)
                    self.preview_slot = None
                    self.status_var.set(
                        "Drag selected edge outward to create an adjacent slot. Release to insert.")
                    return
            for i in range(len(self.slots) - 1, -1, -1):
                s = self.slots[i]
                ci = s.nearest_corner(ix, iy, int(15 / self.zoom))
                if ci >= 0:
                    self.selected = i
                    self._snapshot()
                    self.dragging_corner = ci
                    self.drag_corner_anchor_start = [float(s.pts[ci][0]), float(s.pts[ci][1])]
                    self.drag_linked_corners = self._collect_linked_corners(self.selected, ci)
                    if len(self.drag_linked_corners) <= 1:
                        self.drag_linked_corners = []
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
            return

        if self.mode == 'grid':
            if len(self.grid_pts) < 4:
                self.grid_pts.append([ix, iy])
                if len(self.grid_pts) == 4:
                    self._generate_grid()
            self._redraw()
            return

        if self.mode == 'select':
            if self.grid_groups:
                group_idx, gci, gei = self._find_nearest_grid_handle(
                    ix, iy, int(15 / self.zoom))
                if group_idx >= 0 and gci >= 0:
                    self._snapshot()
                    self.dragging_grid_group = group_idx
                    self.dragging_grid_corner = gci
                    self.active_grid_group = group_idx
                    self.hovered_grid_group = group_idx
                    self.grid_bounds = self.grid_groups[group_idx]['bounds']
                    self.select_drag_start = None
                    self.select_drag_box = None
                    self.status_var.set("Drag selected grid corner to reshape this auto grid.")
                    return
                if group_idx >= 0:
                    self.active_grid_group = group_idx
                    self.hovered_grid_group = group_idx
                    self.grid_bounds = self.grid_groups[group_idx]['bounds']
                    self.selected = -1
                    self.select_drag_start = None
                    self.select_drag_box = None
                    self._refresh_list()
                    self._redraw()
                    self._save_auto_config()
                    self.status_var.set(f"Selected grid #{group_idx + 1}")
                    return
            self.select_drag_start = (ix, iy)
            self.select_drag_box = None
            self.active_grid_group = -1
            self.grid_bounds = []
            self.selected = -1
            self._refresh_list()
            self._redraw()
            self.status_var.set("Drag to select a grid group, or click its border/corner.")

    def _on_drag(self, event):
        if self.img_cv is None or self.mode not in ('edit', 'select', 'tile'):
            return
        ix, iy = self._canvas2img(event.x, event.y)
        h, w = self.img_cv.shape[:2]
        ix = max(0, min(w - 1, int(ix)))
        iy = max(0, min(h - 1, int(iy)))
        if not (self.mode == 'tile' and self.dragging_tile):
            ix, iy = self._snap_to_detected_line(ix, iy, max(6, int(12 / self.zoom)))

        if self.dragging_grid_corner >= 0 and self.dragging_grid_group >= 0:
            group = self.grid_groups[self.dragging_grid_group]
            group['bounds'][self.dragging_grid_corner] = [ix, iy]
            self.grid_bounds = group['bounds']
            grid_helpers.regenerate_grid_for_group(self.dragging_grid_group, self.grid_groups, self.slots)
            self._refresh_list()
            self._redraw()
            return

        if self.mode == 'tile' and self.dragging_tile and self.tile_box is not None:
            self.tile_preview_slots = self._build_tile_clone_slots(ix, iy)
            self._redraw()
            return

        if self.mode == 'select' and self.select_drag_start is not None and self.dragging_grid_corner < 0 and self.dragging_grid_edge < 0:
            self.select_drag_box = [self.select_drag_start, (ix, iy)]
            self._redraw()
            return

        if self.dragging_grid_edge >= 0 and self.dragging_grid_group >= 0:
            group = self.grid_groups[self.dragging_grid_group]
            dx = ix - self.drag_start[0]
            dy = iy - self.drag_start[1]
            new_bounds = []
            for i, pt in enumerate(self.grid_bounds_orig):
                if i == self.dragging_grid_edge or i == (self.dragging_grid_edge + 1) % 4:
                    new_pt = np.array(pt, dtype=np.float32) + np.array([dx, dy], dtype=np.float32)
                    new_bounds.append([int(round(new_pt[0])), int(round(new_pt[1]))])
                else:
                    new_bounds.append(list(pt))
            group['bounds'] = new_bounds
            self.grid_bounds = new_bounds
            grid_helpers.regenerate_grid_for_group(self.dragging_grid_group, self.grid_groups, self.slots)
            self._refresh_list()
            self._redraw()
            return

        if self.dragging_edge >= 0 and self.selected >= 0:
            self.preview_slot = self._make_adjacent_slot(
                self.slots[self.selected], self.dragging_edge, ix, iy)
            self._redraw()
            return

        if self.dragging_corner >= 0 and self.selected >= 0:
            linked = getattr(self, 'drag_linked_corners', None)
            anchor_start = getattr(self, 'drag_corner_anchor_start', None)
            if linked and anchor_start is not None:
                dx = float(ix) - float(anchor_start[0])
                dy = float(iy) - float(anchor_start[1])
                for si, ci, bx, by in linked:
                    nx = max(0, min(w - 1, int(round(float(bx) + dx))))
                    ny = max(0, min(h - 1, int(round(float(by) + dy))))
                    self.slots[si].pts[ci] = [nx, ny]
            else:
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
        if self.dragging_edge >= 0:
            if self.preview_slot is not None:
                self.slots.append(self.preview_slot)
                self.selected = len(self.slots) - 1
                self._refresh_list()
                self.status_var.set("Added adjacent slot from edge drag.")
            self.preview_slot = None
        if self.mode == 'tile' and self.dragging_tile:
            created_count = self._apply_tile_group()
            self._refresh_list()
            self._save_auto_config()
            self.tile_preview_slots = []
            self.dragging_tile = False
            self.tile_drag_start = None
            self.tile_box = None
            self.tile_group_start = -1
            self.tile_base_count = 0
            self._set_mode('edit')
            if created_count > 0:
                self.status_var.set(
                    f"Tile group applied: +{created_count} boxes. Edit whole group via grid handles.")
            else:
                self.status_var.set("Tile group applied. Edit whole group via grid handles.")
            return
        if self.mode == 'select' and self.select_drag_start is not None and self.select_drag_box is not None:
            x0, y0 = self.select_drag_start
            x1, y1 = self.select_drag_box[1]
            x_min, x_max = sorted((x0, x1))
            y_min, y_max = sorted((y0, y1))
            selected_index = -1
            for group_idx, group in enumerate(self.grid_groups):
                if any(x_min <= px <= x_max and y_min <= py <= y_max
                       for px, py in group['bounds']):
                    selected_index = group_idx
                    break
            if selected_index >= 0:
                self.active_grid_group = selected_index
                self.hovered_grid_group = selected_index
                self.grid_bounds = self.grid_groups[selected_index]['bounds']
                self.status_var.set(f"Selected grid #{selected_index + 1}")
            else:
                self.status_var.set("No grid group found inside selection.")
            self.select_drag_start = None
            self.select_drag_box = None
            self._refresh_list()
            self._redraw()
        if (self.dragging_corner >= 0 or self.dragging_slot or
                self.dragging_grid_corner >= 0 or self.dragging_grid_edge >= 0):
            self._refresh_list()
            self._save_auto_config()
        self.dragging_corner = -1
        self.dragging_slot = False
        self.dragging_edge = -1
        self.dragging_grid_corner = -1
        self.dragging_grid_edge = -1
        self.dragging_grid_group = -1
        self.drag_corner_anchor_start = None
        self.drag_linked_corners = []

    def _on_right_click(self, event):
        if self.mode == 'draw' and self.draw_pts:
            self.draw_pts.pop()
            self._redraw()
        elif self.mode == 'tile' and self.draw_pts:
            self.draw_pts.pop()
            self._redraw()
        elif self.mode == 'grid' and self.grid_pts:
            self.grid_pts.pop()
            self._redraw()
        elif self.mode == 'tile' and self.dragging_tile:
            self.dragging_tile = False
            self.tile_preview_slots = []
            self.tile_drag_start = None
            self.status_var.set("Tile clone canceled.")
            self._redraw()
        elif self.mode == 'tile' and self.tile_box is not None:
            self.tile_box = None
            self.draw_pts.clear()
            self.tile_preview_slots = []
            self.dragging_tile = False
            self.tile_drag_start = None
            self._set_mode('edit')
            self.status_var.set("Exited Tile mode. Switched to Edit mode.")
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
            if self.mode in ('edit', 'draw', 'select', 'tile'):
                best_slot = -1
                best_edge = -1
                threshold = LINE_FOCUS_THRESHOLD
                best_dist = threshold + 1
                self.draw_preview_pt = None

                if self.mode in ('edit', 'select'):
                    if self.grid_groups:
                        group_idx, gci, gei = self._find_nearest_grid_handle(
                            ix, iy, int(12 / self.zoom))
                        if group_idx >= 0 and gci >= 0:
                            self.hovered_grid_group = group_idx
                            self.hovered_grid_corner = gci
                            self.hovered_grid_edge = -1
                        elif group_idx >= 0 and gei >= 0:
                            self.hovered_grid_group = group_idx
                            self.hovered_grid_corner = -1
                            self.hovered_grid_edge = gei
                        else:
                            self.hovered_grid_group = -1
                            self.hovered_grid_corner = -1
                            self.hovered_grid_edge = -1
                    else:
                        self.hovered_grid_group = -1
                        self.hovered_grid_corner = -1
                        self.hovered_grid_edge = -1

                if self.mode == 'edit':
                    if self.selected >= 0:
                        edge_i = self.slots[self.selected].nearest_edge(ix, iy, threshold)
                        if edge_i >= 0:
                            best_slot = self.selected
                            best_edge = edge_i
                            best_dist = 0
                    if best_slot < 0:
                        for idx, slot in enumerate(self.slots):
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

                prev_detected_line = self.hovered_detected_line
                self.hovered_detected_line = -1
                if self.detected_lines and self.mode in ('edit', 'draw', 'select'):
                    line_idx = self._nearest_detected_line(ix, iy, threshold)
                    if line_idx >= 0:
                        self.hovered_detected_line = line_idx
                        if self.mode == 'draw':
                            snapped = self._project_to_line(ix, iy, line_idx)
                            self.draw_preview_pt = snapped

                if (best_edge != self.hovered_edge or best_slot != self.hovered_slot or
                        self.hovered_detected_line != prev_detected_line or
                        self.hovered_grid_group != self._prev_hovered_grid_group or
                        self.hovered_grid_corner != self._prev_hovered_grid_corner or
                        self.hovered_grid_edge != self._prev_hovered_grid_edge):
                    self.hovered_edge = best_edge
                    self.hovered_slot = best_slot
                    self._prev_hovered_grid_group = self.hovered_grid_group
                    self._prev_hovered_grid_corner = self.hovered_grid_corner
                    self._prev_hovered_grid_edge = self.hovered_grid_edge
                    self._redraw()

                if best_edge >= 0:
                    self.canvas.config(cursor='hand2')
                    self.status_var.set(
                        "Hover edge selected — click and drag to create adjacent slot")
                elif self.hovered_grid_corner >= 0:
                    self.canvas.config(cursor='hand2')
                    self.status_var.set(
                        "Hover grid corner — click and drag to reshape the full auto grid")
                elif self.hovered_grid_edge >= 0:
                    self.canvas.config(cursor='hand2')
                    self.status_var.set(
                        "Hover grid edge — click and drag to reshape the full auto grid")
                elif self.hovered_detected_line >= 0:
                    self.canvas.config(cursor='hand2')
                    self.status_var.set(
                        "Hover image line — edge focus activated")
                elif self.mode == 'tile' and self.tile_box is not None and self.tile_box.contains(ix, iy):
                    self.canvas.config(cursor='hand2')
                    self.status_var.set(
                        "Tile base slot — click inside and drag to clone boxes")
                else:
                    self.canvas.config(cursor='crosshair')
                    self.status_var.set(
                        f"{self.mode.title()} mode — Pos: ({int(ix)}, {int(iy)})")
            else:
                if (self.hovered_edge != -1 or self.hovered_slot != -1 or
                        self.hovered_detected_line != -1 or self.hovered_grid_corner != -1 or
                        self.hovered_grid_edge != -1):
                    self.hovered_edge = -1
                    self.hovered_slot = -1
                    self.hovered_detected_line = -1
                    self.hovered_grid_corner = -1
                    self.hovered_grid_edge = -1
                    self._prev_hovered_grid_corner = -1
                    self._prev_hovered_grid_edge = -1
                    self._redraw()
                self.canvas.config(cursor='crosshair')
        else:
            if (self.hovered_edge != -1 or self.hovered_slot != -1 or
                    self.hovered_detected_line != -1 or self.hovered_grid_corner != -1 or
                    self.hovered_grid_edge != -1):
                self.hovered_edge = -1
                self.hovered_slot = -1
                self.hovered_detected_line = -1
                self.hovered_grid_corner = -1
                self.hovered_grid_edge = -1
                self._prev_hovered_grid_corner = -1
                self._prev_hovered_grid_edge = -1
                self._redraw()
            self.canvas.config(cursor='crosshair')

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

import copy
import math
import numpy as np
from typing import List, Tuple
from models import Slot


def nearest_grid_edge_in_bounds(px: float, py: float, bounds, threshold: int = 12) -> int:
    if len(bounds) != 4:
        return -1
    best_i = -1
    best_d = threshold + 1
    p = np.array([px, py], dtype=np.float32)
    for i in range(4):
        a = np.array(bounds[i], dtype=np.float32)
        b = np.array(bounds[(i + 1) % 4], dtype=np.float32)
        d = Slot._point_segment_distance(p, a, b)
        if d < best_d:
            best_d = d
            best_i = i
    return best_i


def nearest_grid_corner_in_bounds(px: float, py: float, bounds, threshold: int = 12) -> int:
    best_i = -1
    best_d = threshold + 1
    for i, (gx, gy) in enumerate(bounds):
        d = math.hypot(px - gx, py - gy)
        if d < best_d:
            best_d = d
            best_i = i
    return best_i


def point_in_polygon(px: float, py: float, polygon: List[List[int]]) -> bool:
    inside = False
    j = len(polygon) - 1
    for i in range(len(polygon)):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        intersect = ((yi > py) != (yj > py)) and (
            px < (xj - xi) * (py - yi) / (yj - yi + 1e-9) + xi)
        if intersect:
            inside = not inside
        j = i
    return inside


def find_nearest_grid_handle(px: float, py: float, grid_groups, threshold: int = 12):
    nearest_corner_group = -1
    nearest_corner_idx = -1
    nearest_corner_dist = threshold + 1
    for group_idx, group in enumerate(grid_groups):
        for idx, (gx, gy) in enumerate(group['bounds']):
            d = math.hypot(px - gx, py - gy)
            if d < nearest_corner_dist:
                nearest_corner_dist = d
                nearest_corner_idx = idx
                nearest_corner_group = group_idx
    if nearest_corner_group >= 0 and nearest_corner_dist <= threshold:
        return nearest_corner_group, nearest_corner_idx, -1

    nearest_edge_group = -1
    nearest_edge_idx = -1
    nearest_edge_dist = threshold + 1
    for group_idx, group in enumerate(grid_groups):
        edge_idx = nearest_grid_edge_in_bounds(px, py, group['bounds'], threshold)
        if edge_idx >= 0:
            a = np.array(group['bounds'][edge_idx], dtype=np.float32)
            b = np.array(group['bounds'][(edge_idx + 1) % 4], dtype=np.float32)
            d = Slot._point_segment_distance(np.array([px, py], dtype=np.float32), a, b)
            if d < nearest_edge_dist:
                nearest_edge_dist = d
                nearest_edge_idx = edge_idx
                nearest_edge_group = group_idx
    if nearest_edge_group >= 0 and nearest_edge_dist <= threshold:
        return nearest_edge_group, -1, nearest_edge_idx

    for group_idx, group in enumerate(grid_groups):
        if point_in_polygon(px, py, group['bounds']):
            return group_idx, -1, -1

    return -1, -1, -1


def regenerate_grid_for_group(group_idx: int, grid_groups, slots):
    if group_idx < 0 or group_idx >= len(grid_groups):
        return
    group = grid_groups[group_idx]
    bounds = group['bounds']
    if len(bounds) != 4:
        return
    tl, tr, br, bl = [np.array(p, dtype=np.float64) for p in bounds]
    rows = group['rows']
    cols = group['cols']
    start = group['start']
    count = group['count']
    old_labels = [s.label for s in slots[start:start + count]]
    new_slots = []
    for r in range(rows):
        t0 = tl + (bl - tl) * (r / rows)
        t1 = tr + (br - tr) * (r / rows)
        b0 = tl + (bl - tl) * ((r + 1) / rows)
        b1 = tr + (br - tr) * ((r + 1) / rows)
        for c in range(cols):
            s0, s1 = c / cols, (c + 1) / cols
            p0 = t0 + (t1 - t0) * s0
            p1 = t0 + (t1 - t0) * s1
            p2 = b0 + (b1 - b0) * s1
            p3 = b0 + (b1 - b0) * s0
            pts = [[int(p0[0]), int(p0[1])], [int(p1[0]), int(p1[1])],
                   [int(p2[0]), int(p2[1])], [int(p3[0]), int(p3[1])]]
            idx = len(new_slots)
            label = old_labels[idx] if idx < len(old_labels) else f'S{start + idx}'
            new_slots.append(Slot(pts=pts, label=label, slot_idx=start + idx))
    slots[start:start + count] = new_slots

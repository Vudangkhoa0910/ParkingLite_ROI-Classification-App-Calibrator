import copy
import cv2
import numpy as np
from dataclasses import dataclass
from typing import List

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
        # Use a small tolerance so edge clicks still count as inside.
        if cv2.pointPolygonTest(pts, (float(px), float(py)), True) >= -4.0:
            return True
        # Fallback for non-simple or nearly degenerate polygons: use convex hull
        hull = cv2.convexHull(np.array(self.pts, dtype=np.float32))
        return cv2.pointPolygonTest(hull, (float(px), float(py)), True) >= -4.0

    def nearest_corner(self, px, py, threshold=15):
        best_i, best_d = -1, threshold + 1
        for i, (cx, cy) in enumerate(self.pts):
            d = ((px - cx) ** 2 + (py - cy) ** 2) ** 0.5
            if d < best_d:
                best_i, best_d = i, d
        return best_i

    def nearest_edge(self, px, py, threshold=15):
        best_i, best_d = -1, threshold + 1
        p = np.array([px, py], dtype=np.float32)
        for i in range(4):
            a = np.array(self.pts[i], dtype=np.float32)
            b = np.array(self.pts[(i + 1) % 4], dtype=np.float32)
            d = Slot._point_segment_distance(p, a, b)
            if d < best_d:
                best_i, best_d = i, d
        return best_i if best_d <= threshold else -1

    def closest_point_on_edge(self, px, py, edge_i):
        p = np.array([px, py], dtype=np.float32)
        a = np.array(self.pts[edge_i], dtype=np.float32)
        b = np.array(self.pts[(edge_i + 1) % 4], dtype=np.float32)
        v = b - a
        v_len2 = np.dot(v, v)
        if v_len2 < 1e-6:
            return a
        t = np.clip(np.dot(p - a, v) / v_len2, 0.0, 1.0)
        return a + t * v

    @staticmethod
    def _point_segment_distance(p, a, b):
        v = b - a
        w = p - a
        v_len2 = np.dot(v, v)
        if v_len2 < 1e-6:
            return float(np.linalg.norm(w))
        t = np.clip(np.dot(w, v) / v_len2, 0.0, 1.0)
        proj = a + t * v
        return float(np.linalg.norm(p - proj))

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
        return Slot(pts=[[x, y], [x + w, y], [x + w, y + h], [x, y + h]],
                    label=d.get('label', ''), slot_idx=d.get('slot_idx', -1))

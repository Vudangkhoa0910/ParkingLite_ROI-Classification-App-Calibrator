import json
import os
import copy
from typing import Tuple, List
from models import Slot
from constants import ROI_SIZE


def auto_config_path(image_path=None, ref_path=None, img_path=None):
    # Per-image config must follow the current working image first.
    path = image_path or img_path or ref_path
    if not path:
        return None
    base = os.path.splitext(path)[0]
    return base + '.roi.json'


def load_config_file(path):
    with open(path) as f:
        return json.load(f)


def load_auto_config(config_path):
    if not config_path or not os.path.exists(config_path):
        return None
    try:
        with open(config_path) as f:
            return json.load(f)
    except Exception:
        return None


def _serialize_state(state):
    return {
        'slots': [s.to_dict() for s in state.get('slots', [])],
        'draw_pts': copy.deepcopy(state.get('draw_pts', [])),
        'grid_bounds': copy.deepcopy(state.get('grid_bounds', [])),
        'grid_groups': copy.deepcopy(state.get('grid_groups', [])),
        'active_grid_group': state.get('active_grid_group', -1),
    }


def _deserialize_state(data):
    return {
        'slots': [Slot.from_dict(d) for d in data.get('slots', [])],
        'draw_pts': copy.deepcopy(data.get('draw_pts', [])),
        'grid_bounds': copy.deepcopy(data.get('grid_bounds', [])),
        'grid_groups': copy.deepcopy(data.get('grid_groups', [])),
        'active_grid_group': data.get('active_grid_group', -1),
    }


def load_history_stacks(data):
    undo_stack = [_deserialize_state(s) for s in data.get('undo_stack', [])]
    redo_stack = [_deserialize_state(s) for s in data.get('redo_stack', [])]
    return undo_stack, redo_stack


def save_auto_config(config_path, img_cv, ref_path, img_path, slots, grid_groups, active_grid_group,
                     undo_stack=None, redo_stack=None):
    if not config_path:
        return
    has_history = bool(undo_stack) or bool(redo_stack)
    if not slots and not grid_groups and not has_history:
        if os.path.exists(config_path):
            try:
                os.remove(config_path)
            except OSError:
                pass
        return
    data = {
        'image': os.path.basename(img_path or ref_path or ''),
        'image_size': [img_cv.shape[1], img_cv.shape[0]] if img_cv is not None else [0, 0],
        'roi_size': ROI_SIZE,
        'n_slots': len(slots),
        'rois': [s.to_dict() for s in slots],
        'grid_groups': [copy.deepcopy(g) for g in grid_groups],
        'active_grid_group': active_grid_group,
        'undo_stack': [_serialize_state(s) for s in (undo_stack or [])[-100:]],
        'redo_stack': [_serialize_state(s) for s in (redo_stack or [])[-100:]],
    }
    try:
        with open(config_path, 'w') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    except Exception:
        pass


def load_from_config_data(data):
    slots = [Slot.from_dict(d) for d in data.get('rois', data.get('slots', []))]
    grid_groups = data.get('grid_groups', []) or []
    active_grid_group = data.get('active_grid_group', -1)
    if grid_groups and (active_grid_group < 0 or active_grid_group >= len(grid_groups)):
        active_grid_group = 0
    grid_bounds = grid_groups[active_grid_group]['bounds'] if active_grid_group >= 0 and active_grid_group < len(grid_groups) else []
    return slots, grid_groups, active_grid_group, grid_bounds

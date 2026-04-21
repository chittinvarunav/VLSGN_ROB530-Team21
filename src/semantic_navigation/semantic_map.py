"""
Semantic map — fixed for single room with 5 known objects.
"""

import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional
import numpy as np

# Room bounds
X_MIN, X_MAX = -5.0, 5.0
Y_MIN, Y_MAX = -5.0, 5.0

KNOWN_LABELS = {
    "red cylinder":     "red cylinder",
    "blue box":         "blue box",
    "yellow box":       "yellow box",
    "white cylinder":   "white cylinder",
    "green cylinder":   "green cylinder",
    "blue cube":        "blue box",
    "red pillar":       "red cylinder",
    "pillar":           "white cylinder",
    "colored obstacle": "green cylinder",
    "cylinder":         "white cylinder",
}

def normalize_label(label):
    label = label.strip().lower()
    if label in KNOWN_LABELS:
        return KNOWN_LABELS[label]
    for key, val in KNOWN_LABELS.items():
        if key in label or label in key:
            return val
    return None

def in_room(position):
    x, y = position[0], position[1]
    return X_MIN <= x <= X_MAX and Y_MIN <= y <= Y_MAX


@dataclass
class SemanticObject:
    label: str
    position: list
    confidence: float
    timestamp: float = field(default_factory=time.time)
    observations: int = 1

    @property
    def position_np(self):
        return np.array(self.position)


class SemanticMap:
    def __init__(self, merge_distance=2.0):
        self.objects = []
        self.merge_distance = merge_distance

    def add_object(self, label, position, confidence):
        normalized = normalize_label(label)
        if normalized is None:
            return None

        pos = np.array(position[:3], dtype=float)

        # Ignore detections outside room bounds
        if not in_room(pos):
            return None

        for obj in self.objects:
            if obj.label == normalized:
                dist = np.linalg.norm(obj.position_np[:2] - pos[:2])
                if dist < self.merge_distance:
                    n = obj.observations
                    new_pos = (obj.position_np * n + pos) / (n + 1)
                    obj.position = new_pos.tolist()
                    obj.confidence = max(obj.confidence, confidence)
                    obj.observations += 1
                    obj.timestamp = time.time()
                    return obj

        new_obj = SemanticObject(label=normalized, position=pos.tolist(), confidence=confidence)
        self.objects.append(new_obj)
        return new_obj

    def query(self, label):
        normalized = normalize_label(label)
        if normalized:
            results = [o for o in self.objects if o.label == normalized]
        else:
            label = label.strip().lower()
            results = [o for o in self.objects if label in o.label or o.label in label]
        results.sort(key=lambda o: o.confidence, reverse=True)
        return results

    def query_exact(self, label):
        normalized = normalize_label(label) or label.strip().lower()
        return [o for o in self.objects if o.label == normalized]

    def query_nearest(self, position, max_distance=5.0):
        pos = np.array(position[:3])
        results = [(np.linalg.norm(o.position_np - pos), o)
                   for o in self.objects
                   if np.linalg.norm(o.position_np - pos) <= max_distance]
        results.sort(key=lambda x: x[0])
        return [o for _, o in results]

    def query_in_region(self, center, radius):
        center = np.array(center[:3])
        return [o for o in self.objects
                if np.linalg.norm(o.position_np - center) <= radius]

    def get_best_match(self, label):
        results = self.query(label)
        return results[0] if results else None

    def get_all(self):
        return list(self.objects)

    def get_labels(self):
        return list(set(o.label for o in self.objects))

    def get_stats(self):
        labels = self.get_labels()
        return {
            "total_objects": len(self.objects),
            "unique_labels": len(labels),
            "labels": labels,
            "total_observations": sum(o.observations for o in self.objects),
        }

    def remove_low_confidence(self, threshold=0.3):
        self.objects = [o for o in self.objects if o.confidence >= threshold]

    def clear(self):
        self.objects.clear()

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "merge_distance": self.merge_distance,
            "objects": [asdict(o) for o in self.objects],
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def load(self, path):
        path = Path(path)
        with open(path) as f:
            data = json.load(f)
        self.merge_distance = data.get("merge_distance", self.merge_distance)
        self.objects = []
        for obj_data in data["objects"]:
            normalized = normalize_label(obj_data["label"])
            if normalized and in_room(obj_data["position"]):
                obj_data["label"] = normalized
                self.objects.append(SemanticObject(**obj_data))

    def to_marker_data(self):
        import colorsys
        markers = []
        for obj in self.objects:
            h = hash(obj.label) % 360
            r, g, b = colorsys.hsv_to_rgb(h / 360.0, 0.8, 0.9)
            markers.append({
                "label": obj.label,
                "position": obj.position,
                "confidence": obj.confidence,
                "color": [r, g, b, 1.0],
            })
        return markers

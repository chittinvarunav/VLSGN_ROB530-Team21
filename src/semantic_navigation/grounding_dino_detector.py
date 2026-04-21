"""
Grounding DINO object detection module.
Fixed: thresholds re-enabled, higher defaults to reduce wall false positives.
"""

import os
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from PIL import Image

try:
    import cv2
except ImportError:
    cv2 = None

try:
    import torch
except ImportError:
    torch = None


@dataclass
class Detection:
    label: str
    bbox: list
    score: float
    center: tuple = field(init=False)

    def __post_init__(self):
        self.center = (
            (self.bbox[0] + self.bbox[2]) / 2,
            (self.bbox[1] + self.bbox[3]) / 2,
        )


class GroundingDINODetector:
    MODEL_ID = "IDEA-Research/grounding-dino-base"

    def __init__(self, model_id=None, device=None,
                 box_threshold=0.30, text_threshold=0.25):
        self.model_id = model_id or self.MODEL_ID
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold

        if torch is None:
            self.device = None
        elif device:
            self.device = torch.device(device)
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        self.model = None
        self.processor = None

    def load_model(self):
        if torch is None:
            raise ImportError("PyTorch not installed.")
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
        print(f"Loading Grounding DINO on {self.device}...")
        t0 = time.time()
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(
            self.model_id).to(self.device)
        self.model.eval()
        print(f"Model loaded in {time.time() - t0:.1f}s")

    def detect(self, image, text_prompt, box_threshold=None, text_threshold=None):
        if self.model is None:
            self.load_model()

        box_thresh  = box_threshold  or self.box_threshold
        text_thresh = text_threshold or self.text_threshold

        text_prompt = text_prompt.strip()
        if not text_prompt.endswith("."):
            text_prompt += "."

        if len(image.shape) == 3 and image.shape[2] == 3 and cv2 is not None:
            pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        else:
            pil_image = Image.fromarray(image)

        inputs = self.processor(
            images=pil_image, text=text_prompt, return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold=box_thresh,
            text_threshold=text_thresh,
            target_sizes=[pil_image.size[::-1]],
        )[0]

        prompt_terms = [t.strip() for t in text_prompt.rstrip(".").split(".") if t.strip()]

        detections = []
        boxes  = results["boxes"].cpu().numpy()
        scores = results["scores"].cpu().numpy()
        labels = results["labels"]

        for box, score, label in zip(boxes, scores, labels):
            detections.append(Detection(
                label=self._clean_label(label.strip(), prompt_terms),
                bbox=box.tolist(),
                score=float(score),
            ))

        detections.sort(key=lambda d: d.score, reverse=True)
        return detections

    @staticmethod
    def _clean_label(label, prompt_terms):
        label_lower = label.lower()
        matches = [t for t in prompt_terms if t.lower() in label_lower]
        if not matches:
            return label
        return max(matches, key=lambda t: len(t))

    def detect_from_file(self, image_path, text_prompt, **kwargs):
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")
        return self.detect(image, text_prompt, **kwargs)

    def visualize(self, image, detections, output_path=None):
        vis = image.copy()
        colors = [(0,255,0),(255,0,0),(0,0,255),(255,255,0),(0,255,255),(255,0,255)]
        for i, det in enumerate(detections):
            color = colors[i % len(colors)]
            x1,y1,x2,y2 = [int(v) for v in det.bbox]
            cv2.rectangle(vis, (x1,y1), (x2,y2), color, 2)
            cv2.putText(vis, f"{det.label} {det.score:.2f}", (x1, y1-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        if output_path:
            cv2.imwrite(output_path, vis)
        return vis

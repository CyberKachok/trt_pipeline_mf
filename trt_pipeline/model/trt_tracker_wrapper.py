"""Tracker wrapper that executes TensorRT engines via :mod:`trt.runner`."""

from __future__ import annotations

import collections
import time
from typing import Deque

import numpy as np

from mixformer_utils.processing_utils import Preprocessor_trt, clip_box, sample_target
from .tracker_wrapper import TrackerWrapper

try:  # pragma: no cover - import resolution differs between entrypoints
    from trt_pipeline.trt.runner import TrtRunner
except ImportError:
    if __package__ and __package__.count("."):
        from ..trt.runner import TrtRunner  # type: ignore[import-not-found]
    else:
        import sys
        from pathlib import Path

        sys.path.append(str(Path(__file__).resolve().parents[1]))
        from trt.runner import TrtRunner  # type: ignore[import-not-found]


class TrtTrackerWrapper(TrackerWrapper):
    """Runtime wrapper around a serialized TensorRT MixFormer engine."""

    def __init__(self, cfg_path: str, engine_path: str) -> None:
        super().__init__(cfg_path)
        self.runner = TrtRunner(engine_path)
        self.preprocessor = Preprocessor_trt()
        self.name = "trt"
        self.fps_history: Deque[float] = collections.deque(maxlen=10)

        self.template_tensor: np.ndarray | None = None
        self.online_template: np.ndarray | None = None
        self.online_max_template: np.ndarray | None = None

        self.template_input = self._select_binding(self.runner.get_input_names(), ["template"])
        self.online_input = self._select_binding(self.runner.get_input_names(), ["online_template", "template_online"])
        self.search_input = self._select_binding(self.runner.get_input_names(), ["search", "input_search"])

        self.bbox_output = self._select_binding(self.runner.get_output_names(), ["bbox", "boxes"])
        self.score_output = self._select_binding(self.runner.get_output_names(), ["confidence", "score", "scores"])

    @staticmethod
    def _select_binding(names, candidates):
        for candidate in candidates:
            for name in names:
                if name == candidate or name.endswith(candidate):
                    return name
        raise KeyError(f"Could not locate binding for {candidates}")

    def initialize(self, image, init_bbox: list) -> None:
        template_patch, _ = sample_target(
            image,
            init_bbox,
            self.template_factor,
            output_sz=self.template_size,
        )
        template = self.preprocessor.process(template_patch)
        self.template_tensor = template
        self.online_template = template.copy()
        self.online_max_template = template.copy()

        self.state = init_bbox
        self.max_pred_score = -1.0

    def track(self, image, frame_id: int):
        if self.template_tensor is None or self.online_template is None:
            raise RuntimeError("Tracker must be initialised before calling track()")

        H, W, _ = image.shape
        search_patch, resize_factor = sample_target(
            image,
            self.state,
            self.search_factor,
            output_sz=self.search_size,
        )
        search = self.preprocessor.process(search_patch)

        start = time.time()
        outputs = self.runner.infer(
            {
                self.template_input: self.template_tensor,
                self.online_input: self.online_template,
                self.search_input: search,
            }
        )
        latency = time.time() - start
        if latency > 0:
            self.fps_history.append(1.0 / latency)
        avg_fps = sum(self.fps_history) / len(self.fps_history) if self.fps_history else 0.0

        pred_boxes = outputs[self.bbox_output].reshape(-1)
        pred_score = float(outputs[self.score_output].reshape(-1)[0])
        pred_box = (pred_boxes * self.search_size / resize_factor)

        self.state = clip_box(self.map_box_back(pred_box, resize_factor), H, W, margin=10)

        if pred_score > 0.5 and pred_score > self.max_pred_score:
            template_patch, _ = sample_target(
                image,
                self.state,
                self.template_factor,
                output_sz=self.template_size,
            )
            self.online_max_template = self.preprocessor.process(template_patch)
            self.max_pred_score = pred_score

        if frame_id % self.update_interval == 0 and self.online_max_template is not None:
            self.online_template = self.online_max_template.copy()
            self.max_pred_score = -1.0
            self.online_max_template = self.template_tensor.copy()

        return self.state, pred_score, avg_fps


__all__ = ["TrtTrackerWrapper"]

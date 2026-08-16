from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from ultralytics import YOLO


class YOLOBoTSORTTracker:
    """
    Phase 1:
    YOLO26m detection + BoT-SORT tracking.

    Output:
        One row per tracked object observation.
    """

    def __init__(
        self,
        model_name: str,
        tracker: str,
        imgsz: int,
        conf: float,
        iou: float,
        target_classes: set[str],
        device: str = "auto",
    ) -> None:
        self.tracker = tracker
        self.imgsz = imgsz
        self.conf = conf
        self.iou = iou

        self.device = (
            0
            if device == "auto" and torch.cuda.is_available()
            else "cpu"
            if device == "auto"
            else device
        )

        self.model_path = self._resolve_model_path(
            model_name
        )

        self.model = YOLO(
            str(self.model_path)
            if Path(self.model_path).exists()
            else model_name
        )

        self.target_classes = set(target_classes)

        self.class_names = self.model.names

        self.target_class_ids = {
            int(class_id)
            for class_id, class_name
            in self.class_names.items()
            if class_name in self.target_classes
        }

        if not self.target_class_ids:
            raise ValueError(
                "No target classes were found in model.names. "
                f"Requested: {sorted(self.target_classes)}"
            )

    @staticmethod
    def _resolve_model_path(
        model_name: str,
    ) -> Path:
        """
        Prefer project-local models/<model_name> when available.

        Otherwise return the original model name so Ultralytics
        can resolve/download it.
        """

        configured = Path(model_name)

        if configured.exists():
            return configured

        local_model = (
            Path("models")
            / configured.name
        )

        if local_model.exists():
            return local_model

        return configured

    def run(
        self,
        video_path: str | Path,
        fps: float,
    ) -> pd.DataFrame:

        if fps <= 0:
            raise ValueError(
                f"fps must be > 0, got {fps}"
            )

        track_records: list[pd.DataFrame] = []

        results_stream = self.model.track(
            source=str(video_path),
            tracker=self.tracker,
            imgsz=self.imgsz,
            conf=self.conf,
            iou=self.iou,
            device=self.device,
            stream=True,
            persist=True,
            verbose=False,
        )

        for frame_id, result in enumerate(
            results_stream,
            start=1,
        ):
            boxes = result.boxes

            if boxes is None or len(boxes) == 0:
                continue

            if boxes.id is None:
                continue

            # Move tensors to CPU once.
            xyxy = (
                boxes.xyxy
                .cpu()
                .numpy()
            )

            conf = (
                boxes.conf
                .cpu()
                .numpy()
            )

            cls = (
                boxes.cls
                .cpu()
                .numpy()
                .astype(np.int16)
            )

            track_ids = (
                boxes.id
                .cpu()
                .numpy()
                .astype(np.int32)
            )

            # Target class filtering.
            mask = np.isin(
                cls,
                list(self.target_class_ids),
            )

            if not np.any(mask):
                continue

            xyxy = xyxy[mask]
            conf = conf[mask]
            cls = cls[mask]
            track_ids = track_ids[mask]

            # BBox extraction.
            x1 = xyxy[:, 0]
            y1 = xyxy[:, 1]
            x2 = xyxy[:, 2]
            y2 = xyxy[:, 3]

            # Exact notebook behavior:
            # bottom-center = ((x1+x2)/2, y2)
            bottom_center_x = (
                (x1 + x2) / 2.0
            )

            bottom_center_y = y2

            timestamp = (
                (frame_id - 1) / fps
            )

            frame_df = pd.DataFrame(
                {
                    "frame_id": frame_id,
                    "timestamp_sec": timestamp,
                    "track_id": track_ids,
                    "class_id": cls,
                    "class_name": [
                        self.class_names[int(class_id)]
                        for class_id in cls
                    ],
                    "confidence": conf,
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "bottom_center_x": bottom_center_x,
                    "bottom_center_y": bottom_center_y,
                }
            )

            track_records.append(
                frame_df
            )

        if not track_records:
            return pd.DataFrame(
                columns=[
                    "frame_id",
                    "timestamp_sec",
                    "track_id",
                    "class_id",
                    "class_name",
                    "confidence",
                    "x1",
                    "y1",
                    "x2",
                    "y2",
                    "bottom_center_x",
                    "bottom_center_y",
                ]
            )

        tracks_raw = pd.concat(
            track_records,
            ignore_index=True,
        )

        tracks_raw.sort_values(
            [
                "track_id",
                "frame_id",
            ],
            inplace=True,
        )

        tracks_raw.reset_index(
            drop=True,
            inplace=True,
        )

        return tracks_raw
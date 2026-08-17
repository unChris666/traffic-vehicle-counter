from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import torch
from ultralytics import YOLO


ProgressCallback = Callable[[float, str], None]


class YOLOBoTSORTTracker:
    """
    Phase 1:
    YOLO26m detection + BoT-SORT tracking.

    Output:
        One row per tracked object observation.

    The detection/tracking behavior remains aligned with the
    Phase 1 notebook baseline.
    """

    OUTPUT_COLUMNS = [
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
        self.imgsz = int(imgsz)
        self.conf = float(conf)
        self.iou = float(iou)

        self.device = self._resolve_device(device)

        self.model_path = self._resolve_model_path(
            model_name
        )

        self.model = YOLO(
            str(self.model_path)
            if self.model_path.exists()
            else model_name
        )

        self.target_classes = set(target_classes)

        if not isinstance(self.model.names, dict):
            raise TypeError(
                "Unexpected model.names type: "
                f"{type(self.model.names)}"
            )

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
    def _resolve_device(device: str):
        if device == "auto":
            return (
                0
                if torch.cuda.is_available()
                else "cpu"
            )

        return device

    @staticmethod
    def _resolve_model_path(
        model_name: str,
    ) -> Path:
        """
        Resolution priority:
        1. Explicit path.
        2. models/<filename>.
        3. Original Ultralytics model name.
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

    @staticmethod
    def _report_progress(
        progress_callback: ProgressCallback | None,
        frame_id: int,
        total_frames: int | None,
    ) -> None:
        if progress_callback is None:
            return

        if not total_frames:
            return

        # Do not update UI on every frame.
        if (
            frame_id != 1
            and frame_id != total_frames
            and frame_id % 30 != 0
        ):
            return

        progress = min(
            1.0,
            frame_id / total_frames,
        )

        progress_callback(
            progress,
            f"Inference "
            f"{frame_id:,}/{total_frames:,}",
        )

    def run(
        self,
        video_path: str | Path,
        fps: float,
        total_frames: int | None = None,
        progress_callback: ProgressCallback | None = None,
    ) -> pd.DataFrame:

        if fps <= 0:
            raise ValueError(
                f"fps must be > 0, got {fps}"
            )

        video_path = Path(video_path)

        if not video_path.exists():
            raise FileNotFoundError(
                f"Video not found: {video_path}"
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
            self._report_progress(
                progress_callback,
                frame_id,
                total_frames,
            )

            boxes = result.boxes

            if boxes is None or len(boxes) == 0:
                continue

            if boxes.id is None:
                continue

            xyxy = (
                boxes.xyxy
                .detach()
                .cpu()
                .numpy()
            )

            conf = (
                boxes.conf
                .detach()
                .cpu()
                .numpy()
            )

            cls = (
                boxes.cls
                .detach()
                .cpu()
                .numpy()
                .astype(np.int16)
            )

            track_ids = (
                boxes.id
                .detach()
                .cpu()
                .numpy()
                .astype(np.int32)
            )

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

            x1 = xyxy[:, 0]
            y1 = xyxy[:, 1]
            x2 = xyxy[:, 2]
            y2 = xyxy[:, 3]

            # Preserve notebook behavior.
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
                    "bottom_center_x":
                        bottom_center_x,
                    "bottom_center_y":
                        bottom_center_y,
                }
            )

            track_records.append(frame_df)

        if not track_records:
            if progress_callback is not None:
                progress_callback(
                    1.0,
                    "Inference complete: "
                    "no target tracks found",
                )

            return pd.DataFrame(
                columns=self.OUTPUT_COLUMNS
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

        if progress_callback is not None:
            progress_callback(
                1.0,
                (
                    "Inference complete: "
                    f"{len(tracks_raw):,} observations"
                ),
            )

        return tracks_raw
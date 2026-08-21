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
    Phase 1 — YOLO26m + BoT-SORT.

    Robust branch:
        - baseline model loading
        - baseline tracking behavior
        - vid_stride=1 by default
        - no TensorRT requirement
        - model can be downloaded automatically by Ultralytics

    Output:
        one row per tracked target observation.
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
        *,
        model_name: str = "yolo26m.pt",
        tracker: str = "botsort.yaml",
        imgsz: int = 640,
        conf: float = 0.20,
        iou: float = 0.70,
        vid_stride: int = 1,
        target_classes: set[str] | None = None,
        device: str = "auto",
    ) -> None:

        if imgsz <= 0:
            raise ValueError(
                f"imgsz must be > 0, got {imgsz}"
            )

        if not 0.0 < conf <= 1.0:
            raise ValueError(
                f"conf must be in (0, 1], got {conf}"
            )

        if not 0.0 < iou <= 1.0:
            raise ValueError(
                f"iou must be in (0, 1], got {iou}"
            )

        if vid_stride < 1:
            raise ValueError(
                f"vid_stride must be >= 1, got {vid_stride}"
            )

        self.tracker = str(tracker)
        self.imgsz = int(imgsz)
        self.conf = float(conf)
        self.iou = float(iou)
        self.vid_stride = int(vid_stride)

        self.device = self._resolve_device(
            device
        )

        self.model_name = str(
            model_name
        )

        self.model_path = (
            self._resolve_model_path(
                self.model_name
            )
        )

        # -----------------------------------------------------
        # If a local file exists, use it.
        #
        # Otherwise let Ultralytics resolve/download:
        #     yolo26m.pt
        # -----------------------------------------------------

        model_source = (
            str(self.model_path)
            if self.model_path.exists()
            else self.model_name
        )

        self.model = YOLO(
            model_source,
            task="detect",
        )

        # -----------------------------------------------------
        # Target classes
        # -----------------------------------------------------

        self.target_classes = set(
            target_classes
            if target_classes is not None
            else {
                "person",
                "motorcycle",
                "car",
                "bus",
                "truck",
            }
        )

        if not isinstance(
            self.model.names,
            dict,
        ):
            raise TypeError(
                "Unexpected model.names type: "
                f"{type(self.model.names)}"
            )

        self.class_names = (
            self.model.names
        )

        self.target_class_ids = {
            int(class_id)
            for class_id, class_name
            in self.class_names.items()
            if class_name
            in self.target_classes
        }

        if not self.target_class_ids:
            raise ValueError(
                "No target classes were found "
                "in model.names. "
                f"Requested: "
                f"{sorted(self.target_classes)}"
            )

        self._processed_frames = 0

    # =========================================================
    # DEVICE
    # =========================================================

    @staticmethod
    def _resolve_device(
        device: str,
    ):

        if device == "auto":

            if torch.cuda.is_available():
                return 0

            return "cpu"

        return device

    # =========================================================
    # MODEL PATH
    # =========================================================

    @staticmethod
    def _resolve_model_path(
        model_name: str,
    ) -> Path:
        """
        Resolution:

            1. explicit path
            2. models/<filename>
            3. original model name

        For example:

            yolo26m.pt

        will remain:

            yolo26m.pt

        and Ultralytics can download it automatically.
        """

        configured = Path(
            model_name
        )

        if configured.exists():
            return configured

        local_model = (
            Path("models")
            /
            configured.name
        )

        if local_model.exists():
            return local_model

        return configured

    # =========================================================
    # PROGRESS
    # =========================================================

    @staticmethod
    def _report_progress(
        progress_callback: (
            ProgressCallback | None
        ),
        frame_id: int,
        total_frames: int | None,
    ) -> None:

        if progress_callback is None:
            return

        if not total_frames:
            return

        if (
            frame_id != 1
            and
            frame_id != total_frames
            and
            frame_id % 30 != 0
        ):
            return

        progress = min(
            1.0,
            frame_id / total_frames,
        )

        progress_callback(
            progress,
            (
                "Inference "
                f"{frame_id:,}/"
                f"{total_frames:,}"
            ),
        )

    # =========================================================
    # RUN
    # =========================================================

    def run(
        self,
        video_path: str | Path,
        fps: float,
        total_frames: int | None = None,
        progress_callback: (
            ProgressCallback | None
        ) = None,
    ) -> pd.DataFrame:

        if fps <= 0:
            raise ValueError(
                f"fps must be > 0, got {fps}"
            )

        video_path = Path(
            video_path
        )

        if not video_path.exists():
            raise FileNotFoundError(
                f"Video not found: "
                f"{video_path}"
            )

        self._processed_frames = 0

        track_records: list[
            pd.DataFrame
        ] = []

        # =====================================================
        # TRACK
        # =====================================================

        results_stream = (
            self.model.track(
                source=str(
                    video_path
                ),
                tracker=self.tracker,
                imgsz=self.imgsz,
                conf=self.conf,
                iou=self.iou,
                device=self.device,
                vid_stride=self.vid_stride,
                stream=True,
                persist=True,
                verbose=False,
            )
        )

        # =====================================================
        # FRAME LOOP
        # =====================================================

        for frame_id, result in enumerate(
            results_stream,
            start=1,
        ):

            self._processed_frames += 1

            self._report_progress(
                progress_callback,
                frame_id,
                total_frames,
            )

            boxes = result.boxes

            if (
                boxes is None
                or
                len(boxes) == 0
            ):
                continue

            if boxes.id is None:
                continue

            # -------------------------------------------------
            # Boxes
            # -------------------------------------------------

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

            # -------------------------------------------------
            # Target classes only
            # -------------------------------------------------

            mask = np.isin(
                cls,
                list(
                    self.target_class_ids
                ),
            )

            if not np.any(mask):
                continue

            xyxy = xyxy[mask]
            conf = conf[mask]
            cls = cls[mask]
            track_ids = (
                track_ids[mask]
            )

            # -------------------------------------------------
            # Bounding box
            # -------------------------------------------------

            x1 = xyxy[:, 0]
            y1 = xyxy[:, 1]
            x2 = xyxy[:, 2]
            y2 = xyxy[:, 3]

            # -------------------------------------------------
            # Bottom-center
            #
            # SAME convention as baseline.
            # -------------------------------------------------

            bottom_center_x = (
                (x1 + x2)
                /
                2.0
            )

            bottom_center_y = y2

            # -------------------------------------------------
            # Source-video timestamp
            # -------------------------------------------------

            timestamp = (
                (frame_id - 1)
                /
                fps
            )

            frame_df = pd.DataFrame(
                {
                    "frame_id": frame_id,

                    "timestamp_sec":
                        timestamp,

                    "track_id":
                        track_ids,

                    "class_id":
                        cls,

                    "class_name": [
                        self.class_names[
                            int(class_id)
                        ]
                        for class_id
                        in cls
                    ],

                    "confidence":
                        conf,

                    "x1":
                        x1,

                    "y1":
                        y1,

                    "x2":
                        x2,

                    "y2":
                        y2,

                    "bottom_center_x":
                        bottom_center_x,

                    "bottom_center_y":
                        bottom_center_y,
                }
            )

            track_records.append(
                frame_df
            )

        # =====================================================
        # NO TRACKS
        # =====================================================

        if not track_records:

            if progress_callback is not None:

                progress_callback(
                    1.0,
                    (
                        "Inference complete: "
                        "no target tracks found"
                    ),
                )

            return pd.DataFrame(
                columns=self.OUTPUT_COLUMNS
            )

        # =====================================================
        # MERGE
        # =====================================================

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

        # =====================================================
        # COMPLETE
        # =====================================================

        if progress_callback is not None:

            progress_callback(
                1.0,
                (
                    "Inference complete: "
                    f"{len(tracks_raw):,} "
                    "observations"
                ),
            )

        return tracks_raw

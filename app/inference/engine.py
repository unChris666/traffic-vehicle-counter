from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from app.core.config import (
    AppConfig,
    build_config,
)
from app.inference.detector_tracker import (
    YOLOBoTSORTTracker,
)
from app.counting.track_classifier import (
    build_track_level,
)
from app.counting.counter import (
    TrafficCounter,
)
from app.video.validator import (
    VideoMetadata,
    read_video_metadata,
)


@dataclass(frozen=True)
class CountingResult:
    status: str
    video: dict
    counts: dict[str, int]
    total: int
    performance: dict


class TrafficCountingEngine:
    """
    Production orchestration layer.

    Phase 1 -> Phase 2 -> Phase 3
    """

    def __init__(
        self,
        config: AppConfig | None = None,
    ) -> None:
        self.config = (
            config
            or build_config()
        )

    def validate(
        self,
        video_path: str | Path,
    ) -> VideoMetadata:
        return read_video_metadata(
            video_path
        )

    def process(
        self,
        video_path: str | Path,
    ) -> CountingResult:

        video_path = Path(video_path)

        total_start = time.perf_counter()

        # ========================================================
        # VIDEO VALIDATION
        # ========================================================

        metadata = self.validate(
            video_path
        )

        if metadata.fps <= 0:
            raise ValueError(
                f"Invalid FPS: {metadata.fps}"
            )

        # ========================================================
        # OUTPUT DIRECTORY
        # ========================================================

        output_dir = (
            Path(self.config.output_dir)
            / video_path.stem
        )

        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        # ========================================================
        # PHASE 1
        # YOLO26m + BoT-SORT
        # ========================================================

        phase1_start = (
            time.perf_counter()
        )

        tracker = YOLOBoTSORTTracker(
            model_name=(
                self.config.detection.model_name
            ),
            tracker=(
                self.config.detection.tracker
            ),
            imgsz=(
                self.config.detection.imgsz
            ),
            conf=(
                self.config.detection.conf_threshold
            ),
            iou=(
                self.config.detection.iou_threshold
            ),
            target_classes=set(
                self.config.target_classes
            ),
            device=(
                self.config.detection.device
            ),
        )

        tracks_raw = tracker.run(
            video_path,
            fps=metadata.fps,
        )

        phase1_elapsed = (
            time.perf_counter()
            - phase1_start
        )

        tracks_raw.to_csv(
            output_dir / "tracks_raw.csv",
            index=False,
        )

        # ========================================================
        # PHASE 2
        # TRACK-LEVEL CLASS + QUALITY
        # ========================================================

        phase2_start = (
            time.perf_counter()
        )

        if tracks_raw.empty:
            raise RuntimeError(
                "No target objects were tracked "
                "in the input video."
            )

        (
            track_level,
            tracks_phase2,
        ) = build_track_level(
            tracks_raw,
            fps=metadata.fps,
        )

        phase2_elapsed = (
            time.perf_counter()
            - phase2_start
        )

        track_level.to_csv(
            output_dir
            / "track_quality_summary.csv",
            index=False,
        )

        tracks_phase2.to_csv(
            output_dir
            / "tracks_with_track_class.csv",
            index=False,
        )

        # ========================================================
        # PHASE 3
        # COUNTING
        # ========================================================

        phase3_start = (
            time.perf_counter()
        )

        line_x1 = (
            metadata.width
            * self.config.counting.line_x1_ratio
        )

        line_y1 = (
            metadata.height
            * self.config.counting.line_y1_ratio
        )

        line_x2 = (
            metadata.width
            * self.config.counting.line_x2_ratio
        )

        line_y2 = (
            metadata.height
            * self.config.counting.line_y2_ratio
        )

        counter = TrafficCounter(
            line_x1=line_x1,
            line_y1=line_y1,
            line_x2=line_x2,
            line_y2=line_y2,
            line_deadband_px=(
                self.config.counting.line_deadband_px
            ),
            max_trajectory_gap_sec=(
                self.config.counting
                .max_trajectory_gap_sec
            ),
            moto_dedup_time_sec=(
                self.config.counting
                .moto_dedup_time_sec
            ),
            moto_dedup_distance_px=(
                self.config.counting
                .moto_dedup_distance_px
            ),
            vehicle_classes=set(
                self.config.vehicle_classes
            ),
            fps=metadata.fps,
        )

        counting_result = counter.count(
            tracks_phase2
        )

        phase3_elapsed = (
            time.perf_counter()
            - phase3_start
        )

        # ========================================================
        # SAVE PHASE 3 AUDIT DATA
        # ========================================================

        counting_result.crossing_events.to_csv(
            output_dir
            / "crossing_events.csv",
            index=False,
        )

        counting_result.vehicle_events.to_csv(
            output_dir
            / "vehicle_events.csv",
            index=False,
        )

        counting_result.final_crossings.to_csv(
            output_dir
            / "final_vehicle_crossings.csv",
            index=False,
        )

        final_counts_df = pd.DataFrame(
            [
                {
                    "class": class_name,
                    "quantity": quantity,
                }
                for class_name, quantity
                in counting_result.counts.items()
            ]
        )

        final_counts_df.to_csv(
            output_dir
            / "final_vehicle_counts.csv",
            index=False,
        )

        with open(
            output_dir / "phase3_audit.json",
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(
                counting_result.audit,
                f,
                indent=2,
            )

        # ========================================================
        # FINAL RESULT
        # ========================================================

        total_elapsed = (
            time.perf_counter()
            - total_start
        )

        processing_fps = (
            metadata.frame_count
            / total_elapsed
            if total_elapsed > 0
            else None
        )

        video_info = {
            "filename": metadata.filename,
            "width": metadata.width,
            "height": metadata.height,
            "fps": metadata.fps,
            "frame_count": metadata.frame_count,
            "duration_sec": metadata.duration_sec,
        }

        performance = {
            "total_processing_time_sec": (
                total_elapsed
            ),
            "processing_fps": (
                processing_fps
            ),
            "phase1_inference_time_sec": (
                phase1_elapsed
            ),
            "phase2_classification_time_sec": (
                phase2_elapsed
            ),
            "phase3_counting_time_sec": (
                phase3_elapsed
            ),
        }

        result_json = {
            "status": "completed",
            "video": video_info,
            "counts": counting_result.counts,
            "total": counting_result.total,
            "performance": performance,
            "audit": counting_result.audit,
        }

        with open(
            output_dir / "result.json",
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(
                result_json,
                f,
                indent=2,
            )

        return CountingResult(
            status="completed",
            video=video_info,
            counts=counting_result.counts,
            total=counting_result.total,
            performance=performance,
        )
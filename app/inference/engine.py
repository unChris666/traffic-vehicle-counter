from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import pandas as pd

from app.core.config import (
    AppConfig,
    build_config,
)
from app.counting.counter import TrafficCounter
from app.counting.track_classifier import (
    build_track_level,
)
from app.inference.confidence import (
    ConfidenceEngine,
)
from app.inference.detector_tracker import (
    YOLOBoTSORTTracker,
)
from app.video.validator import (
    VideoMetadata,
    read_video_metadata,
)


ProgressCallback = Callable[
    [float, str],
    None,
]


@dataclass(frozen=True)
class CountingResult:
    status: str
    video: dict
    counts: dict[str, int]
    direction_counts: list[dict]
    count_confidence: list[dict]
    overall_confidence: dict
    total: int
    performance: dict
    artifacts: dict


class TrafficCountingEngine:
    """
    Production orchestration layer.

    Pipeline:

        Video validation
            ↓
        Phase 1 — YOLO26m + BoT-SORT
            ↓
        Phase 2 — Track-level class
            ↓
        Phase 3 — Counting
            ↓
        Phase 6B — Confidence
            ↓
        Optional Phase 6C — Video rendering
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

    @staticmethod
    def _report(
        progress_callback: (
            ProgressCallback | None
        ),
        progress: float,
        description: str,
    ) -> None:
        if progress_callback is None:
            return

        progress_callback(
            max(
                0.0,
                min(1.0, progress),
            ),
            description,
        )

    def process(
        self,
        video_path: str | Path,
        *,
        render_video: bool = False,
        progress_callback: (
            ProgressCallback | None
        ) = None,
    ) -> CountingResult:

        video_path = Path(video_path)

        if not video_path.exists():
            raise FileNotFoundError(
                f"Video not found: {video_path}"
            )

        total_start = (
            time.perf_counter()
        )

        # ========================================================
        # VIDEO VALIDATION
        # ========================================================

        self._report(
            progress_callback,
            0.01,
            "Validating video...",
        )

        metadata = self.validate(
            video_path
        )

        if metadata.fps <= 0:
            raise ValueError(
                f"Invalid FPS: {metadata.fps}"
            )

        self._report(
            progress_callback,
            0.05,
            (
                f"Video validated: "
                f"{metadata.width}x"
                f"{metadata.height}, "
                f"{metadata.fps:.2f} FPS, "
                f"{metadata.frame_count:,} frames"
            ),
        )

        # ========================================================
        # OUTPUT DIRECTORY
        # ========================================================

        output_dir = (
            Path(
                self.config.output_dir
            )
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

        self._report(
            progress_callback,
            0.08,
            "Starting YOLO26m + BoT-SORT...",
        )

        tracker = YOLOBoTSORTTracker(
            model_name=(
                self.config
                .detection
                .model_name
            ),
            tracker=(
                self.config
                .detection
                .tracker
            ),
            imgsz=(
                self.config
                .detection
                .imgsz
            ),
            conf=(
                self.config
                .detection
                .conf_threshold
            ),
            iou=(
                self.config
                .detection
                .iou_threshold
            ),
            target_classes=set(
                self.config
                .target_classes
            ),
            device=(
                self.config
                .detection
                .device
            ),
        )

        tracks_raw = tracker.run(
            video_path,
            fps=metadata.fps,
            total_frames=(
                metadata.frame_count
            ),
            progress_callback=(
                lambda p, d: self._report(
                    progress_callback,
                    0.08 + (
                        p * 0.70
                    ),
                    d,
                )
            ),
        )

        phase1_elapsed = (
            time.perf_counter()
            - phase1_start
        )

        if tracks_raw.empty:
            raise RuntimeError(
                "No target objects were "
                "tracked in the input video."
            )

        tracks_raw.to_csv(
            output_dir
            / "tracks_raw.csv",
            index=False,
        )

        # ========================================================
        # PHASE 2
        # TRACK-LEVEL CLASS
        # ========================================================

        self._report(
            progress_callback,
            0.80,
            "Building track-level classes...",
        )

        phase2_start = (
            time.perf_counter()
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

        self._report(
            progress_callback,
            0.85,
            "Counting vehicle crossings...",
        )

        phase3_start = (
            time.perf_counter()
        )

        line_x1 = (
            metadata.width
            * self.config.counting
            .line_x1_ratio
        )

        line_y1 = (
            metadata.height
            * self.config.counting
            .line_y1_ratio
        )

        line_x2 = (
            metadata.width
            * self.config.counting
            .line_x2_ratio
        )

        line_y2 = (
            metadata.height
            * self.config.counting
            .line_y2_ratio
        )

        counter = TrafficCounter(
            line_x1=line_x1,
            line_y1=line_y1,
            line_x2=line_x2,
            line_y2=line_y2,
            line_deadband_px=(
                self.config.counting
                .line_deadband_px
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
                self.config
                .vehicle_classes
            ),
            fps=metadata.fps,
        )

        counting_result = (
            counter.count(
                tracks_phase2
            )
        )

        phase3_elapsed = (
            time.perf_counter()
            - phase3_start
        )

        # ========================================================
        # DIRECTION COUNTS
        # ========================================================

        direction_counts_df = (
            counting_result
            .final_crossings
            .groupby(
                [
                    "track_class",
                    "direction",
                ]
            )
            .size()
            .rename("count")
            .reset_index()
        )

        direction_classes = [
            "bus",
            "car",
            "motorcycle",
            "person",
            "truck",
        ]

        direction_values = [
            "side_+1_to_-1",
            "side_-1_to_+1",
        ]

        direction_index = (
            pd.MultiIndex
            .from_product(
                [
                    direction_classes,
                    direction_values,
                ],
                names=[
                    "track_class",
                    "direction",
                ],
            )
        )

        direction_counts_df = (
            direction_counts_df
            .set_index(
                [
                    "track_class",
                    "direction",
                ]
            )
            .reindex(
                direction_index,
                fill_value=0,
            )
            .reset_index()
        )

        # ========================================================
        # PHASE 6B
        # CONFIDENCE
        # ========================================================

        self._report(
            progress_callback,
            0.89,
            "Calculating confidence...",
        )

        confidence_start = (
            time.perf_counter()
        )

        confidence_engine = (
            ConfidenceEngine()
        )

        track_confidence = (
            confidence_engine
            .build_track_confidence(
                track_level
            )
        )

        crossing_confidence = (
            confidence_engine
            .build_crossing_confidence(
                final_crossings=(
                    counting_result
                    .final_crossings
                ),
                trajectory=(
                    counting_result
                    .trajectory
                ),
                track_confidence=(
                    track_confidence
                ),
                fps=metadata.fps,
            )
        )

        count_confidence = (
            confidence_engine
            .build_count_confidence(
                crossing_confidence=(
                    crossing_confidence
                ),
                final_counts=(
                    counting_result
                    .counts
                ),
            )
        )

        overall_confidence = (
            confidence_engine
            .build_overall_confidence(
                crossing_confidence
            )
        )

        confidence_elapsed = (
            time.perf_counter()
            - confidence_start
        )

        # ========================================================
        # SAVE PHASE 3 ARTIFACTS
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

        direction_counts_df.to_csv(
            output_dir
            / "vehicle_direction_counts.csv",
            index=False,
        )

        final_counts_df = pd.DataFrame(
            [
                {
                    "class": class_name,
                    "quantity": quantity,
                }
                for class_name, quantity
                in counting_result
                .counts
                .items()
            ]
        )

        final_counts_df.to_csv(
            output_dir
            / "final_vehicle_counts.csv",
            index=False,
        )

        with open(
            output_dir
            / "phase3_audit.json",
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(
                counting_result.audit,
                f,
                indent=2,
            )

        # ========================================================
        # SAVE PHASE 6B ARTIFACTS
        # ========================================================

        track_confidence.to_csv(
            output_dir
            / "track_confidence.csv",
            index=False,
        )

        crossing_confidence.to_csv(
            output_dir
            / "crossing_confidence.csv",
            index=False,
        )

        pd.DataFrame(
            count_confidence
        ).to_csv(
            output_dir
            / "count_confidence.csv",
            index=False,
        )

        # ========================================================
        # OPTIONAL PHASE 6C VIDEO RENDERING
        #
        # This runs only when render_video=True.
        # FFmpeg is therefore required only in the cloud
        # runtime where rendering is executed.
        # ========================================================

        annotated_video_path: (
            Path | None
        ) = None

        rendering_elapsed = 0.0

        if render_video:
            self._report(
                progress_callback,
                0.94,
                "Rendering annotated video...",
            )

            render_start = (
                time.perf_counter()
            )

            from app.video.renderer import (
                VideoRenderer,
            )

            renderer = VideoRenderer(
                line_x1=line_x1,
                line_y1=line_y1,
                line_x2=line_x2,
                line_y2=line_y2,
            )

            annotated_video_path = (
                renderer.render(
                    input_path=video_path,
                    output_path=(
                        output_dir
                        / "annotated_video.mp4"
                    ),
                    fps=metadata.fps,
                    width=metadata.width,
                    height=metadata.height,
                    total_frames=(
                        metadata.frame_count
                    ),
                    tracks_phase2=(
                        tracks_phase2
                    ),
                    final_crossings=(
                        counting_result
                        .final_crossings
                    ),
                    progress_callback=(
                        lambda p, d: self._report(
                            progress_callback,
                            0.94 + (
                                p * 0.05
                            ),
                            d,
                        )
                    ),
                )
            )

            rendering_elapsed = (
                time.perf_counter()
                - render_start
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

        artifacts = {
            "result_json": str(
                output_dir
                / "result.json"
            ),
            "count_csv": str(
                output_dir
                / "final_vehicle_counts.csv"
            ),
            "direction_csv": str(
                output_dir
                / "vehicle_direction_counts.csv"
            ),
            "count_confidence_csv": str(
                output_dir
                / "count_confidence.csv"
            ),
            "annotated_video": (
                str(
                    annotated_video_path
                )
                if annotated_video_path
                is not None
                else None
            ),
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
            "phase6b_confidence_time_sec": (
                confidence_elapsed
            ),
            "rendering_time_sec": (
                rendering_elapsed
            ),
        }

        direction_counts = [
            {
                "track_class": (
                    row["track_class"]
                ),
                "direction": (
                    row["direction"]
                ),
                "count": int(
                    row["count"]
                ),
            }
            for _, row
            in direction_counts_df.iterrows()
        ]

        result_json = {
            "status": "completed",
            "video": video_info,
            "counts": (
                counting_result.counts
            ),
            "count_confidence": (
                count_confidence
            ),
            "overall_confidence": (
                overall_confidence
            ),
            "direction_counts": (
                direction_counts
            ),
            "total": (
                counting_result.total
            ),
            "performance": performance,
            "audit": (
                counting_result.audit
            ),
            "artifacts": artifacts,
        }

        with open(
            output_dir
            / "result.json",
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(
                result_json,
                f,
                indent=2,
            )

        self._report(
            progress_callback,
            1.0,
            "Processing complete.",
        )

        # ========================================================
        # CLI REPORT
        # ========================================================

        print(
            "\n" + "=" * 70
        )
        print("FINAL VEHICLE COUNT")
        print("=" * 70)

        for (
            class_name,
            count,
        ) in (
            counting_result
            .counts
            .items()
        ):
            print(
                f"{class_name:<15}: "
                f"{count}"
            )

        print("-" * 70)

        print(
            f"{'TOTAL VEHICLES':<15}: "
            f"{counting_result.total}"
        )

        print(
            "\n" + "=" * 70
        )
        print("COUNT CONFIDENCE")
        print("=" * 70)

        for row in count_confidence:
            confidence = (
                row["confidence"]
            )

            confidence_text = (
                f"{confidence:.4f}"
                if confidence is not None
                else "N/A"
            )

            print(
                f"{row['class']:<15}: "
                f"{confidence_text:<8} "
                f"{row['flag']}"
            )

        print(
            "\n" + "=" * 70
        )
        print("COUNT BY DIRECTION")
        print("=" * 70)

        for row in direction_counts:
            print(
                f"{row['track_class']:<12} "
                f"{row['direction']:<20} "
                f"{row['count']}"
            )

        return CountingResult(
            status="completed",
            video=video_info,
            counts=counting_result.counts,
            direction_counts=direction_counts,
            count_confidence=(
                count_confidence
            ),
            overall_confidence=(
                overall_confidence
            ),
            total=counting_result.total,
            performance=performance,
            artifacts=artifacts,
        )
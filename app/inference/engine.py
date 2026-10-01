from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import pandas as pd

from app.core.config import AppConfig, build_config
from app.counting.side_box_counter import (
    Line,
    Polygon,
    SideBoxVehicleCounter,
)
from app.inference.detector_tracker import YOLOBoTSORTTracker
from app.video.validator import VideoMetadata, read_video_metadata


ProgressCallback = Callable[[float, str], None]


@dataclass
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

    # Compatibility/debug artifacts.
    frame_state: pd.DataFrame
    final_crossings: pd.DataFrame
    trajectory: pd.DataFrame
    crossing_events: pd.DataFrame
    vehicle_events: pd.DataFrame


class TrafficCountingEngine:
    """
    Phase3New orchestration.

    The important difference from the previous Phase 3 is that counting
    happens causally while iterating through the video frame-by-frame.

        source frame
             ↓
        YOLO26m + BoT-SORT
             ↓
        tracker observation for this frame
             ↓
        Side A / crossing line / Side B state update
             ↓
        immutable track_id + locked vehicle category
             ↓
        count once

    There is no post-hoc trajectory reconstruction used as the identity
    source of truth.
    """

    def __init__(self, config: AppConfig | None = None) -> None:
        self.config = config or build_config()

        if self.config.detection.vid_stride != 1:
            raise ValueError(
                "Phase3New requires vid_stride=1. "
                "Every source frame must be processed."
            )

    @staticmethod
    def _report(
        callback: ProgressCallback | None,
        progress: float,
        description: str,
    ) -> None:
        if callback is not None:
            callback(
                max(0.0, min(1.0, float(progress))),
                description,
            )

    def validate(self, video_path: str | Path) -> VideoMetadata:
        return read_video_metadata(video_path)

    @staticmethod
    def _polygon_from_ratios(
        width: int,
        height: int,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
    ) -> Polygon:
        px1 = width * x1
        py1 = height * y1
        px2 = width * x2
        py2 = height * y2

        return Polygon(
            (
                (px1, py1),
                (px2, py1),
                (px2, py2),
                (px1, py2),
            )
        )

    def _build_counter(
        self,
        metadata: VideoMetadata,
    ) -> tuple[
        SideBoxVehicleCounter,
        Polygon,
        Polygon,
        Line,
    ]:
        c = self.config.counting

        side_a = self._polygon_from_ratios(
            metadata.width,
            metadata.height,
            c.side_a_x1_ratio,
            c.side_a_y1_ratio,
            c.side_a_x2_ratio,
            c.side_a_y2_ratio,
        )

        side_b = self._polygon_from_ratios(
            metadata.width,
            metadata.height,
            c.side_b_x1_ratio,
            c.side_b_y1_ratio,
            c.side_b_x2_ratio,
            c.side_b_y2_ratio,
        )

        line = Line(
            metadata.width * c.line_x1_ratio,
            metadata.height * c.line_y1_ratio,
            metadata.width * c.line_x2_ratio,
            metadata.height * c.line_y2_ratio,
        )

        counter = SideBoxVehicleCounter(
            side_a=side_a,
            side_b=side_b,
            crossing_line=line,
            fps=metadata.fps,
            vehicle_classes=self.config.vehicle_classes,
            speed_window=c.speed_window_frames,
            max_missing_frames=c.state_max_missing_frames,
        )

        return counter, side_a, side_b, line

    def process(
        self,
        video_path: str | Path,
        *,
        render_video: bool = False,
        progress_callback: ProgressCallback | None = None,
    ) -> CountingResult:
        video_path = Path(video_path)

        if not video_path.exists():
            raise FileNotFoundError(
                f"Video not found: {video_path}"
            )

        total_start = time.perf_counter()

        # =====================================================
        # VIDEO VALIDATION
        # =====================================================
        self._report(
            progress_callback,
            0.01,
            "Validating video...",
        )

        metadata = self.validate(video_path)

        if metadata.fps <= 0:
            raise ValueError(
                f"Invalid FPS: {metadata.fps}"
            )

        output_dir = (
            Path(self.config.output_dir)
            / video_path.stem
        )
        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        # =====================================================
        # DETECTION + TRACKING
        # =====================================================
        self._report(
            progress_callback,
            0.05,
            "Starting YOLO26m + BoT-SORT frame-by-frame...",
        )

        inference_start = time.perf_counter()

        tracker = YOLOBoTSORTTracker(
            model_name=self.config.detection.model_name,
            tracker=self.config.detection.tracker,
            imgsz=self.config.detection.imgsz,
            conf=self.config.detection.conf_threshold,
            iou=self.config.detection.iou_threshold,
            vid_stride=1,
            target_classes=set(self.config.target_classes),
            device=self.config.detection.device,
        )

        tracks_raw = tracker.run(
            video_path,
            fps=metadata.fps,
            total_frames=metadata.frame_count,
            progress_callback=(
                lambda p, d: self._report(
                    progress_callback,
                    0.05 + p * 0.60,
                    d,
                )
            ),
        )

        inference_elapsed = time.perf_counter() - inference_start

        if tracks_raw.empty:
            raise RuntimeError(
                "No target objects were tracked in the input video."
            )

        tracks_raw.to_csv(
            output_dir / "tracks_raw.csv",
            index=False,
        )

        # =====================================================
        # FRAME-BY-FRAME STATE ENGINE
        # =====================================================
        self._report(
            progress_callback,
            0.67,
            "Processing Side A → Line → Side B states...",
        )

        counter, side_a, side_b, line = self._build_counter(
            metadata
        )

        state_start = time.perf_counter()

        frame_outputs: list[pd.DataFrame] = []

        # IMPORTANT:
        # Iterate over ALL source frames, including frames where there
        # are no detections. This keeps the temporal model honest.
        grouped = {
            int(frame_id): frame
            for frame_id, frame
            in tracks_raw.groupby("frame_id", sort=False)
        }

        for frame_id in range(1, metadata.frame_count + 1):
            frame_tracks = grouped.get(
                frame_id,
                pd.DataFrame(),
            )

            state_frame = counter.process_frame(
                frame_id=frame_id,
                timestamp_sec=(frame_id - 1) / metadata.fps,
                frame_tracks=frame_tracks,
            )

            if not state_frame.empty:
                frame_outputs.append(state_frame)

            if (
                progress_callback is not None
                and (
                    frame_id == 1
                    or frame_id == metadata.frame_count
                    or frame_id % 30 == 0
                )
            ):
                self._report(
                    progress_callback,
                    0.67
                    + (
                        0.15
                        * frame_id
                        / metadata.frame_count
                    ),
                    (
                        "Frame state "
                        f"{frame_id:,}/"
                        f"{metadata.frame_count:,}"
                    ),
                )

        state_elapsed = time.perf_counter() - state_start

        if frame_outputs:
            frame_state = pd.concat(
                frame_outputs,
                ignore_index=True,
            )
        else:
            frame_state = pd.DataFrame(
                columns=counter.OUTPUT_COLUMNS
            )

        frame_state.to_csv(
            output_dir / "frame_vehicle_state.csv",
            index=False,
        )

        # =====================================================
        # FINAL EVENTS
        # =====================================================
        events = counter.finalize()
        summary = counter.summary()

        events.to_csv(
            output_dir / "vehicle_events.csv",
            index=False,
        )

        # One row per counted vehicle ID.
        events.to_csv(
            output_dir / "final_vehicle_crossings.csv",
            index=False,
        )

        counts_df = pd.DataFrame(
            [
                {
                    "class": class_name,
                    "quantity": quantity,
                }
                for class_name, quantity
                in summary["counts"].items()
            ]
        )

        counts_df.to_csv(
            output_dir / "final_vehicle_counts.csv",
            index=False,
        )

        direction_df = pd.DataFrame(
            summary["direction_counts"]
        )

        direction_df.to_csv(
            output_dir / "vehicle_direction_counts.csv",
            index=False,
        )

        # =====================================================
        # OPTIONAL RENDER
        # =====================================================
        annotated_video_path: Path | None = None
        rendering_elapsed = 0.0

        if render_video:
            from app.video.renderer import VideoRenderer

            self._report(
                progress_callback,
                0.84,
                "Rendering annotated video...",
            )

            render_start = time.perf_counter()

            renderer = VideoRenderer(
                side_a=side_a,
                side_b=side_b,
                crossing_line=line,
            )

            annotated_video_path = renderer.render(
                input_path=video_path,
                output_path=(
                    output_dir
                    / "annotated_video.mp4"
                ),
                fps=metadata.fps,
                width=metadata.width,
                height=metadata.height,
                total_frames=metadata.frame_count,
                frame_state=frame_state,
                progress_callback=(
                    lambda p, d: self._report(
                        progress_callback,
                        0.84 + p * 0.14,
                        d,
                    )
                ),
            )

            rendering_elapsed = (
                time.perf_counter()
                - render_start
            )

        # =====================================================
        # RESULT / AUDIT
        # =====================================================
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
                time.perf_counter()
                - total_start
            ),
            "processing_fps": (
                metadata.frame_count
                / max(
                    time.perf_counter()
                    - total_start,
                    1e-9,
                )
            ),
            "inference_time_sec": inference_elapsed,
            "frame_state_time_sec": state_elapsed,
            "rendering_time_sec": rendering_elapsed,
            "model": self.config.detection.model_name,
            "tracker": self.config.detection.tracker,
            "imgsz": self.config.detection.imgsz,
            "vid_stride": 1,
        }

        artifacts = {
            "tracks_raw": str(
                output_dir / "tracks_raw.csv"
            ),
            "frame_vehicle_state": str(
                output_dir / "frame_vehicle_state.csv"
            ),
            "vehicle_events": str(
                output_dir / "vehicle_events.csv"
            ),
            "final_vehicle_crossings": str(
                output_dir / "final_vehicle_crossings.csv"
            ),
            "final_vehicle_counts": str(
                output_dir / "final_vehicle_counts.csv"
            ),
            "vehicle_direction_counts": str(
                output_dir
                / "vehicle_direction_counts.csv"
            ),
            "annotated_video": (
                str(annotated_video_path)
                if annotated_video_path is not None
                else None
            ),
            "result_json": str(
                output_dir / "result.json"
            ),
        }

        result_json = {
            "status": "completed",
            "video": video_info,
            "configuration": {
                "side_a": {
                    "meaning": "APPROACHING_CAMERA",
                    "x1_ratio": self.config.counting.side_a_x1_ratio,
                    "y1_ratio": self.config.counting.side_a_y1_ratio,
                    "x2_ratio": self.config.counting.side_a_x2_ratio,
                    "y2_ratio": self.config.counting.side_a_y2_ratio,
                },
                "side_b": {
                    "meaning": "AWAY_FROM_CAMERA",
                    "x1_ratio": self.config.counting.side_b_x1_ratio,
                    "y1_ratio": self.config.counting.side_b_y1_ratio,
                    "x2_ratio": self.config.counting.side_b_x2_ratio,
                    "y2_ratio": self.config.counting.side_b_y2_ratio,
                },
                "crossing_line": {
                    "x1": line.x1,
                    "y1": line.y1,
                    "x2": line.x2,
                    "y2": line.y2,
                },
                "frame_by_frame": True,
                "vid_stride": 1,
                "identity_source": "BoT-SORT track_id",
                "vehicle_category_policy": (
                    "locked for track lifetime"
                ),
            },
            "counts": summary["counts"],
            "direction_counts": summary[
                "direction_counts"
            ],
            "total": summary["total"],
            "tracked_vehicle_ids": summary[
                "tracked_vehicle_ids"
            ],
            "counted_vehicle_ids": summary[
                "counted_vehicle_ids"
            ],
            "performance": performance,
            "artifacts": artifacts,
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
                ensure_ascii=False,
            )

        self._report(
            progress_callback,
            1.0,
            "Processing complete.",
        )

        # Console summary.
        print("\n" + "=" * 70)
        print("PHASE3NEW — FRAME-BY-FRAME VEHICLE COUNTER")
        print("=" * 70)
        print(
            f"Frames processed : {metadata.frame_count:,}"
        )
        print(
            f"Tracked IDs      : {summary['tracked_vehicle_ids']:,}"
        )
        print(
            f"Counted IDs      : {summary['counted_vehicle_ids']:,}"
        )
        print("-" * 70)

        for class_name, count in summary[
            "counts"
        ].items():
            print(
                f"{class_name:<15}: {count}"
            )

        print("-" * 70)
        print(
            f"{'TOTAL VEHICLES':<15}: "
            f"{summary['total']}"
        )

        print("\nDirection:")
        for row in summary[
            "direction_counts"
        ]:
            print(
                f"{row['vehicle_category']:<15} "
                f"{row['direction']:<10} "
                f"{row['count']}"
            )

        return CountingResult(
            status="completed",
            video=video_info,
            counts=summary["counts"],
            direction_counts=summary[
                "direction_counts"
            ],
            count_confidence=[],
            overall_confidence={
                "confidence": 1.0,
                "flag": "FRAME_STATE_COUNT",
            },
            total=summary["total"],
            performance=performance,
            artifacts=artifacts,
            frame_state=frame_state,
            final_crossings=events,
            trajectory=frame_state,
            crossing_events=events,
            vehicle_events=events,
        )

from __future__ import annotations

import inspect
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import pandas as pd

from app.core.config import AppConfig, build_config
from app.counting.counter import TrafficCounter
from app.counting.track_classifier import build_track_level
from app.inference.confidence import ConfidenceEngine
from app.inference.detector_tracker import YOLOBoTSORTTracker
from app.video.validator import VideoMetadata, read_video_metadata


ProgressCallback = Callable[[float, str], None]


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
    Production orchestration layer for the robust branch.

    Pipeline
    --------
    Video validation
        ↓
    Phase 1 — YOLO26m + existing BoT-SORT adapter
        ↓
    Phase 2 — track-level class assignment
        ↓
    Phase 3 — RobustCrossing/TrafficCounter
        ↓
    Phase 6B — confidence
        ↓
    Optional Phase 6C — annotated video

    Compatibility rule
    ------------------
    The robust branch detector_tracker.py exposes a compatible
    `vid_stride` parameter. The robust config keeps vid_stride=1
    so Phase 1 remains baseline-like while Phase 3 is changed.
    """

    def __init__(self, config: AppConfig | None = None) -> None:
        self.config = config or build_config()

    def validate(self, video_path: str | Path) -> VideoMetadata:
        return read_video_metadata(video_path)

    @staticmethod
    def _report(
        progress_callback: ProgressCallback | None,
        progress: float,
        description: str,
    ) -> None:
        if progress_callback is None:
            return
        progress_callback(
            max(0.0, min(1.0, float(progress))),
            description,
        )

    @staticmethod
    def _safe_direction_counts(
        final_crossings: pd.DataFrame,
    ) -> pd.DataFrame:
        columns = ["track_class", "direction", "count"]
        if final_crossings.empty:
            return pd.DataFrame(columns=columns)

        required = {"track_class", "direction"}
        missing = required - set(final_crossings.columns)
        if missing:
            raise ValueError(
                "final_crossings missing direction columns: "
                f"{sorted(missing)}"
            )

        return (
            final_crossings
            .groupby(["track_class", "direction"], dropna=False)
            .size()
            .rename("count")
            .reset_index()
        )

    @staticmethod
    def _save_track_audit(
        counting_result,
        output_dir: Path,
    ) -> Path | None:
        track_audit = getattr(counting_result, "track_audit", None)
        if track_audit is None or not isinstance(track_audit, pd.DataFrame):
            return None

        path = output_dir / "track_crossing_audit.csv"
        track_audit.to_csv(path, index=False)
        return path

    def _build_counter(
        self,
        *,
        metadata: VideoMetadata,
    ) -> TrafficCounter:
        """
        Build TrafficCounter using only parameters supported by the
        runtime counter.py.

        This protects the orchestration layer from stale/mismatched
        counter implementations while still enforcing the required
        baseline counting interface.
        """

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

        candidate_kwargs = {
            "line_x1": line_x1,
            "line_y1": line_y1,
            "line_x2": line_x2,
            "line_y2": line_y2,
            "line_deadband_px": (
                self.config.counting.line_deadband_px
            ),
            "max_trajectory_gap_sec": (
                self.config.counting.max_trajectory_gap_sec
            ),
            "moto_dedup_time_sec": (
                self.config.counting.moto_dedup_time_sec
            ),
            "moto_dedup_distance_px": (
                self.config.counting.moto_dedup_distance_px
            ),
            "vehicle_classes": set(
                self.config.vehicle_classes
            ),
            "fps": metadata.fps,

            # Crossing identity / fragmentation.
            "pre_crossing_distance_px": (
                self.config.counting.pre_crossing_distance_px
            ),
            "max_identity_reconnect_gap_sec": (
                self.config.counting.max_identity_reconnect_gap_sec
            ),
            "max_identity_reconnect_distance_px": (
                self.config.counting.max_identity_reconnect_distance_px
            ),
            "identity_match_threshold": (
                self.config.counting.identity_match_threshold
            ),
            "identity_match_margin": (
                self.config.counting.identity_match_margin
            ),
            "velocity_gate_px_per_frame": (
                self.config.counting.velocity_gate_px_per_frame
            ),
            "min_pre_crossing_observations": (
                self.config.counting.min_pre_crossing_observations
            ),

            # Robust crossing geometry.
            "crossing_corridor_px": (
                self.config.counting.crossing_corridor_px
            ),
            "min_direction_displacement_px": (
                self.config.counting.min_direction_displacement_px
            ),
            "direction_window": (
                self.config.counting.direction_window
            ),

            # Phase 1 — trajectory engine.
            "trajectory_smoothing_alpha": (
                self.config.counting.trajectory_smoothing_alpha
            ),
            "trajectory_velocity_window": (
                self.config.counting.trajectory_velocity_window
            ),
            "max_velocity_px_per_frame": (
                self.config.counting.max_velocity_px_per_frame
            ),

            # Phase 2 — crossing corridor.
            "min_pre_zone_observations": (
                self.config.counting.min_pre_zone_observations
            ),
            "min_corridor_observations": (
                self.config.counting.min_corridor_observations
            ),
            "min_post_zone_observations": (
                self.config.counting.min_post_zone_observations
            ),
            "require_post_zone": (
                self.config.counting.require_post_zone
            ),

            # Conservative final duplicate suppression.
            "duplicate_time_sec": (
                self.config.counting.duplicate_time_sec
            ),
            "duplicate_distance_px": (
                self.config.counting.duplicate_distance_px
            ),
        }

        signature = inspect.signature(
            TrafficCounter.__init__
        )

        supported_parameters = {
            name
            for name in signature.parameters
            if name != "self"
        }

        kwargs = {
            key: value
            for key, value in candidate_kwargs.items()
            if key in supported_parameters
        }

        required_parameters = {
            "line_x1",
            "line_y1",
            "line_x2",
            "line_y2",
            "line_deadband_px",
            "max_trajectory_gap_sec",
            "moto_dedup_time_sec",
            "moto_dedup_distance_px",
            "vehicle_classes",
            "fps",
        }

        missing_required = (
            required_parameters
            - set(kwargs)
        )

        if missing_required:
            raise RuntimeError(
                "Runtime TrafficCounter interface is incompatible.\n\n"
                f"Missing required parameters: "
                f"{sorted(missing_required)}\n\n"
                f"Actual signature:\n{signature}"
            )

        optional_robust = {
            "pre_crossing_distance_px",
            "max_identity_reconnect_gap_sec",
            "max_identity_reconnect_distance_px",
            "identity_match_threshold",
            "identity_match_margin",
            "velocity_gate_px_per_frame",
            "min_pre_crossing_observations",
            "crossing_corridor_px",
            "min_direction_displacement_px",
            "direction_window",
        }

        missing_robust = (
            optional_robust
            - set(kwargs)
        )

        print()
        print("=" * 80)
        print("TRAFFIC COUNTER RUNTIME INTERFACE")
        print("=" * 80)
        print(
            f"Module: {TrafficCounter.__module__}"
        )
        print(
            f"Signature: {signature}"
        )

        if missing_robust:
            print(
                "WARNING: runtime TrafficCounter is missing "
                "robust parameters:"
            )
            for name in sorted(missing_robust):
                print(f"  - {name}")

        print("=" * 80)
        print()

        return TrafficCounter(
            **kwargs
        )

    @staticmethod
    def _normalize_confidence_inputs(
        final_crossings: pd.DataFrame,
        trajectory: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Normalize Phase-3 numeric columns before Phase-6B confidence
        scoring so pandas nullable/object values never reach NumPy ufuncs.
        """
        final_crossings = final_crossings.copy()
        trajectory = trajectory.copy()

        crossing_numeric = [
            "crossing_frame",
            "crossing_time_sec",
            "crossing_x",
            "crossing_y",
            "line_distance_px",
            "frame_gap",
            "track_class_ratio",
        ]

        for column in crossing_numeric:
            if column in final_crossings.columns:
                final_crossings[column] = pd.to_numeric(
                    final_crossings[column],
                    errors="coerce",
                )

        trajectory_numeric = [
            "frame_id",
            "timestamp_sec",
            "bottom_center_x",
            "bottom_center_y",
            "line_value",
            "line_distance_px",
            "dx",
            "dy",
            "frame_delta",
            "time_delta_sec",
        ]

        for column in trajectory_numeric:
            if column in trajectory.columns:
                trajectory[column] = pd.to_numeric(
                    trajectory[column],
                    errors="coerce",
                )

        return final_crossings, trajectory

    def process(
        self,
        video_path: str | Path,
        *,
        render_video: bool = False,
        progress_callback: ProgressCallback | None = None,
    ) -> CountingResult:
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        total_start = time.perf_counter()

        # =====================================================
        # VIDEO VALIDATION
        # =====================================================
        self._report(progress_callback, 0.01, "Validating video...")
        metadata = self.validate(video_path)

        if metadata.fps <= 0:
            raise ValueError(f"Invalid FPS: {metadata.fps}")
        if metadata.width <= 0 or metadata.height <= 0:
            raise ValueError(
                f"Invalid video dimensions: {metadata.width}x{metadata.height}"
            )

        self._report(
            progress_callback,
            0.05,
            (
                f"Video validated: {metadata.width}x{metadata.height}, "
                f"{metadata.fps:.2f} FPS, {metadata.frame_count:,} frames"
            ),
        )

        # =====================================================
        # OUTPUT DIRECTORY
        # =====================================================
        output_dir = Path(self.config.output_dir) / video_path.stem
        output_dir.mkdir(parents=True, exist_ok=True)

        # =====================================================
        # PHASE 1 — DETECTION + TRACKING
        # =====================================================
        phase1_start = time.perf_counter()
        self._report(
            progress_callback,
            0.08,
            "Starting YOLO26m + BoT-SORT...",
        )

        # Robust-branch detector_tracker.py accepts vid_stride.
        # Keep it at the configured baseline value (robust branch:
        # vid_stride=1) so Phase 1 remains baseline-compatible.
        tracker = YOLOBoTSORTTracker(
            model_name=self.config.detection.model_name,
            tracker=self.config.detection.tracker,
            imgsz=self.config.detection.imgsz,
            conf=self.config.detection.conf_threshold,
            iou=self.config.detection.iou_threshold,
            vid_stride=self.config.detection.vid_stride,
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
                    0.08 + p * 0.67,
                    d,
                )
            ),
        )

        phase1_elapsed = time.perf_counter() - phase1_start

        if tracks_raw.empty:
            raise RuntimeError(
                "No target objects were tracked in the input video."
            )

        tracks_raw.to_csv(output_dir / "tracks_raw.csv", index=False)

        # =====================================================
        # PHASE 2 — TRACK-LEVEL CLASSIFICATION
        # =====================================================
        phase2_start = time.perf_counter()
        self._report(
            progress_callback,
            0.77,
            "Building track-level classes...",
        )

        track_level, tracks_phase2 = build_track_level(
            tracks_raw,
            fps=metadata.fps,
        )

        phase2_elapsed = time.perf_counter() - phase2_start

        track_level.to_csv(
            output_dir / "track_quality_summary.csv",
            index=False,
        )
        tracks_phase2.to_csv(
            output_dir / "tracks_with_track_class.csv",
            index=False,
        )

        # =====================================================
        # PHASE 3 — ROBUST COUNTING
        # =====================================================
        phase3_start = time.perf_counter()
        self._report(
            progress_callback,
            0.83,
            "Running robust vehicle crossing engine...",
        )

        line_x1 = metadata.width * self.config.counting.line_x1_ratio
        line_y1 = metadata.height * self.config.counting.line_y1_ratio
        line_x2 = metadata.width * self.config.counting.line_x2_ratio
        line_y2 = metadata.height * self.config.counting.line_y2_ratio

        # These are the parameters supported by the current
        # robust TrafficCounter implementation.
        counter = self._build_counter(
            metadata=metadata
        )

        counting_result = counter.count(
            tracks_phase2
        )

        (
            final_crossings_for_confidence,
            trajectory_for_confidence,
        ) = self._normalize_confidence_inputs(
            counting_result.final_crossings,
            counting_result.trajectory,
        )

        phase3_elapsed = time.perf_counter() - phase3_start

        # =====================================================
        # SAVE PHASE 3 ARTIFACTS
        # =====================================================
        counting_result.crossing_events.to_csv(
            output_dir / "crossing_events.csv",
            index=False,
        )
        counting_result.vehicle_events.to_csv(
            output_dir / "vehicle_events.csv",
            index=False,
        )
        counting_result.final_crossings.to_csv(
            output_dir / "final_vehicle_crossings.csv",
            index=False,
        )

        # =====================================================
        # PHASE 1/2 DIAGNOSTIC ARTIFACTS
        # =====================================================
        phase12_trajectory_path = output_dir / "phase12_trajectory.csv"
        phase12_audit_path = output_dir / "phase12_crossing_corridor_audit.csv"

        phase12_trajectory = getattr(
            counting_result,
            "phase12_trajectory",
            None,
        )
        phase12_audit = getattr(
            counting_result,
            "phase12_audit",
            None,
        )

        if isinstance(phase12_trajectory, pd.DataFrame):
            phase12_trajectory.to_csv(
                phase12_trajectory_path,
                index=False,
            )
        else:
            phase12_trajectory_path = None

        if isinstance(phase12_audit, pd.DataFrame):
            phase12_audit.to_csv(
                phase12_audit_path,
                index=False,
            )
        else:
            phase12_audit_path = None

        direction_counts_df = self._safe_direction_counts(
            counting_result.final_crossings
        )
        direction_counts_df.to_csv(
            output_dir / "vehicle_direction_counts.csv",
            index=False,
        )

        track_audit_path = self._save_track_audit(
            counting_result,
            output_dir,
        )

        final_counts_df = pd.DataFrame(
            [
                {
                    "class": class_name,
                    "quantity": quantity,
                }
                for class_name, quantity in counting_result.counts.items()
            ]
        )
        final_counts_df.to_csv(
            output_dir / "final_vehicle_counts.csv",
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
                ensure_ascii=False,
                default=str,
            )

        # =====================================================
        # PHASE 6B — CONFIDENCE
        # =====================================================
        confidence_start = time.perf_counter()
        self._report(
            progress_callback,
            0.88,
            "Calculating confidence...",
        )

        confidence_engine = ConfidenceEngine()

        track_confidence = confidence_engine.build_track_confidence(
            track_level
        )

        crossing_confidence = confidence_engine.build_crossing_confidence(
            final_crossings=final_crossings_for_confidence,
            trajectory=trajectory_for_confidence,
            track_confidence=track_confidence,
            fps=metadata.fps,
        )

        count_confidence = confidence_engine.build_count_confidence(
            crossing_confidence=crossing_confidence,
            final_counts=counting_result.counts,
        )

        overall_confidence = confidence_engine.build_overall_confidence(
            crossing_confidence
        )

        confidence_elapsed = time.perf_counter() - confidence_start

        track_confidence.to_csv(
            output_dir / "track_confidence.csv",
            index=False,
        )
        crossing_confidence.to_csv(
            output_dir / "crossing_confidence.csv",
            index=False,
        )
        pd.DataFrame(count_confidence).to_csv(
            output_dir / "count_confidence.csv",
            index=False,
        )

        # =====================================================
        # OPTIONAL PHASE 6C — RENDERING
        # =====================================================
        annotated_video_path: Path | None = None
        rendering_elapsed = 0.0

        if render_video:
            self._report(
                progress_callback,
                0.94,
                "Rendering annotated video...",
            )
            render_start = time.perf_counter()

            from app.video.renderer import VideoRenderer

            renderer = VideoRenderer(
                line_x1=line_x1,
                line_y1=line_y1,
                line_x2=line_x2,
                line_y2=line_y2,
            )

            annotated_video_path = renderer.render(
                input_path=video_path,
                output_path=output_dir / "annotated_video.mp4",
                fps=metadata.fps,
                width=metadata.width,
                height=metadata.height,
                total_frames=metadata.frame_count,
                tracks_phase2=tracks_phase2,
                final_crossings=counting_result.final_crossings,
                progress_callback=(
                    lambda p, d: self._report(
                        progress_callback,
                        0.94 + p * 0.05,
                        d,
                    )
                ),
            )

            rendering_elapsed = time.perf_counter() - render_start

        # =====================================================
        # FINAL RESULT
        # =====================================================
        total_elapsed = time.perf_counter() - total_start
        processing_fps = (
            metadata.frame_count / total_elapsed
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
            "result_json": str(output_dir / "result.json"),
            "count_csv": str(output_dir / "final_vehicle_counts.csv"),
            "direction_csv": str(output_dir / "vehicle_direction_counts.csv"),
            "count_confidence_csv": str(output_dir / "count_confidence.csv"),
            "crossing_events_csv": str(output_dir / "crossing_events.csv"),
            "vehicle_events_csv": str(output_dir / "vehicle_events.csv"),
            "final_crossings_csv": str(
                output_dir / "final_vehicle_crossings.csv"
            ),
            "phase12_trajectory_csv": (
                str(phase12_trajectory_path)
                if phase12_trajectory_path is not None
                else None
            ),
            "phase12_crossing_corridor_audit_csv": (
                str(phase12_audit_path)
                if phase12_audit_path is not None
                else None
            ),
            "track_crossing_audit_csv": (
                str(track_audit_path)
                if track_audit_path is not None
                else None
            ),
            "annotated_video": (
                str(annotated_video_path)
                if annotated_video_path is not None
                else None
            ),
        }

        performance = {
            "total_processing_time_sec": total_elapsed,
            "processing_fps": processing_fps,
            "phase1_inference_time_sec": phase1_elapsed,
            "phase2_classification_time_sec": phase2_elapsed,
            "phase3_counting_time_sec": phase3_elapsed,
            "phase6b_confidence_time_sec": confidence_elapsed,
            "rendering_time_sec": rendering_elapsed,
            "model": self.config.detection.model_name,
            "tracker": self.config.detection.tracker,
            "imgsz": self.config.detection.imgsz,
            "configured_vid_stride": self.config.detection.vid_stride,
            "configured_conf_threshold": self.config.detection.conf_threshold,
            "configured_iou_threshold": self.config.detection.iou_threshold,
            "crossing_corridor_px": self.config.counting.crossing_corridor_px,
            "line_deadband_px": self.config.counting.line_deadband_px,
            "duplicate_time_sec": self.config.counting.duplicate_time_sec,
            "duplicate_distance_px": self.config.counting.duplicate_distance_px,
        }

        direction_counts = [
            {
                "track_class": row["track_class"],
                "direction": row["direction"],
                "count": int(row["count"]),
            }
            for _, row in direction_counts_df.iterrows()
        ]

        result_json = {
            "status": "completed",
            "video": video_info,
            "counts": counting_result.counts,
            "count_confidence": count_confidence,
            "overall_confidence": overall_confidence,
            "direction_counts": direction_counts,
            "total": counting_result.total,
            "performance": performance,
            "audit": counting_result.audit,
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
                default=str,
            )

        self._report(
            progress_callback,
            1.0,
            "Processing complete.",
        )

        # =====================================================
        # CONSOLE REPORT
        # =====================================================
        if isinstance(phase12_trajectory, pd.DataFrame) and isinstance(phase12_audit, pd.DataFrame):
            print("\n" + "=" * 70)
            print("PHASE 1/2 TRAJECTORY + CROSSING CORRIDOR AUDIT")
            print("=" * 70)
            print(
                f"Tracks analyzed      : {len(phase12_audit):,}"
            )
            print(
                f"Phase 1 PASS         : {int((phase12_audit['phase1_status'] == 'PASS').sum()):,}"
            )
            print(
                f"Phase 1 REVIEW       : {int((phase12_audit['phase1_status'] == 'REVIEW').sum()):,}"
            )
            print(
                f"Phase 1 FAIL         : {int((phase12_audit['phase1_status'] == 'FAIL').sum()):,}"
            )
            print(
                f"Phase 2 PASS         : {int((phase12_audit['phase2_status'] == 'PASS').sum()):,}"
            )
            print(
                f"Phase 2 REVIEW       : {int((phase12_audit['phase2_status'] == 'REVIEW').sum()):,}"
            )
            print(
                f"Phase 2 FAIL         : {int((phase12_audit['phase2_status'] == 'FAIL').sum()):,}"
            )
            print(
                f"P1 + P2 PASS         : {int(phase12_audit['counted'].sum()):,}"
            )
            print("\nZone path examples:")
            print(
                phase12_audit["zone_path"]
                .value_counts()
                .head(10)
                .to_string()
            )
            print(
                f"\nPhase 1/2 audit: {phase12_audit_path}"
            )

        print("\n" + "=" * 70)
        print("FINAL VEHICLE COUNT")
        print("=" * 70)

        for class_name, count in counting_result.counts.items():
            print(f"{class_name:<15}: {count}")

        print("-" * 70)
        print(f"{'TOTAL VEHICLES':<15}: {counting_result.total}")

        print("\n" + "=" * 70)
        print("COUNT CONFIDENCE")
        print("=" * 70)

        for row in count_confidence:
            confidence = row.get("confidence")
            confidence_text = (
                f"{confidence:.4f}"
                if confidence is not None
                else "N/A"
            )
            print(
                f"{row['class']:<15}: "
                f"{confidence_text:<8} "
                f"{row.get('flag', '')}"
            )

        print("\n" + "=" * 70)
        print("COUNT BY DIRECTION")
        print("=" * 70)

        for row in direction_counts:
            print(
                f"{str(row['track_class']):<12} "
                f"{str(row['direction']):<20} "
                f"{row['count']}"
            )

        if track_audit_path is not None:
            print(f"\nTrack audit: {track_audit_path}")

        return CountingResult(
            status="completed",
            video=video_info,
            counts=counting_result.counts,
            direction_counts=direction_counts,
            count_confidence=count_confidence,
            overall_confidence=overall_confidence,
            total=counting_result.total,
            performance=performance,
            artifacts=artifacts,
        )

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import math
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CrossingConfig:
    """Phase 1 + Phase 2 engine.

    Architecture:

        TRACK
          |
          v
        TRAJECTORY
          |- signed distance
          |- normal velocity
          |- speed
          |- trajectory continuity
          |
          v
        ZONE CONTEXT
          |- NO_CROSSING
          |- APPROACHING
          |- NEAR_LINE
          |- CROSSING_CANDIDATE
          |
          v
        CROSSING DETECTOR
          |- stable-side transition
          |- raw segment intersection
          |- deadband-aware transition
          |- gap/velocity bridge
          |
          v
        CROSSING EVIDENCE
          |- pre evidence
          |- corridor evidence
          |- post evidence
          |- direction evidence
          |- fast/sparse evidence
          |
          v
        AUDIT

    Identity management is intentionally outside this module. A single
    physical identity is expected to arrive under ``identity_column``.
    """

    # ------------------------------------------------------------------
    # LINE / ZONE GEOMETRY
    # ------------------------------------------------------------------
    line_deadband_px: float = 8.0
    corridor_px: float = 45.0
    corridor_exit_px: float = 60.0
    approach_distance_px: float = 120.0

    # ------------------------------------------------------------------
    # TRAJECTORY
    # ------------------------------------------------------------------
    max_trajectory_gap_sec: float = 1.50
    smoothing_alpha: float = 0.35
    velocity_window: int = 5
    min_direction_displacement_px: float = 8.0
    direction_window: int = 3
    max_velocity_px_per_frame: float = 80.0
    max_velocity_bridge_px_per_frame: float = 140.0
    min_normal_velocity_px_per_frame: float = 1.0

    # ------------------------------------------------------------------
    # EVIDENCE
    # ------------------------------------------------------------------
    min_track_observations: int = 2
    min_pre_zone_observations: int = 2
    min_corridor_observations: int = 1
    min_post_zone_observations: int = 1
    min_direction_confidence: float = 0.50
    min_normal_displacement_px: float = 8.0

    # Evidence is diagnostic. It does NOT invalidate geometry.
    require_post_zone: bool = False
    allow_crossing_without_pre: bool = True
    allow_crossing_without_post: bool = True
    allow_zero_corridor_crossing: bool = True

    # ------------------------------------------------------------------
    # GAP / FAST OBJECT HANDLING
    # ------------------------------------------------------------------
    gap_bridge_enabled: bool = True
    gap_bridge_max_frames: int = 6
    fast_speed_multiplier: float = 0.85

    # ------------------------------------------------------------------
    # BUSINESS LABELS
    # ------------------------------------------------------------------
    positive_normal_label: str = "L→R"
    negative_normal_label: str = "R→L"

    vehicle_classes: tuple[str, ...] = (
        "motorcycle",
        "car",
        "truck",
        "bus",
    )


class RobustCrossingEngine:
    """Candidate-preserving trajectory and crossing engine.

    Important design rule:
    A track is NOT discarded just because it lacks PRE/CORRIDOR/POST evidence.
    Crossing geometry and crossing evidence are separate concepts.
    """

    TRAJECTORY_COLUMNS = [
        "raw_x",
        "raw_y",
        "smooth_x",
        "smooth_y",
        "dx",
        "dy",
        "frame_delta",
        "time_delta_sec",
        "speed_px_per_frame",
        "velocity_normal_px_per_frame",
        "velocity_tangent_px_per_frame",
        "signed_distance_px",
        "line_distance_px",
        "raw_signed_distance_px",
        "raw_line_distance_px",
        "raw_side",
        "stable_side",
        "zone",
        "zone_context",
        "direction_local",
        "normal_direction_local",
        "trajectory_continuity",
        "gap_bridge_candidate",
        "speed_anomaly",
        "trajectory_quality",
    ]

    EVENT_COLUMNS = [
        "crossing_id",
        "track_id",
        "track_ids",
        "first_frame",
        "last_frame",
        "crossing_frame",
        "crossing_time_sec",
        "crossing_x",
        "crossing_y",
        "direction",
        "normal_direction",
        "line_direction",
        "side_transition",
        "track_class",
        "track_class_ratio",
        "class_ambiguous",
        "line_distance_px",
        "previous_side",
        "current_side",
        "frame_gap",
        "crossing_method",
        "crossing_candidate_class",
        "fast_crossing",
        "sparse_crossing",
        "gap_bridge_used",
        "track_observations",
        "crossing_index",
        "pre_zone_observations",
        "corridor_observations",
        "post_zone_observations",
        "pre_zone_evidence",
        "corridor_evidence",
        "post_zone_evidence",
        "zone_path",
        "zone_chatter_count",
        "trajectory_quality",
        "direction_confidence",
        "normal_displacement_px",
        "corridor_confidence",
        "phase1_status",
        "phase2_status",
        "count_eligibility",
        "counted",
    ]

    AUDIT_COLUMNS = [
        "crossing_id",
        "track_ids",
        "first_frame",
        "last_frame",
        "track_class",
        "track_observations",
        "first_side",
        "last_side",
        "min_distance_px",
        "max_speed_px_per_frame",
        "mean_speed_px_per_frame",
        "max_abs_normal_velocity_px_per_frame",
        "mean_abs_normal_velocity_px_per_frame",
        "mean_abs_tangent_velocity_px_per_frame",
        "trajectory_direction",
        "normal_direction",
        "direction_confidence",
        "normal_displacement_px",
        "zone_path",
        "zone_chatter_count",
        "pre_zone_observations",
        "corridor_observations",
        "post_zone_observations",
        "pre_zone_evidence",
        "corridor_evidence",
        "post_zone_evidence",
        "crossing_detected",
        "crossing_candidate_class",
        "fast_crossing",
        "sparse_crossing",
        "gap_bridge_used",
        "crossing_frame",
        "frame_gap",
        "crossing_method",
        "crossing_direction",
        "phase1_status",
        "phase2_status",
        "phase1_pass",
        "phase2_pass",
        "count_eligibility",
        "counted",
        "failure_reason",
    ]

    def __init__(
        self,
        *,
        line_x1: float,
        line_y1: float,
        line_x2: float,
        line_y2: float,
        fps: float,
        config: Optional[CrossingConfig] = None,
    ) -> None:
        if fps <= 0:
            raise ValueError(f"fps must be > 0, got {fps}")

        self.fps = float(fps)
        self.config = config or CrossingConfig()

        if not 0.0 < self.config.smoothing_alpha <= 1.0:
            raise ValueError("smoothing_alpha must be in (0, 1].")
        if self.config.line_deadband_px < 0:
            raise ValueError("line_deadband_px must be >= 0.")
        if self.config.corridor_px <= self.config.line_deadband_px:
            raise ValueError("corridor_px must be > line_deadband_px.")
        if self.config.corridor_exit_px < self.config.corridor_px:
            raise ValueError("corridor_exit_px must be >= corridor_px.")
        if self.config.approach_distance_px < self.config.corridor_exit_px:
            raise ValueError("approach_distance_px must be >= corridor_exit_px.")

        self.x1 = float(line_x1)
        self.y1 = float(line_y1)
        self.x2 = float(line_x2)
        self.y2 = float(line_y2)

        self.line_dx = self.x2 - self.x1
        self.line_dy = self.y2 - self.y1
        self.line_length = math.hypot(self.line_dx, self.line_dy)

        if self.line_length <= 0:
            raise ValueError("Counting line length must be > 0.")

        # Unit tangent along the counting line.
        self.tangent_x = self.line_dx / self.line_length
        self.tangent_y = self.line_dy / self.line_length

        # Unit normal to the counting line.
        # Positive normal corresponds to positive signed distance.
        self.normal_x = -self.line_dy / self.line_length
        self.normal_y = self.line_dx / self.line_length

    # ==================================================================
    # GEOMETRY
    # ==================================================================

    def signed_line_value(self, x: float, y: float) -> float:
        return (
            self.line_dx * (y - self.y1)
            - self.line_dy * (x - self.x1)
        )

    def signed_distance(self, x: float, y: float) -> float:
        return self.signed_line_value(x, y) / self.line_length

    def line_distance(self, x: float, y: float) -> float:
        return abs(self.signed_distance(x, y))

    def side(self, x: float, y: float) -> int:
        distance = self.signed_distance(x, y)
        if abs(distance) <= self.config.line_deadband_px:
            return 0
        return 1 if distance > 0.0 else -1

    # ==================================================================
    # SMOOTHING
    # ==================================================================

    def _ema(self, values: np.ndarray) -> np.ndarray:
        if values.size == 0:
            return values.copy()

        alpha = float(self.config.smoothing_alpha)
        output = np.empty_like(values, dtype=np.float64)
        output[0] = values[0]

        for i in range(1, len(values)):
            output[i] = alpha * values[i] + (1.0 - alpha) * output[i - 1]

        return output

    # ==================================================================
    # SEGMENT GEOMETRY
    # ==================================================================

    @staticmethod
    def _orientation(
        ax: float,
        ay: float,
        bx: float,
        by: float,
        cx: float,
        cy: float,
    ) -> float:
        return (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)

    @staticmethod
    def _on_segment(
        ax: float,
        ay: float,
        bx: float,
        by: float,
        px: float,
        py: float,
    ) -> bool:
        eps = 1e-9
        return (
            min(ax, bx) - eps <= px <= max(ax, bx) + eps
            and min(ay, by) - eps <= py <= max(ay, by) + eps
        )

    def segments_intersect(
        self,
        p1: tuple[float, float],
        p2: tuple[float, float],
        q1: tuple[float, float],
        q2: tuple[float, float],
    ) -> bool:
        eps = 1e-9

        o1 = self._orientation(*p1, *p2, *q1)
        o2 = self._orientation(*p1, *p2, *q2)
        o3 = self._orientation(*q1, *q2, *p1)
        o4 = self._orientation(*q1, *q2, *p2)

        if (
            ((o1 > eps and o2 < -eps) or (o1 < -eps and o2 > eps))
            and ((o3 > eps and o4 < -eps) or (o3 < -eps and o4 > eps))
        ):
            return True

        if abs(o1) <= eps and self._on_segment(*p1, *p2, *q1):
            return True
        if abs(o2) <= eps and self._on_segment(*p1, *p2, *q2):
            return True
        if abs(o3) <= eps and self._on_segment(*q1, *q2, *p1):
            return True
        if abs(o4) <= eps and self._on_segment(*q1, *q2, *p2):
            return True

        return False

    def trajectory_intersects_line(
        self,
        previous_point: tuple[float, float],
        current_point: tuple[float, float],
    ) -> bool:
        return self.segments_intersect(
            previous_point,
            current_point,
            (self.x1, self.y1),
            (self.x2, self.y2),
        )

    def estimate_crossing_point(
        self,
        previous_point: tuple[float, float],
        current_point: tuple[float, float],
    ) -> tuple[float, float]:
        d_prev = self.signed_distance(*previous_point)
        d_curr = self.signed_distance(*current_point)
        denominator = d_prev - d_curr

        if abs(denominator) < 1e-9:
            return current_point

        alpha = max(0.0, min(1.0, d_prev / denominator))

        return (
            float(
                previous_point[0]
                + alpha * (current_point[0] - previous_point[0])
            ),
            float(
                previous_point[1]
                + alpha * (current_point[1] - previous_point[1])
            ),
        )

    # ==================================================================
    # STABLE SIDE / DEAD-BAND AWARE LOGIC
    # ==================================================================

    def _stable_side_series(self, raw_signed_distance: np.ndarray) -> np.ndarray:
        """Return a side series that preserves the last known non-zero side.

        This explicitly handles:

            +1 -> 0 -> -1
            -1 -> 0 -> +1

        as a valid side transition rather than losing the crossing because
        an intermediate observation lies inside the deadband.
        """
        result = np.zeros(len(raw_signed_distance), dtype=np.int8)
        last_stable = 0

        for i, distance in enumerate(raw_signed_distance):
            if abs(distance) > self.config.line_deadband_px:
                last_stable = 1 if distance > 0 else -1
                result[i] = last_stable
            else:
                result[i] = last_stable

        return result

    def _side_transition(
        self,
        before_side: int,
        after_side: int,
    ) -> str:
        if before_side == -1 and after_side == 1:
            return "side_-1_to_+1"
        if before_side == 1 and after_side == -1:
            return "side_+1_to_-1"
        return "UNKNOWN"

    def _find_stable_side_transition(
        self,
        group: pd.DataFrame,
        end_index: int,
    ) -> tuple[bool, int, int, int | None]:
        """Find the nearest stable-side transition before/at end_index."""
        if end_index < 1:
            return False, 0, 0, None

        stable = group["stable_side"].to_numpy(dtype=np.int8)
        previous_stable = 0

        for i in range(0, end_index + 1):
            current = int(stable[i])
            if current == 0:
                continue
            if previous_stable != 0 and current != previous_stable:
                return True, previous_stable, current, i
            previous_stable = current

        return False, 0, 0, None

    # ==================================================================
    # ZONE STATE
    # ==================================================================

    def _zone_from_distance(self, distance: float, previous_zone: str) -> str:
        """Classify zone with spatial hysteresis.

        PRE / CORRIDOR / POST are relative to stable side. We preserve side
        separately; zone only describes proximity/context around the line.
        """
        d = abs(float(distance))

        if previous_zone == "CORRIDOR":
            if d <= self.config.corridor_exit_px:
                return "CORRIDOR"
        elif previous_zone == "NEAR_LINE":
            if d <= self.config.corridor_exit_px:
                return "NEAR_LINE"

        if d <= self.config.corridor_px:
            return "CORRIDOR"
        if d <= self.config.approach_distance_px:
            return "NEAR_LINE"
        return "PRE"

    @staticmethod
    def _zone_context(
        stable_side: int,
        zone: str,
        has_crossed: bool,
    ) -> str:
        if has_crossed:
            return "POST"
        if zone == "CORRIDOR":
            return "CROSSING_CANDIDATE"
        if zone == "NEAR_LINE":
            return "APPROACHING"
        return "NO_CROSSING"

    @staticmethod
    def _collapse_zone_path(zones: list[str]) -> tuple[str, int]:
        cleaned: list[str] = []
        chatter = 0

        for zone in zones:
            if not zone:
                continue
            if not cleaned or zone != cleaned[-1]:
                if cleaned:
                    previous = cleaned[-1]
                    if {previous, zone} <= {"PRE", "NEAR_LINE", "CORRIDOR"}:
                        chatter += 1
                    if {previous, zone} <= {"CORRIDOR", "POST"}:
                        chatter += 1
                cleaned.append(zone)

        return (
            " → ".join(cleaned) if cleaned else "UNKNOWN",
            chatter,
        )

    # ==================================================================
    # DIRECTION / TRAJECTORY METRICS
    # ==================================================================

    def _infer_trajectory_direction(self, group: pd.DataFrame) -> str:
        if len(group) < 2:
            return "UNKNOWN"

        first = group.iloc[0]
        last = group.iloc[-1]

        dx = float(last["raw_x"] - first["raw_x"])
        dy = float(last["raw_y"] - first["raw_y"])

        tangent_displacement = dx * self.tangent_x + dy * self.tangent_y

        if abs(tangent_displacement) < self.config.min_direction_displacement_px:
            return "UNKNOWN"

        return "L→R" if tangent_displacement > 0 else "R→L"

    def _infer_crossing_direction(
        self,
        group: pd.DataFrame,
        crossing_index: int,
    ) -> tuple[str, float, float]:
        window = max(1, int(self.config.direction_window))
        start = max(0, crossing_index - window)
        end = min(len(group) - 1, crossing_index + window)

        before = group.iloc[start : crossing_index + 1]
        after = group.iloc[crossing_index : end + 1]

        if before.empty or after.empty:
            return "UNKNOWN", 0.0, 0.0

        start_d = float(before.iloc[0]["raw_signed_distance_px"])
        end_d = float(after.iloc[-1]["raw_signed_distance_px"])
        normal_displacement = end_d - start_d

        before_side = 1 if start_d > 0 else -1 if start_d < 0 else 0
        after_side = 1 if end_d > 0 else -1 if end_d < 0 else 0

        displacement_conf = min(
            1.0,
            abs(normal_displacement)
            / max(self.config.min_normal_displacement_px, 1e-6),
        )

        velocity_values = pd.to_numeric(
            group.iloc[start : end + 1]["velocity_normal_px_per_frame"],
            errors="coerce",
        ).dropna()

        if velocity_values.empty:
            velocity_conf = 0.0
        else:
            mean_abs_velocity = float(velocity_values.abs().mean())
            velocity_conf = min(
                1.0,
                mean_abs_velocity
                / max(self.config.min_normal_velocity_px_per_frame, 1.0),
            )

        confidence = float(
            np.clip(
                0.70 * displacement_conf + 0.30 * velocity_conf,
                0.0,
                1.0,
            )
        )

        if before_side == -1 and after_side == 1:
            return self.config.positive_normal_label, confidence, normal_displacement
        if before_side == 1 and after_side == -1:
            return self.config.negative_normal_label, confidence, normal_displacement

        # When endpoints are in the deadband, use signed displacement.
        if normal_displacement > self.config.min_normal_displacement_px:
            return self.config.positive_normal_label, confidence, normal_displacement
        if normal_displacement < -self.config.min_normal_displacement_px:
            return self.config.negative_normal_label, confidence, normal_displacement

        return "UNKNOWN", confidence, normal_displacement

    # ==================================================================
    # TRAJECTORY PREPARATION
    # ==================================================================

    def prepare(self, trajectory: pd.DataFrame) -> pd.DataFrame:
        required = {
            "track_id",
            "frame_id",
            "timestamp_sec",
            "bottom_center_x",
            "bottom_center_y",
            "track_class",
        }

        missing = required - set(trajectory.columns)
        if missing:
            raise ValueError(
                f"Trajectory missing required columns: {sorted(missing)}"
            )

        if trajectory.empty:
            return trajectory.copy()

        df = (
            trajectory.copy()
            .sort_values(["track_id", "frame_id"])
            .reset_index(drop=True)
        )

        if "confidence" not in df.columns:
            df["confidence"] = 1.0

        if "class_name" not in df.columns:
            df["class_name"] = df["track_class"]

        if "track_class_ratio" not in df.columns:
            df["track_class_ratio"] = 1.0

        if "class_ambiguous" not in df.columns:
            df["class_ambiguous"] = False

        df["raw_x"] = pd.to_numeric(
            df["bottom_center_x"], errors="coerce"
        )
        df["raw_y"] = pd.to_numeric(
            df["bottom_center_y"], errors="coerce"
        )
        df["frame_id"] = pd.to_numeric(
            df["frame_id"], errors="coerce"
        )
        df["timestamp_sec"] = pd.to_numeric(
            df["timestamp_sec"], errors="coerce"
        )

        if df[["raw_x", "raw_y", "frame_id"]].isna().any().any():
            raise ValueError(
                "Trajectory contains NaN values in raw position/frame_id."
            )

        # --------------------------------------------------------------
        # Per-track trajectory features.
        # --------------------------------------------------------------
        frames: list[pd.DataFrame] = []

        for track_id, group in df.groupby("track_id", sort=False):
            group = group.sort_values("frame_id").copy().reset_index(drop=True)

            x = group["raw_x"].to_numpy(dtype=np.float64)
            y = group["raw_y"].to_numpy(dtype=np.float64)
            frames_arr = group["frame_id"].to_numpy(dtype=np.float64)
            time_arr = group["timestamp_sec"].to_numpy(dtype=np.float64)

            smooth_x = self._ema(x)
            smooth_y = self._ema(y)

            frame_delta = np.diff(frames_arr, prepend=frames_arr[0])
            time_delta = np.diff(time_arr, prepend=time_arr[0])

            frame_delta[0] = 0.0
            time_delta[0] = 0.0

            safe_frame_delta = np.where(frame_delta > 0.0, frame_delta, np.nan)

            raw_dx = np.diff(x, prepend=x[0])
            raw_dy = np.diff(y, prepend=y[0])

            smooth_dx = np.diff(smooth_x, prepend=smooth_x[0])
            smooth_dy = np.diff(smooth_y, prepend=smooth_y[0])

            speed = np.sqrt(raw_dx**2 + raw_dy**2) / np.where(
                np.isnan(safe_frame_delta), 1.0, np.maximum(safe_frame_delta, 1.0)
            )

            velocity_normal = (
                smooth_dx * self.normal_x
                + smooth_dy * self.normal_y
            ) / np.where(
                np.isnan(safe_frame_delta), 1.0, np.maximum(safe_frame_delta, 1.0)
            )

            velocity_tangent = (
                smooth_dx * self.tangent_x
                + smooth_dy * self.tangent_y
            ) / np.where(
                np.isnan(safe_frame_delta), 1.0, np.maximum(safe_frame_delta, 1.0)
            )

            signed_distance_raw = np.array(
                [self.signed_distance(float(px), float(py)) for px, py in zip(x, y)],
                dtype=np.float64,
            )
            signed_distance_smooth = np.array(
                [
                    self.signed_distance(float(px), float(py))
                    for px, py in zip(smooth_x, smooth_y)
                ],
                dtype=np.float64,
            )

            raw_side = np.array(
                [
                    self.side(float(px), float(py))
                    for px, py in zip(x, y)
                ],
                dtype=np.int8,
            )

            stable_side = self._stable_side_series(signed_distance_raw)

            # ----------------------------------------------------------
            # Hysteretic spatial zone.
            # ----------------------------------------------------------
            zones: list[str] = []
            previous_zone = "PRE"
            for distance in np.abs(signed_distance_raw):
                zone = self._zone_from_distance(
                    float(distance), previous_zone
                )
                zones.append(zone)
                previous_zone = zone

            # Determine whether a stable side transition already occurred.
            crossed_seen = False
            context: list[str] = []
            previous_stable_side = 0

            for side_value, zone_value in zip(stable_side, zones):
                current_side = int(side_value)
                if (
                    previous_stable_side != 0
                    and current_side != 0
                    and current_side != previous_stable_side
                ):
                    crossed_seen = True
                if current_side != 0:
                    previous_stable_side = current_side

                context.append(
                    self._zone_context(
                        current_side,
                        zone_value,
                        crossed_seen,
                    )
                )

            continuity = np.ones(len(group), dtype=np.float64)
            speed_anomaly = np.zeros(len(group), dtype=bool)
            gap_bridge = np.zeros(len(group), dtype=bool)
            quality = np.ones(len(group), dtype=np.float64)

            for i in range(1, len(group)):
                gap = int(frames_arr[i] - frames_arr[i - 1])
                continuity[i] = 1.0 if gap <= 1 else max(
                    0.0,
                    1.0
                    - (
                        (gap - 1)
                        / max(
                            self.config.max_trajectory_gap_sec * self.fps,
                            1.0,
                        )
                    ),
                )

                if (
                    gap > 1
                    and gap <= self.config.gap_bridge_max_frames
                    and self.config.gap_bridge_enabled
                ):
                    gap_bridge[i] = True

            speed_limit = max(self.config.max_velocity_px_per_frame, 1e-6)
            bridge_limit = max(self.config.max_velocity_bridge_px_per_frame, speed_limit)

            for i in range(len(group)):
                speed_value = float(speed[i])
                limit = bridge_limit if gap_bridge[i] else speed_limit

                if speed_value > limit:
                    speed_anomaly[i] = True
                    quality[i] *= max(
                        0.0,
                        min(1.0, limit / max(speed_value, 1e-6)),
                    )

                quality[i] *= float(0.75 + 0.25 * continuity[i])

            group["smooth_x"] = smooth_x
            group["smooth_y"] = smooth_y
            group["dx"] = raw_dx
            group["dy"] = raw_dy
            group["frame_delta"] = frame_delta
            group["time_delta_sec"] = time_delta
            group["speed_px_per_frame"] = speed
            group["velocity_normal_px_per_frame"] = velocity_normal
            group["velocity_tangent_px_per_frame"] = velocity_tangent
            group["raw_signed_distance_px"] = signed_distance_raw
            group["raw_line_distance_px"] = np.abs(signed_distance_raw)
            group["signed_distance_px"] = signed_distance_smooth
            group["line_distance_px"] = np.abs(signed_distance_smooth)
            group["raw_side"] = raw_side
            group["stable_side"] = stable_side
            group["zone"] = zones
            group["zone_context"] = context
            group["trajectory_continuity"] = continuity
            group["gap_bridge_candidate"] = gap_bridge
            group["speed_anomaly"] = speed_anomaly
            group["trajectory_quality"] = quality

            # Local direction is computed from normal/tangent velocity.
            group["direction_local"] = np.where(
                velocity_tangent > self.config.min_direction_displacement_px,
                "L→R",
                np.where(
                    velocity_tangent < -self.config.min_direction_displacement_px,
                    "R→L",
                    "UNKNOWN",
                ),
            )
            group["normal_direction_local"] = np.where(
                velocity_normal > self.config.min_normal_velocity_px_per_frame,
                self.config.positive_normal_label,
                np.where(
                    velocity_normal < -self.config.min_normal_velocity_px_per_frame,
                    self.config.negative_normal_label,
                    "UNKNOWN",
                ),
            )

            frames.append(group)

        result = pd.concat(frames, ignore_index=True)

        return result

    # ==================================================================
    # CROSSING DETECTION
    # ==================================================================

    def _gap_velocity_bridge(
        self,
        previous: pd.Series,
        current: pd.Series,
    ) -> tuple[bool, str]:
        """Detect crossing across a sparse observation gap."""
        if not self.config.gap_bridge_enabled:
            return False, ""

        frame_gap = int(current["frame_id"] - previous["frame_id"])
        if frame_gap <= 1 or frame_gap > self.config.gap_bridge_max_frames:
            return False, ""

        previous_distance = float(previous["raw_signed_distance_px"])
        current_distance = float(current["raw_signed_distance_px"])

        # The strongest sparse crossing evidence is a sign change.
        sign_change = (
            previous_distance != 0.0
            and current_distance != 0.0
            and np.sign(previous_distance) != np.sign(current_distance)
        )

        dx = float(current["raw_x"] - previous["raw_x"])
        dy = float(current["raw_y"] - previous["raw_y"])
        speed = math.hypot(dx, dy) / max(frame_gap, 1)

        fast_motion = speed >= (
            self.config.fast_speed_multiplier
            * self.config.max_velocity_px_per_frame
        )

        if sign_change and fast_motion:
            return True, "gap_velocity_bridge"

        # Even without high speed, a sparse sign transition is a valid bridge.
        if sign_change:
            return True, "gap_side_transition"

        return False, ""

    def _detect_crossing_candidates(
        self,
        group: pd.DataFrame,
    ) -> list[dict]:
        """Return every geometric crossing candidate; do not stop at first.

        Candidate types are based on geometry first, not PRE/POST eligibility.
        """
        candidates: list[dict] = []

        if len(group) < self.config.min_track_observations:
            return candidates

        max_gap_frames = max(
            1.0,
            self.config.max_trajectory_gap_sec * self.fps,
        )

        last_nonzero_index: int | None = None
        last_nonzero_side: int = 0

        for i in range(1, len(group)):
            previous = group.iloc[i - 1]
            current = group.iloc[i]

            previous_frame = int(previous["frame_id"])
            current_frame = int(current["frame_id"])
            frame_gap = current_frame - previous_frame

            if frame_gap <= 0 or frame_gap > max_gap_frames:
                continue

            previous_point = (
                float(previous["raw_x"]),
                float(previous["raw_y"]),
            )
            current_point = (
                float(current["raw_x"]),
                float(current["raw_y"]),
            )

            prev_distance = float(previous["raw_signed_distance_px"])
            curr_distance = float(current["raw_signed_distance_px"])

            prev_raw_side = int(previous["raw_side"])
            curr_raw_side = int(current["raw_side"])

            prev_stable_side = int(previous["stable_side"])
            curr_stable_side = int(current["stable_side"])

            # ----------------------------------------------------------
            # A. Raw segment intersection.
            # ----------------------------------------------------------
            line_intersection = self.trajectory_intersects_line(
                previous_point,
                current_point,
            )

            # ----------------------------------------------------------
            # B. Stable-side crossing. This catches + -> 0 -> -.
            # ----------------------------------------------------------
            stable_side_change = (
                prev_stable_side != 0
                and curr_stable_side != 0
                and prev_stable_side != curr_stable_side
            )

            # ----------------------------------------------------------
            # C. Deadband-aware signed-distance transition.
            # ----------------------------------------------------------
            deadband_aware_transition = False

            if last_nonzero_index is not None and i >= last_nonzero_index:
                if (
                    last_nonzero_side != 0
                    and curr_stable_side != 0
                    and last_nonzero_side != curr_stable_side
                ):
                    deadband_aware_transition = True

            if curr_raw_side != 0:
                last_nonzero_index = i
                last_nonzero_side = curr_raw_side

            # Another direct form: previous/current signed distances have
            # opposite signs, even if both are inside/outside a deadband.
            signed_distance_sign_change = (
                prev_distance != 0.0
                and curr_distance != 0.0
                and np.sign(prev_distance) != np.sign(curr_distance)
            )

            # ----------------------------------------------------------
            # D. Gap / velocity bridge.
            # ----------------------------------------------------------
            gap_bridge, gap_method = self._gap_velocity_bridge(
                previous,
                current,
            )

            crossing = bool(
                line_intersection
                or stable_side_change
                or deadband_aware_transition
                or signed_distance_sign_change
                or gap_bridge
            )

            if not crossing:
                continue

            method_parts: list[str] = []
            if line_intersection:
                method_parts.append("raw_segment_intersection")
            if stable_side_change:
                method_parts.append("stable_side_transition")
            if deadband_aware_transition:
                method_parts.append("deadband_aware_transition")
            if signed_distance_sign_change:
                method_parts.append("signed_distance_sign_change")
            if gap_bridge:
                method_parts.append(gap_method)

            min_distance = min(
                abs(prev_distance),
                abs(curr_distance),
            )

            corridor_touch = min_distance <= self.config.corridor_px
            previous_zone = str(previous["zone"])
            current_zone = str(current["zone"])
            zero_corridor = not corridor_touch

            speed_now = float(current["speed_px_per_frame"])
            speed_fast_threshold = (
                self.config.fast_speed_multiplier
                * self.config.max_velocity_px_per_frame
            )
            speed_fast = speed_now >= speed_fast_threshold

            sparse_crossing = (
                frame_gap > 1
                or zero_corridor
                or gap_bridge
            )

            fast_crossing = bool(
                sparse_crossing
                and (
                    speed_fast
                    or gap_bridge
                    or zero_corridor
                )
            )

            if zero_corridor and not self.config.allow_zero_corridor_crossing:
                # Geometry remains a candidate; do not discard it. Mark as
                # sparse so Phase 3 can decide later.
                method_parts.append("corridor_skipped")
            elif zero_corridor:
                method_parts.append("corridor_skipped")

            crossing_x, crossing_y = self.estimate_crossing_point(
                previous_point,
                current_point,
            )

            candidates.append(
                {
                    "index": i,
                    "crossing_frame": current_frame,
                    "crossing_time_sec": float(current["timestamp_sec"]),
                    "crossing_x": crossing_x,
                    "crossing_y": crossing_y,
                    "previous_side": prev_stable_side,
                    "current_side": curr_stable_side,
                    "side_transition": self._side_transition(
                        prev_stable_side,
                        curr_stable_side,
                    ),
                    "frame_gap": frame_gap,
                    "crossing_method": "+".join(dict.fromkeys(method_parts)),
                    "corridor_touch": bool(corridor_touch),
                    "fast_crossing": bool(fast_crossing),
                    "sparse_crossing": bool(sparse_crossing),
                    "gap_bridge_used": bool(gap_bridge),
                    "previous_zone": previous_zone,
                    "current_zone": current_zone,
                }
            )

        return candidates

    # ==================================================================
    # CROSSING EVIDENCE
    # ==================================================================

    def _zone_evidence(
        self,
        group: pd.DataFrame,
        crossing_index: int | None,
    ) -> dict:
        if group.empty:
            return {
                "pre_zone_observations": 0,
                "corridor_observations": 0,
                "post_zone_observations": 0,
                "pre_zone_evidence": False,
                "corridor_evidence": False,
                "post_zone_evidence": False,
                "zone_path": "UNKNOWN",
                "zone_chatter_count": 0,
                "normal_direction": "UNKNOWN",
                "direction_confidence": 0.0,
                "normal_displacement_px": 0.0,
                "corridor_confidence": 0.0,
            }

        zones = group["zone"].astype(str).tolist()

        # Use first geometric crossing position as the temporal split.
        split = crossing_index if crossing_index is not None else len(group)

        before = group.iloc[: split + 1]
        after = group.iloc[split:]

        pre_obs = int((before["zone"] == "PRE").sum())
        corridor_obs = int((group["zone"] == "CORRIDOR").sum())

        # POST evidence is derived from a stable-side change after the
        # crossing candidate. This avoids relying on the precomputed
        # zone_context, because the trajectory was not yet labelled as
        # crossed when zone_context was originally generated.
        if crossing_index is None or crossing_index <= 0:
            post_obs = 0
        else:
            pre_stable_values = group.iloc[:crossing_index]["stable_side"]
            stable_pre = pre_stable_values[pre_stable_values != 0]
            reference_side = int(stable_pre.iloc[-1]) if not stable_pre.empty else 0
            if reference_side == 0:
                post_obs = 0
            else:
                post_obs = int(
                    (group.iloc[crossing_index + 1 :]["stable_side"] != reference_side).sum()
                )

        pre_evidence = pre_obs >= self.config.min_pre_zone_observations
        corridor_evidence = corridor_obs >= self.config.min_corridor_observations
        post_evidence = post_obs >= self.config.min_post_zone_observations

        path, chatter = self._collapse_zone_path(zones)

        if crossing_index is None:
            normal_direction = "UNKNOWN"
            direction_confidence = 0.0
            normal_displacement = 0.0
        else:
            normal_direction, direction_confidence, normal_displacement = (
                self._infer_crossing_direction(group, crossing_index)
            )

        corridor_density = corridor_obs / max(len(group), 1)
        corridor_confidence = float(
            np.clip(
                0.40 * min(1.0, corridor_obs / max(self.config.min_corridor_observations, 1))
                + 0.30 * min(1.0, corridor_density * 10.0)
                + 0.30 * direction_confidence,
                0.0,
                1.0,
            )
        )

        return {
            "pre_zone_observations": pre_obs,
            "corridor_observations": corridor_obs,
            "post_zone_observations": post_obs,
            "pre_zone_evidence": bool(pre_evidence),
            "corridor_evidence": bool(corridor_evidence),
            "post_zone_evidence": bool(post_evidence),
            "zone_path": path,
            "zone_chatter_count": int(chatter),
            "normal_direction": normal_direction,
            "direction_confidence": float(direction_confidence),
            "normal_displacement_px": float(normal_displacement),
            "corridor_confidence": corridor_confidence,
        }

    # ==================================================================
    # PHASE STATUS
    # ==================================================================

    def _phase1_status(self, group: pd.DataFrame) -> tuple[str, str, bool]:
        if len(group) < self.config.min_track_observations:
            return "FAIL", "insufficient_track_observations", False

        finite_cols = [
            "raw_x",
            "raw_y",
            "speed_px_per_frame",
            "velocity_normal_px_per_frame",
            "trajectory_continuity",
        ]
        if group[finite_cols].isna().any().any():
            return "FAIL", "non_finite_trajectory_values", False

        mean_quality = float(group["trajectory_quality"].mean())
        anomaly_count = int(group["speed_anomaly"].sum())
        continuity_mean = float(group["trajectory_continuity"].mean())

        if anomaly_count > max(2, int(len(group) * 0.25)):
            return "REVIEW", "frequent_speed_anomalies", False

        if continuity_mean < 0.50:
            return "REVIEW", "poor_trajectory_continuity", False

        if mean_quality < 0.65:
            return "REVIEW", "low_trajectory_quality", False

        return "PASS", "", True

    def _phase2_status(
        self,
        crossing: dict | None,
        evidence: dict,
    ) -> tuple[str, str, bool, bool]:
        """Return status, reason, phase2_pass, count_eligibility.

        Important: geometry and evidence remain separate.
        A true geometric crossing can be REVIEW without being deleted.
        """
        if crossing is None:
            return "NOT_CROSSING", "no_geometric_crossing_detected", False, False

        reasons: list[str] = []

        if not evidence["pre_zone_evidence"]:
            reasons.append("insufficient_pre_zone_evidence")

        if not evidence["corridor_evidence"]:
            reasons.append("insufficient_corridor_evidence")

        if not evidence["post_zone_evidence"]:
            reasons.append("insufficient_post_zone_evidence")

        if (
            evidence["normal_direction"] == "UNKNOWN"
            or evidence["direction_confidence"] < self.config.min_direction_confidence
        ):
            reasons.append("low_crossing_direction_confidence")

        # Fast/sparse crossing is still a geometric crossing. Missing zone
        # evidence becomes REVIEW, never automatic rejection.
        if crossing["fast_crossing"]:
            reasons.append("fast_or_sparse_crossing")

        if reasons:
            return "REVIEW", ";".join(reasons), False, True

        return "PASS", "", True, True

    # ==================================================================
    # TRACK EVENT
    # ==================================================================

    def _build_event_for_track(
        self,
        group: pd.DataFrame,
        identity_id: int,
    ) -> dict:
        track_class = str(group.iloc[0].get("track_class", "unknown"))
        track_class_ratio = float(group.iloc[0].get("track_class_ratio", 1.0))
        class_ambiguous = bool(group.iloc[0].get("class_ambiguous", False))

        phase1_status, phase1_reason, phase1_pass = self._phase1_status(group)

        candidates = self._detect_crossing_candidates(group)

        # We preserve ALL geometric candidates for audit. For backward
        # compatibility with one-event-per-identity consumers, the primary
        # event is the first geometrically valid crossing.
        crossing = candidates[0] if candidates else None
        crossing_index = crossing["index"] if crossing else None

        evidence = self._zone_evidence(
            group,
            crossing_index,
        )

        phase2_status, phase2_reason, phase2_pass, count_eligibility = (
            self._phase2_status(crossing, evidence)
        )

        max_speed = float(group["speed_px_per_frame"].max())
        mean_speed = float(group["speed_px_per_frame"].mean())
        max_normal = float(group["velocity_normal_px_per_frame"].abs().max())
        mean_abs_normal = float(group["velocity_normal_px_per_frame"].abs().mean())
        mean_abs_tangent = float(group["velocity_tangent_px_per_frame"].abs().mean())
        trajectory_direction = self._infer_trajectory_direction(group)
        trajectory_quality = float(group["trajectory_quality"].mean())

        crossing_detected = crossing is not None
        candidate_class = (
            "TRUE_CROSSING"
            if crossing_detected and not crossing["fast_crossing"]
            else "FAST_CROSSING"
            if crossing_detected
            else "NEAR_LINE"
            if float(group["raw_line_distance_px"].min()) <= self.config.corridor_exit_px
            else "APPROACHING"
            if float(group["raw_line_distance_px"].min()) <= self.config.approach_distance_px
            else "NO_CROSSING"
        )

        # A track can be a TRUE_CROSSING even if evidence is incomplete.
        # This is exactly the candidate-preserving behavior needed before
        # State Machine.
        count_eligible = bool(
            crossing_detected
            and evidence["direction_confidence"] >= self.config.min_direction_confidence
        )

        # Current module deliberately does NOT perform final counting.
        # `counted` mirrors geometry eligibility only so downstream consumers
        # can inspect candidates without silently deleting them. Phase 3 will
        # own the final count decision.
        counted = count_eligible

        reasons: list[str] = []
        if phase1_reason:
            reasons.append(f"P1:{phase1_reason}")
        if phase2_reason:
            reasons.append(f"P2:{phase2_reason}")

        event = {
            "crossing_id": int(identity_id),
            "track_id": int(group.iloc[-1]["track_id"]),
            "track_ids": str(group["track_id"].drop_duplicates().tolist()),
            "first_frame": int(group["frame_id"].min()),
            "last_frame": int(group["frame_id"].max()),
            "crossing_frame": crossing["crossing_frame"] if crossing else pd.NA,
            "crossing_time_sec": crossing["crossing_time_sec"] if crossing else pd.NA,
            "crossing_x": crossing["crossing_x"] if crossing else pd.NA,
            "crossing_y": crossing["crossing_y"] if crossing else pd.NA,
            "direction": evidence["normal_direction"],
            "normal_direction": evidence["normal_direction"],
            "line_direction": trajectory_direction,
            "side_transition": crossing["side_transition"] if crossing else "UNKNOWN",
            "track_class": track_class,
            "track_class_ratio": track_class_ratio,
            "class_ambiguous": class_ambiguous,
            "line_distance_px": (
                min(
                    float(group.iloc[max(0, crossing_index - 1)]["raw_line_distance_px"]),
                    float(group.iloc[crossing_index]["raw_line_distance_px"]),
                )
                if crossing_index is not None
                else float(group["raw_line_distance_px"].min())
            ),
            "previous_side": crossing["previous_side"] if crossing else pd.NA,
            "current_side": crossing["current_side"] if crossing else pd.NA,
            "frame_gap": crossing["frame_gap"] if crossing else pd.NA,
            "crossing_method": crossing["crossing_method"] if crossing else "",
            "crossing_candidate_class": candidate_class,
            "fast_crossing": bool(crossing["fast_crossing"]) if crossing else False,
            "sparse_crossing": bool(crossing["sparse_crossing"]) if crossing else False,
            "gap_bridge_used": bool(crossing["gap_bridge_used"]) if crossing else False,
            "track_observations": int(len(group)),
            "crossing_index": crossing_index if crossing_index is not None else pd.NA,
            "pre_zone_observations": evidence["pre_zone_observations"],
            "corridor_observations": evidence["corridor_observations"],
            "post_zone_observations": evidence["post_zone_observations"],
            "pre_zone_evidence": evidence["pre_zone_evidence"],
            "corridor_evidence": evidence["corridor_evidence"],
            "post_zone_evidence": evidence["post_zone_evidence"],
            "zone_path": evidence["zone_path"],
            "zone_chatter_count": evidence["zone_chatter_count"],
            "trajectory_quality": round(trajectory_quality, 4),
            "direction_confidence": round(float(evidence["direction_confidence"]), 4),
            "normal_displacement_px": round(float(evidence["normal_displacement_px"]), 4),
            "corridor_confidence": round(float(evidence["corridor_confidence"]), 4),
            "phase1_status": phase1_status,
            "phase2_status": phase2_status,
            "count_eligibility": bool(count_eligibility),
            "counted": bool(counted),
            "_phase1_pass": bool(phase1_pass),
            "_phase2_pass": bool(phase2_pass),
            "_phase1_reason": phase1_reason,
            "_phase2_reason": phase2_reason,
            "_failure_reason": ";".join(reasons),
            "_max_speed": max_speed,
            "_mean_speed": mean_speed,
            "_max_normal": max_normal,
            "_mean_abs_normal": mean_abs_normal,
            "_mean_abs_tangent": mean_abs_tangent,
            "_trajectory_direction": trajectory_direction,
            "_candidates": candidates,
        }

        return event

    # ==================================================================
    # AUDIT
    # ==================================================================

    def _build_audit_row(
        self,
        group: pd.DataFrame,
        event: dict,
    ) -> dict:
        stable = group.loc[group["stable_side"] != 0, "stable_side"]
        first_side = int(stable.iloc[0]) if not stable.empty else 0
        last_side = int(stable.iloc[-1]) if not stable.empty else 0

        crossing_detected = bool(event.get("crossing_candidate_class") in {
            "TRUE_CROSSING",
            "FAST_CROSSING",
        })

        return {
            "crossing_id": int(event["crossing_id"]),
            "track_ids": event["track_ids"],
            "first_frame": int(group["frame_id"].min()),
            "last_frame": int(group["frame_id"].max()),
            "track_class": event["track_class"],
            "track_observations": int(len(group)),
            "first_side": first_side,
            "last_side": last_side,
            "min_distance_px": float(group["raw_line_distance_px"].min()),
            "max_speed_px_per_frame": float(event["_max_speed"]),
            "mean_speed_px_per_frame": float(event["_mean_speed"]),
            "max_abs_normal_velocity_px_per_frame": float(event["_max_normal"]),
            "mean_abs_normal_velocity_px_per_frame": float(event["_mean_abs_normal"]),
            "mean_abs_tangent_velocity_px_per_frame": float(event["_mean_abs_tangent"]),
            "trajectory_direction": event["_trajectory_direction"],
            "normal_direction": event["normal_direction"],
            "direction_confidence": float(event["direction_confidence"]),
            "normal_displacement_px": float(event["normal_displacement_px"]),
            "zone_path": event["zone_path"],
            "zone_chatter_count": int(event["zone_chatter_count"]),
            "pre_zone_observations": int(event["pre_zone_observations"]),
            "corridor_observations": int(event["corridor_observations"]),
            "post_zone_observations": int(event["post_zone_observations"]),
            "pre_zone_evidence": bool(event["pre_zone_evidence"]),
            "corridor_evidence": bool(event["corridor_evidence"]),
            "post_zone_evidence": bool(event["post_zone_evidence"]),
            "crossing_detected": crossing_detected,
            "crossing_candidate_class": event["crossing_candidate_class"],
            "fast_crossing": bool(event["fast_crossing"]),
            "sparse_crossing": bool(event["sparse_crossing"]),
            "gap_bridge_used": bool(event["gap_bridge_used"]),
            "crossing_frame": event["crossing_frame"],
            "frame_gap": event["frame_gap"],
            "crossing_method": event["crossing_method"],
            "crossing_direction": event["normal_direction"],
            "phase1_status": event["phase1_status"],
            "phase2_status": event["phase2_status"],
            "phase1_pass": bool(event["_phase1_pass"]),
            "phase2_pass": bool(event["_phase2_pass"]),
            "count_eligibility": bool(event["count_eligibility"]),
            "counted": bool(event["counted"]),
            "failure_reason": event["_failure_reason"],
        }

    # ==================================================================
    # BATCH
    # ==================================================================

    def process(
        self,
        trajectory: pd.DataFrame,
        identity_column: str = "crossing_id",
        return_diagnostics: bool = False,
    ):
        if identity_column not in trajectory.columns:
            raise ValueError(
                f"Trajectory missing identity column: {identity_column}"
            )

        if trajectory.empty:
            events = pd.DataFrame(columns=self.EVENT_COLUMNS)
            audits = pd.DataFrame(columns=self.AUDIT_COLUMNS)
            prepared = trajectory.copy()
            if return_diagnostics:
                return events, audits, prepared
            return events, audits

        prepared = self.prepare(trajectory)

        events: list[dict] = []
        audits: list[dict] = []

        for identity_id, group in prepared.groupby(identity_column, sort=False):
            group = (
                group.sort_values("frame_id")
                .reset_index(drop=True)
            )

            event = self._build_event_for_track(
                group,
                int(identity_id),
            )

            # Every identity gets an audit row. This fixes the previous
            # problem where no-crossing tracks vanished from the audit.
            audits.append(
                self._build_audit_row(
                    group,
                    event,
                )
            )

            # Only geometric crossing candidates become events. Non-crossing
            # and merely approaching tracks stay in the audit table and do not
            # become fake crossing events.
            if event["crossing_candidate_class"] in {
                "TRUE_CROSSING",
                "FAST_CROSSING",
            }:
                events.append(
                    {
                        key: event.get(key, pd.NA)
                        for key in self.EVENT_COLUMNS
                    }
                )

        events_df = pd.DataFrame(
            events,
            columns=self.EVENT_COLUMNS,
        )

        audit_df = pd.DataFrame(
            audits,
            columns=self.AUDIT_COLUMNS,
        )

        if return_diagnostics:
            return events_df, audit_df, prepared

        return events_df, audit_df

    # ==================================================================
    # REPORT
    # ==================================================================

    @staticmethod
    def print_phase_report(
        events_df: pd.DataFrame,
        audit_df: pd.DataFrame,
    ) -> None:
        print("\n" + "=" * 96)
        print("PHASE 1/2 TRAJECTORY + CROSSING CANDIDATE AUDIT v3")
        print("=" * 96)

        if audit_df.empty:
            print("No tracks available.")
            return

        total = len(audit_df)

        p1_pass = int((audit_df["phase1_status"] == "PASS").sum())
        p1_review = int((audit_df["phase1_status"] == "REVIEW").sum())
        p1_fail = int((audit_df["phase1_status"] == "FAIL").sum())

        candidate_counts = (
            audit_df["crossing_candidate_class"]
            .value_counts()
            .to_dict()
        )

        no_cross = int(candidate_counts.get("NO_CROSSING", 0))
        approaching = int(candidate_counts.get("APPROACHING", 0))
        near_line = int(candidate_counts.get("NEAR_LINE", 0))
        true_cross = int(candidate_counts.get("TRUE_CROSSING", 0))
        fast_cross = int(candidate_counts.get("FAST_CROSSING", 0))

        p2_pass = int((audit_df["phase2_status"] == "PASS").sum())
        p2_review = int((audit_df["phase2_status"] == "REVIEW").sum())
        p2_not_cross = int((audit_df["phase2_status"] == "NOT_CROSSING").sum())

        known_direction = int(
            audit_df["normal_direction"].ne("UNKNOWN").sum()
        )
        crossing_total = true_cross + fast_cross

        chatter = int(
            (pd.to_numeric(audit_df["zone_chatter_count"], errors="coerce") > 0).sum()
        )

        fast = audit_df[
            audit_df["crossing_candidate_class"] == "FAST_CROSSING"
        ]

        sparse = audit_df[
            audit_df["sparse_crossing"].astype(bool)
            & audit_df["crossing_detected"].astype(bool)
        ]

        print(f"Tracks analysed                    : {total:,}")
        print(f"NO_CROSSING tracks                  : {no_cross:,}")
        print(f"APPROACHING tracks                  : {approaching:,}")
        print(f"NEAR_LINE tracks                    : {near_line:,}")
        print(f"TRUE_CROSSING candidates            : {true_cross:,}")
        print(f"FAST_CROSSING candidates            : {fast_cross:,}")
        print()
        print(
            f"PHASE 1                            : "
            f"PASS={p1_pass:,} | REVIEW={p1_review:,} | FAIL={p1_fail:,}"
        )
        print(
            f"PHASE 2                            : "
            f"PASS={p2_pass:,} | REVIEW={p2_review:,} | "
            f"NOT_CROSSING={p2_not_cross:,}"
        )
        print(
            f"Known normal direction              : "
            f"{known_direction:,}/{crossing_total:,}"
        )
        print(f"Zone chatter tracks                 : {chatter:,}")
        print(f"Sparse crossing candidates           : {len(sparse):,}")
        print()

        if not fast.empty:
            print("FAST CROSSING DIAGNOSTIC:")
            print(
                "  Mean max speed (px/frame)         : "
                f"{fast['max_speed_px_per_frame'].mean():.2f}"
            )
            print(
                "  Mean crossing frame gap           : "
                f"{pd.to_numeric(fast['frame_gap'], errors='coerce').mean():.2f}"
            )
            print(
                "  Zero corridor observations        : "
                f"{int((fast['corridor_observations'] == 0).sum()):,}"
            )

        review = audit_df[
            audit_df["phase2_status"] == "REVIEW"
        ]

        if not review.empty:
            print("\nTOP REVIEW REASONS:")
            print(
                review["failure_reason"]
                .replace("", "NO_REASON")
                .value_counts()
                .head(10)
                .to_string()
            )

        print("\nCANDIDATE CLASS DISTRIBUTION:")
        print(
            audit_df["crossing_candidate_class"]
            .value_counts()
            .to_string()
        )

        print("\nCROSSING METHOD DISTRIBUTION:")
        crossing_rows = audit_df[
            audit_df["crossing_detected"].astype(bool)
        ]
        if crossing_rows.empty:
            print("No geometric crossings.")
        else:
            print(
                crossing_rows["crossing_method"]
                .fillna("")
                .value_counts()
                .head(15)
                .to_string()
            )

        print("=" * 96)


__all__ = [
    "CrossingConfig",
    "RobustCrossingEngine",
]
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

    # --------------------------------------------------------------
    # IDENTITY-GAP CROSSING
    # --------------------------------------------------------------
    # If the same physical identity is visible on side A, disappears,
    # and reappears on side B, this may be a valid crossing even when
    # there is no bbox observation inside the line corridor.
    identity_gap_crossing_enabled: bool = True
    identity_gap_max_frames: int = 8
    identity_gap_max_endpoint_distance_px: float = 140.0
    identity_gap_min_side_displacement_px: float = 12.0

    # --------------------------------------------------------------
    # TRAJECTORY-AWARE DUPLICATE IDENTITY AUDIT
    # --------------------------------------------------------------
    candidate_duplicate_max_frame_gap: int = 8
    candidate_duplicate_max_endpoint_distance_px: float = 55.0
    candidate_duplicate_max_crossing_distance_px: float = 55.0
    candidate_duplicate_min_direction_cosine: float = 0.75
    candidate_duplicate_require_non_overlapping_tracks: bool = True

    # Multi-crossing resolution. A raw track may produce several geometric
    # crossings because of zone chatter; only the strongest physical crossing
    # is canonical for that raw track, while all alternatives remain in audit.
    multi_crossing_max_candidates_per_track: int = 32
    multi_crossing_min_separation_frames: int = 2


    # Short tracks are diagnostic, not automatic rejection.
    short_track_observation_threshold: int = 8

    # --------------------------------------------------------------
    # CLASS EVIDENCE
    # --------------------------------------------------------------
    # Use raw detector classes near the crossing, rather than relying
    # only on the global majority `track_class`. This handles cases such
    # as person -> motorcycle during one physical track.
    class_evidence_window_frames: int = 8
    class_recency_decay: float = 0.18
    min_counting_class_confidence: float = 0.45

    # --------------------------------------------------------------
    # ZONE TEMPORAL HYSTERESIS
    # --------------------------------------------------------------
    # A zone transition must persist for N consecutive observations before
    # the stable zone changes. Geometry still uses raw coordinates; this
    # only stabilizes zone labels/evidence.
    zone_enter_confirm_observations: int = 2
    zone_exit_confirm_observations: int = 2

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
        "identity_id",
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
        "detector_track_class",
        "track_class_ratio",
        "class_ambiguous",
        "counting_class",
        "counting_class_confidence",
        "class_transition",
        "class_evidence",
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
        "short_track",
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
        "candidate_quality",
        "geometry_crossing",
        "identity_gap_side_transition",
        "identity_gap_frames",
        "identity_gap_identity_confirmed",
        "candidate_duplicate_of",
        "candidate_duplicate_confidence",
        "candidate_duplicate_reason",
        "candidate_duplicate_suppressed",
        "crossing_candidate_count",
        "multi_crossing_resolved",
        "alternative_crossing_count",
        "gap_count",
        "counted",
    ]

    AUDIT_COLUMNS = [
        "crossing_id",
        "identity_id",
        "track_ids",
        "first_frame",
        "last_frame",
        "track_class",
        "detector_track_class",
        "counting_class",
        "counting_class_confidence",
        "class_transition",
        "class_evidence",
        "track_observations",
        "short_track",
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
        "candidate_quality",
        "geometry_crossing",
        "gap_count",
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
        """Raw zone proposal using spatial hysteresis."""
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

    def _stable_zone_series(self, raw_signed_distance: np.ndarray) -> np.ndarray:
        """Temporally stabilize PRE/NEAR_LINE/CORRIDOR.

        The raw distance remains untouched for crossing geometry. A new zone
        only becomes the stable zone after it is observed consecutively for
        the configured confirmation count. This attacks PRE<->NEAR chatter
        without delaying line-intersection detection.
        """
        n = len(raw_signed_distance)
        if n == 0:
            return np.array([], dtype=object)

        stable: list[str] = []
        current = "PRE"
        pending = None
        pending_count = 0
        enter_need = max(1, int(self.config.zone_enter_confirm_observations))
        exit_need = max(1, int(self.config.zone_exit_confirm_observations))

        for distance in raw_signed_distance:
            proposal = self._zone_from_distance(float(distance), current)

            if proposal == current:
                pending = None
                pending_count = 0
                stable.append(current)
                continue

            if proposal != pending:
                pending = proposal
                pending_count = 1
            else:
                pending_count += 1

            # Entering CORRIDOR/NEAR_LINE is noisy, especially with small
            # motorcycles. Exiting is also stabilized, but use the same
            # explicit configurable minimum rather than a hard one-frame hop.
            need = enter_need if proposal in {"CORRIDOR", "NEAR_LINE"} else exit_need

            if pending_count >= need:
                current = proposal
                pending = None
                pending_count = 0

            stable.append(current)

        return np.asarray(stable, dtype=object)

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
        """Collapse stable zones and count only genuine backtracking chatter.

        A normal progression such as PRE -> NEAR_LINE -> CORRIDOR is NOT
        chatter. Chatter means the trajectory reverses between adjacent
        proximity zones, e.g. PRE -> NEAR_LINE -> PRE or CORRIDOR ->
        NEAR_LINE -> CORRIDOR.
        """
        cleaned: list[str] = []
        for zone in zones:
            if zone and (not cleaned or zone != cleaned[-1]):
                cleaned.append(zone)

        order = {
            "PRE": 0,
            "NEAR_LINE": 1,
            "CORRIDOR": 2,
        }

        chatter = 0
        previous_direction = 0
        for i in range(1, len(cleaned)):
            a = cleaned[i - 1]
            b = cleaned[i]
            if a not in order or b not in order:
                continue

            delta = order[b] - order[a]
            direction = 1 if delta > 0 else -1 if delta < 0 else 0

            if direction != 0 and previous_direction != 0 and direction != previous_direction:
                chatter += 1

            if direction != 0:
                previous_direction = direction

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
            zones = self._stable_zone_series(
                signed_distance_raw
            ).tolist()

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
        """Detect a crossing across a sparse observation gap.

        The strongest special case is the identity-gap side transition:
        the same physical identity was stably on side A, disappeared,
        then reappeared on side B.  This must be treated as crossing
        evidence even when no bbox exists inside the corridor.
        """
        if not self.config.gap_bridge_enabled:
            return False, ""

        frame_gap = int(current["frame_id"] - previous["frame_id"])
        if frame_gap <= 1:
            return False, ""

        max_gap = min(
            int(self.config.gap_bridge_max_frames),
            int(self.config.identity_gap_max_frames),
        )
        if frame_gap > max_gap:
            return False, ""

        prev_d = float(previous["raw_signed_distance_px"])
        curr_d = float(current["raw_signed_distance_px"])
        prev_side = int(previous.get("stable_side", previous.get("raw_side", 0)))
        curr_side = int(current.get("stable_side", current.get("raw_side", 0)))

        dx = float(current["raw_x"] - previous["raw_x"])
        dy = float(current["raw_y"] - previous["raw_y"])
        endpoint_distance = math.hypot(dx, dy)
        speed = endpoint_distance / max(frame_gap, 1)

        sign_change = (
            prev_d != 0.0
            and curr_d != 0.0
            and np.sign(prev_d) != np.sign(curr_d)
        )

        explicit_side_change = (
            prev_side != 0
            and curr_side != 0
            and prev_side != curr_side
        )

        near_boundary = (
            min(abs(prev_d), abs(curr_d))
            <= max(
                self.config.approach_distance_px,
                self.config.corridor_px * 3.0,
            )
        )

        identity_gap_side = bool(
            self.config.identity_gap_crossing_enabled
            and explicit_side_change
            and near_boundary
            and endpoint_distance <= self.config.identity_gap_max_endpoint_distance_px
            and abs(prev_d - curr_d) >= self.config.identity_gap_min_side_displacement_px
        )

        if identity_gap_side:
            return True, "identity_gap_side_transition"

        velocity_bridge = bool(
            sign_change
            and (
                speed >= self.config.fast_speed_multiplier * self.config.max_velocity_px_per_frame
                or speed >= self.config.min_normal_velocity_px_per_frame
            )
            and endpoint_distance <= self.config.identity_gap_max_endpoint_distance_px
        )

        if velocity_bridge:
            return True, "gap_velocity_bridge"

        if explicit_side_change and near_boundary:
            return True, "gap_side_transition"

        if sign_change:
            return True, "gap_side_sign_change"

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
            min(
                self.config.max_trajectory_gap_sec * self.fps,
                max(
                    self.config.gap_bridge_max_frames,
                    self.config.max_trajectory_gap_sec * self.fps,
                ),
            ),
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
            # An identity gap is a first-class crossing signal whenever the
            # same physical identity has a valid non-zero side before the gap
            # and the opposite side after the gap. This remains true even if
            # the raw segment itself also intersects the line.
            endpoint_distance = math.hypot(
                current_point[0] - previous_point[0],
                current_point[1] - previous_point[1],
            )
            identity_gap_side = bool(
                self.config.identity_gap_crossing_enabled
                and frame_gap > 1
                and (
                    stable_side_change
                    or deadband_aware_transition
                    or (
                        prev_raw_side != 0
                        and curr_raw_side != 0
                        and prev_raw_side != curr_raw_side
                    )
                )
                and endpoint_distance <= self.config.identity_gap_max_endpoint_distance_px
                and min(abs(prev_distance), abs(curr_distance))
                <= max(self.config.approach_distance_px, self.config.corridor_px * 3.0)
            )
            if identity_gap_side:
                method_parts.append("identity_gap_side_transition")
            elif gap_bridge:
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
                    "identity_gap_side_transition": bool(identity_gap_side),
                    "identity_gap_frames": int(frame_gap) if identity_gap_side else 0,
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
    # CLASS EVIDENCE
    # ==================================================================

    def _class_evidence(
        self,
        group: pd.DataFrame,
        crossing_index: int | None,
    ) -> dict:
        """Estimate the physical object's class from raw class evidence.

        The global `track_class` is retained as a diagnostic, but the class
        used downstream for counting is learned from detector observations
        near the crossing. This avoids losing a motorcycle whose early
        frames were classified as person.
        """
        if group.empty:
            return {
                "counting_class": "unknown",
                "counting_class_confidence": 0.0,
                "class_transition": "",
                "class_evidence": "",
            }

        raw = group.copy()
        raw["class_name"] = raw.get("class_name", raw.get("track_class", "unknown"))
        raw["class_name"] = raw["class_name"].astype(str).str.lower().str.strip()
        raw["confidence"] = pd.to_numeric(
            raw.get("confidence", 1.0), errors="coerce"
        ).fillna(1.0).clip(0.0, 1.0)
        raw["frame_id"] = pd.to_numeric(raw["frame_id"], errors="coerce")

        if crossing_index is None:
            end_frame = int(raw["frame_id"].max())
        else:
            end_frame = int(raw.iloc[crossing_index]["frame_id"])

        window = max(1, int(self.config.class_evidence_window_frames))
        selected = raw[
            raw["frame_id"] >= (end_frame - window)
        ].copy()

        if selected.empty:
            selected = raw.tail(min(len(raw), window)).copy()

        decay = max(1e-6, float(self.config.class_recency_decay))
        age = (end_frame - selected["frame_id"]).clip(lower=0)
        selected["recency_weight"] = np.exp(-decay * age.astype(float))
        selected["evidence_weight"] = (
            selected["confidence"] * selected["recency_weight"]
        )

        scores = (
            selected.groupby("class_name")["evidence_weight"]
            .sum()
            .sort_values(ascending=False)
        )

        if scores.empty or float(scores.sum()) <= 0.0:
            return {
                "counting_class": "unknown",
                "counting_class_confidence": 0.0,
                "class_transition": "",
                "class_evidence": "",
            }

        top_class = str(scores.index[0])
        total_score = float(scores.sum())
        top_score = float(scores.iloc[0])
        top_conf = top_score / total_score

        # Evidence string is intentionally compact for the audit CSV.
        evidence_parts = [
            f"{cls}:{float(score):.3f}"
            for cls, score in scores.items()
        ]
        evidence_text = " | ".join(evidence_parts)

        ordered_classes = raw["class_name"].tolist()
        transitions: list[str] = []
        for prev, curr in zip(ordered_classes, ordered_classes[1:]):
            if prev != curr:
                pair = f"{prev}->{curr}"
                if pair not in transitions:
                    transitions.append(pair)

        transition_text = " | ".join(transitions)

        return {
            "counting_class": top_class,
            "counting_class_confidence": float(top_conf),
            "class_transition": transition_text,
            "class_evidence": evidence_text,
        }

    # ==================================================================
    # PHASE STATUS
    # ==================================================================

    def _phase1_status(self, group: pd.DataFrame) -> tuple[str, str, bool]:
        """Assess trajectory usability without equating short tracks to failure."""
        if group.empty:
            return "FAIL", "empty_trajectory", False

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
        observations = len(group)

        reasons: list[str] = []

        # Short track is not automatic invalidation. Keep it visible as
        # REVIEW, and let geometric crossing evidence rescue it downstream.
        if observations < self.config.short_track_observation_threshold:
            reasons.append("short_track")

        if anomaly_count > max(2, int(observations * 0.35)):
            reasons.append("frequent_speed_anomalies")

        if continuity_mean < 0.50:
            reasons.append("poor_trajectory_continuity")

        if mean_quality < 0.50:
            reasons.append("low_trajectory_quality")

        if reasons:
            return "REVIEW", ";".join(reasons), False

        return "PASS", "", True

    def _phase2_status(
        self,
        crossing: dict | None,
        evidence: dict,
        class_evidence: dict,
        phase1_status: str,
        phase1_reason: str,
        track_observations: int,
    ) -> tuple[str, str, bool, bool, float]:
        """Canonical candidate status.

        Geometry creates the candidate. PRE/CORRIDOR/POST are evidence only.
        A candidate is count-eligible when geometry + direction + class are
        usable. Short/fast tracks do not fail solely because their evidence is
        sparse.
        """
        if crossing is None:
            return "NOT_CROSSING", "no_geometric_crossing_detected", False, False, 0.0

        reasons: list[str] = []
        if not evidence["pre_zone_evidence"]:
            reasons.append("insufficient_pre_zone_evidence")
        if not evidence["corridor_evidence"]:
            reasons.append("insufficient_corridor_evidence")
        if not evidence["post_zone_evidence"]:
            reasons.append("insufficient_post_zone_evidence")
        if evidence["normal_direction"] == "UNKNOWN" or evidence["direction_confidence"] < self.config.min_direction_confidence:
            reasons.append("low_crossing_direction_confidence")
        if crossing["fast_crossing"]:
            reasons.append("fast_or_sparse_crossing")
        if track_observations < self.config.short_track_observation_threshold:
            reasons.append("short_track_crossing")
        if class_evidence["counting_class"] == "unknown":
            reasons.append("unknown_counting_class")
        elif class_evidence["counting_class_confidence"] < self.config.min_counting_class_confidence:
            reasons.append("low_counting_class_confidence")
        if phase1_status == "REVIEW" and phase1_reason:
            reasons.append(f"trajectory_review:{phase1_reason}")

        direction_score = float(evidence["direction_confidence"])
        class_score = float(class_evidence["counting_class_confidence"])
        continuity_score = float(group_quality := 0.0)  # replaced by caller via candidate_quality

        geometry_score = 1.0
        zone_score = float(np.clip(
            0.45 * float(evidence["pre_zone_evidence"]) +
            0.25 * float(evidence["corridor_evidence"]) +
            0.30 * float(evidence["post_zone_evidence"]),
            0.0, 1.0
        ))
        # Geometry + direction + class dominate. Zone evidence is intentionally secondary.
        candidate_quality = float(np.clip(
            0.40 * geometry_score +
            0.25 * direction_score +
            0.25 * class_score +
            0.10 * zone_score,
            0.0, 1.0,
        ))

        count_eligible = bool(
            evidence["normal_direction"] != "UNKNOWN"
            and evidence["direction_confidence"] >= self.config.min_direction_confidence
            and class_evidence["counting_class"] != "unknown"
            and class_evidence["counting_class_confidence"] >= self.config.min_counting_class_confidence
        )

        # PASS now means the canonical candidate is usable by the Counter.
        # Zone evidence remains visible in review_reason but cannot suppress
        # a real geometric crossing.
        status = "PASS" if count_eligible else "REVIEW"
        return status, ";".join(dict.fromkeys(reasons)), count_eligible, count_eligible, candidate_quality

    # ==================================================================
    # TRACK EVENT
    # ==================================================================

    def _resolve_track_crossing(self, candidates: list[dict]) -> tuple[dict | None, list[dict]]:
        """Resolve multiple geometric crossings within one raw track.

        This is intentionally PER TRACK. It never compares candidates from
        different raw tracks, so simultaneous vehicles are preserved.
        """
        if not candidates:
            return None, []
        candidates = list(candidates[: max(1, int(self.config.multi_crossing_max_candidates_per_track))])

        def score(c: dict) -> float:
            method = str(c.get("crossing_method", ""))
            stable = 1.0 if "stable_side_transition" in method else 0.0
            intersection = 1.0 if "raw_segment_intersection" in method else 0.0
            deadband = 1.0 if "deadband_aware_transition" in method else 0.0
            identity_gap = 1.0 if "identity_gap_side_transition" in method else 0.0
            gap = int(c.get("frame_gap", 1))
            return (
                0.35 * stable
                + 0.25 * intersection
                + 0.15 * deadband
                + 0.10 * identity_gap
                + 0.15 * (1.0 / max(gap, 1))
            )

        ranked = sorted(candidates, key=lambda c: (score(c), -int(c.get("crossing_frame", 0))), reverse=True)
        selected = ranked[0]
        alternatives = [c for c in candidates if c is not selected]
        return selected, alternatives

    def _build_event_for_track(
        self,
        group: pd.DataFrame,
        candidate_id: int,
        identity_id: int,
    ) -> dict:
        detector_track_class = str(group.iloc[0].get("track_class", "unknown"))
        track_class_ratio = float(group.iloc[0].get("track_class_ratio", 1.0))
        class_ambiguous = bool(group.iloc[0].get("class_ambiguous", False))

        phase1_status, phase1_reason, phase1_pass = self._phase1_status(group)
        all_crossing_candidates = self._detect_crossing_candidates(group)
        crossing, alternative_crossings = self._resolve_track_crossing(all_crossing_candidates)
        crossing_index = crossing["index"] if crossing else None

        evidence = self._zone_evidence(group, crossing_index)
        class_evidence = self._class_evidence(group, crossing_index)

        phase2_status, phase2_reason, phase2_pass, count_eligibility, candidate_quality = (
            self._phase2_status(
                crossing,
                evidence,
                class_evidence,
                phase1_status,
                phase1_reason,
                len(group),
            )
        )

        max_speed = float(group["speed_px_per_frame"].max())
        mean_speed = float(group["speed_px_per_frame"].mean())
        max_normal = float(group["velocity_normal_px_per_frame"].abs().max())
        mean_abs_normal = float(group["velocity_normal_px_per_frame"].abs().mean())
        mean_abs_tangent = float(group["velocity_tangent_px_per_frame"].abs().mean())
        trajectory_direction = self._infer_trajectory_direction(group)
        trajectory_quality = float(group["trajectory_quality"].mean())
        short_track = len(group) < self.config.short_track_observation_threshold

        candidate_class = (
            "TRUE_CROSSING"
            if crossing is not None and not crossing["fast_crossing"]
            else "FAST_CROSSING"
            if crossing is not None
            else "NEAR_LINE"
            if float(group["raw_line_distance_px"].min()) <= self.config.corridor_exit_px
            else "APPROACHING"
            if float(group["raw_line_distance_px"].min()) <= self.config.approach_distance_px
            else "NO_CROSSING"
        )

        reasons: list[str] = []
        if phase1_reason:
            reasons.append(f"P1:{phase1_reason}")
        if phase2_reason:
            reasons.append(f"P2:{phase2_reason}")

        return {
            "crossing_id": int(candidate_id),
            "identity_id": int(identity_id),
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
            # IMPORTANT: expose counting_class through track_class so the
            # existing counter remains API-compatible, while preserving the
            # original detector-level track class in a separate field.
            "track_class": class_evidence["counting_class"],
            "detector_track_class": detector_track_class,
            "track_class_ratio": track_class_ratio,
            "class_ambiguous": class_ambiguous,
            "counting_class": class_evidence["counting_class"],
            "counting_class_confidence": float(class_evidence["counting_class_confidence"]),
            "class_transition": class_evidence["class_transition"],
            "class_evidence": class_evidence["class_evidence"],
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
            "short_track": bool(short_track),
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
            "candidate_quality": round(float(candidate_quality), 4),
            "geometry_crossing": bool(crossing is not None),
            "identity_gap_side_transition": bool(crossing.get("identity_gap_side_transition", False)) if crossing else False,
            "identity_gap_frames": int(crossing.get("identity_gap_frames", 0)) if crossing else 0,
            "identity_gap_identity_confirmed": bool(crossing.get("identity_gap_identity_confirmed", False)) if crossing else False,
            "candidate_duplicate_of": pd.NA,
            "candidate_duplicate_confidence": 0.0,
            "candidate_duplicate_reason": "",
            "candidate_duplicate_suppressed": False,
            "gap_count": int((group["frame_delta"] > 1).sum()),
            "counted": bool(count_eligibility),
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
            "_candidates": all_crossing_candidates,
            "crossing_candidate_count": int(len(all_crossing_candidates)),
            "multi_crossing_resolved": bool(len(all_crossing_candidates) > 1),
            "alternative_crossing_count": int(len(alternative_crossings)),
            "alternative_crossings": alternative_crossings,
        }

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
            "identity_id": int(event.get("identity_id", event["crossing_id"])),
            "track_ids": event["track_ids"],
            "first_frame": int(group["frame_id"].min()),
            "last_frame": int(group["frame_id"].max()),
            "track_class": event["track_class"],
            "detector_track_class": event["detector_track_class"],
            "counting_class": event["counting_class"],
            "counting_class_confidence": float(event["counting_class_confidence"]),
            "class_transition": event["class_transition"],
            "class_evidence": event["class_evidence"],
            "track_observations": int(len(group)),
            "crossing_candidate_count": int(event.get("crossing_candidate_count", 0)),
            "multi_crossing_resolved": bool(event.get("multi_crossing_resolved", False)),
            "alternative_crossing_count": int(event.get("alternative_crossing_count", 0)),
            "short_track": bool(event["short_track"]),
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
            "identity_gap_side_transition": bool(event.get("identity_gap_side_transition", False)),
            "identity_gap_frames": int(event.get("identity_gap_frames", 0)),
            "candidate_duplicate_of": event.get("candidate_duplicate_of", pd.NA),
            "candidate_duplicate_confidence": float(event.get("candidate_duplicate_confidence", 0.0)),
            "candidate_duplicate_reason": event.get("candidate_duplicate_reason", ""),
            "candidate_duplicate_suppressed": bool(event.get("candidate_duplicate_suppressed", False)),
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
    # TRAJECTORY-AWARE DUPLICATE IDENTITY AUDIT
    # ==================================================================

    @staticmethod
    def _direction_vector(label: str) -> np.ndarray:
        if label == "L→R":
            return np.array([1.0, 0.0], dtype=float)
        if label == "R→L":
            return np.array([-1.0, 0.0], dtype=float)
        return np.zeros(2, dtype=float)

    def _event_track_id_list(self, event: pd.Series) -> list[int]:
        """Return raw tracker track ids represented by a candidate."""
        raw = event.get("track_ids", "")
        if isinstance(raw, (list, tuple, set)):
            return [int(v) for v in raw]
        text = str(raw).strip()
        if not text:
            tid = event.get("track_id")
            return [int(tid)] if pd.notna(tid) else []
        try:
            parsed = eval(text, {"__builtins__": {}}, {})
            if isinstance(parsed, (list, tuple, set)):
                return [int(v) for v in parsed]
        except Exception:
            pass
        tid = event.get("track_id")
        return [int(tid)] if pd.notna(tid) else []

    def _candidate_rows(self, event: pd.Series, prepared: pd.DataFrame) -> pd.DataFrame:
        track_ids = self._event_track_id_list(event)
        if not track_ids:
            return prepared.iloc[0:0].copy()
        return prepared[
            prepared["track_id"].isin(track_ids)
        ].sort_values("frame_id")

    def _candidate_duplicate_score(
        self,
        a: pd.Series,
        b: pd.Series,
        prepared: pd.DataFrame,
    ) -> tuple[bool, float, str]:
        """Detect a *same-physical-object* duplicate without same-frame suppression.

        The crucial v8 rule is that crossing candidates are generated per raw
        tracker track first. Therefore two vehicles crossing simultaneously
        remain separate candidates even when the identity engine associated
        their fragments to the same physical identity by mistake.
        """
        if int(a["crossing_id"]) == int(b["crossing_id"]):
            return False, 0.0, "same_candidate"

        fa = int(a["crossing_frame"])
        fb = int(b["crossing_frame"])
        if fa == fb:
            a_tracks = set(self._event_track_id_list(a))
            b_tracks = set(self._event_track_id_list(b))
            if a_tracks and b_tracks and a_tracks.intersection(b_tracks):
                return True, 1.0, "same_frame_same_raw_track_duplicate"
            return False, 0.0, "same_frame_independent_candidate"

        if fa < fb:
            earlier, later = a, b
        else:
            earlier, later = b, a

        frame_gap = int(later["crossing_frame"] - earlier["crossing_frame"])
        if frame_gap <= 0 or frame_gap > self.config.candidate_duplicate_max_frame_gap:
            return False, 0.0, "candidate_time_gap_outside_window"

        class_a = str(a.get("counting_class", a.get("track_class", "unknown"))).lower().strip()
        class_b = str(b.get("counting_class", b.get("track_class", "unknown"))).lower().strip()
        if class_a != class_b:
            return False, 0.0, "class_mismatch"

        dir_a = str(a.get("direction", "UNKNOWN"))
        dir_b = str(b.get("direction", "UNKNOWN"))
        if dir_a == "UNKNOWN" or dir_b == "UNKNOWN" or dir_a != dir_b:
            return False, 0.0, "direction_mismatch"

        early_rows = self._candidate_rows(earlier, prepared)
        late_rows = self._candidate_rows(later, prepared)
        if early_rows.empty or late_rows.empty:
            return False, 0.0, "missing_raw_track_trajectory"

        early_last = early_rows.iloc[-1]
        late_first = late_rows.iloc[0]
        temporal_gap = int(late_first["frame_id"] - early_last["frame_id"])

        # Simultaneous/overlapping raw tracks are independent vehicles. Never
        # suppress one based on proximity alone.
        if temporal_gap <= 0:
            return False, 0.0, "overlapping_raw_tracks_independent"

        endpoint_distance = math.hypot(
            float(late_first["raw_x"]) - float(early_last["raw_x"]),
            float(late_first["raw_y"]) - float(early_last["raw_y"]),
        )
        if endpoint_distance > self.config.candidate_duplicate_max_endpoint_distance_px:
            return False, 0.0, "endpoint_distance_too_large"

        crossing_distance = math.hypot(
            float(a["crossing_x"]) - float(b["crossing_x"]),
            float(a["crossing_y"]) - float(b["crossing_y"]),
        )
        if crossing_distance > self.config.candidate_duplicate_max_crossing_distance_px:
            return False, 0.0, "crossing_distance_too_large"

        va = np.array([
            float(early_last.get("dx", 0.0)),
            float(early_last.get("dy", 0.0)),
        ], dtype=float)
        vb = np.array([
            float(late_first.get("dx", 0.0)),
            float(late_first.get("dy", 0.0)),
        ], dtype=float)
        na = float(np.linalg.norm(va))
        nb = float(np.linalg.norm(vb))
        cosine = 1.0
        if na > 1e-6 and nb > 1e-6:
            cosine = float(np.dot(va, vb) / (na * nb))
        if cosine < self.config.candidate_duplicate_min_direction_cosine:
            return False, 0.0, "trajectory_direction_incompatible"

        # If both candidates explicitly belong to the same physical identity,
        # a sequential raw-track split is strong duplicate evidence. If they
        # have different identities, require the stricter geometric signature.
        same_identity = int(a.get("identity_id", -1)) == int(b.get("identity_id", -2))

        identity_gap_score = max(
            0.0,
            1.0 - min(temporal_gap, 8) / 8.0,
        )
        frame_score = max(
            0.0,
            1.0 - frame_gap / max(self.config.candidate_duplicate_max_frame_gap, 1),
        )
        endpoint_score = max(
            0.0,
            1.0 - endpoint_distance / max(self.config.candidate_duplicate_max_endpoint_distance_px, 1e-6),
        )
        crossing_score = max(
            0.0,
            1.0 - crossing_distance / max(self.config.candidate_duplicate_max_crossing_distance_px, 1e-6),
        )
        direction_score = max(0.0, min(1.0, (cosine + 1.0) / 2.0))

        score = float(
            0.25 * identity_gap_score
            + 0.15 * frame_score
            + 0.25 * endpoint_score
            + 0.25 * crossing_score
            + 0.10 * direction_score
        )

        hard_signature = bool(
            temporal_gap <= 2
            and frame_gap <= 5
            and endpoint_distance <= self.config.candidate_duplicate_max_endpoint_distance_px
            and crossing_distance <= self.config.candidate_duplicate_max_crossing_distance_px
            and cosine >= self.config.candidate_duplicate_min_direction_cosine
        )

        # Different physical identities require an extra-high-confidence
        # signature. This prevents two close same-direction vehicles from
        # being merged merely because their crossings are near one another.
        if same_identity and hard_signature:
            return True, max(score, 0.85), (
                f"same_identity_sequential_duplicate;identity_id={int(a.get('identity_id', -1))};"
                f"frame_gap={frame_gap};raw_gap={temporal_gap};"
                f"endpoint_distance={endpoint_distance:.1f};crossing_distance={crossing_distance:.1f};"
                f"direction_cosine={cosine:.3f}"
            )

        if (not same_identity) and hard_signature and score >= 0.86:
            return True, float(score), (
                f"trajectory_duplicate_identity;different_identity;"
                f"frame_gap={frame_gap};raw_gap={temporal_gap};"
                f"endpoint_distance={endpoint_distance:.1f};crossing_distance={crossing_distance:.1f};"
                f"direction_cosine={cosine:.3f}"
            )

        return False, score, "duplicate_signature_not_strong_enough"

    def _apply_duplicate_identity_audit(
        self,
        events: pd.DataFrame,
        prepared: pd.DataFrame,
    ) -> pd.DataFrame:
        if events.empty or len(events) < 2:
            events = events.copy()
            if not events.empty:
                events["candidate_duplicate_of"] = pd.NA
                events["candidate_duplicate_confidence"] = 0.0
                events["candidate_duplicate_reason"] = ""
                events["candidate_duplicate_suppressed"] = False
            return events

        events = events.copy().reset_index(drop=True)
        events["candidate_duplicate_of"] = pd.NA
        events["candidate_duplicate_confidence"] = 0.0
        events["candidate_duplicate_reason"] = ""
        events["candidate_duplicate_suppressed"] = False

        # Process chronological candidates. Earlier candidate wins only when
        # the later candidate is strongly explained as its continuation.
        order = events.sort_values(["crossing_frame", "crossing_id"]).index.tolist()
        for pos, i in enumerate(order):
            if bool(events.loc[i, "candidate_duplicate_suppressed"]):
                continue
            for j in order[pos + 1 :]:
                if bool(events.loc[j, "candidate_duplicate_suppressed"]):
                    continue
                is_dup, score, reason = self._candidate_duplicate_score(
                    events.loc[i], events.loc[j], prepared
                )
                if not is_dup:
                    continue
                events.loc[j, "candidate_duplicate_of"] = int(events.loc[i, "crossing_id"])
                events.loc[j, "candidate_duplicate_confidence"] = float(score)
                events.loc[j, "candidate_duplicate_reason"] = reason
                events.loc[j, "candidate_duplicate_suppressed"] = True
                events.loc[j, "count_eligibility"] = False
                events.loc[j, "counted"] = False
                events.loc[j, "phase2_status"] = "REVIEW"
                existing = str(events.loc[j, "crossing_method"])
                events.loc[j, "crossing_method"] = (
                    existing + "+trajectory_duplicate_identity"
                    if existing else "trajectory_duplicate_identity"
                )

        return events

    def _build_identity_gap_candidate(
        self,
        identity_id: int,
        early: pd.DataFrame,
        late: pd.DataFrame,
        candidate_id: int,
    ) -> dict | None:
        """Create a candidate from side-A -> gap -> side-B for one identity."""
        early = early.sort_values("frame_id").reset_index(drop=True)
        late = late.sort_values("frame_id").reset_index(drop=True)
        if early.empty or late.empty:
            return None

        prev = early.iloc[-1]
        curr = late.iloc[0]
        gap = int(curr["frame_id"] - prev["frame_id"])
        if gap <= 1 or gap > int(self.config.identity_gap_max_frames):
            return None

        prev_side = int(prev.get("stable_side", prev.get("raw_side", 0)))
        curr_side = int(curr.get("stable_side", curr.get("raw_side", 0)))
        if prev_side == 0 or curr_side == 0 or prev_side == curr_side:
            return None

        endpoint_distance = math.hypot(
            float(curr["raw_x"] - prev["raw_x"]),
            float(curr["raw_y"] - prev["raw_y"]),
        )
        if endpoint_distance > self.config.identity_gap_max_endpoint_distance_px:
            return None

        prev_distance = abs(float(prev["raw_signed_distance_px"]))
        curr_distance = abs(float(curr["raw_signed_distance_px"]))
        if min(prev_distance, curr_distance) > max(
            self.config.approach_distance_px,
            self.config.corridor_px * 3.0,
        ):
            return None

        normal_displacement = abs(
            float(curr["raw_signed_distance_px"])
            - float(prev["raw_signed_distance_px"])
        )
        if normal_displacement < self.config.identity_gap_min_side_displacement_px:
            return None

        # Build a compact synthetic trajectory around the gap so class and
        # zone evidence remain available in the canonical candidate.
        combined = pd.concat(
            [early.tail(6), late.head(6)],
            ignore_index=True,
        ).drop_duplicates(
            subset=["frame_id", "track_id"],
            keep="first",
        ).sort_values("frame_id").reset_index(drop=True)
        crossing_index = max(0, len(early.tail(6)) - 1)

        evidence = self._zone_evidence(combined, crossing_index)
        class_evidence = self._class_evidence(combined, crossing_index)
        phase1_status, phase1_reason, phase1_pass = self._phase1_status(combined)

        direction = self.config.positive_normal_label if prev_side < curr_side else self.config.negative_normal_label
        direction_confidence = 1.0
        candidate_quality = float(np.clip(
            0.45
            + 0.20 * min(1.0, normal_displacement / max(self.config.identity_gap_min_side_displacement_px, 1.0))
            + 0.15 * min(1.0, endpoint_distance / max(self.config.identity_gap_max_endpoint_distance_px, 1.0)) ** -1
            + 0.20 * class_evidence["counting_class_confidence"],
            0.0,
            1.0,
        ))

        track_ids = [
            int(early["track_id"].iloc[0]),
            int(late["track_id"].iloc[0]),
        ]
        crossing_x, crossing_y = self.estimate_crossing_point(
            (float(prev["raw_x"]), float(prev["raw_y"])),
            (float(curr["raw_x"]), float(curr["raw_y"])),
        )

        return {
            "crossing_id": int(candidate_id),
            "identity_id": int(identity_id),
            "track_id": int(curr["track_id"]),
            "track_ids": str(track_ids),
            "first_frame": int(early["frame_id"].min()),
            "last_frame": int(late["frame_id"].max()),
            "crossing_frame": int(curr["frame_id"]),
            "crossing_time_sec": float(curr["timestamp_sec"]),
            "crossing_x": float(crossing_x),
            "crossing_y": float(crossing_y),
            "direction": direction,
            "normal_direction": direction,
            "line_direction": direction,
            "side_transition": self._side_transition(prev_side, curr_side),
            "track_class": class_evidence["counting_class"],
            "detector_track_class": str(curr.get("track_class", "unknown")),
            "track_class_ratio": float(curr.get("track_class_ratio", 1.0)),
            "class_ambiguous": bool(curr.get("class_ambiguous", False)),
            "counting_class": class_evidence["counting_class"],
            "counting_class_confidence": float(class_evidence["counting_class_confidence"]),
            "class_transition": class_evidence["class_transition"],
            "class_evidence": class_evidence["class_evidence"],
            "line_distance_px": float(min(prev_distance, curr_distance)),
            "previous_side": int(prev_side),
            "current_side": int(curr_side),
            "frame_gap": int(gap),
            "crossing_method": "identity_gap_side_transition",
            "crossing_candidate_class": "FAST_CROSSING" if gap > 1 else "TRUE_CROSSING",
            "fast_crossing": True,
            "sparse_crossing": True,
            "gap_bridge_used": True,
            "track_observations": int(len(combined)),
            "short_track": bool(len(combined) < self.config.short_track_observation_threshold),
            "crossing_index": crossing_index,
            "pre_zone_observations": int(evidence["pre_zone_observations"]),
            "corridor_observations": int(evidence["corridor_observations"]),
            "post_zone_observations": int(evidence["post_zone_observations"]),
            "pre_zone_evidence": bool(evidence["pre_zone_evidence"]),
            "corridor_evidence": bool(evidence["corridor_evidence"]),
            "post_zone_evidence": bool(evidence["post_zone_evidence"]),
            "zone_path": evidence["zone_path"],
            "zone_chatter_count": int(evidence["zone_chatter_count"]),
            "trajectory_quality": float(combined["trajectory_quality"].mean()),
            "direction_confidence": direction_confidence,
            "normal_displacement_px": float(normal_displacement),
            "corridor_confidence": float(evidence["corridor_confidence"]),
            "phase1_status": phase1_status,
            "phase2_status": "PASS",
            "count_eligibility": bool(
                class_evidence["counting_class"] != "unknown"
                and class_evidence["counting_class_confidence"] >= self.config.min_counting_class_confidence
            ),
            "candidate_quality": candidate_quality,
            "geometry_crossing": True,
            "identity_gap_side_transition": True,
            "identity_gap_frames": int(gap),
            "identity_gap_identity_confirmed": True,
            "candidate_duplicate_of": pd.NA,
            "candidate_duplicate_confidence": 0.0,
            "candidate_duplicate_reason": "",
            "candidate_duplicate_suppressed": False,
            "gap_count": int(gap - 1),
            "counted": bool(
                class_evidence["counting_class"] != "unknown"
                and class_evidence["counting_class_confidence"] >= self.config.min_counting_class_confidence
            ),
            "_phase1_pass": bool(phase1_pass),
            "_phase2_pass": True,
            "_phase1_reason": phase1_reason,
            "_phase2_reason": "identity_gap_side_transition",
            "_failure_reason": "identity_gap_side_transition",
            "_max_speed": float(combined["speed_px_per_frame"].max()),
            "_mean_speed": float(combined["speed_px_per_frame"].mean()),
            "_max_normal": float(combined["velocity_normal_px_per_frame"].abs().max()),
            "_mean_abs_normal": float(combined["velocity_normal_px_per_frame"].abs().mean()),
            "_mean_abs_tangent": float(combined["velocity_tangent_px_per_frame"].abs().mean()),
            "_trajectory_direction": direction,
            "_candidates": [],
        }

    def _fragment_gap_score(self, early: pd.DataFrame, late: pd.DataFrame) -> tuple[bool, float, dict]:
        """Strong raw-fragment continuity check used when physical identity failed to reconnect."""
        if early.empty or late.empty:
            return False, 0.0, {}
        e = early.sort_values("frame_id").iloc[-1]
        l = late.sort_values("frame_id").iloc[0]
        gap = int(l["frame_id"] - e["frame_id"])
        if gap <= 1 or gap > int(self.config.identity_gap_max_frames):
            return False, 0.0, {}

        e_side = int(e.get("stable_side", e.get("raw_side", 0)))
        l_side = int(l.get("stable_side", l.get("raw_side", 0)))
        if e_side == 0 or l_side == 0 or e_side == l_side:
            return False, 0.0, {}

        ex, ey = float(e["raw_x"]), float(e["raw_y"])
        lx, ly = float(l["raw_x"]), float(l["raw_y"])
        raw_distance = math.hypot(lx-ex, ly-ey)
        if raw_distance > self.config.identity_gap_max_endpoint_distance_px:
            return False, 0.0, {}

        near = min(abs(float(e["raw_signed_distance_px"])), abs(float(l["raw_signed_distance_px"])))
        if near > max(self.config.approach_distance_px, self.config.corridor_px * 3.0):
            return False, 0.0, {}

        # Compare tail motion to displacement across the gap.
        ev = np.array([float(e.get("dx",0.0)), float(e.get("dy",0.0))], dtype=float)
        bridge = np.array([lx-ex, ly-ey], dtype=float) / max(gap,1)
        en = float(np.linalg.norm(ev)); bn = float(np.linalg.norm(bridge))
        cosine = 1.0 if en < 1e-6 or bn < 1e-6 else float(np.dot(ev,bridge)/(en*bn))
        if cosine < 0.55:
            return False, 0.0, {}

        endpoint_score = max(0.0, 1.0 - raw_distance / max(self.config.identity_gap_max_endpoint_distance_px,1e-6))
        gap_score = max(0.0, 1.0 - gap / max(self.config.identity_gap_max_frames,1))
        direction_score = max(0.0, min(1.0,(cosine+1.0)/2.0))
        near_score = max(0.0, 1.0 - near / max(self.config.approach_distance_px,1.0))
        score = 0.35*endpoint_score + 0.20*gap_score + 0.30*direction_score + 0.15*near_score
        return True, float(score), {
            "gap": gap, "endpoint_distance": raw_distance, "cosine": cosine,
            "early_side": e_side, "late_side": l_side,
        }

    def _build_unlinked_fragment_gap_candidate(
        self, early: pd.DataFrame, late: pd.DataFrame, candidate_id: int,
    ) -> dict | None:
        ok, score, info = self._fragment_gap_score(early, late)
        if not ok or score < 0.60:
            return None

        # Only use this fallback when the identity engine failed to reconnect.
        early_identity = int(early["identity_id"].iloc[0])
        late_identity = int(late["identity_id"].iloc[0])
        if early_identity == late_identity:
            return None

        combined = pd.concat([early.tail(6), late.head(6)], ignore_index=True).drop_duplicates(
            subset=["frame_id", "track_id"], keep="first"
        ).sort_values("frame_id").reset_index(drop=True)
        split = max(0, len(early.tail(6)) - 1)
        evidence = self._zone_evidence(combined, split)
        class_evidence = self._class_evidence(combined, split)
        phase1_status, phase1_reason, phase1_pass = self._phase1_status(combined)

        direction = self.config.positive_normal_label if info["early_side"] < info["late_side"] else self.config.negative_normal_label
        normal_displacement = abs(float(late["raw_signed_distance_px"].iloc[0]) - float(early["raw_signed_distance_px"].iloc[-1]))
        confidence = min(1.0, max(0.0, score))
        crossing_x, crossing_y = self.estimate_crossing_point(
            (float(early["raw_x"].iloc[-1]), float(early["raw_y"].iloc[-1])),
            (float(late["raw_x"].iloc[0]), float(late["raw_y"].iloc[0])),
        )
        ids = [int(early["track_id"].iloc[0]), int(late["track_id"].iloc[0])]
        cc = class_evidence["counting_class"]
        eligible = bool(cc != "unknown" and class_evidence["counting_class_confidence"] >= self.config.min_counting_class_confidence)

        return {
            "crossing_id": int(candidate_id),
            "identity_id": int(late_identity),
            "track_id": int(late["track_id"].iloc[0]),
            "track_ids": str(ids),
            "first_frame": int(early["frame_id"].min()),
            "last_frame": int(late["frame_id"].max()),
            "crossing_frame": int(late["frame_id"].iloc[0]),
            "crossing_time_sec": float(late["timestamp_sec"].iloc[0]),
            "crossing_x": float(crossing_x), "crossing_y": float(crossing_y),
            "direction": direction, "normal_direction": direction, "line_direction": direction,
            "side_transition": self._side_transition(info["early_side"], info["late_side"]),
            "track_class": cc, "detector_track_class": str(late["track_class"].iloc[0]),
            "track_class_ratio": float(late.get("track_class_ratio", pd.Series([1.0])).iloc[0]),
            "class_ambiguous": bool(late.get("class_ambiguous", pd.Series([False])).iloc[0]),
            "counting_class": cc,
            "counting_class_confidence": float(class_evidence["counting_class_confidence"]),
            "class_transition": class_evidence["class_transition"],
            "class_evidence": class_evidence["class_evidence"],
            "line_distance_px": float(min(abs(float(early["raw_signed_distance_px"].iloc[-1])),abs(float(late["raw_signed_distance_px"].iloc[0])))),
            "previous_side": int(info["early_side"]), "current_side": int(info["late_side"]),
            "frame_gap": int(info["gap"]), "crossing_method": "fragment_gap_side_transition",
            "crossing_candidate_class": "FAST_CROSSING", "fast_crossing": True, "sparse_crossing": True,
            "gap_bridge_used": True, "track_observations": int(len(combined)),
            "short_track": bool(len(combined) < self.config.short_track_observation_threshold),
            "crossing_index": split,
            "pre_zone_observations": int(evidence["pre_zone_observations"]),
            "corridor_observations": int(evidence["corridor_observations"]),
            "post_zone_observations": int(evidence["post_zone_observations"]),
            "pre_zone_evidence": bool(evidence["pre_zone_evidence"]),
            "corridor_evidence": bool(evidence["corridor_evidence"]),
            "post_zone_evidence": bool(evidence["post_zone_evidence"]),
            "zone_path": evidence["zone_path"], "zone_chatter_count": int(evidence["zone_chatter_count"]),
            "trajectory_quality": float(combined["trajectory_quality"].mean()),
            "direction_confidence": float(confidence), "normal_displacement_px": float(normal_displacement),
            "corridor_confidence": float(evidence["corridor_confidence"]),
            "phase1_status": phase1_status, "phase2_status": "PASS", "count_eligibility": eligible,
            "candidate_quality": float(min(1.0,0.5*confidence+0.5*class_evidence["counting_class_confidence"])),
            "geometry_crossing": True, "identity_gap_side_transition": True,
            "identity_gap_frames": int(info["gap"]), "identity_gap_identity_confirmed": False,
            "candidate_duplicate_of": pd.NA, "candidate_duplicate_confidence": 0.0,
            "candidate_duplicate_reason": "unlinked_fragment_gap", "candidate_duplicate_suppressed": False,
            "gap_count": int(max(0,info["gap"]-1)), "counted": eligible,
            "_phase1_pass": bool(phase1_pass), "_phase2_pass": True,
            "_phase1_reason": phase1_reason, "_phase2_reason": "fragment_gap_side_transition",
            "_failure_reason": "fragment_gap_side_transition", "_max_speed": float(combined["speed_px_per_frame"].max()),
            "_mean_speed": float(combined["speed_px_per_frame"].mean()),
            "_max_normal": float(combined["velocity_normal_px_per_frame"].abs().max()),
            "_mean_abs_normal": float(combined["velocity_normal_px_per_frame"].abs().mean()),
            "_mean_abs_tangent": float(combined["velocity_tangent_px_per_frame"].abs().mean()),
            "_trajectory_direction": direction, "_candidates": [],
        }

    # ==================================================================
    # BATCH
    # ==================================================================

    def process(
        self,
        trajectory: pd.DataFrame,
        identity_column: str = "track_id",
        physical_identity_column: str | None = "crossing_id",
        return_diagnostics: bool = False,
    ):
        """Produce canonical crossing candidates without collapsing simultaneous tracks.

        v8 critical design:
        - Direct crossing detection is performed PER RAW TRACK (`identity_column`).
        - Physical identity is an annotation, not a grouping key.
        - Identity-gap crossing candidates are generated separately from the
          physical identity map.
        - Therefore two vehicles that cross simultaneously cannot disappear
          simply because their physical-identity metadata was merged upstream.
        """
        if identity_column not in trajectory.columns:
            raise ValueError(f"Trajectory missing identity column: {identity_column}")

        if physical_identity_column is not None and physical_identity_column not in trajectory.columns:
            physical_identity_column = None

        if trajectory.empty:
            events = pd.DataFrame(columns=self.EVENT_COLUMNS)
            audits = pd.DataFrame(columns=self.AUDIT_COLUMNS)
            prepared = trajectory.copy()
            if return_diagnostics:
                return events, audits, prepared
            return events, audits

        prepared = self.prepare(trajectory)
        if physical_identity_column is not None:
            prepared["identity_id"] = pd.to_numeric(
                prepared[physical_identity_column], errors="coerce"
            ).fillna(prepared[identity_column]).astype(int)
        else:
            prepared["identity_id"] = pd.to_numeric(
                prepared[identity_column], errors="coerce"
            ).astype(int)

        raw_events: list[dict] = []
        audits: list[dict] = []
        next_candidate_id = 1

        # --------------------------------------------------------------
        # A. DIRECT CANDIDATES: one source candidate per RAW TRACK.
        # --------------------------------------------------------------
        for raw_track_id, group in prepared.groupby(identity_column, sort=False):
            group = group.sort_values("frame_id").reset_index(drop=True)
            identity_id = int(group["identity_id"].iloc[0])

            event = self._build_event_for_track(
                group,
                candidate_id=next_candidate_id,
                identity_id=identity_id,
            )
            next_candidate_id += 1

            audits.append(self._build_audit_row(group, event))

            if event["crossing_candidate_class"] in {"TRUE_CROSSING", "FAST_CROSSING"}:
                raw_events.append({
                    key: event.get(key, pd.NA)
                    for key in self.EVENT_COLUMNS
                })

        # --------------------------------------------------------------
        # B. IDENTITY-GAP CANDIDATES: side A -> disappearance -> side B.
        # --------------------------------------------------------------
        if self.config.identity_gap_crossing_enabled and physical_identity_column is not None:
            for identity_id, identity_rows in prepared.groupby("identity_id", sort=False):
                fragments = []
                for track_id, fragment in identity_rows.groupby("track_id", sort=False):
                    fragment = fragment.sort_values("frame_id").reset_index(drop=True)
                    fragments.append(fragment)
                fragments.sort(key=lambda g: int(g["frame_id"].min()))

                for early, late in zip(fragments, fragments[1:]):
                    gap_event = self._build_identity_gap_candidate(
                        int(identity_id),
                        early,
                        late,
                        candidate_id=next_candidate_id,
                    )
                    if gap_event is None:
                        continue
                    next_candidate_id += 1
                    raw_events.append({
                        key: gap_event.get(key, pd.NA)
                        for key in self.EVENT_COLUMNS
                    })

        # --------------------------------------------------------------
        # C. FALLBACK FRAGMENT-GAP CANDIDATES.
        # If BoT-SORT changed track_id and identity reconnect failed, do not
        # lose a real crossing. Only strong, non-overlapping, opposite-side,
        # motion-consistent fragment pairs are allowed.
        # --------------------------------------------------------------
        if self.config.identity_gap_crossing_enabled:
            fragments = []
            for track_id, fragment in prepared.groupby("track_id", sort=False):
                fragment = fragment.sort_values("frame_id").reset_index(drop=True)
                fragments.append(fragment)
            fragments.sort(key=lambda g: int(g["frame_id"].min()))

            used_pairs = set()
            for j, late in enumerate(fragments):
                best = None
                best_score = 0.0
                for early in fragments[:j]:
                    key = (int(early["track_id"].iloc[0]), int(late["track_id"].iloc[0]))
                    if key in used_pairs:
                        continue
                    ok, score, _info = self._fragment_gap_score(early, late)
                    if ok and score > best_score:
                        # Fallback only for physically different identities.
                        if int(early["identity_id"].iloc[0]) != int(late["identity_id"].iloc[0]):
                            best = early
                            best_score = score
                if best is not None and best_score >= 0.60:
                    event = self._build_unlinked_fragment_gap_candidate(
                        best, late, candidate_id=next_candidate_id
                    )
                    if event is not None:
                        next_candidate_id += 1
                        raw_events.append({
                            key: event.get(key, pd.NA)
                            for key in self.EVENT_COLUMNS
                        })
                        used_pairs.add((int(best["track_id"].iloc[0]), int(late["track_id"].iloc[0])))

        events_df = pd.DataFrame(raw_events)
        if events_df.empty:
            events_df = pd.DataFrame(columns=self.EVENT_COLUMNS)
        else:
            # Preserve integer-safe candidate IDs and deterministic ordering.
            events_df = events_df.sort_values(
                ["crossing_frame", "crossing_id"],
                na_position="last",
            ).reset_index(drop=True)

            events_df = self._apply_duplicate_identity_audit(
                events_df,
                prepared,
            )

            for column in self.EVENT_COLUMNS:
                if column not in events_df.columns:
                    events_df[column] = pd.NA
            events_df = events_df[self.EVENT_COLUMNS]

        audit_df = pd.DataFrame(audits)
        if not audit_df.empty and not events_df.empty:
            merge_cols = [
                "crossing_id",
                "identity_id",
                "identity_gap_side_transition",
                "identity_gap_frames",
                "identity_gap_identity_confirmed",
                "candidate_duplicate_of",
                "candidate_duplicate_confidence",
                "candidate_duplicate_reason",
                "candidate_duplicate_suppressed",
                "count_eligibility",
                "counted",
            ]
            available = [c for c in merge_cols if c in events_df.columns]
            # Do not merge raw-track candidates onto an audit row by physical
            # identity. The audit is per raw track, so only identity-gap events
            # are kept as a separate canonical-event record.
            canonical_track_ids = set()
            for _, e in events_df.iterrows():
                ids = self._event_track_id_list(e)
                if len(ids) == 1:
                    canonical_track_ids.add((int(ids[0]), int(e["crossing_id"])))
            audit_df["canonical_candidate_ids"] = ""
            for i, row in audit_df.iterrows():
                tid_list = self._event_track_id_list(pd.Series({"track_ids": row["track_ids"]}))
                matched = [
                    int(e["crossing_id"])
                    for _, e in events_df.iterrows()
                    if set(tid_list).intersection(self._event_track_id_list(e))
                ]
                audit_df.loc[i, "canonical_candidate_ids"] = str(matched)

        if return_diagnostics:
            return events_df, audit_df, prepared
        return events_df, audit_df


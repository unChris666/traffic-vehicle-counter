from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import math
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CrossingConfig:
    """
    Phase 1 + Phase 2 configuration.

    Phase 1: Trajectory Engine
        - raw trajectory history
        - EMA-smoothed trajectory
        - signed distance to counting line
        - velocity in image coordinates
        - velocity normal/tangent to the counting line
        - movement direction

    Phase 2: Crossing Corridor
        - pre-zone
        - crossing corridor
        - post-zone
        - zone transition diagnostics

    This module remains identity-agnostic. The identity column is supplied
    by the caller (normally ``crossing_id`` from CrossingIdentityEngine).

    Important compatibility rule:
        ``process()`` keeps the same two-value return signature as the
        previous robust_crossing.py so the current TrafficCounter does not
        need to change just to evaluate Phase 1/2.
    """

    # ---------------------------------------------------------------
    # LINE GEOMETRY
    # ---------------------------------------------------------------
    line_deadband_px: float = 8.0
    corridor_px: float = 45.0

    # ---------------------------------------------------------------
    # TRAJECTORY
    # ---------------------------------------------------------------
    max_trajectory_gap_sec: float = 1.50
    smoothing_alpha: float = 0.35
    velocity_window: int = 5
    min_direction_displacement_px: float = 8.0
    direction_window: int = 3

    # ---------------------------------------------------------------
    # CROSSING / ZONE VALIDATION
    # ---------------------------------------------------------------
    min_track_observations: int = 2
    min_pre_zone_observations: int = 2
    min_corridor_observations: int = 1
    min_post_zone_observations: int = 1
    require_post_zone: bool = True

    # Maximum acceptable jump between consecutive observations for a
    # trajectory to be considered continuous. This is diagnostic only.
    max_velocity_px_per_frame: float = 80.0

    vehicle_classes: tuple[str, ...] = (
        "motorcycle",
        "car",
        "truck",
        "bus",
    )


class RobustCrossingEngine:
    """
    Phase 1 + Phase 2 trajectory/corridor engine.

    The engine deliberately does NOT implement:
        - physical identity matching
        - ID-switch correction
        - fragment reconnect
        - state machine transitions
        - final vehicle/person counting

    Those remain separate phases.

    Output philosophy:
        Every trajectory observation receives explicit diagnostics so the
        notebook/UI can answer:

          "Did Phase 1 work?"
          "Did Phase 2 work?"
          "Why did this track fail?"

    The public ``process()`` API remains compatible with the previous
    robust_crossing.py implementation:

        events_df, audit_df = engine.process(...)
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
        "side",
        "zone",
        "direction_local",
        "trajectory_continuity",
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
        "track_observations",
        "crossing_index",
        "pre_zone_observations",
        "corridor_observations",
        "post_zone_observations",
        "pre_zone_evidence",
        "corridor_evidence",
        "post_zone_evidence",
        "zone_path",
        "trajectory_quality",
        "direction_confidence",
        "corridor_confidence",
        "phase1_status",
        "phase2_status",
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
        "first_distance_px",
        "last_distance_px",
        "min_distance_px",
        "max_speed_px_per_frame",
        "mean_speed_px_per_frame",
        "mean_abs_normal_velocity_px_per_frame",
        "mean_abs_tangent_velocity_px_per_frame",
        "trajectory_direction",
        "direction_confidence",
        "zone_path",
        "pre_zone_observations",
        "corridor_observations",
        "post_zone_observations",
        "pre_zone_evidence",
        "corridor_evidence",
        "post_zone_evidence",
        "crossing_detected",
        "crossing_frame",
        "crossing_method",
        "crossing_direction",
        "phase1_status",
        "phase2_status",
        "phase1_pass",
        "phase2_pass",
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
            raise ValueError(
                "smoothing_alpha must be in the interval (0, 1]."
            )

        if self.config.corridor_px <= self.config.line_deadband_px:
            raise ValueError(
                "corridor_px must be greater than line_deadband_px."
            )

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

        # Unit normal. Its sign follows the signed-line convention.
        self.normal_x = -self.line_dy / self.line_length
        self.normal_y = self.line_dx / self.line_length

    # ==================================================================
    # LINE GEOMETRY
    # ==================================================================

    def signed_line_value(self, x: float, y: float) -> float:
        """Unnormalized signed line value."""
        return (
            self.line_dx * (y - self.y1)
            - self.line_dy * (x - self.x1)
        )

    def signed_distance(self, x: float, y: float) -> float:
        """Signed perpendicular distance to the counting line in pixels."""
        return self.signed_line_value(x, y) / self.line_length

    def line_distance(self, x: float, y: float) -> float:
        return abs(self.signed_distance(x, y))

    def side(self, x: float, y: float) -> int:
        distance = self.line_distance(x, y)

        if distance <= self.config.line_deadband_px:
            return 0

        return 1 if self.signed_distance(x, y) > 0 else -1

    # ==================================================================
    # TRAJECTORY SMOOTHING
    # ==================================================================

    def _ema(self, values: np.ndarray) -> np.ndarray:
        """Simple causal EMA used only for counting trajectory analysis."""
        if len(values) == 0:
            return values.copy()

        alpha = float(self.config.smoothing_alpha)
        output = np.empty_like(values, dtype=np.float64)
        output[0] = values[0]

        for i in range(1, len(values)):
            output[i] = (
                alpha * values[i]
                + (1.0 - alpha) * output[i - 1]
            )

        return output

    # ==================================================================
    # SEGMENT / LINE INTERSECTION
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
        x_prev, y_prev = previous_point
        x_curr, y_curr = current_point

        v_prev = self.signed_distance(x_prev, y_prev)
        v_curr = self.signed_distance(x_curr, y_curr)
        denominator = v_prev - v_curr

        if abs(denominator) < 1e-9:
            return current_point

        alpha = v_prev / denominator
        alpha = max(0.0, min(1.0, alpha))

        return (
            float(x_prev + alpha * (x_curr - x_prev)),
            float(y_prev + alpha * (y_curr - y_prev)),
        )

    # ==================================================================
    # DIRECTION / VELOCITY
    # ==================================================================

    @staticmethod
    def side_transition(
        previous_side: int,
        current_side: int,
    ) -> str:
        if previous_side == -1 and current_side == 1:
            return "side_-1_to_+1"
        if previous_side == 1 and current_side == -1:
            return "side_+1_to_-1"
        return "UNKNOWN"

    def estimate_motion_direction(
        self,
        group: pd.DataFrame,
        crossing_index: int,
    ) -> str:
        """
        Business/display direction.

        Kept as L→R / R→L for compatibility with the existing UI.
        For a diagonal counting line, ``line_direction`` and
        ``side_transition`` provide the geometrically correct crossing
        direction diagnostics.
        """
        window = max(1, int(self.config.direction_window))

        start_idx = max(0, crossing_index - window)
        end_idx = min(len(group) - 1, crossing_index + window)

        before = group.iloc[start_idx : crossing_index + 1]
        after = group.iloc[crossing_index : end_idx + 1]

        if before.empty or after.empty:
            return "UNKNOWN"

        dx = float(after.iloc[-1]["smooth_x"] - before.iloc[0]["smooth_x"])
        dy = float(after.iloc[-1]["smooth_y"] - before.iloc[0]["smooth_y"])
        displacement = math.hypot(dx, dy)

        if displacement < self.config.min_direction_displacement_px:
            return "UNKNOWN"

        return "L→R" if dx > 0 else "R→L"

    def estimate_direction_confidence(
        self,
        group: pd.DataFrame,
        crossing_index: int,
    ) -> float:
        """Confidence that motion direction is sufficiently observable."""
        window = max(1, int(self.config.direction_window))
        start_idx = max(0, crossing_index - window)
        end_idx = min(len(group) - 1, crossing_index + window)

        if end_idx <= start_idx:
            return 0.0

        sample = group.iloc[start_idx : end_idx + 1]
        dx = float(sample.iloc[-1]["smooth_x"] - sample.iloc[0]["smooth_x"])
        dy = float(sample.iloc[-1]["smooth_y"] - sample.iloc[0]["smooth_y"])
        displacement = math.hypot(dx, dy)

        threshold = max(self.config.min_direction_displacement_px, 1e-6)
        displacement_score = min(1.0, displacement / (threshold * 3.0))

        speeds = pd.to_numeric(
            sample["speed_px_per_frame"],
            errors="coerce",
        ).fillna(0.0)
        speed_score = float((speeds > 0.5).mean()) if len(speeds) else 0.0

        return float(max(0.0, min(1.0, 0.7 * displacement_score + 0.3 * speed_score)))

    # ==================================================================
    # PHASE 1: TRAJECTORY PREPARATION
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
                "Trajectory missing required columns: "
                f"{sorted(missing)}"
            )

        df = trajectory.copy()

        if df.empty:
            for column in self.TRAJECTORY_COLUMNS:
                df[column] = pd.Series(dtype=float)
            return df

        df = (
            df.sort_values(["track_id", "frame_id"])
            .reset_index(drop=True)
        )

        df["raw_x"] = pd.to_numeric(
            df["bottom_center_x"],
            errors="coerce",
        )
        df["raw_y"] = pd.to_numeric(
            df["bottom_center_y"],
            errors="coerce",
        )

        if df[["raw_x", "raw_y"]].isna().any().any():
            raise ValueError(
                "Trajectory contains invalid bottom-center coordinates."
            )

        # --------------------------------------------------------------
        # EMA smoothing per track.
        # --------------------------------------------------------------
        smooth_x_parts: list[pd.Series] = []
        smooth_y_parts: list[pd.Series] = []

        for _, group in df.groupby("track_id", sort=False):
            smooth_x = pd.Series(
                self._ema(group["raw_x"].to_numpy(dtype=np.float64)),
                index=group.index,
            )
            smooth_y = pd.Series(
                self._ema(group["raw_y"].to_numpy(dtype=np.float64)),
                index=group.index,
            )
            smooth_x_parts.append(smooth_x)
            smooth_y_parts.append(smooth_y)

        df["smooth_x"] = pd.concat(smooth_x_parts).sort_index()
        df["smooth_y"] = pd.concat(smooth_y_parts).sort_index()

        # --------------------------------------------------------------
        # Frame / time deltas.
        # --------------------------------------------------------------
        df["frame_delta"] = (
            df.groupby("track_id")["frame_id"].diff()
        )
        df["time_delta_sec"] = (
            df.groupby("track_id")["timestamp_sec"].diff()
        )

        valid_dt = (
            df["frame_delta"].notna()
            & (df["frame_delta"] > 0)
        )

        # --------------------------------------------------------------
        # Raw/smoothed displacement.
        # --------------------------------------------------------------
        df["dx"] = (
            df.groupby("track_id")["smooth_x"].diff()
        )
        df["dy"] = (
            df.groupby("track_id")["smooth_y"].diff()
        )

        df["dx"] = df["dx"].where(valid_dt, 0.0).fillna(0.0)
        df["dy"] = df["dy"].where(valid_dt, 0.0).fillna(0.0)

        frame_delta_safe = df["frame_delta"].where(
            valid_dt,
            1.0,
        ).fillna(1.0)

        df["speed_px_per_frame"] = (
            np.hypot(df["dx"], df["dy"])
            / frame_delta_safe
        )

        # --------------------------------------------------------------
        # Velocity decomposed relative to the counting line.
        # --------------------------------------------------------------
        vx_instant = df["dx"] / frame_delta_safe
        vy_instant = df["dy"] / frame_delta_safe

        velocity_window = max(1, int(self.config.velocity_window))

        # Median filtering of recent velocity is less sensitive to a single
        # noisy bbox jump than raw frame-to-frame velocity.
        vx = (
            df.assign(_vx=vx_instant)
            .groupby("track_id", sort=False)["_vx"]
            .transform(
                lambda series: series.rolling(
                    velocity_window,
                    min_periods=1,
                ).median()
            )
        )
        vy = (
            df.assign(_vy=vy_instant)
            .groupby("track_id", sort=False)["_vy"]
            .transform(
                lambda series: series.rolling(
                    velocity_window,
                    min_periods=1,
                ).median()
            )
        )

        df["velocity_normal_px_per_frame"] = (
            vx * self.normal_x
            + vy * self.normal_y
        )

        df["velocity_tangent_px_per_frame"] = (
            vx * self.tangent_x
            + vy * self.tangent_y
        )

        # --------------------------------------------------------------
        # Signed distance + side.
        #
        # We keep both smoothed geometry and raw geometry. Smoothing is
        # useful for velocity/direction, but zone membership should follow
        # the actual observed bbox position to avoid EMA lag around the
        # crossing boundary.
        # --------------------------------------------------------------
        df["signed_distance_px"] = (
            self.line_dx * (df["smooth_y"] - self.y1)
            - self.line_dy * (df["smooth_x"] - self.x1)
        ) / self.line_length

        df["line_distance_px"] = df["signed_distance_px"].abs()

        df["raw_signed_distance_px"] = (
            self.line_dx * (df["raw_y"] - self.y1)
            - self.line_dy * (df["raw_x"] - self.x1)
        ) / self.line_length

        df["raw_line_distance_px"] = df["raw_signed_distance_px"].abs()

        df["raw_side"] = np.select(
            [
                df["raw_line_distance_px"] <= self.config.line_deadband_px,
                df["raw_signed_distance_px"] > 0,
            ],
            [0, 1],
            default=-1,
        ).astype(int)

        df["side"] = np.select(
            [
                df["line_distance_px"] <= self.config.line_deadband_px,
                df["signed_distance_px"] > 0,
            ],
            [0, 1],
            default=-1,
        ).astype(int)

        # --------------------------------------------------------------
        # Per-track trajectory quality.
        # --------------------------------------------------------------
        df["trajectory_continuity"] = (
            df["frame_delta"].isna()
            | (
                df["frame_delta"]
                <= self.config.max_trajectory_gap_sec * self.fps
            )
        ).astype(float)

        reasonable_speed = (
            df["speed_px_per_frame"]
            <= self.config.max_velocity_px_per_frame
        ).astype(float)

        df["trajectory_quality"] = (
            0.50 * df["trajectory_continuity"]
            + 0.50 * reasonable_speed
        ).clip(0.0, 1.0)

        # --------------------------------------------------------------
        # Zone assignment happens per track after the initial side is
        # established. This produces an auditable PRE/CORRIDOR/POST path.
        # --------------------------------------------------------------
        zones = pd.Series(
            "UNKNOWN",
            index=df.index,
            dtype="object",
        )

        for _, group in df.groupby("track_id", sort=False):
            stable_sides = group.loc[group["raw_side"] != 0, "raw_side"]
            initial_side = (
                int(stable_sides.iloc[0])
                if not stable_sides.empty
                else 0
            )

            for idx in group.index:
                # Zone membership uses actual observed bbox geometry.
                distance = float(df.at[idx, "raw_line_distance_px"])
                side_value = int(df.at[idx, "raw_side"])

                if distance <= self.config.corridor_px:
                    zones.at[idx] = "CORRIDOR"
                elif initial_side == 0:
                    zones.at[idx] = "UNKNOWN"
                elif side_value == 0 or side_value == initial_side:
                    zones.at[idx] = "PRE"
                else:
                    zones.at[idx] = "POST"

        df["zone"] = zones

        # Local direction is derived from recent displacement.
        direction_values: list[str] = []
        for _, group in df.groupby("track_id", sort=False):
            values = ["UNKNOWN"] * len(group)
            idxs = list(group.index)
            window = max(1, int(self.config.direction_window))

            for local_i in range(len(group)):
                start = max(0, local_i - window)
                end = min(len(group) - 1, local_i + window)
                if end <= start:
                    values[local_i] = "UNKNOWN"
                    continue

                dx = float(
                    group.iloc[end]["smooth_x"]
                    - group.iloc[start]["smooth_x"]
                )
                dy = float(
                    group.iloc[end]["smooth_y"]
                    - group.iloc[start]["smooth_y"]
                )
                if math.hypot(dx, dy) < self.config.min_direction_displacement_px:
                    values[local_i] = "UNKNOWN"
                else:
                    values[local_i] = "L→R" if dx > 0 else "R→L"

            for idx, value in zip(idxs, values):
                direction_values.append(value)

        # The groupby iteration above is sorted in the same order as df,
        # so the generated list is aligned with the DataFrame rows.
        df["direction_local"] = direction_values

        return df

    # ==================================================================
    # PHASE 2: CROSSING CORRIDOR / ZONE ANALYSIS
    # ==================================================================

    def _zone_summary(self, group: pd.DataFrame) -> dict:
        zones = group["zone"].astype(str).tolist()
        compact: list[str] = []
        for zone in zones:
            if not compact or compact[-1] != zone:
                compact.append(zone)

        counts = group["zone"].value_counts()
        return {
            "zone_path": " → ".join(compact),
            "pre_zone_observations": int(counts.get("PRE", 0)),
            "corridor_observations": int(counts.get("CORRIDOR", 0)),
            "post_zone_observations": int(counts.get("POST", 0)),
        }

    def _find_zone_transition_index(
        self,
        group: pd.DataFrame,
    ) -> int | None:
        """Find the first observation that has evidence of crossing."""
        if len(group) < 2:
            return None

        for i in range(1, len(group)):
            previous = group.iloc[i - 1]
            current = group.iloc[i]

            frame_gap = int(current["frame_id"] - previous["frame_id"])
            if (
                frame_gap <= 0
                or frame_gap > self.config.max_trajectory_gap_sec * self.fps
            ):
                continue

            previous_point = (
                float(previous["smooth_x"]),
                float(previous["smooth_y"]),
            )
            current_point = (
                float(current["smooth_x"]),
                float(current["smooth_y"]),
            )

            previous_side = int(previous["side"])
            current_side = int(current["side"])

            side_change = (
                previous_side != 0
                and current_side != 0
                and previous_side != current_side
            )

            line_intersection = self.trajectory_intersects_line(
                previous_point,
                current_point,
            )

            # Strong evidence means actual line crossing OR a true side
            # transition. Corridor alone is never enough.
            if side_change or line_intersection:
                return i

        return None

    def _zone_evidence(
        self,
        group: pd.DataFrame,
        crossing_index: int | None,
    ) -> dict:
        summary = self._zone_summary(group)
        pre = summary["pre_zone_observations"]
        corridor = summary["corridor_observations"]
        post = summary["post_zone_observations"]

        if crossing_index is not None:
            before = group.iloc[: crossing_index + 1]
            after = group.iloc[crossing_index:]
            pre_before = int((before["zone"] == "PRE").sum())
            corridor_before = int((before["zone"] == "CORRIDOR").sum())
            post_after = int((after["zone"] == "POST").sum())
        else:
            pre_before = pre
            corridor_before = corridor
            post_after = post

        return {
            **summary,
            "pre_zone_evidence": pre_before >= self.config.min_pre_zone_observations,
            "corridor_evidence": corridor_before >= self.config.min_corridor_observations,
            "post_zone_evidence": post_after >= self.config.min_post_zone_observations,
        }

    def _crossing_method(
        self,
        previous_side: int,
        current_side: int,
        line_intersection: bool,
        corridor_support: bool,
    ) -> str:
        methods: list[str] = []

        if previous_side != 0 and current_side != 0 and previous_side != current_side:
            methods.append("side_change")
        if line_intersection:
            methods.append("line_intersection")
        if corridor_support:
            methods.append("corridor_support")

        return "+".join(methods)

    def _phase1_status(self, group: pd.DataFrame) -> tuple[str, bool, str]:
        observations = len(group)
        if observations < self.config.min_track_observations:
            return "FAIL", False, "insufficient_track_observations"

        continuity = float(group["trajectory_continuity"].mean())
        finite_speed = np.isfinite(group["speed_px_per_frame"].to_numpy()).all()
        has_signed_distance = np.isfinite(
            group["signed_distance_px"].to_numpy()
        ).all()
        has_velocity = np.isfinite(
            group["velocity_normal_px_per_frame"].to_numpy()
        ).all()

        if not finite_speed:
            return "FAIL", False, "non_finite_velocity"
        if not has_signed_distance:
            return "FAIL", False, "non_finite_signed_distance"
        if not has_velocity:
            return "FAIL", False, "non_finite_normal_velocity"
        if continuity < 0.80:
            return "REVIEW", False, "trajectory_has_large_gaps"

        return "PASS", True, ""

    def _phase2_status(self, zone: dict, crossing_index: int | None) -> tuple[str, bool, str]:
        if crossing_index is None:
            return "FAIL", False, "no_geometric_crossing_detected"

        if not zone["pre_zone_evidence"]:
            return "REVIEW", False, "insufficient_pre_zone_evidence"

        if not zone["corridor_evidence"]:
            return "REVIEW", False, "insufficient_corridor_evidence"

        if self.config.require_post_zone and not zone["post_zone_evidence"]:
            return "REVIEW", False, "insufficient_post_zone_evidence"

        return "PASS", True, ""

    # ==================================================================
    # SINGLE IDENTITY ANALYSIS
    # ==================================================================

    def detect_track_crossing(
        self,
        group: pd.DataFrame,
        *,
        identity_id: int | None = None,
    ) -> Optional[dict]:
        group = (
            group.sort_values("frame_id")
            .reset_index(drop=True)
            .copy()
        )

        if len(group) < self.config.min_track_observations:
            return None

        if identity_id is None:
            identity_id = int(group.iloc[0]["track_id"])

        track_class = str(group.iloc[0]["track_class"])
        first_frame = int(group["frame_id"].min())
        last_frame = int(group["frame_id"].max())

        zone = self._zone_evidence(group, None)
        phase1_status, phase1_pass, phase1_reason = self._phase1_status(group)

        crossing_index = self._find_zone_transition_index(group)
        zone = self._zone_evidence(group, crossing_index)

        phase2_status, phase2_pass, phase2_reason = self._phase2_status(
            zone,
            crossing_index,
        )

        failure_reason = phase1_reason or phase2_reason

        max_speed = float(group["speed_px_per_frame"].max())
        mean_speed = float(group["speed_px_per_frame"].mean())
        mean_abs_normal = float(
            group["velocity_normal_px_per_frame"].abs().mean()
        )
        mean_abs_tangent = float(
            group["velocity_tangent_px_per_frame"].abs().mean()
        )

        trajectory_direction = self._infer_track_direction(group)

        if crossing_index is None:
            return {
                "crossing_id": int(identity_id),
                "track_id": int(group.iloc[-1]["track_id"]),
                "track_ids": "",
                "first_frame": first_frame,
                "last_frame": last_frame,
                "crossing_frame": pd.NA,
                "crossing_time_sec": pd.NA,
                "crossing_x": pd.NA,
                "crossing_y": pd.NA,
                "direction": trajectory_direction,
                "line_direction": "UNKNOWN",
                "side_transition": "UNKNOWN",
                "track_class": track_class,
                "track_class_ratio": float(
                    group.iloc[0].get("track_class_ratio", np.nan)
                ),
                "class_ambiguous": bool(
                    group.iloc[0].get("class_ambiguous", False)
                ),
                "line_distance_px": pd.NA,
                "previous_side": pd.NA,
                "current_side": pd.NA,
                "frame_gap": pd.NA,
                "crossing_method": "",
                "track_observations": len(group),
                "crossing_index": pd.NA,
                **zone,
                "trajectory_quality": round(float(group["trajectory_quality"].mean()), 4),
                "direction_confidence": 0.0,
                "corridor_confidence": round(
                    min(
                        1.0,
                        zone["corridor_observations"]
                        / max(1, self.config.min_corridor_observations),
                    ),
                    4,
                ),
                "phase1_status": phase1_status,
                "phase2_status": phase2_status,
                "counted": False,
                "_phase1_pass": phase1_pass,
                "_phase2_pass": phase2_pass,
                "_phase1_reason": phase1_reason,
                "_phase2_reason": phase2_reason,
                "_max_speed": max_speed,
                "_mean_speed": mean_speed,
                "_mean_abs_normal": mean_abs_normal,
                "_mean_abs_tangent": mean_abs_tangent,
            }

        previous = group.iloc[crossing_index - 1]
        current = group.iloc[crossing_index]

        previous_point = (
            float(previous["smooth_x"]),
            float(previous["smooth_y"]),
        )
        current_point = (
            float(current["smooth_x"]),
            float(current["smooth_y"]),
        )

        previous_side = int(previous["side"])
        current_side = int(current["side"])
        frame_gap = int(current["frame_id"] - previous["frame_id"])

        side_change = (
            previous_side != 0
            and current_side != 0
            and previous_side != current_side
        )

        line_intersection = self.trajectory_intersects_line(
            previous_point,
            current_point,
        )

        corridor_support = (
            min(
                float(previous["line_distance_px"]),
                float(current["line_distance_px"]),
            )
            <= self.config.corridor_px
        )

        crossing_x, crossing_y = self.estimate_crossing_point(
            previous_point,
            current_point,
        ) if line_intersection else current_point

        side_transition = self.side_transition(
            previous_side,
            current_side,
        )

        direction = self.estimate_motion_direction(
            group,
            crossing_index,
        )
        direction_confidence = self.estimate_direction_confidence(
            group,
            crossing_index,
        )

        line_direction = side_transition

        crossing_method = self._crossing_method(
            previous_side,
            current_side,
            line_intersection,
            corridor_support,
        )

        corridor_confidence = min(
            1.0,
            zone["corridor_observations"]
            / max(1, self.config.min_corridor_observations),
        )

        counted = bool(phase1_pass and phase2_pass)

        return {
            "crossing_id": int(identity_id),
            "track_id": int(current["track_id"]),
            "track_ids": "",
            "first_frame": first_frame,
            "last_frame": last_frame,
            "crossing_frame": int(current["frame_id"]),
            "crossing_time_sec": float(current["timestamp_sec"]),
            "crossing_x": float(crossing_x),
            "crossing_y": float(crossing_y),
            "direction": direction,
            "line_direction": line_direction,
            "side_transition": side_transition,
            "track_class": track_class,
            "track_class_ratio": float(
                current.get("track_class_ratio", np.nan)
            ),
            "class_ambiguous": bool(
                current.get("class_ambiguous", False)
            ),
            "line_distance_px": float(
                min(
                    previous["line_distance_px"],
                    current["line_distance_px"],
                )
            ),
            "previous_side": (
                previous_side if previous_side != 0 else None
            ),
            "current_side": (
                current_side if current_side != 0 else None
            ),
            "frame_gap": frame_gap,
            "crossing_method": crossing_method,
            "track_observations": len(group),
            "crossing_index": int(crossing_index),
            **zone,
            "trajectory_quality": round(float(group["trajectory_quality"].mean()), 4),
            "direction_confidence": round(float(direction_confidence), 4),
            "corridor_confidence": round(float(corridor_confidence), 4),
            "phase1_status": phase1_status,
            "phase2_status": phase2_status,
            "counted": counted,
            # Private diagnostic values are intentionally added later in
            # the audit rather than exposed in the legacy event interface.
            "_phase1_pass": phase1_pass,
            "_phase2_pass": phase2_pass,
            "_phase1_reason": phase1_reason,
            "_phase2_reason": phase2_reason,
            "_max_speed": max_speed,
            "_mean_speed": mean_speed,
            "_mean_abs_normal": mean_abs_normal,
            "_mean_abs_tangent": mean_abs_tangent,
        }

    def _infer_track_direction(self, group: pd.DataFrame) -> str:
        if len(group) < 2:
            return "UNKNOWN"

        dx = float(group.iloc[-1]["smooth_x"] - group.iloc[0]["smooth_x"])
        dy = float(group.iloc[-1]["smooth_y"] - group.iloc[0]["smooth_y"])

        if math.hypot(dx, dy) < self.config.min_direction_displacement_px:
            return "UNKNOWN"

        return "L→R" if dx > 0 else "R→L"

    # ==================================================================
    # AUDIT BUILDER
    # ==================================================================

    def _build_track_audit_row(
        self,
        group: pd.DataFrame,
        event: dict,
        identity_id: int,
    ) -> dict:
        first_frame = int(group["frame_id"].min())
        last_frame = int(group["frame_id"].max())

        stable_sides = group.loc[group["side"] != 0, "side"]
        first_side = int(stable_sides.iloc[0]) if not stable_sides.empty else 0
        last_side = int(stable_sides.iloc[-1]) if not stable_sides.empty else 0

        trajectory_direction = self._infer_track_direction(group)

        phase1_pass = bool(event.get("_phase1_pass", False))
        phase2_pass = bool(event.get("_phase2_pass", False))

        reasons = []
        phase1_reason = str(event.get("_phase1_reason", ""))
        phase2_reason = str(event.get("_phase2_reason", ""))
        if phase1_reason:
            reasons.append(f"P1:{phase1_reason}")
        if phase2_reason:
            reasons.append(f"P2:{phase2_reason}")

        return {
            "crossing_id": int(identity_id),
            "track_ids": str(group["track_id"].drop_duplicates().tolist()),
            "first_frame": first_frame,
            "last_frame": last_frame,
            "track_class": str(event.get("track_class", "unknown")),
            "track_observations": int(len(group)),
            "first_side": first_side,
            "last_side": last_side,
            "first_distance_px": float(group.iloc[0]["line_distance_px"]),
            "last_distance_px": float(group.iloc[-1]["line_distance_px"]),
            "min_distance_px": float(group["line_distance_px"].min()),
            "max_speed_px_per_frame": float(event.get("_max_speed", 0.0)),
            "mean_speed_px_per_frame": float(event.get("_mean_speed", 0.0)),
            "mean_abs_normal_velocity_px_per_frame": float(
                event.get("_mean_abs_normal", 0.0)
            ),
            "mean_abs_tangent_velocity_px_per_frame": float(
                event.get("_mean_abs_tangent", 0.0)
            ),
            "trajectory_direction": trajectory_direction,
            "direction_confidence": float(
                event.get("direction_confidence", 0.0)
            ),
            "zone_path": str(event.get("zone_path", "UNKNOWN")),
            "pre_zone_observations": int(
                event.get("pre_zone_observations", 0)
            ),
            "corridor_observations": int(
                event.get("corridor_observations", 0)
            ),
            "post_zone_observations": int(
                event.get("post_zone_observations", 0)
            ),
            "pre_zone_evidence": bool(event.get("pre_zone_evidence", False)),
            "corridor_evidence": bool(event.get("corridor_evidence", False)),
            "post_zone_evidence": bool(event.get("post_zone_evidence", False)),
            "crossing_detected": bool(
                pd.notna(event.get("crossing_frame", pd.NA))
            ),
            "crossing_frame": event.get("crossing_frame", pd.NA),
            "crossing_method": str(event.get("crossing_method", "")),
            "crossing_direction": str(
                event.get("line_direction", "UNKNOWN")
            ),
            "phase1_status": str(event.get("phase1_status", "FAIL")),
            "phase2_status": str(event.get("phase2_status", "FAIL")),
            "phase1_pass": phase1_pass,
            "phase2_pass": phase2_pass,
            "counted": bool(event.get("counted", False)),
            "failure_reason": ";".join(reasons),
        }

    # ==================================================================
    # BATCH
    # ==================================================================

    def process(
        self,
        trajectory: pd.DataFrame,
        identity_column: str = "crossing_id",
        return_diagnostics: bool = False,
    ) -> tuple[pd.DataFrame, pd.DataFrame] | tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        if identity_column not in trajectory.columns:
            raise ValueError(
                f"Trajectory missing identity column: {identity_column}"
            )

        if trajectory.empty:
            events_empty = pd.DataFrame(columns=self.EVENT_COLUMNS)
            audit_empty = pd.DataFrame(columns=self.AUDIT_COLUMNS)
            prepared_empty = pd.DataFrame(columns=list(trajectory.columns) + self.TRAJECTORY_COLUMNS)
            if return_diagnostics:
                return events_empty, audit_empty, prepared_empty
            return events_empty, audit_empty

        prepared = self.prepare(trajectory)

        events: list[dict] = []
        audits: list[dict] = []

        for identity_id, group in prepared.groupby(
            identity_column,
            sort=False,
        ):
            group = (
                group.sort_values("frame_id")
                .reset_index(drop=True)
            )

            event = self.detect_track_crossing(
                group,
                identity_id=int(identity_id),
            )

            if event is None:
                continue

            # Fill legacy track_ids field expected downstream.
            event["track_ids"] = str(
                group["track_id"].drop_duplicates().tolist()
            )

            # Remove private diagnostic keys from public event output.
            public_event = {
                key: event.get(key, pd.NA)
                for key in self.EVENT_COLUMNS
            }
            events.append(public_event)

            audits.append(
                self._build_track_audit_row(
                    group,
                    event,
                    int(identity_id),
                )
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
    # OPTIONAL HUMAN-READABLE REPORT
    # ==================================================================

    @staticmethod
    def print_phase_report(
        events_df: pd.DataFrame,
        audit_df: pd.DataFrame,
    ) -> None:
        """
        Print a compact notebook-friendly Phase 1/2 report.

        This function does not change any data. It exists specifically so
        Kaggle notebook users can judge PASS / REVIEW / FAIL before moving
        to the next development phase.
        """
        print("\n" + "=" * 88)
        print("TRAJECTORY + CROSSING CORRIDOR PHASE REPORT")
        print("=" * 88)

        if audit_df.empty:
            print("No tracks available for trajectory/corridor analysis.")
            return

        p1_pass = int((audit_df["phase1_status"] == "PASS").sum())
        p1_review = int((audit_df["phase1_status"] == "REVIEW").sum())
        p1_fail = int((audit_df["phase1_status"] == "FAIL").sum())

        p2_pass = int((audit_df["phase2_status"] == "PASS").sum())
        p2_review = int((audit_df["phase2_status"] == "REVIEW").sum())
        p2_fail = int((audit_df["phase2_status"] == "FAIL").sum())

        crossing_count = int(audit_df["crossing_detected"].sum())
        phase12_count = int(audit_df["counted"].sum())

        print(f"Tracks analysed                  : {len(audit_df):,}")
        print(f"Geometric crossings detected     : {crossing_count:,}")
        print()
        print(
            f"PHASE 1 Trajectory Engine        : "
            f"PASS={p1_pass:,} | REVIEW={p1_review:,} | FAIL={p1_fail:,}"
        )
        print(
            f"PHASE 2 Crossing Corridor        : "
            f"PASS={p2_pass:,} | REVIEW={p2_review:,} | FAIL={p2_fail:,}"
        )
        print(f"Tracks passing P1 + P2           : {phase12_count:,}")

        if len(audit_df):
            print()
            print("Zone evidence totals:")
            print(
                f"  PRE-zone evidence              : "
                f"{int(audit_df['pre_zone_evidence'].sum()):,}"
            )
            print(
                f"  CORRIDOR evidence              : "
                f"{int(audit_df['corridor_evidence'].sum()):,}"
            )
            print(
                f"  POST-zone evidence             : "
                f"{int(audit_df['post_zone_evidence'].sum()):,}"
            )

        failed = audit_df[~audit_df["phase1_pass"] | ~audit_df["phase2_pass"]]
        if not failed.empty:
            print()
            print("Top failure reasons:")
            print(
                failed["failure_reason"]
                .replace("", "NO_REASON")
                .value_counts()
                .head(10)
                .to_string()
            )

        print("=" * 88)


# ======================================================================
# BACKWARD-COMPATIBILITY ALIAS
# ======================================================================

__all__ = [
    "CrossingConfig",
    "RobustCrossingEngine",
]

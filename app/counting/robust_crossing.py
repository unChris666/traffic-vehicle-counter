from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import math
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CrossingConfig:
    """Phase 1 Trajectory Engine + Phase 2 Crossing Corridor.

    This module is intentionally identity-agnostic. Identity management is
    handled by CrossingIdentityEngine before this engine is called.

    Key design changes in v2:
      - NOT_CROSSING is a valid outcome, not a Phase-2 failure.
      - Direction is computed from signed distance / line normal.
      - Corridor membership uses hysteresis to prevent PRE/CORRIDOR chatter.
      - Crossing uses RAW trajectory geometry, so fast objects can cross the
        line between two observations without requiring a bbox observation
        inside the corridor.
      - FAST_CROSSING is explicitly reported when a valid line crossing skips
        the observed corridor.
    """

    # Geometry
    line_deadband_px: float = 8.0
    corridor_px: float = 45.0
    corridor_exit_px: float = 60.0

    # Trajectory
    max_trajectory_gap_sec: float = 1.50
    smoothing_alpha: float = 0.35
    velocity_window: int = 5
    min_direction_displacement_px: float = 8.0
    direction_window: int = 3
    max_velocity_px_per_frame: float = 80.0

    # Crossing validation
    min_track_observations: int = 2
    min_pre_zone_observations: int = 2
    min_corridor_observations: int = 1
    min_post_zone_observations: int = 1
    require_post_zone: bool = True

    # Fast crossing
    # A crossing with zero corridor observations is classified as FAST_CROSSING
    # when the raw segment intersects the line or the signed side changes.
    fast_crossing_allow_zero_corridor: bool = True

    # Direction confidence
    min_normal_displacement_px: float = 8.0
    min_direction_confidence: float = 0.50

    # Business labels. The geometrically meaningful fields are normal_direction
    # and side_transition. `direction` keeps the existing UI labels.
    positive_normal_label: str = "L→R"
    negative_normal_label: str = "R→L"

    vehicle_classes: tuple[str, ...] = (
        "motorcycle",
        "car",
        "truck",
        "bus",
    )


class RobustCrossingEngine:
    """Trajectory + crossing-corridor engine for pre-Phase-3 validation."""

    TRAJECTORY_COLUMNS = [
        "raw_x", "raw_y", "smooth_x", "smooth_y",
        "dx", "dy", "frame_delta", "time_delta_sec",
        "speed_px_per_frame", "velocity_normal_px_per_frame",
        "velocity_tangent_px_per_frame", "signed_distance_px",
        "line_distance_px", "raw_signed_distance_px",
        "raw_line_distance_px", "raw_side", "side", "zone",
        "direction_local", "normal_direction_local",
        "trajectory_continuity", "speed_anomaly", "trajectory_quality",
    ]

    EVENT_COLUMNS = [
        "crossing_id", "track_id", "track_ids", "first_frame", "last_frame",
        "crossing_frame", "crossing_time_sec", "crossing_x", "crossing_y",
        "direction", "normal_direction", "line_direction", "side_transition",
        "track_class", "track_class_ratio", "class_ambiguous",
        "line_distance_px", "previous_side", "current_side", "frame_gap",
        "crossing_method", "crossing_candidate_class", "fast_crossing",
        "track_observations", "crossing_index",
        "pre_zone_observations", "corridor_observations", "post_zone_observations",
        "pre_zone_evidence", "corridor_evidence", "post_zone_evidence",
        "zone_path", "zone_chatter_count", "trajectory_quality",
        "direction_confidence", "corridor_confidence",
        "phase1_status", "phase2_status", "counted",
    ]

    AUDIT_COLUMNS = [
        "crossing_id", "track_ids", "first_frame", "last_frame", "track_class",
        "track_observations", "first_side", "last_side",
        "first_distance_px", "last_distance_px", "min_distance_px",
        "max_speed_px_per_frame", "mean_speed_px_per_frame",
        "max_normal_velocity_px_per_frame", "mean_abs_normal_velocity_px_per_frame",
        "mean_abs_tangent_velocity_px_per_frame", "trajectory_direction",
        "normal_direction", "direction_confidence", "zone_path",
        "zone_chatter_count", "pre_zone_observations", "corridor_observations",
        "post_zone_observations", "pre_zone_evidence", "corridor_evidence",
        "post_zone_evidence", "crossing_detected", "crossing_candidate_class",
        "fast_crossing", "crossing_frame", "frame_gap", "crossing_method",
        "crossing_direction", "phase1_status", "phase2_status",
        "phase1_pass", "phase2_pass", "counted", "failure_reason",
    ]

    def __init__(self, *, line_x1: float, line_y1: float,
                 line_x2: float, line_y2: float, fps: float,
                 config: Optional[CrossingConfig] = None) -> None:
        if fps <= 0:
            raise ValueError(f"fps must be > 0, got {fps}")
        self.fps = float(fps)
        self.config = config or CrossingConfig()
        if not 0.0 < self.config.smoothing_alpha <= 1.0:
            raise ValueError("smoothing_alpha must be in (0, 1].")
        if self.config.corridor_px <= self.config.line_deadband_px:
            raise ValueError("corridor_px must be > line_deadband_px.")
        if self.config.corridor_exit_px < self.config.corridor_px:
            raise ValueError("corridor_exit_px must be >= corridor_px.")

        self.x1, self.y1 = float(line_x1), float(line_y1)
        self.x2, self.y2 = float(line_x2), float(line_y2)
        self.line_dx = self.x2 - self.x1
        self.line_dy = self.y2 - self.y1
        self.line_length = math.hypot(self.line_dx, self.line_dy)
        if self.line_length <= 0:
            raise ValueError("Counting line length must be > 0.")

        self.tangent_x = self.line_dx / self.line_length
        self.tangent_y = self.line_dy / self.line_length
        self.normal_x = -self.line_dy / self.line_length
        self.normal_y = self.line_dx / self.line_length

    # ------------------------------------------------------------------
    # Geometry
    # ------------------------------------------------------------------
    def signed_line_value(self, x: float, y: float) -> float:
        return self.line_dx * (y - self.y1) - self.line_dy * (x - self.x1)

    def signed_distance(self, x: float, y: float) -> float:
        return self.signed_line_value(x, y) / self.line_length

    def line_distance(self, x: float, y: float) -> float:
        return abs(self.signed_distance(x, y))

    def side(self, x: float, y: float) -> int:
        d = self.signed_distance(x, y)
        if abs(d) <= self.config.line_deadband_px:
            return 0
        return 1 if d > 0 else -1

    # ------------------------------------------------------------------
    # Smoothing
    # ------------------------------------------------------------------
    def _ema(self, values: np.ndarray) -> np.ndarray:
        if len(values) == 0:
            return values.copy()
        alpha = float(self.config.smoothing_alpha)
        out = np.empty_like(values, dtype=np.float64)
        out[0] = values[0]
        for i in range(1, len(values)):
            out[i] = alpha * values[i] + (1.0 - alpha) * out[i - 1]
        return out

    # ------------------------------------------------------------------
    # Segment / line intersection
    # ------------------------------------------------------------------
    @staticmethod
    def _orientation(ax, ay, bx, by, cx, cy) -> float:
        return (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)

    @staticmethod
    def _on_segment(ax, ay, bx, by, px, py) -> bool:
        eps = 1e-9
        return (
            min(ax, bx) - eps <= px <= max(ax, bx) + eps
            and min(ay, by) - eps <= py <= max(ay, by) + eps
        )

    def segments_intersect(self, p1, p2, q1, q2) -> bool:
        eps = 1e-9
        o1 = self._orientation(*p1, *p2, *q1)
        o2 = self._orientation(*p1, *p2, *q2)
        o3 = self._orientation(*q1, *q2, *p1)
        o4 = self._orientation(*q1, *q2, *p2)
        if (((o1 > eps and o2 < -eps) or (o1 < -eps and o2 > eps))
                and ((o3 > eps and o4 < -eps) or (o3 < -eps and o4 > eps))):
            return True
        if abs(o1) <= eps and self._on_segment(*p1, *p2, *q1): return True
        if abs(o2) <= eps and self._on_segment(*p1, *p2, *q2): return True
        if abs(o3) <= eps and self._on_segment(*q1, *q2, *p1): return True
        if abs(o4) <= eps and self._on_segment(*q1, *q2, *p2): return True
        return False

    def trajectory_intersects_line(self, previous_point, current_point) -> bool:
        return self.segments_intersect(
            previous_point, current_point, (self.x1, self.y1), (self.x2, self.y2)
        )

    def estimate_crossing_point(self, previous_point, current_point):
        d_prev = self.signed_distance(*previous_point)
        d_curr = self.signed_distance(*current_point)
        denom = d_prev - d_curr
        if abs(denom) < 1e-9:
            return current_point
        alpha = max(0.0, min(1.0, d_prev / denom))
        return (
            float(previous_point[0] + alpha * (current_point[0] - previous_point[0])),
            float(previous_point[1] + alpha * (current_point[1] - previous_point[1])),
        )

    # ------------------------------------------------------------------
    # Trajectory preparation
    # ------------------------------------------------------------------
    def prepare(self, trajectory: pd.DataFrame) -> pd.DataFrame:
        required = {
            "track_id", "frame_id", "timestamp_sec",
            "bottom_center_x", "bottom_center_y", "track_class",
        }
        missing = required - set(trajectory.columns)
        if missing:
            raise ValueError(f"Trajectory missing required columns: {sorted(missing)}")
        if trajectory.empty:
            return trajectory.copy()

        df = trajectory.copy().sort_values(["track_id", "frame_id"]).reset_index(drop=True)
        df["raw_x"] = pd.to_numeric(df["bottom_center_x"], errors="coerce")
        df["raw_y"] = pd.to_numeric(df["bottom_center_y"], errors="coerce")
        if df[["raw_x", "raw_y"]].isna().any().any():
            raise ValueError("Trajectory contains invalid bottom-center coordinates.")

        smooth_x = pd.Series(index=df.index, dtype=float)
        smooth_y = pd.Series(index=df.index, dtype=float)
        for _, g in df.groupby("track_id", sort=False):
            smooth_x.loc[g.index] = self._ema(g["raw_x"].to_numpy(float))
            smooth_y.loc[g.index] = self._ema(g["raw_y"].to_numpy(float))
        df["smooth_x"] = smooth_x
        df["smooth_y"] = smooth_y

        df["frame_delta"] = df.groupby("track_id")["frame_id"].diff()
        df["time_delta_sec"] = df.groupby("track_id")["timestamp_sec"].diff()
        valid = df["frame_delta"].notna() & (df["frame_delta"] > 0)
        fd = df["frame_delta"].where(valid, 1.0).fillna(1.0)

        df["dx"] = df.groupby("track_id")["smooth_x"].diff().where(valid, 0.0).fillna(0.0)
        df["dy"] = df.groupby("track_id")["smooth_y"].diff().where(valid, 0.0).fillna(0.0)
        df["speed_px_per_frame"] = np.hypot(df["dx"], df["dy"]) / fd

        vx_i = df["dx"] / fd
        vy_i = df["dy"] / fd
        w = max(1, int(self.config.velocity_window))
        df["_vx"] = vx_i
        df["_vy"] = vy_i
        df["_vx_med"] = df.groupby("track_id", sort=False)["_vx"].transform(
            lambda s: s.rolling(w, min_periods=1).median()
        )
        df["_vy_med"] = df.groupby("track_id", sort=False)["_vy"].transform(
            lambda s: s.rolling(w, min_periods=1).median()
        )

        df["velocity_normal_px_per_frame"] = (
            df["_vx_med"] * self.normal_x + df["_vy_med"] * self.normal_y
        )
        df["velocity_tangent_px_per_frame"] = (
            df["_vx_med"] * self.tangent_x + df["_vy_med"] * self.tangent_y
        )

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
            [df["raw_line_distance_px"] <= self.config.line_deadband_px,
             df["raw_signed_distance_px"] > 0],
            [0, 1], default=-1,
        ).astype(int)
        df["side"] = np.select(
            [df["line_distance_px"] <= self.config.line_deadband_px,
             df["signed_distance_px"] > 0],
            [0, 1], default=-1,
        ).astype(int)

        max_gap = self.config.max_trajectory_gap_sec * self.fps
        df["trajectory_continuity"] = (
            df["frame_delta"].isna() | (df["frame_delta"] <= max_gap)
        ).astype(float)
        df["speed_anomaly"] = (
            df["speed_px_per_frame"] > self.config.max_velocity_px_per_frame
        ).astype(float)
        df["trajectory_quality"] = (
            0.45 * df["trajectory_continuity"]
            + 0.35 * (1.0 - df["speed_anomaly"])
            + 0.20 * np.isfinite(df["velocity_normal_px_per_frame"]).astype(float)
        ).clip(0.0, 1.0)

        df["zone"] = self._assign_zones_hysteresis(df)
        df["direction_local"], df["normal_direction_local"] = self._local_directions(df)

        return df.drop(columns=["_vx", "_vy", "_vx_med", "_vy_med"], errors="ignore")

    def _assign_zones_hysteresis(self, df: pd.DataFrame) -> pd.Series:
        zones = pd.Series("UNKNOWN", index=df.index, dtype="object")
        enter = float(self.config.corridor_px)
        exit_ = float(self.config.corridor_exit_px)

        for _, g in df.groupby("track_id", sort=False):
            stable = g.loc[g["raw_side"] != 0, "raw_side"]
            if stable.empty:
                continue
            initial_side = int(stable.iloc[0])
            state = "PRE"
            for idx in g.index:
                d = float(df.at[idx, "raw_line_distance_px"])
                s = int(df.at[idx, "raw_side"])

                if state == "PRE":
                    if s != 0 and s != initial_side and d >= exit_:
                        state = "POST"
                    elif d <= enter:
                        state = "CORRIDOR"
                    else:
                        state = "PRE"
                elif state == "CORRIDOR":
                    # Only leave corridor once the object is clearly on the
                    # opposite side and outside the wider exit boundary.
                    if s != 0 and s != initial_side and d >= exit_:
                        state = "POST"
                    else:
                        state = "CORRIDOR"
                elif state == "POST":
                    # Do not bounce back to PRE/CORRIDOR because of line jitter.
                    state = "POST"

                zones.at[idx] = state

        return zones

    def _local_directions(self, df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
        business = pd.Series("UNKNOWN", index=df.index, dtype="object")
        normal = pd.Series("UNKNOWN", index=df.index, dtype="object")
        w = max(1, int(self.config.direction_window))

        for _, g in df.groupby("track_id", sort=False):
            idxs = list(g.index)
            for local_i, idx in enumerate(idxs):
                start = max(0, local_i - w)
                end = min(len(g) - 1, local_i + w)
                if end <= start:
                    continue
                a, b = g.iloc[start], g.iloc[end]
                normal_delta = float(b["raw_signed_distance_px"] - a["raw_signed_distance_px"])
                tangent_delta = float(
                    (b["raw_x"] - a["raw_x"]) * self.tangent_x
                    + (b["raw_y"] - a["raw_y"]) * self.tangent_y
                )
                if abs(normal_delta) >= self.config.min_normal_displacement_px:
                    normal.at[idx] = (
                        self.config.positive_normal_label
                        if normal_delta > 0
                        else self.config.negative_normal_label
                    )
                if abs(tangent_delta) >= self.config.min_direction_displacement_px:
                    business.at[idx] = "L→R" if tangent_delta > 0 else "R→L"
        return business, normal

    # ------------------------------------------------------------------
    # Crossing candidate logic
    # ------------------------------------------------------------------
    def _side_transition(self, previous_side: int, current_side: int) -> str:
        if previous_side == -1 and current_side == 1:
            return "side_-1_to_+1"
        if previous_side == 1 and current_side == -1:
            return "side_+1_to_-1"
        return "UNKNOWN"

    def _pair_crossing(self, previous: pd.Series, current: pd.Series) -> dict:
        p1 = (float(previous["raw_x"]), float(previous["raw_y"]))
        p2 = (float(current["raw_x"]), float(current["raw_y"]))
        prev_side = int(previous["raw_side"])
        curr_side = int(current["raw_side"])
        frame_gap = int(current["frame_id"] - previous["frame_id"])

        line_intersection = self.trajectory_intersects_line(p1, p2)
        side_change = prev_side != 0 and curr_side != 0 and prev_side != curr_side
        corridor_support = min(
            float(previous["raw_line_distance_px"]),
            float(current["raw_line_distance_px"]),
        ) <= self.config.corridor_px

        crossing = bool(line_intersection or side_change)
        skipped_corridor = bool(
            crossing
            and self.config.fast_crossing_allow_zero_corridor
            and not corridor_support
        )

        # Swept corridor: the segment may cross the corridor even if neither
        # endpoint has a bbox center inside it.
        d1 = abs(float(previous["raw_signed_distance_px"]))
        d2 = abs(float(current["raw_signed_distance_px"]))
        swept_corridor = crossing and (min(d1, d2) <= self.config.corridor_px or line_intersection)

        if crossing and skipped_corridor:
            candidate_class = "FAST_CROSSING"
        elif crossing:
            candidate_class = "TRUE_CROSSING"
        elif corridor_support:
            candidate_class = "NEAR_LINE"
        else:
            candidate_class = "NOT_CROSSING"

        return {
            "crossing": crossing,
            "line_intersection": line_intersection,
            "side_change": side_change,
            "corridor_support": corridor_support,
            "swept_corridor": bool(swept_corridor),
            "skipped_corridor": skipped_corridor,
            "candidate_class": candidate_class,
            "previous_side": prev_side,
            "current_side": curr_side,
            "frame_gap": frame_gap,
        }

    def _find_crossing(self, group: pd.DataFrame) -> tuple[int | None, dict | None]:
        for i in range(1, len(group)):
            previous = group.iloc[i - 1]
            current = group.iloc[i]
            gap = int(current["frame_id"] - previous["frame_id"])
            if gap <= 0 or gap > self.config.max_trajectory_gap_sec * self.fps:
                continue
            result = self._pair_crossing(previous, current)
            if result["crossing"]:
                return i, result
        return None, None

    def _zone_summary(self, group: pd.DataFrame) -> dict:
        zones = group["zone"].astype(str).tolist()
        compact = []
        for zone in zones:
            if not compact or compact[-1] != zone:
                compact.append(zone)

        chatter = sum(
            1 for a, b in zip(compact, compact[1:])
            if {a, b} == {"PRE", "CORRIDOR"}
        )
        counts = group["zone"].value_counts()
        return {
            "zone_path": " → ".join(compact),
            "zone_chatter_count": int(chatter),
            "pre_zone_observations": int(counts.get("PRE", 0)),
            "corridor_observations": int(counts.get("CORRIDOR", 0)),
            "post_zone_observations": int(counts.get("POST", 0)),
        }

    def _direction_at_crossing(self, group: pd.DataFrame, crossing_index: int) -> tuple[str, str, float]:
        w = max(1, int(self.config.direction_window))
        start = max(0, crossing_index - w)
        end = min(len(group) - 1, crossing_index + w)
        if end <= start:
            return "UNKNOWN", "UNKNOWN", 0.0

        before = group.iloc[start]
        after = group.iloc[end]
        normal_delta = float(after["raw_signed_distance_px"] - before["raw_signed_distance_px"])
        displacement = abs(normal_delta)

        samples = group.iloc[start:end + 1]
        normal_v = pd.to_numeric(samples["velocity_normal_px_per_frame"], errors="coerce").dropna()
        if displacement < self.config.min_normal_displacement_px:
            return "UNKNOWN", "UNKNOWN", 0.0

        normal_direction = (
            self.config.positive_normal_label
            if normal_delta > 0
            else self.config.negative_normal_label
        )

        same_sign_fraction = float(
            (np.sign(normal_v) == (1 if normal_delta > 0 else -1)).mean()
        ) if not normal_v.empty else 0.0
        displacement_score = min(1.0, displacement / (3.0 * self.config.min_normal_displacement_px))
        confidence = float(np.clip(0.65 * displacement_score + 0.35 * same_sign_fraction, 0.0, 1.0))

        return normal_direction, normal_direction, confidence

    def _phase1_status(self, group: pd.DataFrame) -> tuple[str, bool, str]:
        if len(group) < self.config.min_track_observations:
            return "FAIL", False, "insufficient_track_observations"
        if not np.isfinite(group["speed_px_per_frame"].to_numpy()).all():
            return "FAIL", False, "non_finite_velocity"
        if not np.isfinite(group["signed_distance_px"].to_numpy()).all():
            return "FAIL", False, "non_finite_signed_distance"
        if not np.isfinite(group["velocity_normal_px_per_frame"].to_numpy()).all():
            return "FAIL", False, "non_finite_normal_velocity"

        continuity = float(group["trajectory_continuity"].mean())
        anomaly_fraction = float(group["speed_anomaly"].mean())
        if continuity < 0.80:
            return "REVIEW", False, "trajectory_has_large_gaps"
        if anomaly_fraction > 0.25:
            return "REVIEW", False, "frequent_speed_anomalies"
        return "PASS", True, ""

    def _zone_evidence(self, group: pd.DataFrame, crossing_index: int | None) -> dict:
        zone = self._zone_summary(group)
        if crossing_index is None:
            pre_n = zone["pre_zone_observations"]
            corridor_n = zone["corridor_observations"]
            post_n = zone["post_zone_observations"]
        else:
            before = group.iloc[:crossing_index + 1]
            after = group.iloc[crossing_index:]
            pre_n = int((before["zone"] == "PRE").sum())
            corridor_n = int((before["zone"] == "CORRIDOR").sum())
            post_n = int((after["zone"] == "POST").sum())

        zone["pre_zone_evidence"] = pre_n >= self.config.min_pre_zone_observations
        zone["corridor_evidence"] = corridor_n >= self.config.min_corridor_observations
        zone["post_zone_evidence"] = post_n >= self.config.min_post_zone_observations
        return zone

    def _phase2_status(self, *, candidate_class: str, crossing: bool, zone: dict,
                       normal_direction: str, direction_confidence: float) -> tuple[str, bool, str]:
        if not crossing:
            return "NOT_CROSSING", False, ""
        if normal_direction == "UNKNOWN" or direction_confidence < self.config.min_direction_confidence:
            return "REVIEW", False, "low_crossing_direction_confidence"

        # Fast crossing is valid geometric evidence even when there is no
        # actual bbox-center observation inside the corridor.
        corridor_ok = zone["corridor_evidence"] or candidate_class == "FAST_CROSSING"
        if not corridor_ok:
            return "REVIEW", False, "insufficient_corridor_evidence"

        if not zone["pre_zone_evidence"]:
            return "REVIEW", False, "insufficient_pre_zone_evidence"
        if self.config.require_post_zone and not zone["post_zone_evidence"]:
            return "REVIEW", False, "insufficient_post_zone_evidence"
        return "PASS", True, ""

    def detect_track_crossing(self, group: pd.DataFrame, *, identity_id: int | None = None) -> Optional[dict]:
        group = group.sort_values("frame_id").reset_index(drop=True).copy()
        if len(group) < self.config.min_track_observations:
            return None
        if identity_id is None:
            identity_id = int(group.iloc[0]["track_id"])

        track_class = str(group.iloc[0]["track_class"])
        first_frame = int(group["frame_id"].min())
        last_frame = int(group["frame_id"].max())
        phase1_status, phase1_pass, phase1_reason = self._phase1_status(group)
        crossing_index, pair = self._find_crossing(group)
        zone = self._zone_evidence(group, crossing_index)

        if crossing_index is None or pair is None:
            phase2_status, phase2_pass, phase2_reason = self._phase2_status(
                candidate_class="NOT_CROSSING", crossing=False, zone=zone,
                normal_direction="UNKNOWN", direction_confidence=0.0,
            )
            candidate_class = "NEAR_LINE" if zone["corridor_observations"] else "NOT_CROSSING"
            return self._build_event(
                group, identity_id, track_class, zone, phase1_status, phase1_pass,
                phase1_reason, phase2_status, phase2_pass, phase2_reason,
                crossing_index=None, pair=None, candidate_class=candidate_class,
                normal_direction="UNKNOWN", direction_confidence=0.0,
            )

        normal_direction, line_direction, direction_confidence = self._direction_at_crossing(group, crossing_index)
        phase2_status, phase2_pass, phase2_reason = self._phase2_status(
            candidate_class=pair["candidate_class"], crossing=True, zone=zone,
            normal_direction=normal_direction, direction_confidence=direction_confidence,
        )

        return self._build_event(
            group, identity_id, track_class, zone, phase1_status, phase1_pass,
            phase1_reason, phase2_status, phase2_pass, phase2_reason,
            crossing_index=crossing_index, pair=pair,
            candidate_class=pair["candidate_class"],
            normal_direction=normal_direction,
            direction_confidence=direction_confidence,
            line_direction=line_direction,
        )

    def _build_event(self, group, identity_id, track_class, zone,
                     phase1_status, phase1_pass, phase1_reason,
                     phase2_status, phase2_pass, phase2_reason,
                     crossing_index, pair, candidate_class,
                     normal_direction, direction_confidence,
                     line_direction="UNKNOWN") -> dict:
        first_frame = int(group["frame_id"].min())
        last_frame = int(group["frame_id"].max())
        max_speed = float(group["speed_px_per_frame"].max())
        mean_speed = float(group["speed_px_per_frame"].mean())
        max_normal = float(group["velocity_normal_px_per_frame"].abs().max())
        mean_abs_normal = float(group["velocity_normal_px_per_frame"].abs().mean())
        mean_abs_tangent = float(group["velocity_tangent_px_per_frame"].abs().mean())
        trajectory_direction = self._infer_track_direction(group)

        if crossing_index is None or pair is None:
            crossing_frame = pd.NA
            crossing_time = pd.NA
            crossing_x = pd.NA
            crossing_y = pd.NA
            previous_side = pd.NA
            current_side = pd.NA
            frame_gap = pd.NA
            method = ""
            side_transition = "UNKNOWN"
            direction = trajectory_direction
            fast_crossing = False
            line_distance = float(group["raw_line_distance_px"].min())
        else:
            previous = group.iloc[crossing_index - 1]
            current = group.iloc[crossing_index]
            p1 = (float(previous["raw_x"]), float(previous["raw_y"]))
            p2 = (float(current["raw_x"]), float(current["raw_y"]))
            cross_point = self.estimate_crossing_point(p1, p2)
            crossing_frame = int(current["frame_id"])
            crossing_time = float(current["timestamp_sec"])
            crossing_x, crossing_y = cross_point
            previous_side = pair["previous_side"] if pair["previous_side"] != 0 else None
            current_side = pair["current_side"] if pair["current_side"] != 0 else None
            frame_gap = pair["frame_gap"]
            side_transition = self._side_transition(previous_side or 0, current_side or 0)
            methods = []
            if pair["side_change"]: methods.append("side_change")
            if pair["line_intersection"]: methods.append("line_intersection")
            if pair["corridor_support"]: methods.append("corridor_support")
            if pair["skipped_corridor"]: methods.append("corridor_skipped")
            method = "+".join(methods)
            direction = normal_direction if normal_direction != "UNKNOWN" else trajectory_direction
            fast_crossing = bool(pair["skipped_corridor"])
            line_distance = min(float(previous["raw_line_distance_px"]), float(current["raw_line_distance_px"]))

        corridor_confidence = float(np.clip(
            zone["corridor_observations"] / max(1, self.config.min_corridor_observations), 0.0, 1.0
        ))
        counted = bool(phase1_pass and phase2_pass)

        return {
            "crossing_id": int(identity_id),
            "track_id": int(group.iloc[-1]["track_id"]),
            "track_ids": "",
            "first_frame": first_frame,
            "last_frame": last_frame,
            "crossing_frame": crossing_frame,
            "crossing_time_sec": crossing_time,
            "crossing_x": crossing_x,
            "crossing_y": crossing_y,
            "direction": direction,
            "normal_direction": normal_direction,
            "line_direction": line_direction,
            "side_transition": side_transition,
            "track_class": track_class,
            "track_class_ratio": float(group.iloc[0].get("track_class_ratio", np.nan)),
            "class_ambiguous": bool(group.iloc[0].get("class_ambiguous", False)),
            "line_distance_px": line_distance,
            "previous_side": previous_side,
            "current_side": current_side,
            "frame_gap": frame_gap,
            "crossing_method": method,
            "crossing_candidate_class": candidate_class,
            "fast_crossing": fast_crossing,
            "track_observations": len(group),
            "crossing_index": crossing_index if crossing_index is not None else pd.NA,
            **zone,
            "trajectory_quality": round(float(group["trajectory_quality"].mean()), 4),
            "direction_confidence": round(float(direction_confidence), 4),
            "corridor_confidence": round(corridor_confidence, 4),
            "phase1_status": phase1_status,
            "phase2_status": phase2_status,
            "counted": counted,
            "_phase1_pass": phase1_pass,
            "_phase2_pass": phase2_pass,
            "_phase1_reason": phase1_reason,
            "_phase2_reason": phase2_reason,
            "_max_speed": max_speed,
            "_mean_speed": mean_speed,
            "_max_normal": max_normal,
            "_mean_abs_normal": mean_abs_normal,
            "_mean_abs_tangent": mean_abs_tangent,
            "_trajectory_direction": trajectory_direction,
        }

    def _infer_track_direction(self, group: pd.DataFrame) -> str:
        if len(group) < 2:
            return "UNKNOWN"
        tangent_delta = (
            float(group.iloc[-1]["raw_x"] - group.iloc[0]["raw_x"]) * self.tangent_x
            + float(group.iloc[-1]["raw_y"] - group.iloc[0]["raw_y"]) * self.tangent_y
        )
        if abs(tangent_delta) < self.config.min_direction_displacement_px:
            return "UNKNOWN"
        return "L→R" if tangent_delta > 0 else "R→L"

    # ------------------------------------------------------------------
    # Audit
    # ------------------------------------------------------------------
    def _build_track_audit_row(self, group: pd.DataFrame, event: dict, identity_id: int) -> dict:
        stable = group.loc[group["raw_side"] != 0, "raw_side"]
        first_side = int(stable.iloc[0]) if not stable.empty else 0
        last_side = int(stable.iloc[-1]) if not stable.empty else 0
        reasons = []
        if event.get("_phase1_reason"):
            reasons.append(f"P1:{event['_phase1_reason']}")
        if event.get("_phase2_reason"):
            reasons.append(f"P2:{event['_phase2_reason']}")

        crossing_detected = pd.notna(event.get("crossing_frame", pd.NA))
        return {
            "crossing_id": int(identity_id),
            "track_ids": str(group["track_id"].drop_duplicates().tolist()),
            "first_frame": int(group["frame_id"].min()),
            "last_frame": int(group["frame_id"].max()),
            "track_class": str(event.get("track_class", "unknown")),
            "track_observations": int(len(group)),
            "first_side": first_side,
            "last_side": last_side,
            "first_distance_px": float(group.iloc[0]["raw_line_distance_px"]),
            "last_distance_px": float(group.iloc[-1]["raw_line_distance_px"]),
            "min_distance_px": float(group["raw_line_distance_px"].min()),
            "max_speed_px_per_frame": float(event.get("_max_speed", 0.0)),
            "mean_speed_px_per_frame": float(event.get("_mean_speed", 0.0)),
            "max_normal_velocity_px_per_frame": float(event.get("_max_normal", 0.0)),
            "mean_abs_normal_velocity_px_per_frame": float(event.get("_mean_abs_normal", 0.0)),
            "mean_abs_tangent_velocity_px_per_frame": float(event.get("_mean_abs_tangent", 0.0)),
            "trajectory_direction": str(event.get("_trajectory_direction", "UNKNOWN")),
            "normal_direction": str(event.get("normal_direction", "UNKNOWN")),
            "direction_confidence": float(event.get("direction_confidence", 0.0)),
            "zone_path": str(event.get("zone_path", "UNKNOWN")),
            "zone_chatter_count": int(event.get("zone_chatter_count", 0)),
            "pre_zone_observations": int(event.get("pre_zone_observations", 0)),
            "corridor_observations": int(event.get("corridor_observations", 0)),
            "post_zone_observations": int(event.get("post_zone_observations", 0)),
            "pre_zone_evidence": bool(event.get("pre_zone_evidence", False)),
            "corridor_evidence": bool(event.get("corridor_evidence", False)),
            "post_zone_evidence": bool(event.get("post_zone_evidence", False)),
            "crossing_detected": bool(crossing_detected),
            "crossing_candidate_class": str(event.get("crossing_candidate_class", "NOT_CROSSING")),
            "fast_crossing": bool(event.get("fast_crossing", False)),
            "crossing_frame": event.get("crossing_frame", pd.NA),
            "frame_gap": event.get("frame_gap", pd.NA),
            "crossing_method": str(event.get("crossing_method", "")),
            "crossing_direction": str(event.get("normal_direction", "UNKNOWN")),
            "phase1_status": str(event.get("phase1_status", "FAIL")),
            "phase2_status": str(event.get("phase2_status", "NOT_CROSSING")),
            "phase1_pass": bool(event.get("_phase1_pass", False)),
            "phase2_pass": bool(event.get("_phase2_pass", False)),
            "counted": bool(event.get("counted", False)),
            "failure_reason": ";".join(reasons),
        }

    # ------------------------------------------------------------------
    # Batch
    # ------------------------------------------------------------------
    def process(self, trajectory: pd.DataFrame, identity_column: str = "crossing_id",
                return_diagnostics: bool = False):
        if identity_column not in trajectory.columns:
            raise ValueError(f"Trajectory missing identity column: {identity_column}")
        if trajectory.empty:
            e = pd.DataFrame(columns=self.EVENT_COLUMNS)
            a = pd.DataFrame(columns=self.AUDIT_COLUMNS)
            p = trajectory.copy()
            if return_diagnostics:
                return e, a, p
            return e, a

        prepared = self.prepare(trajectory)
        events, audits = [], []
        for identity_id, group in prepared.groupby(identity_column, sort=False):
            group = group.sort_values("frame_id").reset_index(drop=True)
            event = self.detect_track_crossing(group, identity_id=int(identity_id))
            if event is None:
                continue
            event["track_ids"] = str(group["track_id"].drop_duplicates().tolist())
            events.append({k: event.get(k, pd.NA) for k in self.EVENT_COLUMNS})
            audits.append(self._build_track_audit_row(group, event, int(identity_id)))

        events_df = pd.DataFrame(events, columns=self.EVENT_COLUMNS)
        audit_df = pd.DataFrame(audits, columns=self.AUDIT_COLUMNS)
        if return_diagnostics:
            return events_df, audit_df, prepared
        return events_df, audit_df

    @staticmethod
    def print_phase_report(events_df: pd.DataFrame, audit_df: pd.DataFrame) -> None:
        print("\n" + "=" * 96)
        print("PHASE 1/2 TRAJECTORY + CROSSING CANDIDATE AUDIT v2")
        print("=" * 96)
        if audit_df.empty:
            print("No tracks available.")
            return

        total = len(audit_df)
        p1p = int((audit_df.phase1_status == "PASS").sum())
        p1r = int((audit_df.phase1_status == "REVIEW").sum())
        p1f = int((audit_df.phase1_status == "FAIL").sum())
        not_cross = int((audit_df.crossing_candidate_class == "NOT_CROSSING").sum())
        near_line = int((audit_df.crossing_candidate_class == "NEAR_LINE").sum())
        true_cross = int((audit_df.crossing_candidate_class == "TRUE_CROSSING").sum())
        fast_cross = int((audit_df.crossing_candidate_class == "FAST_CROSSING").sum())
        p2p = int((audit_df.phase2_status == "PASS").sum())
        p2r = int((audit_df.phase2_status == "REVIEW").sum())
        p2f = int((audit_df.phase2_status == "FAIL").sum())
        direction_known = int((audit_df.normal_direction != "UNKNOWN").sum())
        chatter = int((audit_df.zone_chatter_count > 0).sum())

        print(f"Tracks analysed                    : {total:,}")
        print(f"NOT_CROSSING tracks                 : {not_cross:,}")
        print(f"NEAR_LINE tracks                    : {near_line:,}")
        print(f"TRUE_CROSSING candidates            : {true_cross:,}")
        print(f"FAST_CROSSING candidates            : {fast_cross:,}")
        print()
        print(f"PHASE 1                            : PASS={p1p:,} | REVIEW={p1r:,} | FAIL={p1f:,}")
        print(f"PHASE 2                            : PASS={p2p:,} | REVIEW={p2r:,} | FAIL={p2f:,} | NOT_CROSSING={not_cross:,}")
        print(f"Normal direction known              : {direction_known:,}/{true_cross + fast_cross:,}")
        print(f"Zone chatter tracks                 : {chatter:,}")
        print()
        if fast_cross:
            fast = audit_df[audit_df.crossing_candidate_class == "FAST_CROSSING"]
            print("FAST CROSSING DIAGNOSTIC:")
            print(f"  Zero/low corridor evidence        : {int((fast.corridor_observations == 0).sum()):,}")
            print(f"  Mean frame gap at crossing         : {pd.to_numeric(fast.frame_gap, errors='coerce').mean():.2f}")
            print(f"  Mean max speed (px/frame)          : {fast.max_speed_px_per_frame.mean():.2f}")

        review = audit_df[audit_df.phase2_status == "REVIEW"]
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
        print(audit_df["crossing_candidate_class"].value_counts().to_string())
        print("=" * 96)


__all__ = ["CrossingConfig", "RobustCrossingEngine"]
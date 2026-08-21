from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import math
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CrossingConfig:
    """
    Configuration for geometric crossing detection.

    This module is deliberately identity-agnostic:
    one physical identity should already be represented by
    one `counting_id`/`crossing_id` before detection.
    """

    line_deadband_px: float = 20.0
    corridor_px: float = 45.0
    max_trajectory_gap_sec: float = 1.50

    min_direction_displacement_px: float = 8.0
    direction_window: int = 3

    min_track_observations: int = 2

    vehicle_classes: tuple[str, ...] = (
        "motorcycle",
        "car",
        "truck",
        "bus",
    )


class RobustCrossingEngine:
    """
    Geometric crossing detector.

    Detection mechanisms:
      1. side transition
      2. segment / counting-line intersection
      3. corridor-assisted crossing

    Important:
      - No vehicle deduplication happens here.
      - No physical-identity matching happens here.
      - Each input identity can generate at most one crossing event.
    """

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
            raise ValueError(
                f"fps must be > 0, got {fps}"
            )

        self.fps = float(fps)
        self.config = config or CrossingConfig()

        self.x1 = float(line_x1)
        self.y1 = float(line_y1)
        self.x2 = float(line_x2)
        self.y2 = float(line_y2)

        self.line_dx = self.x2 - self.x1
        self.line_dy = self.y2 - self.y1

        self.line_length = math.hypot(
            self.line_dx,
            self.line_dy,
        )

        if self.line_length <= 0:
            raise ValueError(
                "Counting line length must be > 0."
            )

    # ------------------------------------------------------------------
    # LINE GEOMETRY
    # ------------------------------------------------------------------

    def signed_line_value(
        self,
        x: float,
        y: float,
    ) -> float:
        return (
            self.line_dx * (y - self.y1)
            - self.line_dy * (x - self.x1)
        )

    def line_distance(
        self,
        x: float,
        y: float,
    ) -> float:
        return (
            abs(
                self.signed_line_value(x, y)
            )
            / self.line_length
        )

    def side(
        self,
        x: float,
        y: float,
    ) -> int:
        distance = self.line_distance(x, y)

        if distance <= self.config.line_deadband_px:
            return 0

        return (
            1
            if self.signed_line_value(x, y) > 0
            else -1
        )

    # ------------------------------------------------------------------
    # SEGMENT INTERSECTION
    # ------------------------------------------------------------------

    @staticmethod
    def _orientation(
        ax: float,
        ay: float,
        bx: float,
        by: float,
        cx: float,
        cy: float,
    ) -> float:
        return (
            (bx - ax) * (cy - ay)
            - (by - ay) * (cx - ax)
        )

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
            and
            min(ay, by) - eps <= py <= max(ay, by) + eps
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
            and
            ((o3 > eps and o4 < -eps) or (o3 < -eps and o4 > eps))
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

        v_prev = self.signed_line_value(x_prev, y_prev)
        v_curr = self.signed_line_value(x_curr, y_curr)

        denominator = v_prev - v_curr

        if abs(denominator) < 1e-9:
            return current_point

        alpha = v_prev / denominator
        alpha = max(0.0, min(1.0, alpha))

        return (
            float(x_prev + alpha * (x_curr - x_prev)),
            float(y_prev + alpha * (y_curr - y_prev)),
        )

    # ------------------------------------------------------------------
    # DIRECTION
    # ------------------------------------------------------------------

    def side_transition(
        self,
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
        window = max(1, int(self.config.direction_window))

        start_idx = max(
            0,
            crossing_index - window,
        )

        end_idx = min(
            len(group) - 1,
            crossing_index + window,
        )

        before = group.iloc[
            start_idx : crossing_index + 1
        ]

        after = group.iloc[
            crossing_index : end_idx + 1
        ]

        if before.empty or after.empty:
            return "UNKNOWN"

        dx = (
            float(after.iloc[-1]["bottom_center_x"])
            -
            float(before.iloc[0]["bottom_center_x"])
        )

        dy = (
            float(after.iloc[-1]["bottom_center_y"])
            -
            float(before.iloc[0]["bottom_center_y"])
        )

        displacement = math.hypot(dx, dy)

        if displacement < self.config.min_direction_displacement_px:
            return "UNKNOWN"

        return "L→R" if dx > 0 else "R→L"

    def direction(
        self,
        group: pd.DataFrame,
        crossing_index: int,
        previous_side: int,
        current_side: int,
    ) -> tuple[str, str]:
        """
        Returns:
          direction        -> L→R / R→L / UNKNOWN
          side_transition  -> side_-1_to_+1 / side_+1_to_-1 / UNKNOWN
        """

        side_transition = self.side_transition(
            previous_side,
            current_side,
        )

        motion_direction = self.estimate_motion_direction(
            group,
            crossing_index,
        )

        # Motion direction is the display/business direction.
        # Side transition remains available for geometry auditing.
        if motion_direction != "UNKNOWN":
            return motion_direction, side_transition

        if side_transition == "side_-1_to_+1":
            return "L→R", side_transition

        if side_transition == "side_+1_to_-1":
            return "R→L", side_transition

        return "UNKNOWN", side_transition

    # ------------------------------------------------------------------
    # SINGLE IDENTITY CROSSING
    # ------------------------------------------------------------------

    def detect_track_crossing(
        self,
        group: pd.DataFrame,
        *,
        identity_id: int | None = None,
    ) -> Optional[dict]:
        group = (
            group
            .sort_values("frame_id")
            .reset_index(drop=True)
            .copy()
        )

        if len(group) < self.config.min_track_observations:
            return None

        if identity_id is None:
            identity_id = int(group.iloc[0]["track_id"])

        track_class = str(
            group.iloc[0]["track_class"]
        )

        first_frame = int(
            group["frame_id"].min()
        )

        last_frame = int(
            group["frame_id"].max()
        )

        # Geometry for all observations.
        if "line_value" not in group.columns:
            group["line_value"] = (
                group.apply(
                    lambda row: self.signed_line_value(
                        float(row["bottom_center_x"]),
                        float(row["bottom_center_y"]),
                    ),
                    axis=1,
                )
            )

        if "line_distance_px" not in group.columns:
            group["line_distance_px"] = (
                group["line_value"].abs()
                / self.line_length
            )

        if "side" not in group.columns:
            group["side"] = group.apply(
                lambda row: self.side(
                    float(row["bottom_center_x"]),
                    float(row["bottom_center_y"]),
                ),
                axis=1,
            )

        max_gap_frames = (
            self.config.max_trajectory_gap_sec
            * self.fps
        )

        for i in range(1, len(group)):
            previous = group.iloc[i - 1]
            current = group.iloc[i]

            previous_frame = int(previous["frame_id"])
            current_frame = int(current["frame_id"])

            frame_gap = (
                current_frame
                - previous_frame
            )

            if frame_gap <= 0 or frame_gap > max_gap_frames:
                continue

            previous_point = (
                float(previous["bottom_center_x"]),
                float(previous["bottom_center_y"]),
            )

            current_point = (
                float(current["bottom_center_x"]),
                float(current["bottom_center_y"]),
            )

            previous_side = int(previous["side"])
            current_side = int(current["side"])

            previous_distance = float(
                previous["line_distance_px"]
            )
            current_distance = float(
                current["line_distance_px"]
            )

            side_change = (
                previous_side != 0
                and
                current_side != 0
                and
                previous_side != current_side
            )

            line_intersection = (
                self.trajectory_intersects_line(
                    previous_point,
                    current_point,
                )
            )

            # Corridor is a supporting signal, never a standalone
            # "count because near line" trigger.
            corridor_support = (
                min(
                    previous_distance,
                    current_distance,
                )
                <= self.config.corridor_px
            )

            # Crossing is strongest when the actual trajectory
            # intersects the line, or the side changes.
            crossed = (
                side_change
                or line_intersection
            )

            # If both observations are inside the corridor but the
            # segment did not intersect, do not count. This avoids
            # counting vehicles travelling parallel to the line.
            if not crossed:
                continue

            direction, side_transition = self.direction(
                group,
                i,
                previous_side,
                current_side,
            )

            if direction == "UNKNOWN":
                continue

            if line_intersection:
                crossing_x, crossing_y = (
                    self.estimate_crossing_point(
                        previous_point,
                        current_point,
                    )
                )
            else:
                crossing_x = float(
                    current["bottom_center_x"]
                )
                crossing_y = float(
                    current["bottom_center_y"]
                )

            methods: list[str] = []

            if side_change:
                methods.append("side_change")

            if line_intersection:
                methods.append("line_intersection")

            if corridor_support:
                methods.append("corridor_support")

            return {
                "crossing_id": int(identity_id),
                "track_id": int(current["track_id"]),
                "track_ids": "",
                "first_frame": first_frame,
                "last_frame": last_frame,
                "crossing_frame": current_frame,
                "crossing_time_sec": float(
                    current["timestamp_sec"]
                ),
                "crossing_x": crossing_x,
                "crossing_y": crossing_y,
                "direction": direction,
                "side_transition": side_transition,
                "track_class": track_class,
                "track_class_ratio": float(
                    current.get(
                        "track_class_ratio",
                        np.nan,
                    )
                ),
                "class_ambiguous": bool(
                    current.get(
                        "class_ambiguous",
                        False,
                    )
                ),
                "line_distance_px": min(
                    previous_distance,
                    current_distance,
                ),
                "previous_side": (
                    previous_side
                    if previous_side != 0
                    else None
                ),
                "current_side": (
                    current_side
                    if current_side != 0
                    else None
                ),
                "frame_gap": frame_gap,
                "crossing_method": "+".join(methods),
                "track_observations": len(group),
                "counted": True,
            }

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
            "direction": self._infer_track_direction(group),
            "side_transition": "UNKNOWN",
            "track_class": track_class,
            "track_class_ratio": float(
                group.iloc[0].get(
                    "track_class_ratio",
                    np.nan,
                )
            ),
            "class_ambiguous": bool(
                group.iloc[0].get(
                    "class_ambiguous",
                    False,
                )
            ),
            "line_distance_px": pd.NA,
            "previous_side": pd.NA,
            "current_side": pd.NA,
            "frame_gap": pd.NA,
            "crossing_method": "",
            "track_observations": len(group),
            "counted": False,
        }

    def _infer_track_direction(
        self,
        group: pd.DataFrame,
    ) -> str:
        if len(group) < 2:
            return "UNKNOWN"

        first = group.iloc[0]
        last = group.iloc[-1]

        dx = (
            float(last["bottom_center_x"])
            - float(first["bottom_center_x"])
        )

        dy = (
            float(last["bottom_center_y"])
            - float(first["bottom_center_y"])
        )

        if math.hypot(dx, dy) < self.config.min_direction_displacement_px:
            return "UNKNOWN"

        return "L→R" if dx > 0 else "R→L"

    # ------------------------------------------------------------------
    # BATCH
    # ------------------------------------------------------------------

    def process(
        self,
        trajectory: pd.DataFrame,
        identity_column: str = "crossing_id",
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        if identity_column not in trajectory.columns:
            raise ValueError(
                f"Trajectory missing identity column: {identity_column}"
            )

        events: list[dict] = []
        audits: list[dict] = []

        for identity_id, group in trajectory.groupby(
            identity_column,
            sort=False,
        ):
            group = (
                group
                .sort_values("frame_id")
                .reset_index(drop=True)
            )

            event = self.detect_track_crossing(
                group,
                identity_id=int(identity_id),
            )

            if event is None:
                continue

            events.append(event)

            audits.append(
                {
                    "crossing_id": int(identity_id),
                    "track_ids": (
                        str(
                            group["track_id"]
                            .drop_duplicates()
                            .tolist()
                        )
                    ),
                    "first_frame": int(
                        group["frame_id"].min()
                    ),
                    "last_frame": int(
                        group["frame_id"].max()
                    ),
                    "crossing_frame": event["crossing_frame"],
                    "direction": event["direction"],
                    "counted": bool(event["counted"]),
                    "crossing_method": event["crossing_method"],
                    "track_observations": len(group),
                }
            )

        columns = [
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
            "counted",
        ]

        events_df = pd.DataFrame(events, columns=columns)

        if audits:
            audit_df = pd.DataFrame(audits)
        else:
            audit_df = pd.DataFrame(
                columns=[
                    "crossing_id",
                    "track_ids",
                    "first_frame",
                    "last_frame",
                    "crossing_frame",
                    "direction",
                    "counted",
                    "crossing_method",
                    "track_observations",
                ]
            )

        return events_df, audit_df

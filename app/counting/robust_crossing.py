from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import math
import numpy as np
import pandas as pd


# ============================================================
# CONFIG
# ============================================================

@dataclass
class CrossingConfig:
    # --------------------------------------------------------
    # Counting line
    # --------------------------------------------------------
    line_deadband_px: float = 20.0

    # Corridor around the counting line.
    #
    # Example:
    #
    #              BEFORE
    # ==============================
    #          CORRIDOR
    # ==============================
    #              AFTER
    #
    corridor_px: float = 45.0

    # --------------------------------------------------------
    # Temporal constraints
    # --------------------------------------------------------
    max_trajectory_gap_sec: float = 1.50

    # --------------------------------------------------------
    # Direction validation
    # --------------------------------------------------------
    min_direction_displacement_px: float = 8.0

    # Number of observations used to estimate direction.
    direction_window: int = 3

    # --------------------------------------------------------
    # Duplicate suppression
    # --------------------------------------------------------
    dedup_time_sec: float = 1.50
    dedup_distance_px: float = 100.0

    # --------------------------------------------------------
    # Track quality
    # --------------------------------------------------------
    min_track_observations: int = 2

    # --------------------------------------------------------
    # Classes
    # --------------------------------------------------------
    vehicle_classes: tuple[str, ...] = (
        "motorcycle",
        "car",
        "truck",
        "bus",
    )


# ============================================================
# ROBUST CROSSING ENGINE
# ============================================================

class RobustCrossingEngine:

    def __init__(
        self,
        line: dict,
        fps: float,
        config: Optional[CrossingConfig] = None,
    ):
        self.line = line
        self.fps = float(fps)
        self.config = config or CrossingConfig()

        self.x1 = float(line["x1"])
        self.y1 = float(line["y1"])
        self.x2 = float(line["x2"])
        self.y2 = float(line["y2"])

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

    # ========================================================
    # LINE GEOMETRY
    # ========================================================

    def signed_line_value(
        self,
        x: float,
        y: float,
    ) -> float:
        """
        Cross product relative to counting line.

        > 0 : one side
        < 0 : opposite side
        = 0 : exactly on line
        """

        return (
            self.line_dx * (y - self.y1)
            -
            self.line_dy * (x - self.x1)
        )

    def line_distance(
        self,
        x: float,
        y: float,
    ) -> float:
        """
        Perpendicular pixel distance from point to line.
        """

        value = abs(
            self.signed_line_value(x, y)
        )

        return value / self.line_length

    def side(
        self,
        x: float,
        y: float,
    ) -> int:
        """
        Returns:

        0  = inside deadband
        +1 = positive side
        -1 = negative side
        """

        distance = self.line_distance(x, y)

        if distance <= self.config.line_deadband_px:
            return 0

        value = self.signed_line_value(
            x,
            y,
        )

        return 1 if value > 0 else -1

    # ========================================================
    # SEGMENT INTERSECTION
    # ========================================================

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
            -
            (by - ay) * (cx - ax)
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

        return (
            min(ax, bx) <= px <= max(ax, bx)
            and
            min(ay, by) <= py <= max(ay, by)
        )

    def segments_intersect(
        self,
        p1: tuple[float, float],
        p2: tuple[float, float],
        q1: tuple[float, float],
        q2: tuple[float, float],
    ) -> bool:

        o1 = self._orientation(
            *p1,
            *p2,
            *q1,
        )

        o2 = self._orientation(
            *p1,
            *p2,
            *q2,
        )

        o3 = self._orientation(
            *q1,
            *q2,
            *p1,
        )

        o4 = self._orientation(
            *q1,
            *q2,
            *p2,
        )

        eps = 1e-9

        # General intersection
        if (
            ((o1 > eps and o2 < -eps) or
             (o1 < -eps and o2 > eps))
            and
            ((o3 > eps and o4 < -eps) or
             (o3 < -eps and o4 > eps))
        ):
            return True

        # Collinear cases
        if abs(o1) <= eps and self._on_segment(
            *p1,
            *p2,
            *q1,
        ):
            return True

        if abs(o2) <= eps and self._on_segment(
            *p1,
            *p2,
            *q2,
        ):
            return True

        if abs(o3) <= eps and self._on_segment(
            *q1,
            *q2,
            *p1,
        ):
            return True

        if abs(o4) <= eps and self._on_segment(
            *q1,
            *q2,
            *p2,
        ):
            return True

        return False

    def trajectory_intersects_line(
        self,
        previous_point: tuple[float, float],
        current_point: tuple[float, float],
    ) -> bool:

        line_start = (
            self.x1,
            self.y1,
        )

        line_end = (
            self.x2,
            self.y2,
        )

        return self.segments_intersect(
            previous_point,
            current_point,
            line_start,
            line_end,
        )

    # ========================================================
    # CROSSING POINT
    # ========================================================

    def estimate_crossing_point(
        self,
        previous_point: tuple[float, float],
        current_point: tuple[float, float],
    ) -> tuple[float, float]:

        x_prev, y_prev = previous_point
        x_curr, y_curr = current_point

        v_prev = self.signed_line_value(
            x_prev,
            y_prev,
        )

        v_curr = self.signed_line_value(
            x_curr,
            y_curr,
        )

        denominator = v_prev - v_curr

        if abs(denominator) < 1e-9:
            return current_point

        alpha = v_prev / denominator

        alpha = max(
            0.0,
            min(
                1.0,
                alpha,
            ),
        )

        x = (
            x_prev
            +
            alpha * (x_curr - x_prev)
        )

        y = (
            y_prev
            +
            alpha * (y_curr - y_prev)
        )

        return (
            float(x),
            float(y),
        )

    # ========================================================
    # DIRECTION
    # ========================================================

    def direction_from_sides(
        self,
        previous_side: int,
        current_side: int,
    ) -> str:

        if (
            previous_side < 0
            and current_side > 0
        ):
            return "L→R"

        if (
            previous_side > 0
            and current_side < 0
        ):
            return "R→L"

        return "UNKNOWN"

    def estimate_direction(
        self,
        group: pd.DataFrame,
        crossing_index: int,
    ) -> str:

        window = max(
            1,
            int(self.config.direction_window),
        )

        start_idx = max(
            0,
            crossing_index - window,
        )

        end_idx = min(
            len(group) - 1,
            crossing_index + window,
        )

        before = group.iloc[
            start_idx
            :
            crossing_index + 1
        ]

        after = group.iloc[
            crossing_index
            :
            end_idx + 1
        ]

        if len(before) == 0 or len(after) == 0:
            return "UNKNOWN"

        x_before = float(
            before.iloc[0]["bottom_center_x"]
        )

        y_before = float(
            before.iloc[0]["bottom_center_y"]
        )

        x_after = float(
            after.iloc[-1]["bottom_center_x"]
        )

        y_after = float(
            after.iloc[-1]["bottom_center_y"]
        )

        dx = x_after - x_before
        dy = y_after - y_before

        # Camera-specific semantic direction.
        #
        # For your current camera geometry,
        # horizontal movement is the primary signal.
        #
        # We still use line-side transition as the
        # primary direction signal whenever possible.

        displacement = math.hypot(
            dx,
            dy,
        )

        if displacement < self.config.min_direction_displacement_px:
            return "UNKNOWN"

        if abs(dx) >= abs(dy):
            return (
                "L→R"
                if dx > 0
                else "R→L"
            )

        return (
            "L→R"
            if dx > 0
            else "R→L"
        )

    # ========================================================
    # CROSSING CANDIDATE DETECTION
    # ========================================================

    def detect_track_crossing(
        self,
        group: pd.DataFrame,
    ) -> Optional[dict]:

        group = (
            group
            .sort_values("frame_id")
            .reset_index(drop=True)
        )

        if len(group) < self.config.min_track_observations:
            return None

        track_id = group.iloc[0]["track_id"]

        track_class = group.iloc[0]["track_class"]

        first_frame = int(
            group["frame_id"].min()
        )

        last_frame = int(
            group["frame_id"].max()
        )

        # ----------------------------------------------------
        # Calculate geometry
        # ----------------------------------------------------

        group = group.copy()

        group["line_value"] = (
            group.apply(
                lambda row:
                self.signed_line_value(
                    float(row["bottom_center_x"]),
                    float(row["bottom_center_y"]),
                ),
                axis=1,
            )
        )

        group["line_distance_px"] = (
            group["line_value"].abs()
            /
            self.line_length
        )

        group["side"] = group.apply(
            lambda row:
            self.side(
                float(row["bottom_center_x"]),
                float(row["bottom_center_y"]),
            ),
            axis=1,
        )

        # ----------------------------------------------------
        # Search ALL consecutive observations.
        #
        # This is different from the old implementation,
        # which removed the deadband observations first.
        #
        # Keeping them allows us to detect:
        #
        # BEFORE → DEADZONE → AFTER
        #
        # ----------------------------------------------------

        for i in range(1, len(group)):

            previous = group.iloc[i - 1]
            current = group.iloc[i]

            previous_frame = int(
                previous["frame_id"]
            )

            current_frame = int(
                current["frame_id"]
            )

            frame_gap = (
                current_frame
                -
                previous_frame
            )

            max_gap_frames = (
                self.config.max_trajectory_gap_sec
                *
                self.fps
            )

            if frame_gap > max_gap_frames:
                continue

            previous_point = (
                float(
                    previous["bottom_center_x"]
                ),
                float(
                    previous["bottom_center_y"]
                ),
            )

            current_point = (
                float(
                    current["bottom_center_x"]
                ),
                float(
                    current["bottom_center_y"]
                ),
            )

            previous_side = int(
                previous["side"]
            )

            current_side = int(
                current["side"]
            )

            # =================================================
            # METHOD 1
            # SIDE CHANGE
            # =================================================

            side_change = (
                previous_side != 0
                and
                current_side != 0
                and
                previous_side != current_side
            )

            # =================================================
            # METHOD 2
            # LINE INTERSECTION
            # =================================================

            line_intersection = (
                self.trajectory_intersects_line(
                    previous_point,
                    current_point,
                )
            )

            # =================================================
            # METHOD 3
            # CROSSING CORRIDOR
            # =================================================

            previous_distance = float(
                previous["line_distance_px"]
            )

            current_distance = float(
                current["line_distance_px"]
            )

            corridor_crossing = (
                (
                    previous_distance
                    <=
                    self.config.corridor_px
                )
                and
                (
                    current_distance
                    <=
                    self.config.corridor_px
                )
                and
                (
                    previous_side
                    !=
                    current_side
                )
                and
                previous_side != 0
                and
                current_side != 0
            )

            # ------------------------------------------------
            # Main crossing decision
            #
            # Any of these mechanisms can rescue a fast object:
            #
            # side_change
            # line_intersection
            # corridor_crossing
            # ------------------------------------------------

            crossed = (
                side_change
                or
                line_intersection
                or
                corridor_crossing
            )

            if not crossed:
                continue

            # =================================================
            # DIRECTION VALIDATION
            # =================================================

            direction = self.direction_from_sides(
                previous_side,
                current_side,
            )

            if direction == "UNKNOWN":
                direction = self.estimate_direction(
                    group,
                    i,
                )

            # If we still cannot determine direction,
            # reject the event.
            #
            # This prevents jitter from becoming a count.

            if direction == "UNKNOWN":
                continue

            # =================================================
            # CROSSING POINT
            # =================================================

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

            # =================================================
            # METHOD USED
            # =================================================

            methods = []

            if side_change:
                methods.append(
                    "side_change"
                )

            if line_intersection:
                methods.append(
                    "line_intersection"
                )

            if corridor_crossing:
                methods.append(
                    "corridor"
                )

            method = "+".join(methods)

            return {
                "track_id": track_id,
                "track_class": track_class,
                "first_frame": first_frame,
                "last_frame": last_frame,
                "crossing_frame": current_frame,
                "crossing_time_sec": float(
                    current["timestamp_sec"]
                ),
                "crossing_x": crossing_x,
                "crossing_y": crossing_y,
                "direction": direction,
                "crossing_method": method,
                "previous_side": previous_side,
                "current_side": current_side,
                "frame_gap": frame_gap,
                "track_observations": len(group),
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
                "counted": True,
                "duplicate_of_track_id": pd.NA,
                "dedup_reason": "",
            }

        # ----------------------------------------------------
        # Track existed but never crossed
        # ----------------------------------------------------

        return {
            "track_id": track_id,
            "track_class": track_class,
            "first_frame": first_frame,
            "last_frame": last_frame,
            "crossing_frame": pd.NA,
            "crossing_time_sec": pd.NA,
            "crossing_x": pd.NA,
            "crossing_y": pd.NA,
            "direction": self._infer_track_direction(group),
            "crossing_method": "",
            "previous_side": pd.NA,
            "current_side": pd.NA,
            "frame_gap": pd.NA,
            "track_observations": len(group),
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
            "counted": False,
            "duplicate_of_track_id": pd.NA,
            "dedup_reason": "",
        }

    # ========================================================
    # TRACK DIRECTION WITHOUT CROSSING
    # ========================================================

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
            -
            float(first["bottom_center_x"])
        )

        dy = (
            float(last["bottom_center_y"])
            -
            float(first["bottom_center_y"])
        )

        displacement = math.hypot(
            dx,
            dy,
        )

        if displacement < self.config.min_direction_displacement_px:
            return "UNKNOWN"

        if abs(dx) >= abs(dy):
            return (
                "L→R"
                if dx > 0
                else "R→L"
            )

        return (
            "L→R"
            if dx > 0
            else "R→L"
        )

    # ========================================================
    # PROCESS ALL TRACKS
    # ========================================================

    def process(
        self,
        trajectory: pd.DataFrame,
    ) -> pd.DataFrame:

        required_columns = {
            "track_id",
            "frame_id",
            "timestamp_sec",
            "bottom_center_x",
            "bottom_center_y",
            "track_class",
        }

        missing = (
            required_columns
            -
            set(trajectory.columns)
        )

        if missing:
            raise ValueError(
                "Trajectory missing columns: "
                f"{sorted(missing)}"
            )

        results = []

        for track_id, group in trajectory.groupby(
            "track_id",
            sort=False,
        ):

            event = self.detect_track_crossing(
                group
            )

            if event is not None:
                results.append(event)

        if not results:
            return pd.DataFrame(
                columns=[
                    "track_id",
                    "track_class",
                    "first_frame",
                    "last_frame",
                    "crossing_frame",
                    "crossing_time_sec",
                    "crossing_x",
                    "crossing_y",
                    "direction",
                    "crossing_method",
                    "previous_side",
                    "current_side",
                    "frame_gap",
                    "track_observations",
                    "track_class_ratio",
                    "class_ambiguous",
                    "counted",
                    "duplicate_of_track_id",
                    "dedup_reason",
                ]
            )

        return pd.DataFrame(results)

    # ========================================================
    # DUPLICATE SUPPRESSION
    # ========================================================

    def deduplicate(
        self,
        events: pd.DataFrame,
    ) -> pd.DataFrame:

        events = events.copy()

        if events.empty:
            return events

        events["duplicate_of_track_id"] = pd.NA
        events["dedup_reason"] = ""

        # Only actual crossings
        crossing_mask = (
            events["counted"]
            &
            events["crossing_time_sec"].notna()
        )

        crossing_indices = (
            events.loc[
                crossing_mask
            ]
            .sort_values(
                "crossing_time_sec"
            )
            .index
            .tolist()
        )

        accepted = []

        for idx in crossing_indices:

            current = events.loc[idx]

            # ------------------------------------------------
            # Compare against accepted events
            # ------------------------------------------------

            duplicate_found = False

            for accepted_idx in accepted:

                previous = events.loc[
                    accepted_idx
                ]

                # Same class only
                if (
                    current["track_class"]
                    !=
                    previous["track_class"]
                ):
                    continue

                # Same direction only
                if (
                    current["direction"]
                    !=
                    previous["direction"]
                ):
                    continue

                # Temporal distance
                dt = abs(
                    float(
                        current[
                            "crossing_time_sec"
                        ]
                    )
                    -
                    float(
                        previous[
                            "crossing_time_sec"
                        ]
                    )
                )

                if (
                    dt
                    >
                    self.config.dedup_time_sec
                ):
                    continue

                # Spatial distance
                if (
                    pd.isna(
                        current["crossing_x"]
                    )
                    or
                    pd.isna(
                        current["crossing_y"]
                    )
                    or
                    pd.isna(
                        previous["crossing_x"]
                    )
                    or
                    pd.isna(
                        previous["crossing_y"]
                    )
                ):
                    continue

                distance = math.hypot(
                    float(
                        current["crossing_x"]
                    )
                    -
                    float(
                        previous["crossing_x"]
                    ),
                    float(
                        current["crossing_y"]
                    )
                    -
                    float(
                        previous["crossing_y"]
                    ),
                )

                if (
                    distance
                    >
                    self.config.dedup_distance_px
                ):
                    continue

                duplicate_found = True

                events.at[
                    idx,
                    "duplicate_of_track_id",
                ] = previous[
                    "track_id"
                ]

                events.at[
                    idx,
                    "dedup_reason",
                ] = (
                    "fragmentation_duplicate"
                )

                events.at[
                    idx,
                    "counted",
                ] = False

                break

            if not duplicate_found:
                accepted.append(idx)

        return events

    # ========================================================
    # FINAL VEHICLE EVENTS
    # ========================================================

    def vehicle_events(
        self,
        events: pd.DataFrame,
    ) -> pd.DataFrame:

        if events.empty:
            return events.copy()

        return events[
            events["track_class"].isin(
                self.config.vehicle_classes
            )
        ].copy()

    # ========================================================
    # FINAL COUNT
    # ========================================================

    def count(
        self,
        events: pd.DataFrame,
    ) -> pd.DataFrame:

        vehicle_events = (
            self.vehicle_events(events)
        )

        if vehicle_events.empty:

            return pd.DataFrame({
                "track_class": [
                    "motorcycle",
                    "car",
                    "truck",
                    "bus",
                ],
                "vehicle_count": [
                    0,
                    0,
                    0,
                    0,
                ],
            })

        final = vehicle_events[
            vehicle_events["counted"]
        ]

        counts = (
            final
            .groupby("track_class")
            .size()
            .reindex(
                [
                    "motorcycle",
                    "car",
                    "truck",
                    "bus",
                ],
                fill_value=0,
            )
            .astype(int)
            .rename("vehicle_count")
            .reset_index()
        )

        return counts

    # ========================================================
    # AUDIT LOG
    # ========================================================

    def build_audit_log(
        self,
        events: pd.DataFrame,
    ) -> pd.DataFrame:

        if events.empty:
            return pd.DataFrame(
                columns=[
                    "TRACK ID",
                    "CLASS",
                    "FIRST FRAME",
                    "LAST FRAME",
                    "CROSS FRAME",
                    "DIRECTION",
                    "COUNTED",
                ]
            )

        audit = events.copy()

        audit["CROSS FRAME"] = (
            audit["crossing_frame"]
            .apply(
                lambda x:
                "--"
                if pd.isna(x)
                else str(int(x))
            )
        )

        audit["COUNTED"] = np.where(
            audit["counted"],
            "YES",
            "NO",
        )

        audit = audit[
            [
                "track_id",
                "track_class",
                "first_frame",
                "last_frame",
                "CROSS FRAME",
                "direction",
                "COUNTED",
            ]
        ].copy()

        audit.columns = [
            "TRACK ID",
            "CLASS",
            "FIRST FRAME",
            "LAST FRAME",
            "CROSS FRAME",
            "DIRECTION",
            "COUNTED",
        ]

        return audit

    # ========================================================
    # COMPLETE PIPELINE
    # ========================================================

    def run(
        self,
        trajectory: pd.DataFrame,
    ) -> dict:

        # ----------------------------------------------------
        # 1. Detect crossing / non-crossing per track
        # ----------------------------------------------------

        events = self.process(
            trajectory
        )

        # ----------------------------------------------------
        # 2. Deduplicate
        # ----------------------------------------------------

        events = self.deduplicate(
            events
        )

        # ----------------------------------------------------
        # 3. Final vehicle events
        # ----------------------------------------------------

        vehicle_events = (
            self.vehicle_events(
                events
            )
        )

        # ----------------------------------------------------
        # 4. Final counts
        # ----------------------------------------------------

        counts = self.count(
            events
        )

        # ----------------------------------------------------
        # 5. Audit
        # ----------------------------------------------------

        audit = self.build_audit_log(
            events
        )

        return {
            "events": events,
            "vehicle_events": vehicle_events,
            "counts": counts,
            "audit": audit,
        }

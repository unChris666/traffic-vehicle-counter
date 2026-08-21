from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd


# ============================================================
# RESULT
# ============================================================

@dataclass(frozen=True)
class CountingResult:
    counts: dict[str, int]
    total: int

    trajectory: pd.DataFrame

    crossing_candidates: pd.DataFrame
    crossing_events: pd.DataFrame

    crossing_vehicle: pd.DataFrame
    crossing_person: pd.DataFrame

    vehicle_events: pd.DataFrame
    final_crossings: pd.DataFrame

    audit: dict[str, int]

    # NEW
    track_audit: pd.DataFrame


# ============================================================
# TRAFFIC COUNTER
# ============================================================

class TrafficCounter:
    """
    Phase 3 Robust Traffic Counting Engine.

    Responsibilities:
        1. Build track trajectories.
        2. Calculate counting-line geometry.
        3. Detect crossing using:
           - side change
           - line intersection
           - crossing corridor
        4. Validate direction.
        5. Keep first crossing per track.
        6. Suppress fragmented duplicates.
        7. Aggregate vehicle counts.
        8. Produce track-level audit log.

    IMPORTANT:
        Detection and tracking are NOT performed here.
        This class only consumes tracks_phase2.
    """

    def __init__(
        self,
        *,
        line_x1: float,
        line_y1: float,
        line_x2: float,
        line_y2: float,
        line_deadband_px: float,
        max_trajectory_gap_sec: float,
        moto_dedup_time_sec: float,
        moto_dedup_distance_px: float,
        vehicle_classes: set[str],
        fps: float,

        # ====================================================
        # NEW ROBUST CROSSING PARAMETERS
        # ====================================================

        crossing_corridor_px: float = 45.0,

        min_direction_displacement_px: float = 8.0,

        direction_window: int = 3,

        # Duplicate suppression for fragmented tracks.
        duplicate_time_sec: float = 1.50,

        duplicate_distance_px: float = 100.0,

    ) -> None:

        if fps <= 0:
            raise ValueError(
                f"fps must be > 0, got {fps}"
            )

        # ====================================================
        # COUNTING LINE
        # ====================================================

        self.x1 = float(line_x1)
        self.y1 = float(line_y1)

        self.x2 = float(line_x2)
        self.y2 = float(line_y2)

        self.line_dx = (
            self.x2 - self.x1
        )

        self.line_dy = (
            self.y2 - self.y1
        )

        self.line_length = math.hypot(
            self.line_dx,
            self.line_dy,
        )

        if self.line_length <= 0:
            raise ValueError(
                "Counting line cannot have zero length"
            )

        # ====================================================
        # EXISTING PARAMETERS
        # ====================================================

        self.line_deadband_px = float(
            line_deadband_px
        )

        self.max_trajectory_gap_sec = float(
            max_trajectory_gap_sec
        )

        self.moto_dedup_time_sec = float(
            moto_dedup_time_sec
        )

        self.moto_dedup_distance_px = float(
            moto_dedup_distance_px
        )

        self.vehicle_classes = set(
            vehicle_classes
        )

        self.fps = float(fps)

        # ====================================================
        # NEW PARAMETERS
        # ====================================================

        self.crossing_corridor_px = float(
            crossing_corridor_px
        )

        self.min_direction_displacement_px = float(
            min_direction_displacement_px
        )

        self.direction_window = max(
            1,
            int(direction_window),
        )

        self.duplicate_time_sec = float(
            duplicate_time_sec
        )

        self.duplicate_distance_px = float(
            duplicate_distance_px
        )

    # ========================================================
    # TRAJECTORY
    # ========================================================

    def _build_trajectory(
        self,
        tracks_phase2: pd.DataFrame,
    ) -> pd.DataFrame:

        required = {
            "track_id",
            "frame_id",
            "timestamp_sec",
            "bottom_center_x",
            "bottom_center_y",
            "track_class",
            "track_class_ratio",
            "class_ambiguous",
        }

        missing = (
            required
            -
            set(tracks_phase2.columns)
        )

        if missing:
            raise ValueError(
                "tracks_phase2 missing required "
                f"columns: {sorted(missing)}"
            )

        trajectory = (
            tracks_phase2[
                [
                    "track_id",
                    "frame_id",
                    "timestamp_sec",
                    "bottom_center_x",
                    "bottom_center_y",
                    "track_class",
                    "track_class_ratio",
                    "class_ambiguous",
                ]
            ]
            .sort_values(
                [
                    "track_id",
                    "frame_id",
                ]
            )
            .copy()
        )

        trajectory["dx"] = (
            trajectory
            .groupby("track_id")[
                "bottom_center_x"
            ]
            .diff()
        )

        trajectory["dy"] = (
            trajectory
            .groupby("track_id")[
                "bottom_center_y"
            ]
            .diff()
        )

        trajectory["frame_delta"] = (
            trajectory
            .groupby("track_id")[
                "frame_id"
            ]
            .diff()
        )

        trajectory["time_delta_sec"] = (
            trajectory["frame_delta"]
            /
            self.fps
        )

        return trajectory

    # ========================================================
    # LINE GEOMETRY
    # ========================================================

    def _apply_line_geometry(
        self,
        trajectory: pd.DataFrame,
    ) -> pd.DataFrame:

        trajectory = trajectory.copy()

        # ----------------------------------------------------
        # Signed line value
        #
        # Positive and negative represent opposite sides.
        # ----------------------------------------------------

        trajectory["line_value"] = (
            self.line_dx
            *
            (
                trajectory["bottom_center_y"]
                -
                self.y1
            )
            -
            self.line_dy
            *
            (
                trajectory["bottom_center_x"]
                -
                self.x1
            )
        )

        # ----------------------------------------------------
        # Perpendicular distance from counting line
        # ----------------------------------------------------

        trajectory["line_distance_px"] = (
            trajectory["line_value"]
            .abs()
            /
            self.line_length
        )

        # ----------------------------------------------------
        # Side
        #
        # 0  = deadband
        # +1 = positive side
        # -1 = negative side
        # ----------------------------------------------------

        trajectory["side"] = np.select(
            [
                (
                    trajectory["line_distance_px"]
                    <=
                    self.line_deadband_px
                ),

                trajectory["line_value"] > 0,
            ],
            [
                0,
                1,
            ],
            default=-1,
        )

        return trajectory

    # ========================================================
    # SEGMENT GEOMETRY
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
            (bx - ax)
            *
            (cy - ay)
            -
            (by - ay)
            *
            (cx - ax)
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
            min(ax, bx)
            <=
            px
            <=
            max(ax, bx)
            and
            min(ay, by)
            <=
            py
            <=
            max(ay, by)
        )

    def _segments_intersect(
        self,
        p1: tuple[float, float],
        p2: tuple[float, float],
        q1: tuple[float, float],
        q2: tuple[float, float],
    ) -> bool:

        eps = 1e-9

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

        # General case
        if (
            (
                (o1 > eps and o2 < -eps)
                or
                (o1 < -eps and o2 > eps)
            )
            and
            (
                (o3 > eps and o4 < -eps)
                or
                (o3 < -eps and o4 > eps)
            )
        ):
            return True

        # Collinear cases
        if (
            abs(o1) <= eps
            and
            self._on_segment(
                *p1,
                *p2,
                *q1,
            )
        ):
            return True

        if (
            abs(o2) <= eps
            and
            self._on_segment(
                *p1,
                *p2,
                *q2,
            )
        ):
            return True

        if (
            abs(o3) <= eps
            and
            self._on_segment(
                *q1,
                *q2,
                *p1,
            )
        ):
            return True

        if (
            abs(o4) <= eps
            and
            self._on_segment(
                *q1,
                *q2,
                *p2,
            )
        ):
            return True

        return False

    # ========================================================
    # LINE INTERSECTION
    # ========================================================

    def _trajectory_intersects_line(
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

        return self._segments_intersect(
            previous_point,
            current_point,
            line_start,
            line_end,
        )

    # ========================================================
    # CROSSING POINT
    # ========================================================

    def _estimate_crossing_point(
        self,
        previous_point: tuple[float, float],
        current_point: tuple[float, float],
    ) -> tuple[float, float]:

        x_prev, y_prev = previous_point

        x_curr, y_curr = current_point

        previous_value = (
            self.line_dx
            *
            (y_prev - self.y1)
            -
            self.line_dy
            *
            (x_prev - self.x1)
        )

        current_value = (
            self.line_dx
            *
            (y_curr - self.y1)
            -
            self.line_dy
            *
            (x_curr - self.x1)
        )

        denominator = (
            previous_value
            -
            current_value
        )

        if abs(denominator) < 1e-9:
            return current_point

        alpha = (
            previous_value
            /
            denominator
        )

        alpha = max(
            0.0,
            min(
                1.0,
                alpha,
            ),
        )

        crossing_x = (
            x_prev
            +
            alpha
            *
            (
                x_curr
                -
                x_prev
            )
        )

        crossing_y = (
            y_prev
            +
            alpha
            *
            (
                y_curr
                -
                y_prev
            )
        )

        return (
            float(crossing_x),
            float(crossing_y),
        )

    # ========================================================
    # DIRECTION
    # ========================================================

    def _direction_from_side_transition(
        self,
        previous_side: int,
        current_side: int,
    ) -> str:

        if (
            previous_side == -1
            and
            current_side == 1
        ):
            return "side_-1_to_+1"

        if (
            previous_side == 1
            and
            current_side == -1
        ):
            return "side_+1_to_-1"

        return "UNKNOWN"

    def _estimate_direction_from_motion(
        self,
        group: pd.DataFrame,
        crossing_index: int,
    ) -> str:

        window = self.direction_window

        start_index = max(
            0,
            crossing_index - window,
        )

        end_index = min(
            len(group) - 1,
            crossing_index + window,
        )

        before = group.iloc[
            start_index
            :
            crossing_index + 1
        ]

        after = group.iloc[
            crossing_index
            :
            end_index + 1
        ]

        if (
            before.empty
            or
            after.empty
        ):
            return "UNKNOWN"

        x_before = float(
            before.iloc[0][
                "bottom_center_x"
            ]
        )

        y_before = float(
            before.iloc[0][
                "bottom_center_y"
            ]
        )

        x_after = float(
            after.iloc[-1][
                "bottom_center_x"
            ]
        )

        y_after = float(
            after.iloc[-1][
                "bottom_center_y"
            ]
        )

        dx = (
            x_after
            -
            x_before
        )

        dy = (
            y_after
            -
            y_before
        )

        displacement = math.hypot(
            dx,
            dy,
        )

        if (
            displacement
            <
            self.min_direction_displacement_px
        ):
            return "UNKNOWN"

        # ----------------------------------------------------
        # Keep direction semantics compatible with existing
        # pipeline.
        #
        # Horizontal movement is dominant for this camera.
        # ----------------------------------------------------

        if abs(dx) >= abs(dy):

            if dx > 0:
                return "L→R"

            return "R→L"

        # For diagonal movement, x movement remains the
        # semantic direction.
        if dx > 0:
            return "L→R"

        return "R→L"

    # ========================================================
    # AUDIT TEMPLATE
    # ========================================================

    @staticmethod
    def _empty_track_audit() -> pd.DataFrame:

        return pd.DataFrame(
            columns=[
                "track_id",
                "track_class",
                "first_frame",
                "last_frame",
                "crossing_frame",
                "direction",
                "counted",
                "crossing_method",
                "duplicate_of_track_id",
                "dedup_reason",
            ]
        )

    # ========================================================
    # CROSSING DETECTION
    # ========================================================

    def _detect_crossings(
        self,
        trajectory: pd.DataFrame,
    ) -> tuple[
        pd.DataFrame,
        pd.DataFrame,
        pd.DataFrame,
    ]:

        # ====================================================
        # IMPORTANT:
        #
        # DO NOT REMOVE SIDE == 0.
        #
        # Deadband observations are retained because:
        #
        # BEFORE → DEADZONE → AFTER
        #
        # can occur for fast vehicles.
        # ====================================================

        candidates = []

        track_audit = []

        # ====================================================
        # PROCESS EACH TRACK
        # ====================================================

        for track_id, group in trajectory.groupby(
            "track_id",
            sort=False,
        ):

            group = (
                group
                .sort_values("frame_id")
                .reset_index(drop=True)
            )

            first_frame = int(
                group["frame_id"].min()
            )

            last_frame = int(
                group["frame_id"].max()
            )

            track_class = (
                group.iloc[0]["track_class"]
            )

            crossing_found = False

            crossing_event = None

            # ------------------------------------------------
            # Search consecutive observations
            # ------------------------------------------------

            for i in range(
                1,
                len(group),
            ):

                previous = group.iloc[
                    i - 1
                ]

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
                    self.max_trajectory_gap_sec
                    *
                    self.fps
                )

                # --------------------------------------------
                # Don't interpolate across extremely large
                # tracking gaps.
                # --------------------------------------------

                if (
                    frame_gap
                    >
                    max_gap_frames
                ):
                    continue

                previous_point = (
                    float(
                        previous[
                            "bottom_center_x"
                        ]
                    ),
                    float(
                        previous[
                            "bottom_center_y"
                        ]
                    ),
                )

                current_point = (
                    float(
                        current[
                            "bottom_center_x"
                        ]
                    ),
                    float(
                        current[
                            "bottom_center_y"
                        ]
                    ),
                )

                previous_side = int(
                    previous["side"]
                )

                current_side = int(
                    current["side"]
                )

                previous_distance = float(
                    previous[
                        "line_distance_px"
                    ]
                )

                current_distance = float(
                    current[
                        "line_distance_px"
                    ]
                )

                # =================================================
                # METHOD 1 — SIDE CHANGE
                # =================================================

                side_change = (
                    previous_side != 0
                    and
                    current_side != 0
                    and
                    previous_side
                    !=
                    current_side
                )

                # =================================================
                # METHOD 2 — LINE INTERSECTION
                # =================================================

                line_intersection = (
                    self._trajectory_intersects_line(
                        previous_point,
                        current_point,
                    )
                )

                # =================================================
                # METHOD 3 — CROSSING CORRIDOR
                # =================================================
                #
                # Handles:
                #
                # BEFORE
                #    ↓
                # CORRIDOR
                #    ↓
                # AFTER
                #
                # especially useful when only a few frames exist.
                # =================================================

                corridor_crossing = (
                    previous_distance
                    <=
                    self.crossing_corridor_px
                    and
                    current_distance
                    <=
                    self.crossing_corridor_px
                    and
                    previous_side != 0
                    and
                    current_side != 0
                    and
                    previous_side
                    !=
                    current_side
                )

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
                # DIRECTION
                # =================================================

                direction = (
                    self
                    ._direction_from_side_transition(
                        previous_side,
                        current_side,
                    )
                )

                # If side information cannot determine direction,
                # estimate from trajectory movement.
                if direction == "UNKNOWN":

                    direction = (
                        self
                        ._estimate_direction_from_motion(
                            group,
                            i,
                        )
                    )

                # -------------------------------------------------
                # If line intersection happened but one/both points
                # are in deadband, estimate direction from motion.
                # -------------------------------------------------

                if (
                    direction == "UNKNOWN"
                    and
                    line_intersection
                ):

                    direction = (
                        self
                        ._estimate_direction_from_motion(
                            group,
                            i,
                        )
                    )

                # Still unknown = reject.
                if direction == "UNKNOWN":
                    continue

                # =================================================
                # CROSSING POINT
                # =================================================

                if line_intersection:

                    crossing_x, crossing_y = (
                        self
                        ._estimate_crossing_point(
                            previous_point,
                            current_point,
                        )
                    )

                else:

                    crossing_x = float(
                        current[
                            "bottom_center_x"
                        ]
                    )

                    crossing_y = float(
                        current[
                            "bottom_center_y"
                        ]
                    )

                # =================================================
                # METHOD LABEL
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

                method = "+".join(
                    methods
                )

                crossing_event = {
                    "track_id": track_id,
                    "frame_id": current_frame,
                    "timestamp_sec": float(
                        current[
                            "timestamp_sec"
                        ]
                    ),
                    "bottom_center_x": crossing_x,
                    "bottom_center_y": crossing_y,
                    "direction": direction,
                    "track_class": track_class,
                    "track_class_ratio": float(
                        current[
                            "track_class_ratio"
                        ]
                    ),
                    "class_ambiguous": bool(
                        current[
                            "class_ambiguous"
                        ]
                    ),
                    "line_distance_px": min(
                        previous_distance,
                        current_distance,
                    ),
                    "previous_side": previous_side,
                    "current_side": current_side,
                    "frame_gap": frame_gap,
                    "crossing_method": method,
                }

                candidates.append(
                    crossing_event
                )

                crossing_found = True

                # ---------------------------------------------
                # First crossing per track.
                # ---------------------------------------------

                break

            # =================================================
            # TRACK AUDIT
            # =================================================

            if crossing_found:

                track_audit.append(
                    {
                        "track_id": track_id,
                        "track_class": track_class,
                        "first_frame": first_frame,
                        "last_frame": last_frame,
                        "crossing_frame": (
                            crossing_event[
                                "frame_id"
                            ]
                        ),
                        "direction": (
                            crossing_event[
                                "direction"
                            ]
                        ),
                        "counted": True,
                        "crossing_method": (
                            crossing_event[
                                "crossing_method"
                            ]
                        ),
                        "duplicate_of_track_id": pd.NA,
                        "dedup_reason": "",
                    }
                )

            else:

                direction = (
                    self
                    ._estimate_direction_from_motion(
                        group,
                        max(
                            0,
                            len(group) // 2,
                        ),
                    )
                )

                track_audit.append(
                    {
                        "track_id": track_id,
                        "track_class": track_class,
                        "first_frame": first_frame,
                        "last_frame": last_frame,
                        "crossing_frame": pd.NA,
                        "direction": direction,
                        "counted": False,
                        "crossing_method": "",
                        "duplicate_of_track_id": pd.NA,
                        "dedup_reason": "",
                    }
                )

        # ====================================================
        # EMPTY RESULT
        # ====================================================

        if not candidates:

            crossing_candidates = pd.DataFrame(
                columns=[
                    "track_id",
                    "frame_id",
                    "timestamp_sec",
                    "bottom_center_x",
                    "bottom_center_y",
                    "direction",
                    "track_class",
                    "track_class_ratio",
                    "class_ambiguous",
                    "line_distance_px",
                    "previous_side",
                    "current_side",
                    "frame_gap",
                    "crossing_method",
                ]
            )

            crossing_events = (
                crossing_candidates
                .rename(
                    columns={
                        "frame_id":
                            "crossing_frame",

                        "timestamp_sec":
                            "crossing_time_sec",

                        "bottom_center_x":
                            "crossing_x",

                        "bottom_center_y":
                            "crossing_y",
                    }
                )
            )

            return (
                crossing_candidates,
                crossing_events,
                pd.DataFrame(track_audit),
            )

        # ====================================================
        # CROSSING CANDIDATES
        # ====================================================

        crossing_candidates = (
            pd.DataFrame(candidates)
            .sort_values(
                [
                    "track_id",
                    "frame_id",
                ]
            )
            .reset_index(drop=True)
        )

        # ====================================================
        # CROSSING EVENTS
        # ====================================================

        crossing_events = (
            crossing_candidates
            .sort_values(
                [
                    "track_id",
                    "frame_id",
                ]
            )
            .drop_duplicates(
                "track_id",
                keep="first",
            )
            [
                [
                    "track_id",
                    "frame_id",
                    "timestamp_sec",
                    "bottom_center_x",
                    "bottom_center_y",
                    "direction",
                    "track_class",
                    "track_class_ratio",
                    "class_ambiguous",
                    "line_distance_px",
                    "previous_side",
                    "current_side",
                    "frame_gap",
                    "crossing_method",
                ]
            ]
            .rename(
                columns={
                    "frame_id":
                        "crossing_frame",

                    "timestamp_sec":
                        "crossing_time_sec",

                    "bottom_center_x":
                        "crossing_x",

                    "bottom_center_y":
                        "crossing_y",
                }
            )
            .reset_index(drop=True)
        )

        track_audit_df = (
            pd.DataFrame(track_audit)
            .sort_values(
                [
                    "track_id",
                ]
            )
            .reset_index(drop=True)
        )

        return (
            crossing_candidates,
            crossing_events,
            track_audit_df,
        )

    # ========================================================
    # VEHICLE / PERSON
    # ========================================================

    def _split_vehicle_person(
        self,
        crossing_events: pd.DataFrame,
    ) -> tuple[
        pd.DataFrame,
        pd.DataFrame,
    ]:

        crossing_vehicle = (
            crossing_events[
                crossing_events[
                    "track_class"
                ].isin(
                    self.vehicle_classes
                )
            ]
            .copy()
        )

        crossing_person = (
            crossing_events[
                crossing_events[
                    "track_class"
                ]
                ==
                "person"
            ]
            .copy()
        )

        return (
            crossing_vehicle,
            crossing_person,
        )

    # ========================================================
    # DUPLICATE SUPPRESSION
    # ========================================================

    def _deduplicate_motorcycles(
        self,
        crossing_vehicle: pd.DataFrame,
    ) -> tuple[
        pd.DataFrame,
        pd.DataFrame,
    ]:

        vehicle_events = (
            crossing_vehicle
            .sort_values(
                "crossing_time_sec"
            )
            .reset_index(drop=True)
            .copy()
        )

        if vehicle_events.empty:

            vehicle_events[
                "duplicate_of_track_id"
            ] = pd.NA

            vehicle_events[
                "dedup_reason"
            ] = ""

            vehicle_events[
                "is_duplicate"
            ] = False

            return (
                vehicle_events,
                vehicle_events.copy(),
            )

        vehicle_events[
            "duplicate_of_track_id"
        ] = pd.NA

        vehicle_events[
            "dedup_reason"
        ] = ""

        # ====================================================
        # ALL VEHICLES
        #
        # We now allow duplicate suppression for fragmented
        # tracks across vehicle classes, not only motorcycles.
        #
        # Same class + same direction + close time + close
        # crossing location.
        # ====================================================

        accepted_indices: list[int] = []

        for idx in range(
            len(vehicle_events)
        ):

            current = (
                vehicle_events
                .iloc[idx]
            )

            duplicate_found = False

            for accepted_idx in accepted_indices:

                previous = (
                    vehicle_events
                    .iloc[
                        accepted_idx
                    ]
                )

                # ------------------------------------------------
                # Class must match.
                # ------------------------------------------------

                if (
                    current[
                        "track_class"
                    ]
                    !=
                    previous[
                        "track_class"
                    ]
                ):
                    continue

                # ------------------------------------------------
                # Direction must match.
                # ------------------------------------------------

                if (
                    current[
                        "direction"
                    ]
                    !=
                    previous[
                        "direction"
                    ]
                ):
                    continue

                # ------------------------------------------------
                # Time distance.
                # ------------------------------------------------

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

                # Use the larger configured value for motorcycles
                # because fragmentation is especially common there.
                dedup_time_limit = max(
                    self.duplicate_time_sec,
                    (
                        self.moto_dedup_time_sec
                        if current[
                            "track_class"
                        ]
                        ==
                        "motorcycle"
                        else 0.0
                    ),
                )

                if (
                    dt
                    >
                    dedup_time_limit
                ):
                    continue

                # ------------------------------------------------
                # Spatial distance.
                # ------------------------------------------------

                distance = math.hypot(
                    float(
                        current[
                            "crossing_x"
                        ]
                    )
                    -
                    float(
                        previous[
                            "crossing_x"
                        ]
                    ),

                    float(
                        current[
                            "crossing_y"
                        ]
                    )
                    -
                    float(
                        previous[
                            "crossing_y"
                        ]
                    ),
                )

                dedup_distance_limit = max(
                    self.duplicate_distance_px,
                    (
                        self.moto_dedup_distance_px
                        if current[
                            "track_class"
                        ]
                        ==
                        "motorcycle"
                        else 0.0
                    ),
                )

                if (
                    distance
                    >
                    dedup_distance_limit
                ):
                    continue

                # ------------------------------------------------
                # Duplicate found.
                # ------------------------------------------------

                duplicate_found = True

                vehicle_events.at[
                    idx,
                    "duplicate_of_track_id",
                ] = previous[
                    "track_id"
                ]

                vehicle_events.at[
                    idx,
                    "dedup_reason",
                ] = (
                    "track_fragmentation"
                )

                break

            if not duplicate_found:

                accepted_indices.append(
                    idx
                )

        vehicle_events[
            "is_duplicate"
        ] = (
            vehicle_events[
                "duplicate_of_track_id"
            ]
            .notna()
        )

        final_crossings = (
            vehicle_events[
                ~vehicle_events[
                    "is_duplicate"
                ]
            ]
            .copy()
        )

        return (
            vehicle_events,
            final_crossings,
        )

    # ========================================================
    # UPDATE AUDIT AFTER DEDUP
    # ========================================================

    def _update_track_audit_after_dedup(
        self,
        track_audit: pd.DataFrame,
        vehicle_events: pd.DataFrame,
    ) -> pd.DataFrame:

        if track_audit.empty:
            return track_audit

        audit = track_audit.copy()

        if vehicle_events.empty:
            return audit

        duplicates = vehicle_events[
            vehicle_events[
                "is_duplicate"
            ]
        ]

        for _, duplicate in (
            duplicates.iterrows()
        ):

            track_id = (
                duplicate[
                    "track_id"
                ]
            )

            mask = (
                audit[
                    "track_id"
                ]
                ==
                track_id
            )

            audit.loc[
                mask,
                "counted",
            ] = False

            audit.loc[
                mask,
                "duplicate_of_track_id",
            ] = duplicate[
                "duplicate_of_track_id"
            ]

            audit.loc[
                mask,
                "dedup_reason",
            ] = duplicate[
                "dedup_reason"
            ]

        return audit

    # ========================================================
    # AGGREGATE COUNTS
    # ========================================================

    def _aggregate_counts(
        self,
        final_crossings: pd.DataFrame,
    ) -> tuple[
        dict[str, int],
        int,
    ]:

        if final_crossings.empty:

            return (
                {
                    "motorcycle": 0,
                    "car": 0,
                    "truck": 0,
                    "bus": 0,
                },
                0,
            )

        final_counts = (
            final_crossings
            .groupby(
                "track_class"
            )
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
        )

        counts = {
            "motorcycle": int(
                final_counts[
                    "motorcycle"
                ]
            ),

            "car": int(
                final_counts[
                    "car"
                ]
            ),

            "truck": int(
                final_counts[
                    "truck"
                ]
            ),

            "bus": int(
                final_counts[
                    "bus"
                ]
            ),
        }

        return (
            counts,
            int(
                sum(
                    counts.values()
                )
            ),
        )

    # ========================================================
    # MAIN COUNT
    # ========================================================

    def count(
        self,
        tracks_phase2: pd.DataFrame,
    ) -> CountingResult:

        # ----------------------------------------------------
        # 1. Build trajectory
        # ----------------------------------------------------

        trajectory = (
            self
            ._build_trajectory(
                tracks_phase2
            )
        )

        # ----------------------------------------------------
        # 2. Apply counting-line geometry
        # ----------------------------------------------------

        trajectory = (
            self
            ._apply_line_geometry(
                trajectory
            )
        )

        # ----------------------------------------------------
        # 3. Robust crossing detection
        # ----------------------------------------------------

        (
            crossing_candidates,
            crossing_events,
            track_audit,
        ) = (
            self
            ._detect_crossings(
                trajectory
            )
        )

        # ----------------------------------------------------
        # 4. Separate vehicle / person
        # ----------------------------------------------------

        (
            crossing_vehicle,
            crossing_person,
        ) = (
            self
            ._split_vehicle_person(
                crossing_events
            )
        )

        # ----------------------------------------------------
        # 5. Duplicate suppression
        # ----------------------------------------------------

        (
            vehicle_events,
            final_crossings,
        ) = (
            self
            ._deduplicate_motorcycles(
                crossing_vehicle
            )
        )

        # ----------------------------------------------------
        # 6. Update audit
        # ----------------------------------------------------

        track_audit = (
            self
            ._update_track_audit_after_dedup(
                track_audit,
                vehicle_events,
            )
        )

        # ----------------------------------------------------
        # 7. Aggregate
        # ----------------------------------------------------

        counts, total = (
            self
            ._aggregate_counts(
                final_crossings
            )
        )

        # ====================================================
        # AUDIT METRICS
        # ====================================================

        audit = {

            "all_tracks_analyzed":
                int(
                    trajectory[
                        "track_id"
                    ]
                    .nunique()
                ),

            "all_crossing_candidates":
                int(
                    len(
                        crossing_candidates
                    )
                ),

            "all_crossing_events":
                int(
                    len(
                        crossing_events
                    )
                ),

            "person_crossings_excluded":
                int(
                    len(
                        crossing_person
                    )
                ),

            "vehicle_crossings_before_dedup":
                int(
                    len(
                        vehicle_events
                    )
                ),

            "track_fragment_duplicates_removed":
                int(
                    vehicle_events[
                        "is_duplicate"
                    ].sum()
                ),

            "final_vehicle_crossings":
                int(
                    len(
                        final_crossings
                    )
                ),

            "final_vehicle_count":
                int(
                    total
                ),
        }

        # ====================================================
        # RETURN
        # ====================================================

        return CountingResult(

            counts=counts,

            total=total,

            trajectory=trajectory,

            crossing_candidates=(
                crossing_candidates
            ),

            crossing_events=(
                crossing_events
            ),

            crossing_vehicle=(
                crossing_vehicle
            ),

            crossing_person=(
                crossing_person
            ),

            vehicle_events=(
                vehicle_events
            ),

            final_crossings=(
                final_crossings
            ),

            audit=audit,

            track_audit=(
                track_audit
            ),
        )

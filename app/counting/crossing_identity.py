from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
import math

import numpy as np
import pandas as pd


VEHICLE_CLASSES = {
    "motorcycle",
    "car",
    "truck",
    "bus",
}


@dataclass
class TrackFragment:
    track_id: int

    class_name: str
    class_ratio: float
    class_ambiguous: bool

    first_frame: int
    last_frame: int

    first_time_sec: float
    last_time_sec: float

    first_x: float
    first_y: float

    last_x: float
    last_y: float

    first_side: int
    last_side: int

    first_distance_px: float
    last_distance_px: float

    velocity_x: float
    velocity_y: float

    first_normal_velocity: float
    last_normal_velocity: float

    observation_count: int
    mean_confidence: float

    rows: pd.DataFrame


@dataclass
class CrossingIdentity:
    crossing_id: int

    track_ids: list[int] = field(default_factory=list)

    vehicle_class: str = "unknown"

    class_ratio: float = 0.0

    class_ambiguous: bool = False

    first_frame: Optional[int] = None
    last_frame: Optional[int] = None

    first_time_sec: Optional[float] = None
    last_time_sec: Optional[float] = None

    last_x: Optional[float] = None
    last_y: Optional[float] = None

    last_side: int = 0
    last_distance_px: float = 0.0

    last_velocity_x: float = 0.0
    last_velocity_y: float = 0.0

    first_normal_velocity: float = 0.0
    last_normal_velocity: float = 0.0

    observation_count: int = 0

    reconnect_count: int = 0

    identity_confidence: float = 0.0

    class_history: list[str] = field(default_factory=list)


class CrossingIdentityEngine:
    """
    Physical identity reconstruction.

    Design goals:

    1. Raw tracker IDs are NOT physical identities.
    2. Sequential fragments can belong to the same physical vehicle.
    3. Overlapping tracks are NEVER merged.
    4. Fast objects receive larger motion-aware tolerance.
    5. Side transition is allowed only when geometry supports crossing.
    6. Person <-> motorcycle is allowed because rider/vehicle ambiguity
       is a known failure mode.
    """

    def __init__(
        self,
        *,
        fps: float,
        line_x1: float,
        line_y1: float,
        line_x2: float,
        line_y2: float,
        line_deadband_px: float = 20.0,

        max_reconnect_gap_sec: float = 1.20,
        max_reconnect_distance_px: float = 140.0,

        identity_match_threshold: float = 0.72,
        identity_match_margin: float = 0.06,

        min_pre_crossing_observations: int = 2,

        velocity_gate_px_per_frame: float = 80.0,

        fast_velocity_px_per_frame: float = 35.0,

        near_line_multiplier: float = 5.0,

        person_motorcycle_compatibility: float = 0.55,
    ) -> None:

        if fps <= 0:
            raise ValueError("fps must be > 0")

        self.fps = float(fps)

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
                "Counting line cannot have zero length."
            )

        self.line_deadband_px = float(
            line_deadband_px
        )

        self.max_reconnect_gap_frames = max(
            1,
            int(
                round(
                    max_reconnect_gap_sec * self.fps
                )
            ),
        )

        self.max_reconnect_distance_px = float(
            max_reconnect_distance_px
        )

        self.identity_match_threshold = float(
            identity_match_threshold
        )

        self.identity_match_margin = float(
            identity_match_margin
        )

        self.min_pre_crossing_observations = int(
            min_pre_crossing_observations
        )

        self.velocity_gate_px_per_frame = float(
            velocity_gate_px_per_frame
        )

        self.fast_velocity_px_per_frame = float(
            fast_velocity_px_per_frame
        )

        self.near_line_multiplier = float(
            near_line_multiplier
        )

        self.person_motorcycle_compatibility = float(
            person_motorcycle_compatibility
        )

    # ============================================================
    # GEOMETRY
    # ============================================================

    def _line_geometry(
        self,
        x: float,
        y: float,
    ) -> tuple[float, float, int]:

        line_value = (
            self.line_dx * (y - self.y1)
            -
            self.line_dy * (x - self.x1)
        )

        distance = (
            abs(line_value)
            /
            self.line_length
        )

        if distance <= self.line_deadband_px:
            side = 0
        elif line_value > 0:
            side = 1
        else:
            side = -1

        return (
            float(line_value),
            float(distance),
            int(side),
        )

    def _normal_velocity(
        self,
        vx: float,
        vy: float,
    ) -> float:

        nx = -self.line_dy / self.line_length
        ny = self.line_dx / self.line_length

        return float(
            vx * nx + vy * ny
        )

    # ============================================================
    # VELOCITY
    # ============================================================

    @staticmethod
    def _estimate_velocity(
        rows: pd.DataFrame,
        *,
        tail: bool,
    ) -> tuple[float, float]:

        if len(rows) < 2:
            return 0.0, 0.0

        sample = (
            rows.tail(7)
            if tail
            else rows.head(7)
        )

        frame_delta = (
            sample["frame_id"]
            .diff()
        )

        dx = (
            sample["bottom_center_x"]
            .diff()
        )

        dy = (
            sample["bottom_center_y"]
            .diff()
        )

        valid = frame_delta > 0

        if not valid.any():
            return 0.0, 0.0

        vx = float(
            (
                dx[valid]
                /
                frame_delta[valid]
            ).median()
        )

        vy = float(
            (
                dy[valid]
                /
                frame_delta[valid]
            ).median()
        )

        return vx, vy

    @staticmethod
    def _velocity_cosine(
        ax: float,
        ay: float,
        bx: float,
        by: float,
    ) -> float:

        na = math.hypot(ax, ay)
        nb = math.hypot(bx, by)

        if na < 1e-6 or nb < 1e-6:
            return 0.5

        return float(
            np.clip(
                (
                    ax * bx
                    +
                    ay * by
                )
                /
                (
                    na * nb
                ),
                -1.0,
                1.0,
            )
        )

    # ============================================================
    # CLASS
    # ============================================================

    def _class_compatibility(
        self,
        identity_class: str,
        fragment_class: str,
    ) -> tuple[bool, float]:

        a = str(
            identity_class
        ).strip().lower()

        b = str(
            fragment_class
        ).strip().lower()

        if (
            a == b
            and a
        ):
            return True, 1.0

        if (
            a in {"", "unknown"}
            or
            b in {"", "unknown"}
        ):
            return True, 0.65

        if {
            a,
            b,
        } == {
            "person",
            "motorcycle",
        }:
            return (
                True,
                self.person_motorcycle_compatibility,
            )

        if (
            a in VEHICLE_CLASSES
            and
            b in VEHICLE_CLASSES
            and
            a != b
        ):
            return False, 0.0

        if a == "person" or b == "person":
            return False, 0.0

        return False, 0.0

    # ============================================================
    # PREPARE
    # ============================================================

    def prepare(
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
                "tracks_phase2 missing columns: "
                f"{sorted(missing)}"
            )

        trajectory = (
            tracks_phase2
            .copy()
            .sort_values(
                [
                    "track_id",
                    "frame_id",
                ]
            )
            .reset_index(drop=True)
        )

        if "confidence" not in trajectory:
            trajectory["confidence"] = 1.0

        geometry = trajectory.apply(
            lambda row:
            self._line_geometry(
                float(
                    row["bottom_center_x"]
                ),
                float(
                    row["bottom_center_y"]
                ),
            ),
            axis=1,
            result_type="expand",
        )

        geometry.columns = [
            "line_value",
            "line_distance_px",
            "side",
        ]

        trajectory = pd.concat(
            [
                trajectory,
                geometry,
            ],
            axis=1,
        )

        trajectory["frame_delta"] = (
            trajectory
            .groupby("track_id")["frame_id"]
            .diff()
            .fillna(0)
        )

        trajectory["dx"] = (
            trajectory
            .groupby("track_id")[
                "bottom_center_x"
            ]
            .diff()
            .fillna(0)
        )

        trajectory["dy"] = (
            trajectory
            .groupby("track_id")[
                "bottom_center_y"
            ]
            .diff()
            .fillna(0)
        )

        trajectory["speed_px_per_frame"] = (
            np.hypot(
                trajectory["dx"],
                trajectory["dy"],
            )
            /
            trajectory["frame_delta"].replace(
                0,
                np.nan,
            )
        ).fillna(0.0)

        trajectory["normal_velocity"] = (
            trajectory.apply(
                lambda r:
                self._normal_velocity(
                    float(
                        r["dx"]
                        /
                        max(
                            float(
                                r["frame_delta"]
                            ),
                            1.0,
                        )
                    ),
                    float(
                        r["dy"]
                        /
                        max(
                            float(
                                r["frame_delta"]
                            ),
                            1.0,
                        )
                    ),
                ),
                axis=1,
            )
        )

        trajectory["track_obs_index"] = (
            trajectory
            .groupby("track_id")
            .cumcount()
        )

        trajectory["track_obs_total"] = (
            trajectory
            .groupby("track_id")["track_id"]
            .transform("size")
        )

        return trajectory

    # ============================================================
    # FRAGMENT
    # ============================================================

    def build_fragments(
        self,
        trajectory: pd.DataFrame,
    ) -> list[TrackFragment]:

        fragments = []

        for track_id, rows in trajectory.groupby(
            "track_id",
            sort=False,
        ):

            rows = (
                rows
                .sort_values("frame_id")
                .reset_index(drop=True)
            )

            stable = rows[
                rows["side"] != 0
            ]

            first_side = (
                int(
                    stable.iloc[0]["side"]
                )
                if not stable.empty
                else 0
            )

            last_side = (
                int(
                    stable.iloc[-1]["side"]
                )
                if not stable.empty
                else 0
            )

            vx_first, vy_first = (
                self._estimate_velocity(
                    rows,
                    tail=False,
                )
            )

            vx_last, vy_last = (
                self._estimate_velocity(
                    rows,
                    tail=True,
                )
            )

            first_nv = (
                self._normal_velocity(
                    vx_first,
                    vy_first,
                )
            )

            last_nv = (
                self._normal_velocity(
                    vx_last,
                    vy_last,
                )
            )

            class_counts = (
                rows["track_class"]
                .astype(str)
                .str.lower()
                .value_counts()
            )

            class_name = (
                str(
                    class_counts.index[0]
                )
                if not class_counts.empty
                else "unknown"
            )

            class_ratio = float(
                class_counts.iloc[0]
                /
                class_counts.sum()
            )

            fragments.append(
                TrackFragment(
                    track_id=int(track_id),
                    class_name=class_name,
                    class_ratio=class_ratio,
                    class_ambiguous=(
                        class_ratio < 0.70
                    ),

                    first_frame=int(
                        rows["frame_id"].min()
                    ),
                    last_frame=int(
                        rows["frame_id"].max()
                    ),

                    first_time_sec=float(
                        rows["timestamp_sec"].min()
                    ),
                    last_time_sec=float(
                        rows["timestamp_sec"].max()
                    ),

                    first_x=float(
                        rows.iloc[0][
                            "bottom_center_x"
                        ]
                    ),
                    first_y=float(
                        rows.iloc[0][
                            "bottom_center_y"
                        ]
                    ),

                    last_x=float(
                        rows.iloc[-1][
                            "bottom_center_x"
                        ]
                    ),
                    last_y=float(
                        rows.iloc[-1][
                            "bottom_center_y"
                        ]
                    ),

                    first_side=first_side,
                    last_side=last_side,

                    first_distance_px=float(
                        rows.iloc[0][
                            "line_distance_px"
                        ]
                    ),
                    last_distance_px=float(
                        rows.iloc[-1][
                            "line_distance_px"
                        ]
                    ),

                    velocity_x=float(
                        vx_last
                    ),
                    velocity_y=float(
                        vy_last
                    ),

                    first_normal_velocity=float(
                        first_nv
                    ),
                    last_normal_velocity=float(
                        last_nv
                    ),

                    observation_count=int(
                        len(rows)
                    ),

                    mean_confidence=float(
                        rows["confidence"].mean()
                    ),

                    rows=rows,
                )
            )

        fragments.sort(
            key=lambda x: (
                x.first_frame,
                x.track_id,
            )
        )

        return fragments

    # ============================================================
    # MATCH SCORE
    # ============================================================

    def _match_score(
        self,
        identity: CrossingIdentity,
        fragment: TrackFragment,
    ) -> tuple[float, dict]:

        if identity.last_frame is None:
            return -1.0, {}

        gap = (
            fragment.first_frame
            -
            identity.last_frame
        )

        if (
            gap <= 0
            or
            gap > self.max_reconnect_gap_frames
        ):
            return -1.0, {
                "reason":
                "temporal_overlap_or_gap"
            }

        allowed, class_score = (
            self._class_compatibility(
                identity.vehicle_class,
                fragment.class_name,
            )
        )

        if not allowed:
            return -1.0, {
                "reason":
                "class_incompatible"
            }

        old_speed = math.hypot(
            identity.last_velocity_x,
            identity.last_velocity_y,
        )

        new_speed = math.hypot(
            fragment.velocity_x,
            fragment.velocity_y,
        )

        predicted_x = (
            identity.last_x
            +
            identity.last_velocity_x
            *
            gap
        )

        predicted_y = (
            identity.last_y
            +
            identity.last_velocity_y
            *
            gap
        )

        predicted_distance = math.hypot(
            fragment.first_x
            -
            predicted_x,
            fragment.first_y
            -
            predicted_y,
        )

        raw_distance = math.hypot(
            fragment.first_x
            -
            identity.last_x,
            fragment.first_y
            -
            identity.last_y,
        )

        continuity_distance = min(
            predicted_distance,
            raw_distance,
        )

        # Fast objects receive a motion-derived tolerance.
        expected_motion = (
            max(
                old_speed,
                new_speed,
                1.0,
            )
            *
            gap
        )

        dynamic_gate = max(
            self.max_reconnect_distance_px,
            min(
                self.velocity_gate_px_per_frame
                *
                gap
                *
                1.35,
                220.0,
            ),
        )

        dynamic_gate = max(
            dynamic_gate,
            expected_motion * 1.15,
        )

        if continuity_distance > dynamic_gate:
            return -1.0, {
                "reason":
                "spatial_gate",
                "distance":
                continuity_distance,
                "gate":
                dynamic_gate,
            }

        # --------------------------------------------------------
        # SIDE
        # --------------------------------------------------------

        side_score = 1.0

        if (
            identity.last_side != 0
            and
            fragment.first_side != 0
            and
            identity.last_side
            !=
            fragment.first_side
        ):

            near_line = (
                identity.last_distance_px
                <=
                self.line_deadband_px
                *
                self.near_line_multiplier
                or
                fragment.first_distance_px
                <=
                self.line_deadband_px
                *
                self.near_line_multiplier
            )

            if not near_line:
                return -1.0, {
                    "reason":
                    "implausible_side_change"
                }

            # Crossing side change is valid.
            side_score = 0.92

        # --------------------------------------------------------
        # VELOCITY
        # --------------------------------------------------------

        velocity_score = 0.55

        if (
            old_speed > 1.0
            and
            new_speed > 1.0
        ):

            cosine = (
                self._velocity_cosine(
                    identity.last_velocity_x,
                    identity.last_velocity_y,
                    fragment.velocity_x,
                    fragment.velocity_y,
                )
            )

            if cosine < 0.15:
                return -1.0, {
                    "reason":
                    "velocity_direction_conflict"
                }

            speed_ratio = (
                min(
                    old_speed,
                    new_speed,
                )
                /
                max(
                    old_speed,
                    new_speed,
                )
            )

            velocity_score = (
                0.55
                *
                (
                    (cosine + 1.0)
                    /
                    2.0
                )
                +
                0.45
                *
                speed_ratio
            )

        # --------------------------------------------------------
        # NORMAL VELOCITY
        # --------------------------------------------------------

        direction_score = 1.0

        old_nv = (
            identity.last_normal_velocity
        )

        new_nv = (
            fragment.first_normal_velocity
        )

        if (
            abs(old_nv) >= 1.0
            and
            abs(new_nv) >= 1.0
        ):

            # If side does not change, normal motion should not reverse.
            if (
                identity.last_side
                ==
                fragment.first_side
                and
                old_nv * new_nv < 0
            ):
                return -1.0, {
                    "reason":
                    "normal_velocity_reverse"
                }

            direction_score = (
                1.0
                if old_nv * new_nv >= 0
                else 0.85
            )

        # --------------------------------------------------------
        # TEMPORAL
        # --------------------------------------------------------

        temporal_score = max(
            0.0,
            1.0
            -
            (
                gap
                /
                self.max_reconnect_gap_frames
            ),
        )

        spatial_score = max(
            0.0,
            1.0
            -
            (
                continuity_distance
                /
                dynamic_gate
            ),
        )

        fast_bonus = (
            1.0
            if
            max(
                old_speed,
                new_speed,
            )
            >=
            self.fast_velocity_px_per_frame
            else
            0.0
        )

        score = (
            0.32 * spatial_score
            +
            0.17 * temporal_score
            +
            0.20 * velocity_score
            +
            0.13 * side_score
            +
            0.10 * class_score
            +
            0.08 * direction_score
        )

        if fast_bonus:
            # Fast motion is evidence, not a free pass.
            score += 0.04

        score = float(
            np.clip(
                score,
                0.0,
                1.0,
            )
        )

        return score, {
            "gap": gap,
            "distance": continuity_distance,
            "gate": dynamic_gate,
            "spatial_score": spatial_score,
            "temporal_score": temporal_score,
            "velocity_score": velocity_score,
            "side_score": side_score,
            "class_score": class_score,
            "direction_score": direction_score,
            "fast": bool(fast_bonus),
        }

    # ============================================================
    # IDENTITY CREATION
    # ============================================================

    def _create_identity(
        self,
        identity_id: int,
        fragment: TrackFragment,
    ) -> CrossingIdentity:

        return CrossingIdentity(
            crossing_id=identity_id,
            track_ids=[
                fragment.track_id
            ],

            vehicle_class=(
                fragment.class_name
            ),

            class_ratio=(
                fragment.class_ratio
            ),

            class_ambiguous=(
                fragment.class_ambiguous
            ),

            first_frame=(
                fragment.first_frame
            ),

            last_frame=(
                fragment.last_frame
            ),

            first_time_sec=(
                fragment.first_time_sec
            ),

            last_time_sec=(
                fragment.last_time_sec
            ),

            last_x=(
                fragment.last_x
            ),

            last_y=(
                fragment.last_y
            ),

            last_side=(
                fragment.last_side
            ),

            last_distance_px=(
                fragment.last_distance_px
            ),

            last_velocity_x=(
                fragment.velocity_x
            ),

            last_velocity_y=(
                fragment.velocity_y
            ),

            first_normal_velocity=(
                fragment.first_normal_velocity
            ),

            last_normal_velocity=(
                fragment.last_normal_velocity
            ),

            observation_count=(
                fragment.observation_count
            ),

            identity_confidence=(
                fragment.class_ratio
            ),

            class_history=[
                fragment.class_name
            ],
        )

    # ============================================================
    # UPDATE
    # ============================================================

    def _update_identity(
        self,
        identity: CrossingIdentity,
        fragment: TrackFragment,
        score: float,
    ) -> None:

        if (
            fragment.track_id
            not in
            identity.track_ids
        ):
            identity.track_ids.append(
                fragment.track_id
            )

        identity.last_frame = (
            fragment.last_frame
        )

        identity.last_time_sec = (
            fragment.last_time_sec
        )

        identity.last_x = (
            fragment.last_x
        )

        identity.last_y = (
            fragment.last_y
        )

        identity.last_side = (
            fragment.last_side
        )

        identity.last_distance_px = (
            fragment.last_distance_px
        )

        identity.last_velocity_x = (
            fragment.velocity_x
        )

        identity.last_velocity_y = (
            fragment.velocity_y
        )

        identity.last_normal_velocity = (
            fragment.last_normal_velocity
        )

        identity.observation_count += (
            fragment.observation_count
        )

        identity.reconnect_count += 1

        identity.identity_confidence = (
            0.70
            *
            identity.identity_confidence
            +
            0.30
            *
            score
        )

        identity.class_history.append(
            fragment.class_name
        )

        # --------------------------------------------------------
        # Resolve class from all fragments.
        # Vehicle evidence is preferred over rider-only evidence.
        # --------------------------------------------------------

        if (
            fragment.class_name
            in VEHICLE_CLASSES
            and
            fragment.class_ratio
            >=
            identity.class_ratio
        ):
            identity.vehicle_class = (
                fragment.class_name
            )

            identity.class_ratio = (
                fragment.class_ratio
            )

        elif (
            identity.vehicle_class
            not in VEHICLE_CLASSES
            and
            fragment.class_name
            != "unknown"
        ):
            identity.vehicle_class = (
                fragment.class_name
            )

        if (
            fragment.class_name
            !=
            identity.vehicle_class
        ):
            identity.class_ambiguous = True

    # ============================================================
    # RUN
    # ============================================================

    def run(
        self,
        tracks_phase2: pd.DataFrame,
    ):

        trajectory = self.prepare(
            tracks_phase2
        )

        fragments = self.build_fragments(
            trajectory
        )

        identities = []

        track_to_identity = {}

        audit = {
            "unique_track_ids": int(
                trajectory["track_id"].nunique()
            ),
            "crossing_identities": 0,
            "track_reconnections": 0,
            "ambiguous_reconnections": 0,
            "rejected_overlap": 0,
            "rejected_spatial": 0,
            "rejected_velocity": 0,
            "rejected_class": 0,
        }

        for fragment in fragments:

            candidates = []

            for identity in identities:

                score, details = (
                    self._match_score(
                        identity,
                        fragment,
                    )
                )

                if score < 0:

                    reason = details.get(
                        "reason",
                        "",
                    )

                    if (
                        reason
                        ==
                        "temporal_overlap_or_gap"
                    ):
                        audit[
                            "rejected_overlap"
                        ] += 1

                    elif (
                        reason
                        ==
                        "class_incompatible"
                    ):
                        audit[
                            "rejected_class"
                        ] += 1

                    elif (
                        reason
                        ==
                        "velocity_direction_conflict"
                        or
                        reason
                        ==
                        "normal_velocity_reverse"
                    ):
                        audit[
                            "rejected_velocity"
                        ] += 1

                    elif (
                        reason
                        ==
                        "spatial_gate"
                    ):
                        audit[
                            "rejected_spatial"
                        ] += 1

                    continue

                candidates.append(
                    (
                        score,
                        identity,
                        details,
                    )
                )

            candidates.sort(
                key=lambda x: x[0],
                reverse=True,
            )

            assigned = None

            if candidates:

                best_score = (
                    candidates[0][0]
                )

                second_score = (
                    candidates[1][0]
                    if len(candidates) > 1
                    else -1.0
                )

                margin = (
                    best_score
                    -
                    second_score
                )

                if (
                    best_score
                    >=
                    self.identity_match_threshold
                    and
                    (
                        second_score < 0
                        or
                        margin
                        >=
                        self.identity_match_margin
                    )
                ):
                    assigned = (
                        candidates[0][1]
                    )

                elif (
                    best_score
                    >=
                    self.identity_match_threshold
                    -
                    0.08
                ):
                    audit[
                        "ambiguous_reconnections"
                    ] += 1

            if assigned is None:

                assigned = (
                    self._create_identity(
                        len(identities) + 1,
                        fragment,
                    )
                )

                identities.append(
                    assigned
                )

            else:

                self._update_identity(
                    assigned,
                    fragment,
                    candidates[0][0],
                )

                audit[
                    "track_reconnections"
                ] += 1

            track_to_identity[
                fragment.track_id
            ] = assigned.crossing_id

        trajectory = trajectory.copy()

        trajectory["crossing_id"] = (
            trajectory["track_id"]
            .map(track_to_identity)
            .astype("Int64")
        )

        trajectory["identity_id"] = (
            trajectory["crossing_id"]
        )

        trajectory["raw_track_id"] = (
            trajectory["track_id"]
        )

        identity_map = {
            identity.crossing_id:
            identity
            for identity
            in identities
        }

        audit[
            "crossing_identities"
        ] = len(identities)

        return (
            trajectory,
            identity_map,
            track_to_identity,
            audit,
        )
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import math

import pandas as pd


class CrossingState(str, Enum):
    NOT_SEEN = "NOT_SEEN"
    APPROACHING = "APPROACHING"
    PRE_CROSSING = "PRE_CROSSING"
    CROSSING = "CROSSING"
    COUNTED = "COUNTED"
    POST_CROSSING = "POST_CROSSING"
    REVIEW = "REVIEW"


@dataclass
class TrackFragment:
    """
    One tracker track_id.

    A physical vehicle can consist of multiple fragments.
    This object is used only for identity association.
    """

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

    observation_count: int
    mean_confidence: float

    rows: pd.DataFrame


@dataclass
class CrossingIdentity:
    crossing_id: int
    track_ids: list[int] = field(default_factory=list)

    state: CrossingState = CrossingState.NOT_SEEN

    vehicle_class: str = "unknown"
    class_ratio: float = 0.0
    class_ambiguous: bool = False

    first_frame: int | None = None
    last_frame: int | None = None

    first_time_sec: float | None = None
    last_time_sec: float | None = None

    last_x: float | None = None
    last_y: float | None = None

    last_side: int = 0
    last_distance_px: float = 0.0

    last_velocity_x: float = 0.0
    last_velocity_y: float = 0.0

    stable_observations_before_crossing: int = 0


class CrossingIdentityEngine:
    """
    Converts raw tracker IDs into conservative physical crossing IDs.

    Critical rule:
        Overlapping tracks are NEVER reconnected.

    This prevents:
        motor A = track 101
        motor B = track 102

    from being merged merely because they enter the line area
    at nearly the same time.
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
        pre_crossing_distance_px: float = 100.0,
        max_reconnect_gap_sec: float = 1.0,
        max_reconnect_distance_px: float = 100.0,
        identity_match_threshold: float = 0.82,
        identity_match_margin: float = 0.08,
        velocity_gate_px_per_frame: float = 30.0,
        min_pre_crossing_observations: int = 2,
        max_crossing_gap_sec: float = 1.0,
    ) -> None:

        if fps <= 0:
            raise ValueError(
                f"fps must be > 0, got {fps}"
            )

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

        self.pre_crossing_distance_px = float(
            pre_crossing_distance_px
        )

        self.max_reconnect_gap_frames = max(
            1,
            int(
                round(
                    max_reconnect_gap_sec
                    * self.fps
                )
            ),
        )

        self.max_crossing_gap_frames = max(
            1,
            int(
                round(
                    max_crossing_gap_sec
                    * self.fps
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

        self.velocity_gate_px_per_frame = float(
            velocity_gate_px_per_frame
        )

        self.min_pre_crossing_observations = max(
            1,
            int(min_pre_crossing_observations),
        )

    # ------------------------------------------------------------------
    # GEOMETRY
    # ------------------------------------------------------------------

    def _line_geometry(
        self,
        x: float,
        y: float,
    ) -> tuple[float, float, int]:

        line_value = (
            self.line_dx * (y - self.y1)
            - self.line_dy * (x - self.x1)
        )

        distance = (
            abs(line_value)
            / self.line_length
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

    @staticmethod
    def _estimate_velocity(
        rows: pd.DataFrame,
        *,
        tail: bool,
    ) -> tuple[float, float]:

        if len(rows) < 2:
            return 0.0, 0.0

        sample = (
            rows.tail(5)
            if tail
            else rows.head(5)
        )

        dx = sample[
            "bottom_center_x"
        ].diff()

        dy = sample[
            "bottom_center_y"
        ].diff()

        dt = sample[
            "frame_id"
        ].diff()

        valid = dt > 0

        if not valid.any():
            return 0.0, 0.0

        vx = float(
            (dx[valid] / dt[valid]).median()
        )

        vy = float(
            (dy[valid] / dt[valid]).median()
        )

        return vx, vy

    @staticmethod
    def _velocity_similarity(
        ax: float,
        ay: float,
        bx: float,
        by: float,
    ) -> float:

        norm_a = math.hypot(ax, ay)
        norm_b = math.hypot(bx, by)

        if norm_a < 1e-6 or norm_b < 1e-6:
            return 0.5

        cosine = (
            ax * bx + ay * by
        ) / (
            norm_a * norm_b
        )

        return max(
            -1.0,
            min(1.0, cosine),
        )

    # ------------------------------------------------------------------
    # PREPARE
    # ------------------------------------------------------------------

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
            - set(tracks_phase2.columns)
        )

        if missing:
            raise ValueError(
                "tracks_phase2 missing required "
                f"columns: {sorted(missing)}"
            )

        trajectory = (
            tracks_phase2.copy()
            .sort_values(
                [
                    "frame_id",
                    "track_id",
                ]
            )
            .reset_index(drop=True)
        )

        if "confidence" not in trajectory.columns:
            trajectory["confidence"] = 1.0

        if "class_name" not in trajectory.columns:
            trajectory["class_name"] = (
                trajectory["track_class"]
            )

        geometry = trajectory.apply(
            lambda row: self._line_geometry(
                float(row["bottom_center_x"]),
                float(row["bottom_center_y"]),
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
            [trajectory, geometry],
            axis=1,
        )

        trajectory[
            "track_obs_index"
        ] = (
            trajectory
            .groupby("track_id")
            .cumcount()
        )

        trajectory[
            "track_obs_total"
        ] = (
            trajectory
            .groupby("track_id")[
                "track_id"
            ]
            .transform("size")
        )

        return trajectory

    # ------------------------------------------------------------------
    # FRAGMENTS
    # ------------------------------------------------------------------

    def build_fragments(
        self,
        trajectory: pd.DataFrame,
    ) -> list[TrackFragment]:

        fragments: list[TrackFragment] = []

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
                int(stable.iloc[0]["side"])
                if not stable.empty
                else 0
            )

            last_side = (
                int(stable.iloc[-1]["side"])
                if not stable.empty
                else 0
            )

            first_velocity = self._estimate_velocity(
                rows,
                tail=False,
            )

            last_velocity = self._estimate_velocity(
                rows,
                tail=True,
            )

            class_counts = (
                rows["track_class"]
                .value_counts()
            )

            class_name = str(
                class_counts.index[0]
            )

            class_ratio = float(
                class_counts.iloc[0]
                / class_counts.sum()
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
                        last_velocity[0]
                    ),
                    velocity_y=float(
                        last_velocity[1]
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
            key=lambda fragment: (
                fragment.first_frame,
                fragment.track_id,
            )
        )

        return fragments

    # ------------------------------------------------------------------
    # MATCH SCORE
    # ------------------------------------------------------------------

    def _candidate_score(
        self,
        identity: CrossingIdentity,
        fragment: TrackFragment,
    ) -> float:

        if (
            identity.last_frame is None
            or identity.last_x is None
            or identity.last_y is None
        ):
            return -1.0

        gap_frames = (
            fragment.first_frame
            - identity.last_frame
        )

        # CRITICAL:
        # overlapping tracks are separate vehicles/fragments.
        if (
            gap_frames <= 0
            or gap_frames > self.max_reconnect_gap_frames
        ):
            return -1.0

        # Predict where the old identity should be now.
        predicted_x = (
            identity.last_x
            +
            identity.last_velocity_x
            * gap_frames
        )

        predicted_y = (
            identity.last_y
            +
            identity.last_velocity_y
            * gap_frames
        )

        predicted_distance = math.hypot(
            fragment.first_x - predicted_x,
            fragment.first_y - predicted_y,
        )

        raw_distance = math.hypot(
            fragment.first_x - identity.last_x,
            fragment.first_y - identity.last_y,
        )

        # Use predicted distance for moving vehicles.
        continuity_distance = min(
            predicted_distance,
            raw_distance,
        )

        if (
            continuity_distance
            >
            self.max_reconnect_distance_px
        ):
            return -1.0

        if (
            identity.vehicle_class != "unknown"
            and fragment.class_name
            != identity.vehicle_class
        ):
            return -1.0

        # Spatial continuity
        spatial_score = max(
            0.0,
            1.0
            -
            (
                continuity_distance
                /
                self.max_reconnect_distance_px
            ),
        )

        temporal_score = max(
            0.0,
            1.0
            -
            (
                gap_frames
                /
                self.max_reconnect_gap_frames
            ),
        )

        # Side continuity.
        side_score = 1.0

        if (
            identity.last_side != 0
            and fragment.first_side != 0
        ):
            if (
                identity.last_side
                ==
                fragment.first_side
            ):
                side_score = 1.0
            else:
                # Side change is only plausible when the
                # identity is already close to the line.
                near_line = (
                    identity.last_distance_px
                    <=
                    self.line_deadband_px * 4
                    or
                    fragment.first_distance_px
                    <=
                    self.line_deadband_px * 4
                )

                if not near_line:
                    return -1.0

                side_score = 0.35

        # Velocity compatibility.
        velocity_score = 0.5

        old_speed = math.hypot(
            identity.last_velocity_x,
            identity.last_velocity_y,
        )

        new_speed = math.hypot(
            fragment.velocity_x,
            fragment.velocity_y,
        )

        if (
            old_speed > 1.0
            and
            new_speed > 1.0
        ):
            cosine = self._velocity_similarity(
                identity.last_velocity_x,
                identity.last_velocity_y,
                fragment.velocity_x,
                fragment.velocity_y,
            )

            if cosine < 0.25:
                return -1.0

            speed_ratio = (
                min(old_speed, new_speed)
                /
                max(old_speed, new_speed)
            )

            velocity_score = (
                0.5 * ((cosine + 1.0) / 2.0)
                +
                0.5 * speed_ratio
            )

        score = (
            0.50 * spatial_score
            +
            0.20 * temporal_score
            +
            0.20 * velocity_score
            +
            0.10 * side_score
        )

        return float(score)

    # ------------------------------------------------------------------
    # IDENTITY
    # ------------------------------------------------------------------

    @staticmethod
    def _create_identity(
        crossing_id: int,
        fragment: TrackFragment,
    ) -> CrossingIdentity:

        return CrossingIdentity(
            crossing_id=crossing_id,
            track_ids=[fragment.track_id],
            vehicle_class=fragment.class_name,
            class_ratio=fragment.class_ratio,
            class_ambiguous=fragment.class_ambiguous,
            first_frame=fragment.first_frame,
            last_frame=fragment.last_frame,
            first_time_sec=fragment.first_time_sec,
            last_time_sec=fragment.last_time_sec,
            last_x=fragment.last_x,
            last_y=fragment.last_y,
            last_side=fragment.last_side,
            last_distance_px=fragment.last_distance_px,
            last_velocity_x=fragment.velocity_x,
            last_velocity_y=fragment.velocity_y,
        )

    @staticmethod
    def _update_identity_meta(
        identity: CrossingIdentity,
        fragment: TrackFragment,
    ) -> None:

        if fragment.track_id not in identity.track_ids:
            identity.track_ids.append(
                fragment.track_id
            )

        if (
            fragment.class_name
            != identity.vehicle_class
        ):
            identity.class_ambiguous = True

        identity.last_frame = fragment.last_frame
        identity.last_time_sec = fragment.last_time_sec
        identity.last_x = fragment.last_x
        identity.last_y = fragment.last_y
        identity.last_side = fragment.last_side
        identity.last_distance_px = fragment.last_distance_px
        identity.last_velocity_x = fragment.velocity_x
        identity.last_velocity_y = fragment.velocity_y

        if (
            fragment.class_ratio
            >
            identity.class_ratio
        ):
            identity.class_ratio = (
                fragment.class_ratio
            )

    # ------------------------------------------------------------------
    # RUN
    # ------------------------------------------------------------------

    def run(
        self,
        tracks_phase2: pd.DataFrame,
    ) -> tuple[
        pd.DataFrame,
        dict[int, CrossingIdentity],
        dict[int, int],
        dict[str, int],
    ]:

        trajectory = self.prepare(
            tracks_phase2
        )

        fragments = self.build_fragments(
            trajectory
        )

        identities: list[CrossingIdentity] = []

        track_to_identity: dict[int, int] = {}

        reconnection_count = 0

        for fragment in fragments:

            candidates: list[
                tuple[float, CrossingIdentity]
            ] = []

            for identity in identities:

                score = self._candidate_score(
                    identity,
                    fragment,
                )

                if score >= 0:
                    candidates.append(
                        (score, identity)
                    )

            candidates.sort(
                key=lambda item: item[0],
                reverse=True,
            )

            assigned_identity = None

            if candidates:

                best_score, best_identity = (
                    candidates[0]
                )

                second_score = (
                    candidates[1][0]
                    if len(candidates) > 1
                    else -1.0
                )

                margin_ok = (
                    second_score < 0
                    or
                    (
                        best_score
                        -
                        second_score
                        >=
                        self.identity_match_margin
                    )
                )

                # Conservative identity merge.
                if (
                    best_score
                    >=
                    self.identity_match_threshold
                    and
                    margin_ok
                ):
                    assigned_identity = (
                        best_identity
                    )

            if assigned_identity is None:

                assigned_identity = (
                    self._create_identity(
                        crossing_id=len(identities) + 1,
                        fragment=fragment,
                    )
                )

                identities.append(
                    assigned_identity
                )

            else:

                assigned_identity.track_ids.append(
                    fragment.track_id
                )

                reconnection_count += 1

                self._update_identity_meta(
                    assigned_identity,
                    fragment,
                )

            track_to_identity[
                fragment.track_id
            ] = assigned_identity.crossing_id

        # ------------------------------------------------------
        # Identity metadata maps.
        # ------------------------------------------------------

        identity_map = {
            identity.crossing_id: identity
            for identity in identities
        }

        trajectory = trajectory.copy()

        trajectory[
            "crossing_id"
        ] = (
            trajectory["track_id"]
            .map(track_to_identity)
        )

        # Useful aliases for downstream code.
        trajectory[
            "raw_track_id"
        ] = trajectory["track_id"]

        return (
            trajectory,
            identity_map,
            track_to_identity,
            {
                "unique_track_ids": int(
                    trajectory["track_id"].nunique()
                ),
                "crossing_identities": int(
                    len(identities)
                ),
                "track_reconnections": int(
                    reconnection_count
                ),
                "fragmented_identities": int(
                    sum(
                        len(identity.track_ids) > 1
                        for identity in identities
                    )
                ),
            },
        )

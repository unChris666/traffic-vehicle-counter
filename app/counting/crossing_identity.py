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
    One raw track_id can be one fragment of a physical vehicle.
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
    """
    Physical-vehicle identity near the counting zone.

    A single crossing identity can contain multiple
    track_ids if BoT-SORT fragmented the track.
    """

    crossing_id: int

    track_ids: list[int] = field(
        default_factory=list
    )

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

    crossing_frame: int | None = None
    crossing_time_sec: float | None = None

    crossing_x: float | None = None
    crossing_y: float | None = None

    crossing_track_id: int | None = None

    direction: str | None = None

    counted: bool = False


class CrossingIdentityEngine:
    """
    Phase 1:

        track_id fragments
                ↓
        crossing identity
                ↓
        crossing state machine

    Important:
    track_id is NOT the final counting identity.
    crossing_id is.
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

        min_pre_crossing_observations: int = 3,

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

        self.min_pre_crossing_observations = int(
            min_pre_crossing_observations
        )

    # =========================================================
    # GEOMETRY
    # =========================================================

    def _line_geometry(
        self,
        x: float,
        y: float,
    ) -> tuple[float, float, int]:

        line_value = (
            self.line_dx
            * (y - self.y1)
            - self.line_dy
            * (x - self.x1)
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

    # =========================================================
    # VELOCITY
    # =========================================================

    @staticmethod
    def _estimate_velocity(
        rows: pd.DataFrame,
        *,
        tail: bool,
    ) -> tuple[float, float]:

        if len(rows) < 2:
            return 0.0, 0.0

        if tail:
            sample = rows.tail(5)

        else:
            sample = rows.head(5)

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
            (
                dx[valid]
                / dt[valid]
            ).median()
        )

        vy = float(
            (
                dy[valid]
                / dt[valid]
            ).median()
        )

        return vx, vy

    @staticmethod
    def _velocity_similarity(
        ax: float,
        ay: float,
        bx: float,
        by: float,
    ) -> float:

        norm_a = math.hypot(
            ax,
            ay,
        )

        norm_b = math.hypot(
            bx,
            by,
        )

        if (
            norm_a < 1e-6
            or norm_b < 1e-6
        ):
            return 0.5

        cosine = (
            ax * bx
            + ay * by
        ) / (
            norm_a
            * norm_b
        )

        return max(
            -1.0,
            min(1.0, cosine),
        )

    # =========================================================
    # PREPARE TRACK OBSERVATIONS
    # =========================================================

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
            "class_name",
            "confidence",
        }

        missing = (
            required
            - set(
                tracks_phase2.columns
            )
        )

        if missing:
            raise ValueError(
                "tracks_phase2 missing "
                f"required columns: {sorted(missing)}"
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

        geometry = trajectory.apply(
            lambda row: self._line_geometry(
                float(
                    row[
                        "bottom_center_x"
                    ]
                ),
                float(
                    row[
                        "bottom_center_y"
                    ]
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

    # =========================================================
    # BUILD TRACK FRAGMENTS
    # =========================================================

    def build_fragments(
        self,
        trajectory: pd.DataFrame,
    ) -> list[TrackFragment]:

        fragments: list[
            TrackFragment
        ] = []

        for (
            track_id,
            rows,
        ) in trajectory.groupby(
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

            if stable.empty:

                first_side = 0
                last_side = 0

            else:

                first_side = int(
                    stable.iloc[0]["side"]
                )

                last_side = int(
                    stable.iloc[-1]["side"]
                )

            first_velocity = (
                self._estimate_velocity(
                    rows,
                    tail=False,
                )
            )

            last_velocity = (
                self._estimate_velocity(
                    rows,
                    tail=True,
                )
            )

            class_counts = (
                rows["class_name"]
                .value_counts()
            )

            if class_counts.empty:

                class_name = str(
                    rows.iloc[0][
                        "track_class"
                    ]
                )

                class_ratio = float(
                    rows.iloc[0][
                        "track_class_ratio"
                    ]
                )

            else:

                class_name = str(
                    class_counts
                    .index[0]
                )

                class_ratio = float(
                    class_counts.iloc[0]
                    / class_counts.sum()
                )

            fragments.append(
                TrackFragment(
                    track_id=int(
                        track_id
                    ),

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
                        rows[
                            "timestamp_sec"
                        ].min()
                    ),

                    last_time_sec=float(
                        rows[
                            "timestamp_sec"
                        ].max()
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
                        first_velocity[0]
                    ),

                    velocity_y=float(
                        first_velocity[1]
                    ),

                    observation_count=int(
                        len(rows)
                    ),

                    mean_confidence=float(
                        rows[
                            "confidence"
                        ].mean()
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

    # =========================================================
    # CROSSING IDENTITY MATCH
    # =========================================================

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

        if (
            gap_frames <= 0
            or gap_frames
            > self.max_reconnect_gap_frames
        ):
            return -1.0

        distance = math.hypot(
            fragment.first_x
            - identity.last_x,

            fragment.first_y
            - identity.last_y,
        )

        if (
            distance
            > self.max_reconnect_distance_px
        ):
            return -1.0

        # -----------------------------------------------------
        # Class gate
        # -----------------------------------------------------

        if (
            identity.vehicle_class
            != "unknown"
            and fragment.class_name
            != identity.vehicle_class
        ):
            return -1.0

        # -----------------------------------------------------
        # Spatial score
        # -----------------------------------------------------

        spatial_score = max(
            0.0,
            1.0
            - (
                distance
                / self.max_reconnect_distance_px
            ),
        )

        # -----------------------------------------------------
        # Temporal score
        # -----------------------------------------------------

        temporal_score = max(
            0.0,
            1.0
            - (
                gap_frames
                / self.max_reconnect_gap_frames
            ),
        )

        # -----------------------------------------------------
        # Side compatibility
        # -----------------------------------------------------

        side_score = 1.0

        if (
            identity.last_side != 0
            and fragment.first_side != 0
        ):

            if (
                identity.last_side
                == fragment.first_side
            ):
                side_score = 1.0

            else:

                # Side can change across an occlusion,
                # but only when both fragments are close
                # to the counting line.
                if (
                    identity.last_distance_px
                    > self.line_deadband_px * 3
                    and
                    fragment.first_distance_px
                    > self.line_deadband_px * 3
                ):
                    return -1.0

                side_score = 0.35

        # -----------------------------------------------------
        # Velocity compatibility
        # -----------------------------------------------------

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
            old_speed
            > 1.0
            and new_speed
            > 1.0
        ):

            cosine = (
                self._velocity_similarity(
                    identity.last_velocity_x,
                    identity.last_velocity_y,

                    fragment.velocity_x,
                    fragment.velocity_y,
                )
            )

            if cosine < 0.50:
                return -1.0

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
                0.5
                * (
                    (cosine + 1.0)
                    / 2.0
                )
                +
                0.5
                * speed_ratio
            )

        # -----------------------------------------------------
        # Final score
        # -----------------------------------------------------

        score = (
            0.45 * spatial_score
            +
            0.20 * temporal_score
            +
            0.20 * velocity_score
            +
            0.15 * side_score
        )

        return float(score)

    # =========================================================
    # CREATE IDENTITY
    # =========================================================

    @staticmethod
    def _create_identity(
        crossing_id: int,
        fragment: TrackFragment,
    ) -> CrossingIdentity:

        return CrossingIdentity(
            crossing_id=crossing_id,

            track_ids=[
                fragment.track_id
            ],

            state=CrossingState.NOT_SEEN,

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
        )

    # =========================================================
    # UPDATE IDENTITY META
    # =========================================================

    def _update_identity_meta(
        self,
        identity: CrossingIdentity,
        fragment: TrackFragment,
    ) -> None:

        if (
            fragment.track_id
            not in identity.track_ids
        ):
            identity.track_ids.append(
                fragment.track_id
            )

        # Keep original identity class,
        # but mark ambiguity if fragments disagree.
        if (
            fragment.class_name
            != identity.vehicle_class
        ):
            identity.class_ambiguous = True

        else:

            identity.class_ratio = float(
                (
                    identity.class_ratio
                    + fragment.class_ratio
                )
                / 2.0
            )

        identity.last_velocity_x = (
            fragment.velocity_x
        )

        identity.last_velocity_y = (
            fragment.velocity_y
        )

    # =========================================================
    # STATE MACHINE
    # =========================================================

    def _process_observation(
        self,
        identity: CrossingIdentity,
        row: pd.Series,
        previous_frame: int | None,
    ) -> None:

        frame_id = int(
            row["frame_id"]
        )

        timestamp_sec = float(
            row["timestamp_sec"]
        )

        x = float(
            row["bottom_center_x"]
        )

        y = float(
            row["bottom_center_y"]
        )

        side = int(
            row["side"]
        )

        distance_px = float(
            row["line_distance_px"]
        )

        # -----------------------------------------------------
        # If already counted, remain post-crossing.
        # -----------------------------------------------------

        if identity.counted:

            identity.state = (
                CrossingState.POST_CROSSING
            )

            identity.last_frame = frame_id
            identity.last_time_sec = timestamp_sec
            identity.last_x = x
            identity.last_y = y
            identity.last_side = (
                side
                if side != 0
                else identity.last_side
            )
            identity.last_distance_px = (
                distance_px
            )

            return

        # -----------------------------------------------------
        # Deadband: keep state but don't change side.
        # -----------------------------------------------------

        if side == 0:

            identity.last_frame = frame_id
            identity.last_time_sec = timestamp_sec
            identity.last_x = x
            identity.last_y = y
            identity.last_distance_px = distance_px

            return

        # -----------------------------------------------------
        # Temporal validity.
        # -----------------------------------------------------

        valid_temporal_transition = True

        if previous_frame is not None:

            frame_gap = (
                frame_id
                - previous_frame
            )

            if (
                frame_gap
                > self.max_crossing_gap_frames
            ):
                valid_temporal_transition = False

        # -----------------------------------------------------
        # NOT_SEEN → APPROACHING
        # -----------------------------------------------------

        if (
            identity.state
            == CrossingState.NOT_SEEN
        ):

            identity.state = (
                CrossingState.APPROACHING
            )

            identity.last_side = side
            identity.last_frame = frame_id
            identity.last_time_sec = timestamp_sec
            identity.last_x = x
            identity.last_y = y
            identity.last_distance_px = distance_px

            return

        # -----------------------------------------------------
        # APPROACHING → PRE_CROSSING
        # -----------------------------------------------------

        if (
            identity.state
            == CrossingState.APPROACHING
        ):

            same_side = (
                side
                == identity.last_side
            )

            close_to_line = (
                distance_px
                <= self.pre_crossing_distance_px
            )

            if (
                same_side
                and close_to_line
            ):

                identity.stable_observations_before_crossing += 1

                if (
                    identity.stable_observations_before_crossing
                    >= self.min_pre_crossing_observations
                ):

                    identity.state = (
                        CrossingState.PRE_CROSSING
                    )

            else:

                identity.stable_observations_before_crossing = 0

        # -----------------------------------------------------
        # PRE_CROSSING → CROSSING → COUNTED
        # -----------------------------------------------------

        elif (
            identity.state
            == CrossingState.PRE_CROSSING
        ):

            if (
                valid_temporal_transition
                and identity.last_side != 0
                and side != identity.last_side
            ):

                identity.state = (
                    CrossingState.CROSSING
                )

                identity.crossing_frame = (
                    frame_id
                )

                identity.crossing_time_sec = (
                    timestamp_sec
                )

                identity.crossing_x = x
                identity.crossing_y = y

                identity.crossing_track_id = (
                    int(row["track_id"])
                )

                identity.direction = (
                    f"side_{identity.last_side:+d}"
                    f"_to_{side:+d}"
                )

                identity.counted = True

                identity.state = (
                    CrossingState.POST_CROSSING
                )

        # -----------------------------------------------------
        # Update previous pose
        # -----------------------------------------------------

        identity.last_frame = frame_id
        identity.last_time_sec = timestamp_sec
        identity.last_x = x
        identity.last_y = y
        identity.last_side = side
        identity.last_distance_px = distance_px

    # =========================================================
    # MAIN
    # =========================================================

    def run(
        self,
        tracks_phase2: pd.DataFrame,
    ) -> tuple[
        pd.DataFrame,
        pd.DataFrame,
        dict[str, int],
    ]:

        trajectory = self.prepare(
            tracks_phase2
        )

        fragments = self.build_fragments(
            trajectory
        )

        identities: list[
            CrossingIdentity
        ] = []

        track_to_identity: dict[
            int,
            int,
        ] = {}

        reconnection_count = 0

        # -----------------------------------------------------
        # Fragment → crossing identity
        # -----------------------------------------------------

        for fragment in fragments:

            candidates: list[
                tuple[
                    float,
                    CrossingIdentity,
                ]
            ] = []

            for identity in identities:

                score = (
                    self._candidate_score(
                        identity,
                        fragment,
                    )
                )

                if score >= 0:
                    candidates.append(
                        (
                            score,
                            identity,
                        )
                    )

            candidates.sort(
                key=lambda item: item[0],
                reverse=True,
            )

            assigned_identity = None

            if candidates:

                best_score = candidates[0][0]
                best_identity = candidates[0][1]

                second_score = (
                    candidates[1][0]
                    if len(candidates) > 1
                    else -1.0
                )

                sufficient_margin = (
                    second_score < 0
                    or (
                        best_score
                        - second_score
                        >= self.identity_match_margin
                    )
                )

                if (
                    best_score
                    >= self.identity_match_threshold
                    and sufficient_margin
                ):

                    assigned_identity = (
                        best_identity
                    )

            # -------------------------------------------------
            # New crossing identity
            # -------------------------------------------------

            if (
                assigned_identity
                is None
            ):

                assigned_identity = (
                    self._create_identity(
                        crossing_id=(
                            len(identities)
                            + 1
                        ),
                        fragment=fragment,
                    )
                )

                identities.append(
                    assigned_identity
                )

            # -------------------------------------------------
            # Reconnected fragment
            # -------------------------------------------------

            else:

                if (
                    fragment.track_id
                    not in assigned_identity.track_ids
                ):

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
            ] = (
                assigned_identity.crossing_id
            )

            # -------------------------------------------------
            # Replay observations
            # -------------------------------------------------

            previous_frame = None

            for _, row in (
                fragment.rows
                .sort_values("frame_id")
                .iterrows()
            ):

                self._process_observation(
                    assigned_identity,
                    row,
                    previous_frame,
                )

                previous_frame = int(
                    row["frame_id"]
                )

        # -----------------------------------------------------
        # Attach crossing ID to each observation.
        # -----------------------------------------------------

        trajectory[
            "crossing_id"
        ] = (
            trajectory["track_id"]
            .map(track_to_identity)
        )

        # -----------------------------------------------------
        # Identity-level state
        # -----------------------------------------------------

        identity_state_map = {
            identity.crossing_id:
                identity.state.value
            for identity in identities
        }

        trajectory[
            "crossing_state"
        ] = (
            trajectory["crossing_id"]
            .map(identity_state_map)
        )

        # -----------------------------------------------------
        # Build crossing events
        # -----------------------------------------------------

        crossing_rows = []

        for identity in identities:

            if not identity.counted:
                continue

            crossing_rows.append(
                {
                    "crossing_id": (
                        identity.crossing_id
                    ),

                    # IMPORTANT:
                    # Keep one raw track_id for compatibility
                    # with existing confidence engine.
                    "track_id": (
                        identity.crossing_track_id
                    ),

                    "track_ids": ",".join(
                        map(
                            str,
                            identity.track_ids,
                        )
                    ),

                    "crossing_frame": (
                        identity.crossing_frame
                    ),

                    "crossing_time_sec": (
                        identity.crossing_time_sec
                    ),

                    "crossing_x": (
                        identity.crossing_x
                    ),

                    "crossing_y": (
                        identity.crossing_y
                    ),

                    "direction": (
                        identity.direction
                    ),

                    "track_class": (
                        identity.vehicle_class
                    ),

                    "track_class_ratio": (
                        identity.class_ratio
                    ),

                    "class_ambiguous": (
                        identity.class_ambiguous
                    ),

                    "state": (
                        identity.state.value
                    ),

                    "num_track_fragments": (
                        len(
                            identity.track_ids
                        )
                    ),
                }
            )

        crossing_events = pd.DataFrame(
            crossing_rows
        )

        if crossing_events.empty:

            crossing_events = pd.DataFrame(
                columns=[
                    "crossing_id",
                    "track_id",
                    "track_ids",
                    "crossing_frame",
                    "crossing_time_sec",
                    "crossing_x",
                    "crossing_y",
                    "direction",
                    "track_class",
                    "track_class_ratio",
                    "class_ambiguous",
                    "state",
                    "num_track_fragments",
                ]
            )

        # -----------------------------------------------------
        # Audit
        # -----------------------------------------------------

        audit = {
            "unique_track_ids": int(
                trajectory["track_id"]
                .nunique()
            ),

            "crossing_identities": int(
                len(identities)
            ),

            "track_reconnections": int(
                reconnection_count
            ),

            "fragmented_identities": int(
                sum(
                    len(
                        identity.track_ids
                    ) > 1
                    for identity
                    in identities
                )
            ),

            "counted_crossing_identities": int(
                sum(
                    identity.counted
                    for identity
                    in identities
                )
            ),

            "not_crossed_identities": int(
                sum(
                    not identity.counted
                    for identity
                    in identities
                )
            ),
        }

        return (
            trajectory,
            crossing_events,
            audit,
        )

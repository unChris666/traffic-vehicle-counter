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

    # Reconstruction audit
    last_match_score: float = 0.0
    last_match_reason: str = ""
    reconnect_count: int = 0


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
        identity_same_side_near_line_block: bool = True,
        identity_prediction_gate_max_px: float = 220.0,
        identity_min_velocity_cosine: float = 0.35,
        identity_min_normal_velocity_px_per_frame: float = 1.0,
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
        self.identity_same_side_near_line_block = bool(identity_same_side_near_line_block)
        self.identity_prediction_gate_max_px = float(identity_prediction_gate_max_px)
        self.identity_min_velocity_cosine = float(identity_min_velocity_cosine)
        self.identity_min_normal_velocity_px_per_frame = float(identity_min_normal_velocity_px_per_frame)

        self._audit_counters = {
            "same_side_near_line_blocks": 0,
            "prediction_gate_used": 0,
            "prediction_gate_rejections": 0,
            "velocity_conflict_rejections": 0,
            "direction_conflict_rejections": 0,
            "class_conflict_rejections": 0,
            "ambiguous_identity_matches": 0,
        }

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

    def _normal_velocity(self, vx: float, vy: float) -> float:
        nx = -self.line_dy / self.line_length
        ny = self.line_dx / self.line_length
        return float(vx * nx + vy * ny)

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
    ) -> tuple[float, str]:
        """Conservative trajectory-aware fragment reconstruction.

        A reconnect is accepted only when temporal continuity, predicted
        position, class compatibility, velocity direction and counting-line
        geometry agree. Near-line same-side fragments are deliberately NOT
        merged: that pattern is more consistent with two vehicles arriving
        close together than with one fragmented vehicle.
        """
        if identity.last_frame is None or identity.last_x is None or identity.last_y is None:
            return -1.0, "missing_identity_state"

        gap = int(fragment.first_frame - identity.last_frame)
        if gap <= 0 or gap > self.max_reconnect_gap_frames:
            return -1.0, "temporal_overlap_or_gap"

        # Hard same-side near-line guard. This is the most important anti-merge
        # rule for two vehicles entering the counting corridor together.
        near_old = identity.last_distance_px <= self.pre_crossing_distance_px
        near_new = fragment.first_distance_px <= self.pre_crossing_distance_px
        same_side = (
            identity.last_side != 0
            and fragment.first_side != 0
            and identity.last_side == fragment.first_side
        )
        if self.identity_same_side_near_line_block and near_old and near_new and same_side:
            self._audit_counters["same_side_near_line_blocks"] += 1
            return -1.0, "same_side_near_line_block"

        # Class compatibility. Unknown is weakly compatible; person↔motorcycle
        # is retained as a rider/vehicle ambiguity case but never preferred over
        # a same-class match. Different vehicle classes are hard incompatible.
        a = str(identity.vehicle_class).strip().lower()
        b = str(fragment.class_name).strip().lower()
        if a not in {"", "unknown"} and b not in {"", "unknown"}:
            if a != b and {a, b} != {"person", "motorcycle"}:
                self._audit_counters["class_conflict_rejections"] += 1
                return -1.0, "class_incompatible"

        old_v = (float(identity.last_velocity_x), float(identity.last_velocity_y))
        new_v = (float(fragment.velocity_x), float(fragment.velocity_y))
        old_speed = math.hypot(*old_v)
        new_speed = math.hypot(*new_v)

        # Predict forward when velocity is reliable. Raw endpoint distance is
        # only used as a fallback for nearly stationary fragments.
        if old_speed >= self.velocity_gate_px_per_frame or new_speed >= self.velocity_gate_px_per_frame:
            predicted_x = identity.last_x + old_v[0] * gap
            predicted_y = identity.last_y + old_v[1] * gap
            continuity_distance = math.hypot(fragment.first_x - predicted_x, fragment.first_y - predicted_y)
            self._audit_counters["prediction_gate_used"] += 1
        else:
            continuity_distance = math.hypot(fragment.first_x - identity.last_x, fragment.first_y - identity.last_y)

        expected_motion = max(old_speed, new_speed, 1.0) * gap
        dynamic_gate = max(
            self.max_reconnect_distance_px,
            min(self.identity_prediction_gate_max_px, self.velocity_gate_px_per_frame * gap * 1.25),
            min(self.identity_prediction_gate_max_px, expected_motion * 1.15),
        )
        if continuity_distance > dynamic_gate:
            self._audit_counters["prediction_gate_rejections"] += 1
            return -1.0, f"prediction_gate:{continuity_distance:.1f}>{dynamic_gate:.1f}"

        spatial_score = max(0.0, 1.0 - continuity_distance / max(dynamic_gate, 1e-6))
        temporal_score = max(0.0, 1.0 - gap / max(self.max_reconnect_gap_frames, 1))

        velocity_score = 0.55
        cosine = 0.5
        if old_speed > 1.0 and new_speed > 1.0:
            cosine = self._velocity_similarity(old_v[0], old_v[1], new_v[0], new_v[1])
            if cosine < self.identity_min_velocity_cosine:
                self._audit_counters["velocity_conflict_rejections"] += 1
                return -1.0, f"velocity_direction_conflict:{cosine:.3f}"
            speed_ratio = min(old_speed, new_speed) / max(old_speed, new_speed)
            velocity_score = 0.60 * ((cosine + 1.0) / 2.0) + 0.40 * speed_ratio

        old_side = identity.last_side
        new_side = fragment.first_side
        side_score = 1.0
        if old_side != 0 and new_side != 0 and old_side != new_side:
            # A side transition is plausible only when at least one endpoint
            # is in the pre-crossing corridor. Otherwise it is an implausible
            # identity jump.
            if not (near_old or near_new):
                self._audit_counters["direction_conflict_rejections"] += 1
                return -1.0, "implausible_side_change"
            side_score = 0.90

        # Normal velocity must not reverse across a short gap. A reversal
        # indicates two different objects or a noisy identity jump.
        old_nv = self._normal_velocity(*old_v)
        new_nv = self._normal_velocity(*new_v)
        if (
            abs(old_nv) >= self.identity_min_normal_velocity_px_per_frame
            and abs(new_nv) >= self.identity_min_normal_velocity_px_per_frame
            and old_nv * new_nv < 0
        ):
            self._audit_counters["direction_conflict_rejections"] += 1
            return -1.0, "normal_velocity_reverse"

        normal_score = 1.0 if old_nv * new_nv >= 0 else 0.85
        class_score = 1.0 if a == b else 0.55 if {a, b} == {"person", "motorcycle"} else 0.65

        score = (
            0.38 * spatial_score
            + 0.18 * temporal_score
            + 0.18 * velocity_score
            + 0.12 * side_score
            + 0.09 * normal_score
            + 0.05 * class_score
        )

        if score < self.identity_match_threshold:
            return float(score), f"below_threshold:{score:.3f}"

        return float(score), "accepted"

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
                tuple[float, CrossingIdentity, str]
            ] = []

            for identity in identities:

                score, reason = self._candidate_score(
                    identity,
                    fragment,
                )

                if score >= 0:
                    candidates.append(
                        (score, identity, reason)
                    )

            candidates.sort(
                key=lambda item: item[0],
                reverse=True,
            )

            assigned_identity = None

            if candidates:

                best_score, best_identity, best_reason = candidates[0]

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
                    best_score >= self.identity_match_threshold
                    and margin_ok
                ):
                    assigned_identity = best_identity
                    assigned_identity.last_match_score = float(best_score)
                    assigned_identity.last_match_reason = str(best_reason)
                elif best_score >= self.identity_match_threshold and not margin_ok:
                    self._audit_counters["ambiguous_identity_matches"] += 1

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

                if fragment.track_id not in assigned_identity.track_ids:
                    assigned_identity.track_ids.append(fragment.track_id)
                assigned_identity.reconnect_count += 1
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
                **{k: int(v) for k, v in self._audit_counters.items()},
            },
        )
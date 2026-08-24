from __future__ import annotations

from dataclasses import dataclass
import ast
import math

import numpy as np
import pandas as pd


VEHICLE_CLASSES = {
    "motorcycle",
    "car",
    "truck",
    "bus",
}


@dataclass(frozen=True)
class ResolverConfig:

    max_sequential_frame_gap: int = 10

    max_crossing_distance_px: float = 65.0

    max_endpoint_distance_px: float = 80.0

    min_direction_cosine: float = 0.65

    duplicate_score_threshold: float = 0.82

    same_identity_duplicate_threshold: float = 0.72

    fast_min_normal_displacement_px: float = 10.0

    fast_max_gap_frames: int = 10

    min_vehicle_class_confidence: float = 0.45

    person_motorcycle_vehicle_confidence: float = 0.60


class CrossingCandidateResolver:

    """
    Resolves candidate-level identity ambiguity.

    Responsibilities:

        1. Fast crossing evidence.
        2. Same-frame multi-crossing resolution.
        3. Sequential duplicate detection.
        4. Shared-track duplicate detection.
        5. Vehicle eligibility.
        6. Candidate lineage audit.

    It does NOT perform final counting.
    """

    def __init__(
        self,
        config: ResolverConfig | None = None,
    ) -> None:

        self.config = (
            config
            or
            ResolverConfig()
        )

    # ============================================================
    # HELPERS
    # ============================================================

    @staticmethod
    def _safe_float(
        value,
        default=0.0,
    ):

        try:
            value = float(value)

            if not math.isfinite(value):
                return default

            return value

        except (
            TypeError,
            ValueError,
        ):
            return default

    @staticmethod
    def _track_ids(
        value,
    ) -> set[int]:

        if value is None:
            return set()

        if isinstance(
            value,
            (list, tuple, set),
        ):
            result = value

        else:

            text = str(value).strip()

            try:
                result = ast.literal_eval(
                    text
                )
            except Exception:
                result = [
                    x.strip()
                    for x in text.split(",")
                    if x.strip()
                ]

        output = set()

        for item in result:

            try:
                output.add(
                    int(item)
                )
            except (
                TypeError,
                ValueError,
            ):
                continue

        return output

    @staticmethod
    def _direction_vector(
        rows: pd.DataFrame,
    ) -> np.ndarray:

        if rows.empty:
            return np.zeros(2)

        rows = (
            rows
            .sort_values("frame_id")
        )

        if len(rows) < 2:
            return np.zeros(2)

        first = rows.iloc[0]
        last = rows.iloc[-1]

        return np.array(
            [
                float(
                    last["raw_x"]
                    -
                    first["raw_x"]
                ),
                float(
                    last["raw_y"]
                    -
                    first["raw_y"]
                ),
            ],
            dtype=float,
        )

    @staticmethod
    def _cosine(
        a: np.ndarray,
        b: np.ndarray,
    ) -> float:

        na = float(
            np.linalg.norm(a)
        )

        nb = float(
            np.linalg.norm(b)
        )

        if (
            na < 1e-6
            or
            nb < 1e-6
        ):
            return 0.5

        return float(
            np.clip(
                np.dot(a, b)
                /
                (na * nb),
                -1.0,
                1.0,
            )
        )

    # ============================================================
    # VEHICLE ELIGIBILITY
    # ============================================================

    def _vehicle_eligibility(
        self,
        row: pd.Series,
    ) -> tuple[
        bool,
        str,
        str,
    ]:

        cls = str(
            row.get(
                "counting_class",
                row.get(
                    "track_class",
                    "unknown",
                ),
            )
        ).lower().strip()

        confidence = self._safe_float(
            row.get(
                "counting_class_confidence",
                row.get(
                    "track_class_ratio",
                    0.0,
                ),
            )
        )

        # BLOCKING RULE.
        if cls == "person":
            return (
                False,
                "PERSON",
                "person_never_enters_vehicle_count",
            )

        if cls not in VEHICLE_CLASSES:
            return (
                False,
                "UNKNOWN",
                "non_vehicle_class",
            )

        if (
            confidence
            <
            self.config.min_vehicle_class_confidence
        ):
            return (
                False,
                "LOW_CLASS_CONFIDENCE",
                "vehicle_class_confidence_below_threshold",
            )

        return (
            True,
            cls,
            "vehicle_eligible",
        )

    # ============================================================
    # FAST CROSSING
    # ============================================================

    def _fast_crossing_evidence(
        self,
        row: pd.Series,
    ) -> dict:

        fast = bool(
            row.get(
                "fast_crossing",
                False,
            )
        )

        sparse = bool(
            row.get(
                "sparse_crossing",
                False,
            )
        )

        short = bool(
            row.get(
                "short_track",
                False,
            )
        )

        gap = int(
            self._safe_float(
                row.get(
                    "frame_gap",
                    0,
                )
            )
        )

        normal_displacement = abs(
            self._safe_float(
                row.get(
                    "normal_displacement_px",
                    0.0,
                )
            )
        )

        direction_confidence = self._safe_float(
            row.get(
                "direction_confidence",
                0.0,
            )
        )

        geometry = bool(
            row.get(
                "geometry_crossing",
                False,
            )
        )

        score = 0.0

        score += (
            0.35
            *
            min(
                normal_displacement
                /
                max(
                    self.config.fast_min_normal_displacement_px,
                    1.0,
                ),
                1.0,
            )
        )

        score += (
            0.30
            *
            direction_confidence
        )

        score += (
            0.20
            *
            float(
                geometry
            )
        )

        score += (
            0.15
            *
            (
                1.0
                -
                min(
                    gap,
                    self.config.fast_max_gap_frames,
                )
                /
                max(
                    self.config.fast_max_gap_frames,
                    1,
                )
            )
        )

        score = float(
            np.clip(
                score,
                0.0,
                1.0,
            )
        )

        return {
            "fast_crossing": fast,
            "sparse_crossing": sparse,
            "short_track": short,
            "fast_crossing_confidence": score,
            "fast_crossing_valid": (
                geometry
                and
                direction_confidence >= 0.45
                and
                normal_displacement
                >=
                self.config.fast_min_normal_displacement_px
            ),
        }

    # ============================================================
    # SAME FRAME
    # ============================================================

    def _same_frame_relation(
        self,
        a: pd.Series,
        b: pd.Series,
    ) -> tuple[
        bool,
        float,
        str,
    ]:

        fa = int(
            self._safe_float(
                a["crossing_frame"]
            )
        )

        fb = int(
            self._safe_float(
                b["crossing_frame"]
            )
        )

        if fa != fb:
            return (
                False,
                0.0,
                "not_same_frame",
            )

        tracks_a = self._track_ids(
            a.get("track_ids")
        )

        tracks_b = self._track_ids(
            b.get("track_ids")
        )

        shared_tracks = (
            tracks_a
            &
            tracks_b
        )

        identity_a = int(
            self._safe_float(
                a.get(
                    "identity_id",
                    -1,
                ),
                -1,
            )
        )

        identity_b = int(
            self._safe_float(
                b.get(
                    "identity_id",
                    -2,
                ),
                -2,
            )
        )

        # --------------------------------------------------------
        # SAME RAW TRACK = DEFINITE DUPLICATE
        # --------------------------------------------------------

        if shared_tracks:

            return (
                True,
                0.99,
                (
                    "same_frame_shared_raw_track:"
                    +
                    ",".join(
                        map(
                            str,
                            sorted(
                                shared_tracks
                            ),
                        )
                    )
                ),
            )

        # --------------------------------------------------------
        # SAME PHYSICAL IDENTITY
        # --------------------------------------------------------

        if (
            identity_a >= 0
            and
            identity_a == identity_b
        ):

            return (
                True,
                0.94,
                "same_frame_same_identity",
            )

        # --------------------------------------------------------
        # DIFFERENT RAW TRACKS / DIFFERENT IDENTITIES
        #
        # IMPORTANT:
        # NEVER suppress merely because same frame.
        # --------------------------------------------------------

        return (
            False,
            0.0,
            "same_frame_independent_tracks",
        )

    # ============================================================
    # SEQUENTIAL DUPLICATE
    # ============================================================

    def _sequential_duplicate(
        self,
        a: pd.Series,
        b: pd.Series,
        prepared: pd.DataFrame,
    ) -> tuple[
        bool,
        float,
        str,
    ]:

        fa = int(
            self._safe_float(
                a["crossing_frame"]
            )
        )

        fb = int(
            self._safe_float(
                b["crossing_frame"]
            )
        )

        if fa == fb:
            return (
                self._same_frame_relation(
                    a,
                    b,
                )
            )

        if fa < fb:
            early = a
            late = b
        else:
            early = b
            late = a

        frame_gap = int(
            self._safe_float(
                late["crossing_frame"]
            )
            -
            self._safe_float(
                early["crossing_frame"]
            )
        )

        if (
            frame_gap <= 0
            or
            frame_gap
            >
            self.config.max_sequential_frame_gap
        ):
            return (
                False,
                0.0,
                "outside_temporal_window",
            )

        class_a = str(
            early.get(
                "counting_class",
                early.get(
                    "track_class",
                    "unknown",
                ),
            )
        ).lower()

        class_b = str(
            late.get(
                "counting_class",
                late.get(
                    "track_class",
                    "unknown",
                ),
            )
        ).lower()

        if class_a != class_b:
            return (
                False,
                0.0,
                "class_mismatch",
            )

        direction_a = str(
            early.get(
                "direction",
                "UNKNOWN",
            )
        )

        direction_b = str(
            late.get(
                "direction",
                "UNKNOWN",
            )
        )

        if (
            direction_a == "UNKNOWN"
            or
            direction_b == "UNKNOWN"
            or
            direction_a
            !=
            direction_b
        ):
            return (
                False,
                0.0,
                "direction_mismatch",
            )

        tracks_early = self._track_ids(
            early.get("track_ids")
        )

        tracks_late = self._track_ids(
            late.get("track_ids")
        )

        shared_tracks = (
            tracks_early
            &
            tracks_late
        )

        if shared_tracks:

            return (
                True,
                0.99,
                "sequential_shared_raw_track",
            )

        identity_a = int(
            self._safe_float(
                early.get(
                    "identity_id",
                    -1,
                ),
                -1,
            )
        )

        identity_b = int(
            self._safe_float(
                late.get(
                    "identity_id",
                    -2,
                ),
                -2,
            )
        )

        same_identity = (
            identity_a >= 0
            and
            identity_a == identity_b
        )

        # --------------------------------------------------------
        # Get raw trajectory endpoints.
        # --------------------------------------------------------

        if tracks_early:
            early_track = next(
                iter(tracks_early)
            )
        else:
            early_track = int(
                early["track_id"]
            )

        if tracks_late:
            late_track = next(
                iter(tracks_late)
            )
        else:
            late_track = int(
                late["track_id"]
            )

        early_rows = prepared[
            prepared["track_id"]
            ==
            early_track
        ].sort_values(
            "frame_id"
        )

        late_rows = prepared[
            prepared["track_id"]
            ==
            late_track
        ].sort_values(
            "frame_id"
        )

        if (
            early_rows.empty
            or
            late_rows.empty
        ):
            return (
                False,
                0.0,
                "missing_raw_trajectory",
            )

        e = early_rows.iloc[-1]
        l = late_rows.iloc[0]

        raw_gap = int(
            l["frame_id"]
            -
            e["frame_id"]
        )

        # --------------------------------------------------------
        # If tracks overlap in time, they are independent.
        # --------------------------------------------------------

        if raw_gap <= 0:

            return (
                False,
                0.0,
                "overlapping_raw_tracks",
            )

        endpoint_distance = math.hypot(
            float(
                l["raw_x"]
            )
            -
            float(
                e["raw_x"]
            ),
            float(
                l["raw_y"]
            )
            -
            float(
                e["raw_y"]
            ),
        )

        crossing_distance = math.hypot(
            float(
                early["crossing_x"]
            )
            -
            float(
                late["crossing_x"]
            ),
            float(
                early["crossing_y"]
            )
            -
            float(
                late["crossing_y"]
            ),
        )

        if (
            endpoint_distance
            >
            self.config.max_endpoint_distance_px
        ):
            return (
                False,
                0.0,
                "endpoint_distance_too_large",
            )

        if (
            crossing_distance
            >
            self.config.max_crossing_distance_px
        ):
            return (
                False,
                0.0,
                "crossing_distance_too_large",
            )

        va = self._direction_vector(
            early_rows
        )

        vb = self._direction_vector(
            late_rows
        )

        cosine = self._cosine(
            va,
            vb,
        )

        if (
            cosine
            <
            self.config.min_direction_cosine
        ):
            return (
                False,
                0.0,
                "trajectory_direction_mismatch",
            )

        endpoint_score = max(
            0.0,
            1.0
            -
            endpoint_distance
            /
            max(
                self.config.max_endpoint_distance_px,
                1.0,
            ),
        )

        crossing_score = max(
            0.0,
            1.0
            -
            crossing_distance
            /
            max(
                self.config.max_crossing_distance_px,
                1.0,
            ),
        )

        temporal_score = max(
            0.0,
            1.0
            -
            frame_gap
            /
            max(
                self.config.max_sequential_frame_gap,
                1,
            ),
        )

        direction_score = (
            cosine + 1.0
        ) / 2.0

        identity_score = (
            1.0
            if same_identity
            else 0.0
        )

        score = (
            0.25 * endpoint_score
            +
            0.25 * crossing_score
            +
            0.15 * temporal_score
            +
            0.15 * direction_score
            +
            0.20 * identity_score
        )

        hard_duplicate = (
            same_identity
            and
            endpoint_distance
            <=
            self.config.max_endpoint_distance_px
            and
            crossing_distance
            <=
            self.config.max_crossing_distance_px
            and
            cosine
            >=
            self.config.min_direction_cosine
        )

        if hard_duplicate:

            return (
                True,
                max(
                    score,
                    self.config.same_identity_duplicate_threshold,
                ),
                "same_identity_trajectory_duplicate",
            )

        if (
            score
            >=
            self.config.duplicate_score_threshold
        ):

            return (
                True,
                score,
                "trajectory_duplicate",
            )

        return (
            False,
            score,
            "trajectory_not_duplicate",
        )

    # ============================================================
    # APPLY
    # ============================================================

    def resolve(
        self,
        candidates: pd.DataFrame,
        prepared: pd.DataFrame,
    ) -> pd.DataFrame:

        if (
            candidates is None
            or
            candidates.empty
        ):
            return candidates.copy()

        events = (
            candidates
            .copy()
            .reset_index(drop=True)
        )

        # --------------------------------------------------------
        # Initialize audit columns.
        # --------------------------------------------------------

        defaults = {
            "candidate_duplicate_of": pd.NA,
            "candidate_duplicate_confidence": 0.0,
            "candidate_duplicate_reason": "",
            "candidate_duplicate_suppressed": False,

            "vehicle_eligible": False,
            "vehicle_eligibility_reason": "",

            "fast_crossing_confidence": 0.0,
            "fast_crossing_valid": False,

            "multi_crossing_frame": False,
            "multi_crossing_group_size": 1,
            "multi_crossing_resolution": "",
        }

        for column, default in defaults.items():

            if column not in events.columns:
                events[column] = default

        # --------------------------------------------------------
        # Fast crossing.
        # --------------------------------------------------------

        for idx, row in events.iterrows():

            fast_info = (
                self._fast_crossing_evidence(
                    row
                )
            )

            events.loc[
                idx,
                "fast_crossing_confidence",
            ] = (
                fast_info[
                    "fast_crossing_confidence"
                ]
            )

            events.loc[
                idx,
                "fast_crossing_valid",
            ] = (
                fast_info[
                    "fast_crossing_valid"
                ]
            )

        # --------------------------------------------------------
        # Vehicle eligibility.
        # --------------------------------------------------------

        for idx, row in events.iterrows():

            eligible, cls, reason = (
                self._vehicle_eligibility(
                    row
                )
            )

            events.loc[
                idx,
                "vehicle_eligible",
            ] = bool(
                eligible
            )

            events.loc[
                idx,
                "vehicle_eligibility_reason",
            ] = reason

            if (
                cls
                !=
                "PERSON"
                and
                cls
                !=
                "UNKNOWN"
                and
                cls
                !=
                "LOW_CLASS_CONFIDENCE"
            ):
                events.loc[
                    idx,
                    "counting_class",
                ] = cls

        # --------------------------------------------------------
        # Multi-crossing groups.
        # --------------------------------------------------------

        frame_counts = (
            events["crossing_frame"]
            .value_counts()
        )

        multi_frames = set(
            frame_counts[
                frame_counts > 1
            ].index.tolist()
        )

        events["multi_crossing_frame"] = (
            events[
                "crossing_frame"
            ].isin(
                multi_frames
            )
        )

        events["multi_crossing_group_size"] = (
            events[
                "crossing_frame"
            ].map(
                frame_counts
            ).fillna(1).astype(int)
        )

        events.loc[
            events[
                "multi_crossing_frame"
            ],
            "multi_crossing_resolution",
        ] = (
            "KEEP_INDEPENDENT_UNTIL_PAIRWISE_RESOLUTION"
        )

        # --------------------------------------------------------
        # Pairwise candidate resolution.
        # --------------------------------------------------------

        order = (
            events
            .sort_values(
                [
                    "crossing_frame",
                    "crossing_id",
                ]
            )
            .index
            .tolist()
        )

        for position, i in enumerate(order):

            if bool(
                events.loc[
                    i,
                    "candidate_duplicate_suppressed",
                ]
            ):
                continue

            for j in order[position + 1:]:

                if bool(
                    events.loc[
                        j,
                        "candidate_duplicate_suppressed",
                    ]
                ):
                    continue

                # Different directions are almost always
                # independent vehicles.
                direction_i = str(
                    events.loc[
                        i,
                        "direction",
                    ]
                )

                direction_j = str(
                    events.loc[
                        j,
                        "direction",
                    ]
                )

                if (
                    direction_i
                    !=
                    direction_j
                ):
                    continue

                is_duplicate, confidence, reason = (
                    self._sequential_duplicate(
                        events.loc[i],
                        events.loc[j],
                        prepared,
                    )
                )

                if not is_duplicate:
                    continue

                # ------------------------------------------------
                # Decide which candidate survives.
                #
                # Earlier candidate normally wins.
                # But if later candidate has substantially
                # stronger evidence, keep later candidate.
                # ------------------------------------------------

                score_i = self._safe_float(
                    events.loc[
                        i,
                        "candidate_quality",
                    ],
                    0.0,
                )

                score_j = self._safe_float(
                    events.loc[
                        j,
                        "candidate_quality",
                    ],
                    0.0,
                )

                if (
                    score_j
                    >
                    score_i + 0.15
                ):
                    winner = j
                    loser = i
                else:
                    winner = i
                    loser = j

                events.loc[
                    loser,
                    "candidate_duplicate_of",
                ] = int(
                    events.loc[
                        winner,
                        "crossing_id",
                    ]
                )

                events.loc[
                    loser,
                    "candidate_duplicate_confidence",
                ] = float(
                    confidence
                )

                events.loc[
                    loser,
                    "candidate_duplicate_reason",
                ] = reason

                events.loc[
                    loser,
                    "candidate_duplicate_suppressed",
                ] = True

                events.loc[
                    loser,
                    "count_eligibility",
                ] = False

                events.loc[
                    loser,
                    "counted",
                ] = False

                events.loc[
                    loser,
                    "multi_crossing_resolution",
                ] = (
                    "SUPPRESS_DUPLICATE"
                )

                events.loc[
                    winner,
                    "multi_crossing_resolution",
                ] = (
                    "KEEP_WINNER"
                )

        # --------------------------------------------------------
        # HARD BLOCK PERSON.
        # --------------------------------------------------------

        person_mask = (
            events[
                "counting_class"
            ]
            .astype(str)
            .str.lower()
            ==
            "person"
        )

        events.loc[
            person_mask,
            "vehicle_eligible",
        ] = False

        events.loc[
            person_mask,
            "count_eligibility",
        ] = False

        events.loc[
            person_mask,
            "counted",
        ] = False

        events.loc[
            person_mask,
            "vehicle_eligibility_reason",
        ] = (
            "person_never_enters_vehicle_count"
        )

        # --------------------------------------------------------
        # Duplicates are always ineligible.
        # --------------------------------------------------------

        duplicate_mask = (
            events[
                "candidate_duplicate_suppressed"
            ]
            .astype(bool)
        )

        events.loc[
            duplicate_mask,
            "count_eligibility",
        ] = False

        events.loc[
            duplicate_mask,
            "counted",
        ] = False

        return events
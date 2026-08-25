from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math

import numpy as np
import pandas as pd


VEHICLE_CLASSES = {
    "motorcycle",
    "car",
    "truck",
    "bus",
}


class CrossingState(str, Enum):

    NEW = "NEW"

    CONFIRMED = "CONFIRMED"

    APPROACHING = "APPROACHING"

    CROSSING = "CROSSING"

    CROSSED = "CROSSED"

    COUNTED = "COUNTED"

    REVIEW = "REVIEW"

    REJECTED = "REJECTED"


@dataclass(frozen=True)
class StateMachineConfig:

    min_confirmed_observations: int = 2

    min_direction_confidence: float = 0.45

    count_threshold: float = 0.64

    review_threshold: float = 0.45

    min_vehicle_class_confidence: float = 0.45

    # Phase 3.1 class arbitration gates.
    identity_class_confidence_floor: float = 0.65
    identity_class_margin_floor: float = 0.15
    identity_class_ambiguous_review: bool = True
    identity_class_allow_strong_rescue: bool = True

    # Fast crossing gates.
    fast_crossing_min_score: float = 0.58
    fast_crossing_min_direction_confidence: float = 0.50
    fast_crossing_min_normal_displacement_px: float = 8.0
    fast_crossing_min_continuity: float = 0.50

    short_crossing_min_score: float = 0.58

    weight_geometry: float = 0.30

    weight_direction: float = 0.18

    weight_continuity: float = 0.18

    weight_class: float = 0.15

    weight_pre: float = 0.05

    weight_corridor: float = 0.05

    weight_post: float = 0.04

    weight_fast_evidence: float = 0.05


class CrossingStateMachine:

    STATE_COLUMNS = [
        "crossing_id",
        "phase3_state",
        "state_history",
        "state_reason",
        "decision_score",
        "count_eligibility",
        "counted",
        "review_reason",
        "reject_reason",
        # Phase 3.1 audit
        "identity_class_confidence",
        "identity_class_margin",
        "identity_class_ambiguous",
        "fast_crossing",
        "fast_crossing_confidence",
        "fast_crossing_reason",
    ]

    def __init__(
        self,
        config: StateMachineConfig | None = None,
    ) -> None:

        self.config = (
            config
            or
            StateMachineConfig()
        )

    # ============================================================
    # SAFE HELPERS
    # ============================================================

    @staticmethod
    def _bool(
        row: pd.Series,
        key: str,
        default=False,
    ) -> bool:

        value = row.get(
            key,
            default,
        )

        if pd.isna(value):
            return default

        return bool(value)

    @staticmethod
    def _float(
        row: pd.Series,
        key: str,
        default=0.0,
    ) -> float:

        value = row.get(
            key,
            default,
        )

        try:
            value = float(value)

            if not math.isfinite(value):
                return float(default)

            return value

        except (
            TypeError,
            ValueError,
        ):
            return float(default)

    @staticmethod
    def _int(
        row: pd.Series,
        key: str,
        default=0,
    ) -> int:

        try:
            return int(
                row.get(
                    key,
                    default,
                )
            )

        except (
            TypeError,
            ValueError,
        ):
            return int(default)

    # ============================================================
    # VEHICLE
    # ============================================================

    def _is_vehicle(
        self,
        row: pd.Series,
    ) -> bool:

        cls = str(
            row.get(
                "counting_class",
                row.get(
                    "track_class",
                    "unknown",
                ),
            )
        ).lower().strip()

        if cls not in VEHICLE_CLASSES:
            return False

        if (
            not
            self._bool(
                row,
                "vehicle_eligible",
                True,
            )
        ):
            return False

        confidence = self._float(
            row,
            "counting_class_confidence",
            self._float(
                row,
                "track_class_ratio",
                0.0,
            ),
        )

        return (
            confidence
            >=
            self.config.min_vehicle_class_confidence
        )

    # ============================================================
    # DIRECTION
    # ============================================================

    def _direction_confidence(
        self,
        row: pd.Series,
    ) -> float:

        explicit = self._float(
            row,
            "direction_confidence",
            -1.0,
        )

        if explicit >= 0:
            return float(
                np.clip(
                    explicit,
                    0.0,
                    1.0,
                )
            )

        direction = str(
            row.get(
                "direction",
                "UNKNOWN",
            )
        )

        if direction == "UNKNOWN":
            return 0.0

        displacement = abs(
            self._float(
                row,
                "normal_displacement_px",
                0.0,
            )
        )

        normal_velocity = abs(
            self._float(
                row,
                "normal_velocity",
                0.0,
            )
        )

        score = (
            0.50
            +
            min(
                0.30,
                displacement / 100.0,
            )
            +
            min(
                0.20,
                normal_velocity / 100.0,
            )
        )

        return float(
            np.clip(
                score,
                0.0,
                1.0,
            )
        )

    # ============================================================
    # CONTINUITY
    # ============================================================

    def _continuity_score(
        self,
        row: pd.Series,
    ) -> float:

        value = self._float(
            row,
            "trajectory_continuity",
            -1.0,
        )

        if value < 0:
            value = self._float(
                row,
                "continuity_score",
                -1.0,
            )

        if value < 0:

            identity_confidence = (
                self._float(
                    row,
                    "identity_confidence",
                    0.70,
                )
            )

            fast_confidence = (
                self._float(
                    row,
                    "fast_crossing_confidence",
                    0.0,
                )
            )

            return float(
                np.clip(
                    0.65
                    *
                    identity_confidence
                    +
                    0.35
                    *
                    fast_confidence,
                    0.0,
                    1.0,
                )
            )

        return float(
            np.clip(
                value,
                0.0,
                1.0,
            )
        )

    # ============================================================
    # CLASS
    # ============================================================

    def _class_score(
        self,
        row: pd.Series,
    ) -> float:

        identity_conf = self._float(
            row,
            "identity_class_confidence",
            -1.0,
        )
        if identity_conf >= 0.0:
            return float(np.clip(identity_conf, 0.0, 1.0))

        return float(
            np.clip(
                self._float(
                    row,
                    "counting_class_confidence",
                    self._float(
                        row,
                        "track_class_ratio",
                        0.0,
                    ),
                ),
                0.0,
                1.0,
            )
        )

    # ============================================================
    # ZONE
    # ============================================================

    def _zone_score(
        self,
        row: pd.Series,
        key: str,
    ) -> float:

        value = self._float(
            row,
            key,
            0.0,
        )

        observations = max(
            self._int(
                row,
                "track_observations",
                1,
            ),
            1,
        )

        return float(
            np.clip(
                value
                /
                min(
                    observations,
                    5,
                ),
                0.0,
                1.0,
            )
        )

    # ============================================================
    # DECISION SCORE
    # ============================================================

    def _decision_score(
        self,
        row: pd.Series,
    ) -> float:

        geometry = float(
            self._bool(
                row,
                "geometry_crossing",
                False,
            )
        )

        direction = (
            self._direction_confidence(
                row
            )
        )

        continuity = (
            self._continuity_score(
                row
            )
        )

        class_score = (
            self._class_score(
                row
            )
        )

        pre = self._zone_score(
            row,
            "pre_zone_evidence",
        )

        corridor = self._zone_score(
            row,
            "corridor_evidence",
        )

        post = self._zone_score(
            row,
            "post_zone_evidence",
        )

        fast_evidence = self._float(
            row,
            "fast_crossing_confidence",
            0.0,
        )

        weights = [
            self.config.weight_geometry,
            self.config.weight_direction,
            self.config.weight_continuity,
            self.config.weight_class,
            self.config.weight_pre,
            self.config.weight_corridor,
            self.config.weight_post,
            self.config.weight_fast_evidence,
        ]

        values = [
            geometry,
            direction,
            continuity,
            class_score,
            pre,
            corridor,
            post,
            fast_evidence,
        ]

        total_weight = sum(
            weights
        )

        score = sum(
            w * v
            for w, v
            in zip(
                weights,
                values,
            )
        )

        return float(
            np.clip(
                score
                /
                max(
                    total_weight,
                    1e-9,
                ),
                0.0,
                1.0,
            )
        )

    # ============================================================
    # EVALUATE
    # ============================================================

    def evaluate_candidate(
        self,
        row: pd.Series,
    ) -> dict:

        candidate_id = int(
            row.get(
                "crossing_id",
                -1,
            )
        )

        history = [
            CrossingState.NEW.value
        ]

        reasons = []

        review_reasons = []

        reject_reasons = []

        score = (
            self._decision_score(
                row
            )
        )

        # --------------------------------------------------------
        # BLOCKING: duplicate
        # --------------------------------------------------------

        if self._bool(
            row,
            "candidate_duplicate_suppressed",
            False,
        ):

            history.append(
                CrossingState.REJECTED.value
            )

            reject_reasons.append(
                "trajectory_duplicate_suppressed"
            )

            return self._result(
                candidate_id,
                CrossingState.REJECTED,
                history,
                "trajectory_duplicate_suppressed",
                score,
                False,
                False,
                "",
                ";".join(
                    reject_reasons
                ),
            )

        # --------------------------------------------------------
        # BLOCKING: PERSON / NON VEHICLE
        # --------------------------------------------------------

        if not self._is_vehicle(
            row
        ):

            cls = str(
                row.get(
                    "counting_class",
                    row.get(
                        "track_class",
                        "unknown",
                    ),
                )
            ).lower().strip()

            if cls == "person":
                reason = (
                    "person_never_enters_vehicle_count"
                )
            else:
                reason = (
                    "non_vehicle_or_low_class_confidence"
                )

            history.append(
                CrossingState.REJECTED.value
            )

            reject_reasons.append(
                reason
            )

            return self._result(
                candidate_id,
                CrossingState.REJECTED,
                history,
                reason,
                score,
                False,
                False,
                "",
                reason,
            )

        # --------------------------------------------------------
        # CONFIRMED
        # --------------------------------------------------------

        observations = self._int(
            row,
            "track_observations",
            0,
        )

        geometry = self._bool(
            row,
            "geometry_crossing",
            False,
        )

        if (
            observations
            >=
            self.config.min_confirmed_observations
            or
            geometry
        ):

            history.append(
                CrossingState.CONFIRMED.value
            )

        else:

            history.append(
                CrossingState.REJECTED.value
            )

            reject_reasons.append(
                "insufficient_observations"
            )

            return self._result(
                candidate_id,
                CrossingState.REJECTED,
                history,
                "insufficient_observations",
                score,
                False,
                False,
                "",
                "insufficient_observations",
            )

        # --------------------------------------------------------
        # APPROACHING
        # --------------------------------------------------------

        pre_evidence = self._float(
            row,
            "pre_zone_evidence",
            0.0,
        )

        if pre_evidence > 0:

            history.append(
                CrossingState.APPROACHING.value
            )

        else:

            history.append(
                CrossingState.APPROACHING.value
            )

            reasons.append(
                "approaching_inferred"
            )

        # --------------------------------------------------------
        # CROSSING
        # --------------------------------------------------------

        if not geometry:

            history.append(
                CrossingState.REVIEW.value
            )

            review_reasons.append(
                "no_geometric_crossing"
            )

            return self._result(
                candidate_id,
                CrossingState.REVIEW,
                history,
                ";".join(
                    reasons
                ),
                score,
                False,
                False,
                ";".join(
                    review_reasons
                ),
                "",
            )

        history.append(
            CrossingState.CROSSING.value
        )

        history.append(
            CrossingState.CROSSED.value
        )

        # --------------------------------------------------------
        # EVIDENCE
        # --------------------------------------------------------

        direction_confidence = (
            self._direction_confidence(
                row
            )
        )

        class_score = (
            self._class_score(
                row
            )
        )

        identity_class_conf = self._float(
            row,
            "identity_class_confidence",
            class_score,
        )
        identity_class_margin = self._float(
            row,
            "identity_class_margin",
            1.0,
        )
        identity_class_ambiguous = self._bool(
            row,
            "identity_class_ambiguous",
            False,
        )

        fast = self._bool(
            row,
            "fast_crossing",
            False,
        )

        sparse = self._bool(
            row,
            "sparse_crossing",
            False,
        )

        short = self._bool(
            row,
            "short_track",
            False,
        )

        fast_score = self._float(
            row,
            "fast_crossing_confidence",
            0.0,
        )

        # --------------------------------------------------------
        # Direction is always mandatory.
        # --------------------------------------------------------

        if (
            direction_confidence
            <
            self.config.min_direction_confidence
        ):

            review_reasons.append(
                "low_direction_confidence"
            )

        # --------------------------------------------------------
        # Class is always mandatory.
        # --------------------------------------------------------

        if (
            class_score
            <
            self.config.min_vehicle_class_confidence
        ):

            review_reasons.append(
                "low_vehicle_class_confidence"
            )

        if identity_class_ambiguous and self.config.identity_class_ambiguous_review:
            review_reasons.append("ambiguous_identity_class")

        if identity_class_conf < self.config.identity_class_confidence_floor:
            review_reasons.append("identity_class_confidence_below_floor")

        if identity_class_margin < self.config.identity_class_margin_floor:
            review_reasons.append("identity_class_margin_below_floor")

        # --------------------------------------------------------
        # Fast crossing:
        #
        # Do NOT lower the threshold.
        # Instead require stronger fast evidence.
        # --------------------------------------------------------

        if fast or sparse or short:

            reasons.append(
                "fast_sparse_mode"
            )

            if (
                fast_score
                <
                self.config.fast_crossing_min_score
            ):

                review_reasons.append(
                    "weak_fast_crossing_evidence"
                )

            if (
                abs(
                    self._float(
                        row,
                        "fast_normal_displacement_px",
                        self._float(row, "normal_displacement_px", 0.0),
                    )
                )
                <
                self.config.fast_crossing_min_normal_displacement_px
            ):

                review_reasons.append(
                    "insufficient_normal_displacement"
                )

            if (
                self._float(row, "fast_direction_confidence", direction_confidence)
                <
                self.config.fast_crossing_min_direction_confidence
            ):

                review_reasons.append(
                    "fast_crossing_low_direction_confidence"
                )

            if (
                self._float(row, "trajectory_continuity", self._continuity_score(row))
                <
                self.config.fast_crossing_min_continuity
            ):

                review_reasons.append(
                    "fast_crossing_low_continuity"
                )

        # --------------------------------------------------------
        # Final decision.
        # --------------------------------------------------------

        if review_reasons:

            # Strong geometry + strong direction +
            # strong class can still rescue.
            rescue = (
                geometry
                and
                direction_confidence
                >=
                self.config.min_direction_confidence
                and
                class_score
                >=
                self.config.min_vehicle_class_confidence
                and
                (
                    not identity_class_ambiguous
                    or self.config.identity_class_allow_strong_rescue
                    and identity_class_conf >= self.config.identity_class_confidence_floor
                    and identity_class_margin >= self.config.identity_class_margin_floor
                )
                and
                score
                >=
                self.config.count_threshold
            )

            if fast or sparse or short:

                rescue = (
                    rescue
                    and
                    fast_score
                    >=
                    self.config.fast_crossing_min_score
                )

            if rescue:

                reasons.append(
                    "strong_evidence_rescue"
                )

            else:

                history.append(
                    CrossingState.REVIEW.value
                )

                return self._result(
                    candidate_id,
                    CrossingState.REVIEW,
                    history,
                    ";".join(
                        reasons
                    ),
                    score,
                    False,
                    False,
                    ";".join(
                        review_reasons
                    ),
                    "",
                )

        # --------------------------------------------------------
        # COUNT
        # --------------------------------------------------------

        if score >= self.config.count_threshold:

            history.append(
                CrossingState.COUNTED.value
            )

            return self._result(
                candidate_id,
                CrossingState.COUNTED,
                history,
                ";".join(
                    reasons
                )
                or
                "sufficient_vehicle_crossing_evidence",
                score,
                True,
                True,
                "",
                "",
            )

        # --------------------------------------------------------
        # REVIEW
        # --------------------------------------------------------

        if score >= self.config.review_threshold:

            history.append(
                CrossingState.REVIEW.value
            )

            return self._result(
                candidate_id,
                CrossingState.REVIEW,
                history,
                ";".join(
                    reasons
                ),
                score,
                False,
                False,
                "decision_score_below_count_threshold",
                "",
            )

        # --------------------------------------------------------
        # REJECT
        # --------------------------------------------------------

        history.append(
            CrossingState.REJECTED.value
        )

        return self._result(
            candidate_id,
            CrossingState.REJECTED,
            history,
            ";".join(
                reasons
            ),
            score,
            False,
            False,
            "",
            "decision_score_too_low",
        )

    # ============================================================
    # RESULT
    # ============================================================

    @staticmethod
    def _result(
        candidate_id,
        state,
        history,
        state_reason,
        score,
        eligibility,
        counted,
        review_reason,
        reject_reason,
        row: pd.Series | None = None,
    ):

        row = row if row is not None else pd.Series(dtype=object)

        return {
            "crossing_id": int(
                candidate_id
            ),

            "phase3_state": (
                state.value
            ),

            "state_history": (
                " → ".join(history)
            ),

            "state_reason": (
                state_reason
            ),

            "decision_score": float(
                score
            ),

            "count_eligibility": bool(
                eligibility
            ),

            "counted": bool(
                counted
            ),

            "review_reason": (
                review_reason
            ),

            "reject_reason": (
                reject_reason
            ),
            "identity_class_confidence": float(
                row.get("identity_class_confidence", row.get("counting_class_confidence", 0.0))
                if row is not None else 0.0
            ),
            "identity_class_margin": float(
                row.get("identity_class_margin", 0.0)
                if row is not None else 0.0
            ),
            "identity_class_ambiguous": bool(
                row.get("identity_class_ambiguous", False)
                if row is not None else False
            ),
            "fast_crossing": bool(
                row.get("fast_crossing", False)
                if row is not None else False
            ),
            "fast_crossing_confidence": float(
                row.get("fast_crossing_confidence", 0.0)
                if row is not None else 0.0
            ),
            "fast_crossing_reason": str(
                row.get("fast_crossing_reason", "")
                if row is not None else ""
            ),
        }

    # ============================================================
    # RUN
    # ============================================================

    def run(
        self,
        candidates: pd.DataFrame,
    ):

        if (
            candidates is None
            or
            candidates.empty
        ):

            empty = pd.DataFrame(
                columns=self.STATE_COLUMNS
            )

            return (
                empty,
                empty.copy(),
            )

        results = [
            self.evaluate_candidate(
                row
            )
            for _, row
            in candidates.iterrows()
        ]

        state_df = pd.DataFrame(
            results
        )

        # Phase 3.1 audit columns are copied from the canonical candidate.
        # This keeps the decision table self-contained even when _result()
        # is called by legacy branches.
        audit_cols = [
            "identity_class_confidence",
            "identity_class_margin",
            "identity_class_ambiguous",
            "fast_crossing",
            "fast_crossing_confidence",
            "fast_crossing_reason",
        ]
        for col in audit_cols:
            if col in candidates.columns:
                state_df[col] = candidates[col].values

        enriched = candidates.merge(
            state_df,
            on="crossing_id",
            how="left",
            suffixes=(
                "",
                "_phase3",
            ),
        )

        enriched[
            "count_eligibility"
        ] = (
            enriched[
                "count_eligibility"
            ]
            .fillna(False)
            .astype(bool)
        )

        enriched[
            "counted"
        ] = (
            enriched[
                "counted"
            ]
            .fillna(False)
            .astype(bool)
        )

        # Final safety gate.
        person_mask = (
            enriched[
                "counting_class"
            ]
            .astype(str)
            .str.lower()
            ==
            "person"
        )

        enriched.loc[
            person_mask,
            "count_eligibility",
        ] = False

        enriched.loc[
            person_mask,
            "counted",
        ] = False

        enriched.loc[
            person_mask,
            "phase3_state",
        ] = (
            CrossingState.REJECTED.value
        )

        enriched.loc[
            person_mask,
            "reject_reason",
        ] = (
            "person_never_enters_vehicle_count"
        )

        return (
            enriched,
            state_df,
        )
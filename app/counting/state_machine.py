from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math

import numpy as np
import pandas as pd


class CrossingState(str, Enum):
    NEW = "NEW"
    CONFIRMED = "CONFIRMED"
    APPROACHING = "APPROACHING"
    CROSSING = "CROSSING"
    CROSSED = "CROSSED"
    COUNTED = "COUNTED"
    REJECTED = "REJECTED"
    REVIEW = "REVIEW"


@dataclass(frozen=True)
class StateMachineConfig:
    min_confirmed_observations: int = 2
    min_direction_confidence: float = 0.45
    count_threshold: float = 0.62
    review_threshold: float = 0.45
    person_count_threshold: float = 0.55
    person_review_threshold: float = 0.42

    weight_geometry: float = 0.32
    weight_direction: float = 0.16
    weight_continuity: float = 0.14
    weight_class: float = 0.12
    weight_pre: float = 0.08
    weight_corridor: float = 0.05
    weight_post: float = 0.07
    weight_fast_sparse: float = 0.06

    fast_crossing_floor: float = 0.55
    short_crossing_floor: float = 0.52


class CrossingStateMachine:
    """
    Phase 3 lifecycle over canonical CrossingCandidate rows.

    The state machine does not perform detection. It only consumes candidate
    evidence already produced by RobustCrossingEngine.
    """

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
    ]

    def __init__(self, config: StateMachineConfig | None = None) -> None:
        self.config = config or StateMachineConfig()

    @staticmethod
    def _bool(row: pd.Series, key: str, default: bool = False) -> bool:
        value = row.get(key, default)
        if pd.isna(value):
            return default
        return bool(value)

    @staticmethod
    def _float(row: pd.Series, key: str, default: float = 0.0) -> float:
        value = row.get(key, default)
        try:
            value = float(value)
        except (TypeError, ValueError):
            return float(default)
        if not math.isfinite(value):
            return float(default)
        return float(value)

    @staticmethod
    def _int(row: pd.Series, key: str, default: int = 0) -> int:
        value = row.get(key, default)
        try:
            return int(value)
        except (TypeError, ValueError):
            return int(default)

    def _direction_confidence(self, row: pd.Series) -> float:
        explicit = self._float(row, "direction_confidence", -1.0)
        if explicit >= 0:
            return float(np.clip(explicit, 0.0, 1.0))

        direction = str(row.get("direction", "UNKNOWN"))
        normal_direction = str(row.get("normal_direction", "UNKNOWN"))
        if direction != "UNKNOWN" or normal_direction != "UNKNOWN":
            displacement = abs(self._float(row, "normal_displacement_px", 0.0))
            velocity = abs(self._float(row, "normal_velocity", 0.0))
            score = 0.55
            score += min(0.30, displacement / 100.0)
            score += min(0.15, velocity / 100.0)
            return float(np.clip(score, 0.0, 1.0))

        return 0.0

    def _continuity_score(self, row: pd.Series) -> float:
        value = self._float(row, "trajectory_continuity", 1.0)
        if value == 1.0:
            value = self._float(row, "continuity_score", 1.0)
        return float(np.clip(value, 0.0, 1.0))

    def _class_score(self, row: pd.Series) -> float:
        return float(
            np.clip(
                self._float(row, "counting_class_confidence", row.get("track_class_ratio", 1.0)),
                0.0,
                1.0,
            )
        )

    def _zone_score(self, row: pd.Series, key: str) -> float:
        value = self._float(row, key, 0.0)
        obs = max(self._int(row, "track_observations", 1), 1)
        return float(np.clip(value / min(obs, 5), 0.0, 1.0))

    def _decision_score(self, row: pd.Series) -> float:
        geometry = 1.0 if self._bool(row, "geometry_crossing") else 0.0
        direction = self._direction_confidence(row)
        continuity = self._continuity_score(row)
        class_score = self._class_score(row)
        pre = self._zone_score(row, "pre_zone_evidence")
        corridor = self._zone_score(row, "corridor_evidence")
        post = self._zone_score(row, "post_zone_evidence")
        fast_sparse = 1.0 if (
            self._bool(row, "fast_crossing")
            or self._bool(row, "sparse_crossing")
            or self._bool(row, "short_track")
        ) else 0.0

        total_weight = (
            self.config.weight_geometry
            + self.config.weight_direction
            + self.config.weight_continuity
            + self.config.weight_class
            + self.config.weight_pre
            + self.config.weight_corridor
            + self.config.weight_post
            + self.config.weight_fast_sparse
        )
        raw = (
            self.config.weight_geometry * geometry
            + self.config.weight_direction * direction
            + self.config.weight_continuity * continuity
            + self.config.weight_class * class_score
            + self.config.weight_pre * pre
            + self.config.weight_corridor * corridor
            + self.config.weight_post * post
            + self.config.weight_fast_sparse * fast_sparse
        )
        return float(np.clip(raw / max(total_weight, 1e-9), 0.0, 1.0))

    def evaluate_candidate(self, row: pd.Series) -> dict:
        candidate_id = int(row.get("crossing_id", -1))
        class_name = str(row.get("counting_class", row.get("track_class", "unknown"))).lower().strip()

        score = self._decision_score(row)
        geometry = self._bool(row, "geometry_crossing")
        direction_conf = self._direction_confidence(row)
        observations = self._int(row, "track_observations", 0)
        duplicate = self._bool(row, "candidate_duplicate_suppressed")
        short_track = self._bool(row, "short_track")
        fast_crossing = self._bool(row, "fast_crossing")
        sparse = self._bool(row, "sparse_crossing")

        history = [CrossingState.NEW.value]
        reasons: list[str] = []
        review_reasons: list[str] = []
        reject_reasons: list[str] = []

        if duplicate:
            history.append(CrossingState.REJECTED.value)
            reject_reasons.append("trajectory_duplicate_suppressed")
            return self._result(
                candidate_id,
                CrossingState.REJECTED,
                history,
                ";".join(reject_reasons),
                score,
                False,
                False,
                "",
                ";".join(reject_reasons),
            )

        if observations >= self.config.min_confirmed_observations or geometry:
            history.append(CrossingState.CONFIRMED.value)
        else:
            history.append(CrossingState.REJECTED.value)
            reject_reasons.append("insufficient_observations")
            return self._result(
                candidate_id,
                CrossingState.REJECTED,
                history,
                ";".join(reject_reasons),
                score,
                False,
                False,
                "",
                ";".join(reject_reasons),
            )

        if self._float(row, "pre_zone_evidence", 0) > 0:
            history.append(CrossingState.APPROACHING.value)
        elif geometry:
            # A sparse object can jump directly into a crossing. Keep the
            # semantic state trace explicit rather than inventing evidence.
            history.append(CrossingState.APPROACHING.value)
            reasons.append("approaching_inferred_from_crossing")

        if geometry:
            history.append(CrossingState.CROSSING.value)
        else:
            history.append(CrossingState.REVIEW.value)
            review_reasons.append("no_geometric_crossing")
            return self._result(
                candidate_id,
                CrossingState.REVIEW,
                history,
                ";".join(reasons),
                score,
                False,
                False,
                ";".join(review_reasons),
                "",
            )

        # Once geometric crossing is confirmed, treat it as crossed. Zone
        # evidence strengthens confidence but does not become a hard gate.
        history.append(CrossingState.CROSSED.value)

        if direction_conf < self.config.min_direction_confidence:
            review_reasons.append("low_direction_confidence")

        if self._class_score(row) < 0.35:
            review_reasons.append("low_class_confidence")

        if short_track:
            reasons.append("short_track_exception")
        if fast_crossing:
            reasons.append("fast_crossing_exception")
        if sparse:
            reasons.append("sparse_crossing_exception")

        # Fast/sparse/short events can be counted with reduced zone evidence,
        # but not with weak geometry or direction.
        floor = self.config.count_threshold
        if class_name == "person":
            floor = self.config.person_count_threshold

        if fast_crossing:
            floor = min(floor, self.config.fast_crossing_floor)
        if short_track:
            floor = min(floor, self.config.short_crossing_floor)

        if review_reasons:
            # Strong geometry + good direction can still rescue a short/fast
            # event unless its class confidence is genuinely unusable.
            can_rescue = (
                geometry
                and direction_conf >= self.config.min_direction_confidence
                and self._class_score(row) >= 0.35
                and score >= floor
            )
            if can_rescue:
                reasons.append("review_evidence_rescued")
                history.append(CrossingState.COUNTED.value)
                return self._result(
                    candidate_id,
                    CrossingState.COUNTED,
                    history,
                    ";".join(reasons),
                    score,
                    True,
                    True,
                    ";".join(review_reasons),
                    "",
                )

            history.append(CrossingState.REVIEW.value)
            return self._result(
                candidate_id,
                CrossingState.REVIEW,
                history,
                ";".join(reasons),
                score,
                False,
                False,
                ";".join(review_reasons),
                "",
            )

        if score >= floor:
            history.append(CrossingState.COUNTED.value)
            return self._result(
                candidate_id,
                CrossingState.COUNTED,
                history,
                ";".join(reasons) or "sufficient_crossing_evidence",
                score,
                True,
                True,
                "",
                "",
            )

        if score >= self.config.review_threshold:
            history.append(CrossingState.REVIEW.value)
            review_reasons.append("decision_score_below_count_threshold")
            return self._result(
                candidate_id,
                CrossingState.REVIEW,
                history,
                ";".join(reasons),
                score,
                False,
                False,
                ";".join(review_reasons),
                "",
            )

        history.append(CrossingState.REJECTED.value)
        reject_reasons.append("decision_score_too_low")
        return self._result(
            candidate_id,
            CrossingState.REJECTED,
            history,
            ";".join(reasons),
            score,
            False,
            False,
            "",
            ";".join(reject_reasons),
        )

    @staticmethod
    def _result(
        candidate_id: int,
        state: CrossingState,
        history: list[str],
        state_reason: str,
        score: float,
        eligibility: bool,
        counted: bool,
        review_reason: str,
        reject_reason: str,
    ) -> dict:
        return {
            "crossing_id": candidate_id,
            "phase3_state": state.value,
            "state_history": " → ".join(history),
            "state_reason": state_reason,
            "decision_score": float(score),
            "count_eligibility": bool(eligibility),
            "counted": bool(counted),
            "review_reason": review_reason,
            "reject_reason": reject_reason,
        }

    def run(self, candidates: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        if candidates is None or candidates.empty:
            state_df = pd.DataFrame(columns=self.STATE_COLUMNS)
            return state_df, state_df.copy()

        results = [
            self.evaluate_candidate(row)
            for _, row in candidates.iterrows()
        ]
        state_df = pd.DataFrame(results)
        enriched = candidates.merge(
            state_df,
            on="crossing_id",
            how="left",
            suffixes=("", "_phase3"),
        )

        # Canonical counter-facing status comes from Phase 3 only.
        enriched["count_eligibility"] = enriched["count_eligibility"].fillna(False).astype(bool)
        enriched["counted"] = enriched["counted"].fillna(False).astype(bool)

        return enriched, state_df
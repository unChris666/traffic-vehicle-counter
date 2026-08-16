from __future__ import annotations

import numpy as np
import pandas as pd
from sympy import fps


class ConfidenceEngine:
    """
    Phase 6B confidence engine.

    Does NOT modify or filter final counts.
    It only assigns quality/confidence scores.
    """

    def __init__(
        self,
        *,
        high_threshold: float = 0.85,
        medium_threshold: float = 0.70,
    ) -> None:
        self.high_threshold = high_threshold
        self.medium_threshold = medium_threshold

    @staticmethod
    def _flag(score: float | None) -> str:
        if score is None:
            return "N/A"

        if score >= 0.85:
            return "HIGH"

        if score >= 0.70:
            return "MEDIUM"

        return "LOW"

    @staticmethod
    def _clip(value: float) -> float:
        return float(np.clip(value, 0.0, 1.0))

    # ============================================================
    # TRACK CONFIDENCE
    # ============================================================

    def build_track_confidence(
        self,
        track_level: pd.DataFrame,
    ) -> pd.DataFrame:

        required = {
            "track_id",
            "track_class",
            "track_class_ratio",
            "observation_ratio",
            "mean_confidence",
        }

        missing = required - set(track_level.columns)

        if missing:
            raise ValueError(
                "track_level missing required columns: "
                f"{sorted(missing)}"
            )

        result = track_level.copy()

        result["detection_confidence"] = (
            result["mean_confidence"]
            .clip(0.0, 1.0)
        )

        result["class_confidence"] = (
            result["track_class_ratio"]
            .clip(0.0, 1.0)
        )

        result["tracking_confidence"] = (
            result["observation_ratio"]
            .clip(0.0, 1.0)
        )

        # Deliberately transparent weighted score.
        result["track_confidence"] = (
            0.40 * result["detection_confidence"]
            + 0.30 * result["class_confidence"]
            + 0.30 * result["tracking_confidence"]
        )

        result["track_confidence_flag"] = (
            result["track_confidence"]
            .apply(self._flag)
        )

        return result

    # ============================================================
    # CROSSING CONFIDENCE
    # ============================================================

    def build_crossing_confidence(
        self,
        final_crossings: pd.DataFrame,
        trajectory: pd.DataFrame,
        track_confidence: pd.DataFrame,
        fps: float,
    ) -> pd.DataFrame:

        if fps <= 0:
            raise ValueError(
                f"fps must be > 0, got {fps}"
            )

        if final_crossings.empty:
            result = final_crossings.copy()

            result["geometry_confidence"] = pd.Series(
                dtype=float
            )
            result["temporal_confidence"] = pd.Series(
                dtype=float
            )
            result["crossing_confidence"] = pd.Series(
                dtype=float
            )
            result["crossing_confidence_flag"] = pd.Series(
                dtype=str
            )

            return result

        required_crossing = {
            "track_id",
            "crossing_frame",
            "crossing_x",
            "crossing_y",
            "track_class",
            "line_distance_px",
            "previous_side",
            "frame_gap",
        }

        missing = (
            required_crossing
            - set(final_crossings.columns)
        )

        if missing:
            raise ValueError(
                "final_crossings missing required columns: "
                f"{sorted(missing)}"
            )

        result = final_crossings.copy()

        # ------------------------------------------------------------
        # Geometry confidence
        #
        # Larger distance from the counting line's deadband means
        # stronger evidence that the crossing is real.
        # ------------------------------------------------------------

        result["geometry_confidence"] = (
            1.0
            - np.exp(
                -result["line_distance_px"] / 20.0
            )
        ).clip(0.0, 1.0)

        # ------------------------------------------------------------
        # Temporal confidence
        #
        # Smaller frame gap = stronger temporal transition.
        # ------------------------------------------------------------

        gap_sec = (
            result["frame_gap"]
            / fps
        )

        result["temporal_confidence"] = (
            1.0 / (1.0 + gap_sec)
        ).clip(0.0, 1.0)

        # ------------------------------------------------------------
        # Track confidence
        # ------------------------------------------------------------

        result = result.merge(
            track_confidence[
                [
                    "track_id",
                    "track_confidence",
                    "detection_confidence",
                    "class_confidence",
                    "tracking_confidence",
                ]
            ],
            on="track_id",
            how="left",
            validate="many_to_one",
        )

        # ------------------------------------------------------------
        # Final crossing confidence
        # ------------------------------------------------------------

        result["crossing_confidence"] = (
            0.60 * result["track_confidence"]
            + 0.25 * result["geometry_confidence"]
            + 0.15 * result["temporal_confidence"]
        ).clip(0.0, 1.0)

        result["crossing_confidence_flag"] = (
            result["crossing_confidence"]
            .apply(self._flag)
        )

        return result

    # ============================================================
    # CLASS-LEVEL COUNT CONFIDENCE
    # ============================================================

    def build_count_confidence(
        self,
        crossing_confidence: pd.DataFrame,
        final_counts: dict[str, int],
    ) -> list[dict]:

        rows: list[dict] = []

        for class_name, quantity in final_counts.items():

            class_events = crossing_confidence[
                crossing_confidence["track_class"]
                == class_name
            ]

            if quantity == 0:
                rows.append(
                    {
                        "class": class_name,
                        "quantity": 0,
                        "confidence": None,
                        "flag": "N/A",
                    }
                )
                continue

            if class_events.empty:
                score = None
            else:
                score = float(
                    class_events[
                        "crossing_confidence"
                    ].mean()
                )

            rows.append(
                {
                    "class": class_name,
                    "quantity": int(quantity),
                    "confidence": (
                        round(score, 4)
                        if score is not None
                        else None
                    ),
                    "flag": self._flag(score),
                }
            )

        return rows

    # ============================================================
    # OVERALL COUNT CONFIDENCE
    # ============================================================

    def build_overall_confidence(
        self,
        crossing_confidence: pd.DataFrame,
    ) -> dict:

        if crossing_confidence.empty:
            return {
                "confidence": None,
                "flag": "N/A",
            }

        score = float(
            crossing_confidence[
                "crossing_confidence"
            ].mean()
        )

        return {
            "confidence": round(score, 4),
            "flag": self._flag(score),
        }
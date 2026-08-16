from __future__ import annotations

import numpy as np
import pandas as pd


class ConfidenceEngine:
    """Phase 6B quality/confidence scoring.

    This layer never changes or filters the final count.
    The scores are internal quality scores, not calibrated probabilities.
    """

    def __init__(
        self,
        *,
        high_threshold: float = 0.85,
        medium_threshold: float = 0.70,
    ) -> None:
        if not 0.0 <= medium_threshold <= 1.0:
            raise ValueError("medium_threshold must be between 0 and 1")
        if not 0.0 <= high_threshold <= 1.0:
            raise ValueError("high_threshold must be between 0 and 1")
        if medium_threshold > high_threshold:
            raise ValueError("medium_threshold cannot exceed high_threshold")

        self.high_threshold = float(high_threshold)
        self.medium_threshold = float(medium_threshold)

    def _flag(self, score: float | None) -> str:
        if score is None:
            return "N/A"
        if score >= self.high_threshold:
            return "HIGH"
        if score >= self.medium_threshold:
            return "MEDIUM"
        return "LOW"

    def build_track_confidence(self, track_level: pd.DataFrame) -> pd.DataFrame:
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
                f"track_level missing required columns: {sorted(missing)}"
            )

        result = track_level.copy()
        result["detection_confidence"] = result["mean_confidence"].clip(0.0, 1.0)
        result["class_confidence"] = result["track_class_ratio"].clip(0.0, 1.0)
        result["tracking_confidence"] = result["observation_ratio"].clip(0.0, 1.0)

        result["track_confidence"] = (
            0.40 * result["detection_confidence"]
            + 0.30 * result["class_confidence"]
            + 0.30 * result["tracking_confidence"]
        )
        result["track_confidence_flag"] = result["track_confidence"].apply(self._flag)
        return result

    def build_crossing_confidence(
        self,
        final_crossings: pd.DataFrame,
        trajectory: pd.DataFrame,
        track_confidence: pd.DataFrame,
        fps: float,
    ) -> pd.DataFrame:
        """Score accepted crossings using final-crossing metadata.

        `trajectory` remains in the API for compatibility and future diagnostics.
        The required crossing state is already carried by `final_crossings`.
        """
        if fps <= 0:
            raise ValueError(f"fps must be > 0, got {fps}")

        if final_crossings.empty:
            result = final_crossings.copy()
            result["geometry_confidence"] = pd.Series(dtype=float)
            result["temporal_confidence"] = pd.Series(dtype=float)
            result["crossing_confidence"] = pd.Series(dtype=float)
            result["crossing_confidence_flag"] = pd.Series(dtype=str)
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
        missing = required_crossing - set(final_crossings.columns)
        if missing:
            raise ValueError(
                "final_crossings missing required columns: "
                f"{sorted(missing)}"
            )

        required_track = {
            "track_id",
            "track_confidence",
            "detection_confidence",
            "class_confidence",
            "tracking_confidence",
        }
        missing = required_track - set(track_confidence.columns)
        if missing:
            raise ValueError(
                "track_confidence missing required columns: "
                f"{sorted(missing)}"
            )

        result = final_crossings.copy()

        # Farther from the line means stronger geometric evidence.
        result["geometry_confidence"] = (
            1.0 - np.exp(-result["line_distance_px"] / 20.0)
        ).clip(0.0, 1.0)

        gap_sec = result["frame_gap"] / float(fps)
        result["temporal_confidence"] = (
            1.0 / (1.0 + gap_sec)
        ).clip(0.0, 1.0)

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

        if result["track_confidence"].isna().any():
            missing_ids = (
                result.loc[
                    result["track_confidence"].isna(),
                    "track_id",
                ]
                .unique()
                .tolist()
            )
            raise ValueError(
                f"Missing track confidence for track IDs: {missing_ids}"
            )

        result["crossing_confidence"] = (
            0.60 * result["track_confidence"]
            + 0.25 * result["geometry_confidence"]
            + 0.15 * result["temporal_confidence"]
        ).clip(0.0, 1.0)

        result["crossing_confidence_flag"] = (
            result["crossing_confidence"].apply(self._flag)
        )
        return result

    def build_count_confidence(
        self,
        crossing_confidence: pd.DataFrame,
        final_counts: dict[str, int],
    ) -> list[dict]:
        rows: list[dict] = []

        if not crossing_confidence.empty:
            required = {"track_class", "crossing_confidence"}
            missing = required - set(crossing_confidence.columns)
            if missing:
                raise ValueError(
                    "crossing_confidence missing required columns: "
                    f"{sorted(missing)}"
                )

        for class_name, quantity in final_counts.items():
            if quantity == 0:
                rows.append({
                    "class": class_name,
                    "quantity": 0,
                    "confidence": None,
                    "flag": "N/A",
                })
                continue

            class_events = crossing_confidence[
                crossing_confidence["track_class"] == class_name
            ]
            score = (
                float(class_events["crossing_confidence"].mean())
                if not class_events.empty
                else None
            )

            rows.append({
                "class": class_name,
                "quantity": int(quantity),
                "confidence": round(score, 4) if score is not None else None,
                "flag": self._flag(score),
            })

        return rows

    def build_overall_confidence(
        self,
        crossing_confidence: pd.DataFrame,
    ) -> dict:
        if crossing_confidence.empty:
            return {"confidence": None, "flag": "N/A"}

        if "crossing_confidence" not in crossing_confidence.columns:
            raise ValueError(
                "crossing_confidence missing 'crossing_confidence' column"
            )

        score = float(crossing_confidence["crossing_confidence"].mean())
        return {
            "confidence": round(score, 4),
            "flag": self._flag(score),
        }
from __future__ import annotations

import numpy as np
import pandas as pd


def build_track_level(
    tracks_raw: pd.DataFrame,
    fps: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Convert frame-level tracking observations into robust track-level class.

    Class decision:
        confidence-weighted voting
        +
        temporal consistency support

    A short class flip caused by glare/headlight is penalized because
    detections supported by the same class in adjacent observations
    receive more voting weight.

    The final `track_class` is therefore NOT a single-frame class.
    """

    if fps <= 0:
        raise ValueError(
            f"fps must be > 0, got {fps}"
        )

    if tracks_raw.empty:
        raise ValueError(
            "tracks_raw is empty"
        )

    required_columns = {
        "track_id",
        "frame_id",
        "class_name",
        "confidence",
    }

    missing = (
        required_columns
        - set(tracks_raw.columns)
    )

    if missing:
        raise ValueError(
            "tracks_raw missing required columns: "
            f"{sorted(missing)}"
        )

    df = (
        tracks_raw
        .sort_values(
            ["track_id", "frame_id"]
        )
        .copy()
    )

    # ------------------------------------------------------------
    # Temporal neighborhood support per observation
    # ------------------------------------------------------------

    grouped = df.groupby(
        "track_id",
        sort=False,
    )

    previous_class = (
        grouped["class_name"]
        .shift(1)
    )

    next_class = (
        grouped["class_name"]
        .shift(-1)
    )

    same_as_previous = (
        df["class_name"] == previous_class
    ).astype(float)

    same_as_next = (
        df["class_name"] == next_class
    ).astype(float)

    # Base = 1.0.
    # +0.5 if previous observation agrees.
    # +0.5 if next observation agrees.
    df["temporal_support"] = (
        1.0
        + 0.5 * same_as_previous
        + 0.5 * same_as_next
    )

    # Confidence is the main signal.
    # Temporal support stabilizes against short-lived class flips.
    df["temporal_weighted_confidence"] = (
        df["confidence"].clip(0.0, 1.0)
        * df["temporal_support"]
    )

    # ------------------------------------------------------------
    # Majority vote (diagnostic / comparison)
    # ------------------------------------------------------------

    majority_counts = (
        df.groupby(
            ["track_id", "class_name"],
            observed=True,
        )
        .size()
        .rename("class_observations")
        .reset_index()
    )

    track_totals = (
        majority_counts
        .groupby("track_id")["class_observations"]
        .sum()
        .rename("total_observations")
    )

    majority_counts = majority_counts.join(
        track_totals,
        on="track_id",
    )

    majority_counts["class_ratio"] = (
        majority_counts["class_observations"]
        / majority_counts["total_observations"]
    )

    majority_class = (
        majority_counts
        .sort_values(
            [
                "track_id",
                "class_observations",
                "class_ratio",
            ],
            ascending=[True, False, False],
        )
        .drop_duplicates("track_id")
        [
            [
                "track_id",
                "class_name",
                "class_ratio",
            ]
        ]
        .rename(
            columns={
                "class_name": "majority_class",
                "class_ratio": "majority_class_ratio",
            }
        )
    )

    # ------------------------------------------------------------
    # Confidence-weighted vote
    # ------------------------------------------------------------

    weighted_votes = (
        df.groupby(
            ["track_id", "class_name"],
            observed=True,
        )["confidence"]
        .sum()
        .rename("weighted_confidence")
        .reset_index()
    )

    weighted_class = (
        weighted_votes
        .sort_values(
            [
                "track_id",
                "weighted_confidence",
            ],
            ascending=[True, False],
        )
        .drop_duplicates("track_id")
        [
            [
                "track_id",
                "class_name",
                "weighted_confidence",
            ]
        ]
        .rename(
            columns={
                "class_name": "confidence_weighted_class",
            }
        )
    )

    # ------------------------------------------------------------
    # Confidence + temporal consistency vote (FINAL)
    # ------------------------------------------------------------

    temporal_votes = (
        df.groupby(
            ["track_id", "class_name"],
            observed=True,
        )["temporal_weighted_confidence"]
        .sum()
        .rename("temporal_weighted_vote")
        .reset_index()
    )

    temporal_class = (
        temporal_votes
        .sort_values(
            [
                "track_id",
                "temporal_weighted_vote",
            ],
            ascending=[True, False],
        )
        .drop_duplicates("track_id")
        [
            [
                "track_id",
                "class_name",
                "temporal_weighted_vote",
            ]
        ]
        .rename(
            columns={
                "class_name": "track_class",
            }
        )
    )

    # ------------------------------------------------------------
    # Final class confidence ratio
    # ------------------------------------------------------------

    total_temporal_vote = (
        temporal_votes
        .groupby("track_id")[
            "temporal_weighted_vote"
        ]
        .sum()
        .rename("total_temporal_vote")
    )

    temporal_votes = temporal_votes.join(
        total_temporal_vote,
        on="track_id",
    )

    temporal_votes["temporal_class_ratio"] = (
        temporal_votes["temporal_weighted_vote"]
        / temporal_votes["total_temporal_vote"]
    )

    temporal_class_ratio = (
        temporal_votes
        .sort_values(
            [
                "track_id",
                "temporal_weighted_vote",
            ],
            ascending=[True, False],
        )
        .drop_duplicates("track_id")
        [
            [
                "track_id",
                "temporal_class_ratio",
            ]
        ]
        .rename(
            columns={
                "temporal_class_ratio":
                    "track_class_ratio",
            }
        )
    )

    track_level = (
        temporal_class
        .merge(
            temporal_class_ratio,
            on="track_id",
            how="left",
        )
        .merge(
            majority_class,
            on="track_id",
            how="left",
        )
        .merge(
            weighted_class,
            on="track_id",
            how="left",
        )
    )

    track_level["class_ambiguous"] = (
        track_level["track_class_ratio"] < 0.70
    )

    # ------------------------------------------------------------
    # Track coverage / confidence
    # ------------------------------------------------------------

    track_frame_stats = (
        df.groupby("track_id")
        .agg(
            first_frame=("frame_id", "min"),
            last_frame=("frame_id", "max"),
            observed_frames=("frame_id", "nunique"),
            mean_confidence=("confidence", "mean"),
            median_confidence=("confidence", "median"),
            mean_temporal_support=(
                "temporal_support",
                "mean",
            ),
        )
    )

    # Source frame ids are preserved, so estimate the actual sampling
    # interval from each track rather than assuming every source frame
    # was processed.
    per_track_median_gap = (
        df.groupby("track_id")["frame_id"]
        .diff()
        .groupby(df["track_id"])
        .median()
        .rename("median_source_frame_gap")
    )

    track_frame_stats = track_frame_stats.merge(
        per_track_median_gap,
        on="track_id",
        how="left",
    )

    track_frame_stats["median_source_frame_gap"] = (
        track_frame_stats["median_source_frame_gap"]
        .fillna(1.0)
        .clip(lower=1.0)
    )

    track_frame_stats["expected_observations"] = (
        (
            track_frame_stats["last_frame"]
            - track_frame_stats["first_frame"]
        )
        / track_frame_stats["median_source_frame_gap"]
    ).round().astype(int) + 1

    track_frame_stats["observation_ratio"] = (
        track_frame_stats["observed_frames"]
        / track_frame_stats["expected_observations"]
    ).clip(0.0, 1.0)

    track_frame_stats["duration_sec"] = (
        (
            track_frame_stats["last_frame"]
            - track_frame_stats["first_frame"]
        )
        / fps
    )

    track_level = (
        track_level
        .merge(
            track_frame_stats,
            on="track_id",
            how="left",
        )
    )

    # ------------------------------------------------------------
    # Gap analysis
    # ------------------------------------------------------------

    df_with_gap = df.copy()

    df_with_gap["frame_gap"] = (
        df_with_gap
        .groupby("track_id")["frame_id"]
        .diff()
    )

    df_with_gap["gap_sec"] = (
        df_with_gap["frame_gap"] / fps
    )

    gap_mask = (
        df_with_gap["gap_sec"] > 1.0
    )

    gap_summary = (
        df_with_gap.loc[gap_mask]
        .groupby("track_id")
        .agg(
            gap_events=("gap_sec", "size"),
            max_gap_sec=("gap_sec", "max"),
            total_gap_sec=("gap_sec", "sum"),
        )
    )

    track_level = (
        track_level
        .merge(
            gap_summary,
            on="track_id",
            how="left",
        )
    )

    for column in [
        "gap_events",
        "max_gap_sec",
        "total_gap_sec",
    ]:
        track_level[column] = (
            track_level[column]
            .fillna(0)
        )

    # ------------------------------------------------------------
    # Quality score
    # ------------------------------------------------------------

    track_level["quality_score"] = (
        0.35
        * track_level["track_class_ratio"]
        + 0.30
        * track_level["observation_ratio"].clip(0, 1)
        + 0.35
        * track_level["mean_confidence"].clip(0, 1)
    )

    track_level["quality_level"] = pd.cut(
        track_level["quality_score"],
        bins=[
            -np.inf,
            0.60,
            0.75,
            0.90,
            np.inf,
        ],
        labels=[
            "poor",
            "moderate",
            "good",
            "excellent",
        ],
        right=False,
    )

    # ------------------------------------------------------------
    # Merge track-level decision back to every frame observation
    # ------------------------------------------------------------

    tracks_phase2 = (
        df
        .merge(
            track_level[
                [
                    "track_id",
                    "track_class",
                    "track_class_ratio",
                    "class_ambiguous",
                    "quality_score",
                    "quality_level",
                ]
            ],
            on="track_id",
            how="left",
            validate="many_to_one",
        )
    )

    return (
        track_level,
        tracks_phase2,
    )

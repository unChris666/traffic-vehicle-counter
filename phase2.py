
# PHASE 2: Track-level class

# ============================================================
# MAJORITY CLASS COUNT
# ============================================================

class_counts = (
    tracks_raw
    .groupby(
        [
            "track_id",
            "class_name"
        ],
        observed=True
    )
    .size()
    .rename("class_observations")
    .reset_index()
)


# ============================================================
# TOTAL OBSERVATIONS PER TRACK
# ============================================================

track_totals = (
    class_counts
    .groupby("track_id")[
        "class_observations"
    ]
    .sum()
    .rename("total_observations")
)


class_counts = (
    class_counts
    .join(
        track_totals,
        on="track_id"
    )
)


class_counts["class_ratio"] = (
    class_counts["class_observations"]
    /
    class_counts["total_observations"]
)


# ============================================================
# MAJORITY CLASS
# ============================================================

majority_class = (
    class_counts
    .sort_values(
        [
            "track_id",
            "class_observations",
            "class_ratio"
        ],
        ascending=[
            True,
            False,
            False
        ]
    )
    .drop_duplicates(
        "track_id"
    )
    [
        [
            "track_id",
            "class_name",
            "class_ratio"
        ]
    ]
    .rename(
        columns={
            "class_name":
                "track_class",

            "class_ratio":
                "track_class_ratio"
        }
    )
)


# ============================================================
# CONFIDENCE-WEIGHTED CLASS
# ============================================================

weighted_votes = (
    tracks_raw
    .groupby(
        [
            "track_id",
            "class_name"
        ],
        observed=True
    )["confidence"]
    .sum()
    .rename(
        "weighted_confidence"
    )
    .reset_index()
)


weighted_class = (
    weighted_votes
    .sort_values(
        [
            "track_id",
            "weighted_confidence"
        ],
        ascending=[
            True,
            False
        ]
    )
    .drop_duplicates(
        "track_id"
    )
    [
        [
            "track_id",
            "class_name"
        ]
    ]
    .rename(
        columns={
            "class_name":
                "confidence_weighted_class"
        }
    )
)


# ============================================================
# CLASS AMBIGUITY
#
# Conservative:
# If majority class ratio < 70%,
# mark ambiguous.
# ============================================================

majority_class["class_ambiguous"] = (
    majority_class["track_class_ratio"]
    < 0.70
)


# ============================================================
# MERGE
# ============================================================

track_level = (
    majority_class
    .merge(
        weighted_class,
        on="track_id",
        how="left"
    )
)


print("=" * 70)
print("TRACK-LEVEL CLASS")
print("=" * 70)

display(
    track_level.head(20)
)

# ============================================================
# TRACK COVERAGE
# ============================================================

track_frame_stats = (
    tracks_raw
    .groupby("track_id")
    .agg(
        first_frame=(
            "frame_id",
            "min"
        ),

        last_frame=(
            "frame_id",
            "max"
        ),

        observed_frames=(
            "frame_id",
            "nunique"
        ),

        mean_confidence=(
            "confidence",
            "mean"
        ),

        median_confidence=(
            "confidence",
            "median"
        ),
    )
)


track_frame_stats["expected_frames"] = (
    track_frame_stats["last_frame"]
    -
    track_frame_stats["first_frame"]
    +
    1
)


track_frame_stats["observation_ratio"] = (
    track_frame_stats["observed_frames"]
    /
    track_frame_stats["expected_frames"]
)


track_frame_stats["duration_sec"] = (
    track_frame_stats["expected_frames"]
    /
    FPS
)


track_level = (
    track_level
    .merge(
        track_frame_stats,
        on="track_id",
        how="left"
    )
)


display(
    track_level.head()
)

# ============================================================
# GAP ANALYSIS
# ============================================================

tracks_raw["frame_gap"] = (
    tracks_raw
    .groupby("track_id")["frame_id"]
    .diff()
)


tracks_raw["gap_sec"] = (
    tracks_raw["frame_gap"]
    /
    FPS
)


gap_mask = (
    tracks_raw["gap_sec"]
    > 1
)


gap_summary = (
    tracks_raw.loc[gap_mask]
    .groupby("track_id")
    .agg(
        gap_events=(
            "gap_sec",
            "size"
        ),

        max_gap_sec=(
            "gap_sec",
            "max"
        ),

        total_gap_sec=(
            "gap_sec",
            "sum"
        ),
    )
)


track_level = (
    track_level
    .merge(
        gap_summary,
        on="track_id",
        how="left"
    )
)


track_level[
    [
        "gap_events",
        "max_gap_sec",
        "total_gap_sec"
    ]
] = (
    track_level[
        [
            "gap_events",
            "max_gap_sec",
            "total_gap_sec"
        ]
    ]
    .fillna(0)
)

# ============================================================
# TRACK QUALITY
# ============================================================

track_level["quality_score"] = (
    0.35
    * track_level["track_class_ratio"]
    +
    0.30
    * track_level["observation_ratio"].clip(0, 1)
    +
    0.35
    * track_level["mean_confidence"].clip(0, 1)
)


# ============================================================
# QUALITY LEVEL
# ============================================================

track_level["quality_level"] = pd.cut(
    track_level["quality_score"],
    bins=[
        -np.inf,
        0.60,
        0.75,
        0.90,
        np.inf
    ],
    labels=[
        "poor",
        "moderate",
        "good",
        "excellent"
    ],
    right=False
)


# ============================================================
# SAVE
# ============================================================

track_level.to_csv(
    PHASE2_SUMMARY_PATH,
    index=False
)


print("=" * 70)
print("PHASE 2 — TRACK QUALITY")
print("=" * 70)

display(
    track_level[
        [
            "track_id",
            "track_class",
            "track_class_ratio",
            "confidence_weighted_class",
            "class_ambiguous",
            "observation_ratio",
            "mean_confidence",
            "quality_score",
            "quality_level"
        ]
    ].head(20)
)


tracks_phase2 = (
    tracks_raw
    .merge(
        track_level[
            [
                "track_id",
                "track_class",
                "track_class_ratio",
                "class_ambiguous",
                "quality_score",
                "quality_level"
            ]
        ],
        on="track_id",
        how="left",
        validate="many_to_one"
    )
)


tracks_phase2.to_csv(
    PHASE2_TRACKS_PATH,
    index=False
)


print(
    "Output:",
    PHASE2_TRACKS_PATH
)

display(
    tracks_phase2.head()
)


# ============================================================
# PHASE 2 VALIDATION
# ============================================================

print("=" * 70)
print("PHASE 2 VALIDATION")
print("=" * 70)


print(
    "Unique tracks:",
    track_level["track_id"].nunique()
)


print(
    "Track-level classes:"
)

display(
    track_level[
        "track_class"
    ]
    .value_counts()
)


print(
    "\nAmbiguous tracks:"
)

print(
    track_level[
        "class_ambiguous"
    ]
    .value_counts()
)


print(
    "\nQuality:"
)

display(
    track_level[
        "quality_level"
    ]
    .value_counts()
    .rename("track_count")
    .to_frame()
)

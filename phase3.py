
# PHASE 3: Bottom-center trajectory

# ============================================================
# BOTTOM-CENTER TRAJECTORY
# ============================================================

trajectory = (
    tracks_phase2[
        [
            "track_id",
            "frame_id",
            "timestamp_sec",
            "bottom_center_x",
            "bottom_center_y",
            "track_class",
            "track_class_ratio",
            "class_ambiguous"
        ]
    ]
    .sort_values(
        [
            "track_id",
            "frame_id"
        ]
    )
    .copy()
)


trajectory["dx"] = (
    trajectory
    .groupby("track_id")[
        "bottom_center_x"
    ]
    .diff()
)


trajectory["dy"] = (
    trajectory
    .groupby("track_id")[
        "bottom_center_y"
    ]
    .diff()
)


trajectory["frame_delta"] = (
    trajectory
    .groupby("track_id")[
        "frame_id"
    ]
    .diff()
)


trajectory["time_delta_sec"] = (
    trajectory["frame_delta"]
    /
    FPS
)


print(
    trajectory.shape
)

display(
    trajectory.head(20)
)

# ============================================================
# LINE GEOMETRY
# ============================================================

X1 = COUNT_LINE["x1"]
Y1 = COUNT_LINE["y1"]

X2 = COUNT_LINE["x2"]
Y2 = COUNT_LINE["y2"]


LINE_DX = X2 - X1
LINE_DY = Y2 - Y1


LINE_LENGTH = math.hypot(
    LINE_DX,
    LINE_DY
)


def signed_line_value(
    x,
    y
):
    """
    Signed cross-product value.

    > 0 = one side
    < 0 = opposite side
    = 0 = exactly on line
    """

    return (
        LINE_DX * (y - Y1)
        -
        LINE_DY * (x - X1)
    )


trajectory["line_value"] = (
    LINE_DX
    * (
        trajectory["bottom_center_y"]
        -
        Y1
    )
    -
    LINE_DY
    * (
        trajectory["bottom_center_x"]
        -
        X1
    )
)


# ------------------------------------------------------------
# Normalize to approximate pixel distance from line
# ------------------------------------------------------------

trajectory["line_distance_px"] = (
    np.abs(
        trajectory["line_value"]
    )
    /
    LINE_LENGTH
)


# ------------------------------------------------------------
# Deadband
# ------------------------------------------------------------

trajectory["side"] = np.select(
    [
        trajectory["line_distance_px"]
        <= LINE_DEADBAND_PX,

        trajectory["line_value"] > 0
    ],
    [
        0,
        1
    ],
    default=-1
)


display(
    trajectory[
        [
            "track_id",
            "frame_id",
            "bottom_center_x",
            "bottom_center_y",
            "line_distance_px",
            "side"
        ]
    ].head(20)
)

# ============================================================
# STABLE SIDE OBSERVATIONS
# ============================================================

trajectory_stable = trajectory[
    trajectory["side"] != 0
].copy()


trajectory_stable["previous_side"] = (
    trajectory_stable
    .groupby("track_id")["side"]
    .shift(1)
)


trajectory_stable["previous_frame"] = (
    trajectory_stable
    .groupby("track_id")["frame_id"]
    .shift(1)
)


trajectory_stable["frame_gap"] = (
    trajectory_stable["frame_id"]
    -
    trajectory_stable["previous_frame"]
)


# ------------------------------------------------------------
# Valid consecutive observations
# ------------------------------------------------------------

trajectory_stable["valid_temporal_transition"] = (
    trajectory_stable["frame_gap"]
    <= (
        MAX_TRAJECTORY_GAP_SEC
        * FPS
    )
)


trajectory_stable["crossed"] = (
    trajectory_stable["valid_temporal_transition"]
    &
    trajectory_stable["previous_side"].notna()
    &
    (
        trajectory_stable["side"]
        !=
        trajectory_stable["previous_side"]
    )
)


crossing_candidates = (
    trajectory_stable[
        trajectory_stable["crossed"]
    ]
    .copy()
)


print(
    "Crossing candidates:",
    len(crossing_candidates)
)


# ============================================================
# CROSSING EVENTS
# ============================================================

crossing_candidates["direction"] = np.where(
    crossing_candidates["previous_side"] < 0,
    "side_-1_to_+1",
    "side_+1_to_-1"
)


# ============================================================
# ONE CROSSING / TRACK
#
# If a track crosses multiple times because of jitter,
# keep the FIRST valid crossing only.
# ============================================================

crossing_events = (
    crossing_candidates
    .sort_values(
        [
            "track_id",
            "frame_id"
        ]
    )
    .drop_duplicates(
        subset=["track_id"],
        keep="first"
    )
    [
        [
            "track_id",
            "frame_id",
            "timestamp_sec",
            "bottom_center_x",
            "bottom_center_y",
            "direction",
            "track_class",
            "track_class_ratio",
            "class_ambiguous"
        ]
    ]
    .rename(
        columns={
            "frame_id":
                "crossing_frame",

            "timestamp_sec":
                "crossing_time_sec",

            "bottom_center_x":
                "crossing_x",

            "bottom_center_y":
                "crossing_y"
        }
    )
    .reset_index(drop=True)
)


print("=" * 70)
print("CROSSING EVENTS")
print("=" * 70)

print(
    "Unique crossing tracks:",
    len(crossing_events)
)

display(
    crossing_events.head(20)
)

# ============================================================
# PERSON EXCLUSION
#
# PERSON NEVER ENTERS VEHICLE COUNT.
#
# Motorcycle + rider:
#     motorcycle = 1 vehicle
#
# Person:
#     pedestrian only
# ============================================================

crossing_vehicle = crossing_events[
    crossing_events["track_class"].isin(
        VEHICLE_CLASSES
    )
].copy()


crossing_person = crossing_events[
    crossing_events["track_class"]
    ==
    "person"
].copy()


print("=" * 70)
print("PERSON EXCLUSION")
print("=" * 70)

print(
    "Total crossing events:",
    len(crossing_events)
)

print(
    "Vehicle crossings:",
    len(crossing_vehicle)
)

print(
    "Person crossings:",
    len(crossing_person)
)


# ============================================================
# MOTORCYCLE FRAGMENTATION DEDUP
#
# CONSERVATIVE
# ============================================================

vehicle_events = (
    crossing_vehicle
    .sort_values(
        "crossing_time_sec"
    )
    .reset_index(
        drop=True
    )
)


vehicle_events["duplicate_of_track_id"] = pd.NA

vehicle_events["dedup_reason"] = ""


# ------------------------------------------------------------
# Only process motorcycles
# ------------------------------------------------------------

motorcycle_idx = np.flatnonzero(
    (
        vehicle_events["track_class"]
        ==
        "motorcycle"
    ).to_numpy()
)


accepted_motorcycles = []


for idx in motorcycle_idx:

    current = vehicle_events.iloc[idx]


    # --------------------------------------------------------
    # Compare only against previously accepted motorcycles
    # --------------------------------------------------------

    duplicate_found = False

    duplicate_track_id = None


    for accepted_idx in accepted_motorcycles:

        previous = (
            vehicle_events
            .iloc[accepted_idx]
        )


        # ----------------------------------------------------
        # Same direction required
        # ----------------------------------------------------

        if (
            current["direction"]
            !=
            previous["direction"]
        ):
            continue


        # ----------------------------------------------------
        # Temporal proximity
        # ----------------------------------------------------

        dt = abs(
            current["crossing_time_sec"]
            -
            previous["crossing_time_sec"]
        )


        if dt > MOTO_DEDUP_TIME_SEC:
            continue


        # ----------------------------------------------------
        # Spatial proximity
        # ----------------------------------------------------

        distance = math.hypot(
            current["crossing_x"]
            -
            previous["crossing_x"],

            current["crossing_y"]
            -
            previous["crossing_y"]
        )


        if distance > MOTO_DEDUP_DISTANCE_PX:
            continue


        # ----------------------------------------------------
        # Candidate duplicate
        # ----------------------------------------------------

        duplicate_found = True

        duplicate_track_id = (
            previous["track_id"]
        )

        break


    if duplicate_found:

        vehicle_events.at[
            idx,
            "duplicate_of_track_id"
        ] = duplicate_track_id

        vehicle_events.at[
            idx,
            "dedup_reason"
        ] = (
            "motorcycle_fragmentation"
        )

    else:

        accepted_motorcycles.append(
            idx
        )


# ------------------------------------------------------------
# Final accepted events
# ------------------------------------------------------------

vehicle_events["is_duplicate"] = (
    vehicle_events[
        "duplicate_of_track_id"
    ]
    .notna()
)


final_crossings = vehicle_events[
    ~vehicle_events["is_duplicate"]
].copy()


print("=" * 70)
print("MOTORCYCLE DEDUP")
print("=" * 70)

print(
    "Before dedup:",
    len(vehicle_events)
)

print(
    "Duplicates:",
    vehicle_events[
        "is_duplicate"
    ].sum()
)

print(
    "After dedup:",
    len(final_crossings)
)

# ============================================================
# FINAL VEHICLE COUNT
# ============================================================

final_counts = (
    final_crossings
    .groupby(
        "track_class"
    )
    .size()
    .reindex(
        [
            "motorcycle",
            "car",
            "truck",
            "bus"
        ],
        fill_value=0
    )
    .rename(
        "vehicle_count"
    )
    .reset_index()
)


final_counts["vehicle_count"] = (
    final_counts["vehicle_count"]
    .astype(int)
)


total_vehicle_count = int(
    final_counts[
        "vehicle_count"
    ].sum()
)


print("=" * 70)
print("FINAL VEHICLE COUNT")
print("=" * 70)

display(
    final_counts
)


print(
    "TOTAL VEHICLES:",
    total_vehicle_count
)


# ============================================================
# SAVE CROSSING EVENTS
# ============================================================

crossing_events.to_csv(
    PHASE3_CROSSINGS_PATH,
    index=False
)


final_crossings.to_csv(
    PHASE3_DIR /
    "final_vehicle_crossings.csv",
    index=False
)


final_counts.to_csv(
    PHASE3_FINAL_COUNTS_PATH,
    index=False
)


print(
    "Saved:"
)

print(
    PHASE3_CROSSINGS_PATH
)

print(
    PHASE3_FINAL_COUNTS_PATH
)


# ============================================================
# PHASE 3 AUDIT
# ============================================================

audit = {
    "all_crossing_events":
        len(crossing_events),

    "person_crossings_excluded":
        len(crossing_person),

    "vehicle_crossings_before_dedup":
        len(vehicle_events),

    "motorcycle_duplicates_removed":
        int(
            vehicle_events[
                "is_duplicate"
            ].sum()
        ),

    "final_vehicle_crossings":
        len(final_crossings),

    "final_vehicle_count":
        total_vehicle_count,
}


print("=" * 70)
print("PHASE 3 AUDIT")
print("=" * 70)

for key, value in audit.items():

    print(
        f"{key:35s}: {value:,}"
    )


with open(
    PHASE3_DIR /
    "phase3_audit.json",
    "w"
) as f:

    json.dump(
        audit,
        f,
        indent=4
    )


# ============================================================
# COUNT BY DIRECTION
# ============================================================

direction_counts = (
    final_crossings
    .groupby(
        [
            "track_class",
            "direction"
        ]
    )
    .size()
    .rename(
        "count"
    )
    .reset_index()
)


display(
    direction_counts
)


# ============================================================
# COUNT BY TRACK QUALITY
# ============================================================

crossing_quality = (
    final_crossings
    .merge(
        track_level[
            [
                "track_id",
                "quality_score",
                "quality_level"
            ]
        ],
        on="track_id",
        how="left",
        validate="one_to_one"
    )
)


quality_count = (
    crossing_quality
    .groupby(
        [
            "track_class",
            "quality_level"
        ],
        observed=True
    )
    .size()
    .rename(
        "count"
    )
    .reset_index()
)


display(
    quality_count
)


# ============================================================
# VIDEO RENDER PREPARATION
# ============================================================

# Only observations needed for visualization
render_df = tracks_phase2[
    [
        "frame_id",
        "track_id",
        "x1",
        "y1",
        "x2",
        "y2",
        "track_class",
        "track_class_ratio",
        "class_ambiguous"
    ]
].copy()


render_df["frame_id"] = (
    render_df["frame_id"]
    .astype(np.int32)
)


# ------------------------------------------------------------
# Group by frame once
# ------------------------------------------------------------

frame_groups = {
    frame_id: group
    for frame_id, group
    in render_df.groupby(
        "frame_id",
        sort=False
    )
}


# ------------------------------------------------------------
# Crossing events by frame
# ------------------------------------------------------------

crossing_frame_groups = {
    frame_id: group
    for frame_id, group
    in final_crossings.groupby(
        "crossing_frame",
        sort=False
    )
}


# ------------------------------------------------------------
# Cumulative count lookup
#
# IMPORTANT:
# This uses INTEGER frame keys.
#
# No tuple-key bug.
# ------------------------------------------------------------

crossing_frame_counts = (
    final_crossings[
        "crossing_frame"
    ]
    .value_counts()
    .sort_index()
)


cumulative_count_by_frame = (
    crossing_frame_counts
    .reindex(
        range(
            1,
            TOTAL_FRAMES + 1
        ),
        fill_value=0
    )
    .cumsum()
    .astype(int)
)


print(
    "Frame groups:",
    len(frame_groups)
)

print(
    "Crossing frames:",
    len(crossing_frame_groups)
)

print(
    "Final count:",
    cumulative_count_by_frame.iloc[-1]
)


# ============================================================
# FFMPEG CHECK
# ============================================================

ffmpeg_path = (
    subprocess
    .check_output(
        [
            "which",
            "ffmpeg"
        ],
        text=True
    )
    .strip()
)


print(
    "FFmpeg:",
    ffmpeg_path
)


version = subprocess.check_output(
    [
        "ffmpeg",
        "-version"
    ],
    text=True
)


print(
    version.splitlines()[0]
)

# ============================================================
# VIDEO RENDER HELPERS
# ============================================================

def draw_counting_line(
    frame
):

    cv2.line(
        frame,

        (
            COUNT_LINE["x1"],
            COUNT_LINE["y1"]
        ),

        (
            COUNT_LINE["x2"],
            COUNT_LINE["y2"]
        ),

        (0, 255, 255),

        4,

        cv2.LINE_AA
    )


    # Endpoints

    cv2.circle(
        frame,

        (
            COUNT_LINE["x1"],
            COUNT_LINE["y1"]
        ),

        7,

        (255, 0, 0),

        -1
    )


    cv2.circle(
        frame,

        (
            COUNT_LINE["x2"],
            COUNT_LINE["y2"]
        ),

        7,

        (255, 0, 0),

        -1
    )


    cv2.putText(
        frame,

        "COUNTING LINE",

        (
            min(
                COUNT_LINE["x1"] + 10,
                FRAME_WIDTH - 200
            ),

            max(
                COUNT_LINE["y1"] - 12,
                25
            )
        ),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.65,

        (0, 255, 255),

        2,

        cv2.LINE_AA
    )


    return frame


def draw_track(
    frame,
    row
):

    x1 = int(row["x1"])
    y1 = int(row["y1"])
    x2 = int(row["x2"])
    y2 = int(row["y2"])

    track_id = int(
        row["track_id"]
    )

    # ========================================================
    # IMPORTANT:
    # USE ONLY TRACK-LEVEL CLASS
    # ========================================================

    track_class = str(
        row["track_class"]
    )

    ratio = float(
        row["track_class_ratio"]
    )

    ambiguous = bool(
        row["class_ambiguous"]
    )


    label = (
        f"ID {track_id} | "
        f"{track_class.upper()} | "
        f"{ratio:.0%}"
    )


    if ambiguous:

        label += " | AMBIG"


    # --------------------------------------------------------
    # Bounding box
    # --------------------------------------------------------

    cv2.rectangle(
        frame,

        (x1, y1),

        (x2, y2),

        (0, 255, 0),

        2
    )


    # --------------------------------------------------------
    # Label
    # --------------------------------------------------------

    (
        text_w,
        text_h
    ), baseline = cv2.getTextSize(
        label,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        1
    )


    text_x = x1

    text_y = max(
        text_h + baseline + 2,
        y1 - 5
    )


    cv2.rectangle(
        frame,

        (
            text_x,
            text_y
            -
            text_h
            -
            baseline
        ),

        (
            text_x
            +
            text_w
            +
            4,

            text_y
            +
            2
        ),

        (0, 0, 0),

        -1
    )


    cv2.putText(
        frame,

        label,

        (
            text_x + 2,
            text_y
        ),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.45,

        (255, 255, 255),

        1,

        cv2.LINE_AA
    )


    return frame

# ============================================================
# PHASE 3 — PREPARE CUMULATIVE COUNT ARRAY
#
# PURPOSE
# ------------------------------------------------------------
# Convert cumulative_count_by_frame into a plain NumPy array.
#
# WHY?
# ------------------------------------------------------------
# Pandas Series uses LABEL-based indexing with [].
# Rendering requires POSITION-based indexing.
#
# Therefore:
#
# Pandas Series
#      ↓
# reset_index(drop=True)
#      ↓
# NumPy array
#      ↓
# cumulative_count_array[frame_id - 1]
#
# This makes rendering independent from Pandas index.
# ============================================================

import numpy as np
import pandas as pd


print("=" * 70)
print("PREPARING CUMULATIVE COUNT ARRAY")
print("=" * 70)


# ============================================================
# INSPECT ORIGINAL OBJECT
# ============================================================

print(
    "Original type :",
    type(cumulative_count_by_frame)
)


if isinstance(
    cumulative_count_by_frame,
    pd.Series
):

    print(
        "Original index type :",
        type(
            cumulative_count_by_frame.index
        )
    )

    print(
        "Original length     :",
        len(
            cumulative_count_by_frame
        )
    )

    print(
        "Original first index:",
        cumulative_count_by_frame.index[:5].tolist()
    )


# ============================================================
# CONVERT TO NUMPY
# ============================================================

if isinstance(
    cumulative_count_by_frame,
    pd.Series
):

    cumulative_count_array = (
        cumulative_count_by_frame
        .reset_index(drop=True)
        .to_numpy(
            dtype=np.int64
        )
    )

else:

    cumulative_count_array = np.asarray(
        cumulative_count_by_frame,
        dtype=np.int64
    )


# ============================================================
# VALIDATION
# ============================================================

print(
    "\nConverted type:",
    type(cumulative_count_array)
)

print(
    "Length:",
    len(cumulative_count_array)
)


# ============================================================
# VIDEO FRAME VALIDATION
# ============================================================

if "TOTAL_FRAMES" not in globals():

    cap_check = cv2.VideoCapture(
        str(VIDEO_PATH)
    )

    if not cap_check.isOpened():

        raise RuntimeError(
            f"Cannot open video: {VIDEO_PATH}"
        )

    TOTAL_FRAMES = int(
        cap_check.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    cap_check.release()


print(
    "Video frames:",
    TOTAL_FRAMES
)


# ============================================================
# IMPORTANT
# ============================================================
#
# We allow either:
#
# len == TOTAL_FRAMES
#
# OR
#
# len == TOTAL_FRAMES + 1
#
# depending on how the cumulative series was constructed.
#
# ============================================================

if len(cumulative_count_array) == TOTAL_FRAMES:

    pass


elif len(cumulative_count_array) == TOTAL_FRAMES + 1:

    print(
        "\nDetected TOTAL_FRAMES + 1 structure."
    )

    # Remove initial frame-0 state.
    cumulative_count_array = (
        cumulative_count_array[1:]
    )


else:

    raise ValueError(
        "\nCumulative count length does not "
        "match video frame count.\n\n"
        f"Video frames       : {TOTAL_FRAMES:,}\n"
        f"Cumulative length  : "
        f"{len(cumulative_count_array):,}\n\n"
        "Rebuild cumulative_count_by_frame "
        "before rendering."
    )


# ============================================================
# FINAL VALIDATION
# ============================================================

assert len(
    cumulative_count_array
) == TOTAL_FRAMES


# ============================================================
# SANITY CHECK
# ============================================================

if np.any(
    cumulative_count_array < 0
):

    raise ValueError(
        "Cumulative vehicle count contains "
        "negative values."
    )


if len(
    cumulative_count_array
) > 1:

    if np.any(
        np.diff(
            cumulative_count_array
        ) < 0
    ):

        raise ValueError(
            "Cumulative count is decreasing. "
            "This should never happen."
        )


print("\n" + "=" * 70)
print("CUMULATIVE COUNT ARRAY READY")
print("=" * 70)

print(
    f"Length        : "
    f"{len(cumulative_count_array):,}"
)

print(
    f"First values  : "
    f"{cumulative_count_array[:10].tolist()}"
)

print(
    f"Last values   : "
    f"{cumulative_count_array[-10:].tolist()}"
)

print(
    f"Final count   : "
    f"{int(cumulative_count_array[-1])}"
)

print(
    "Validation    : PASS"
)
# ============================================================
# PHASE 3 — FINAL CFR TRACKED VIDEO RENDERER
#
# COMPATIBILITY:
#   Kaggle FFmpeg 4.4.2
#
# IMPORTANT:
# ------------------------------------------------------------
# This cell DOES NOT:
#   - run YOLO
#   - run BoT-SORT
#   - recalculate tracking
#   - recalculate counting
#
# It ONLY renders the already-computed Phase 2 + Phase 3 data.
#
# OUTPUT:
#   - every source frame is rendered
#   - no frame dropping
#   - CFR output
#   - track-level class
#   - diagonal counting line
#   - cumulative vehicle count
#   - crossing events
# ============================================================

import cv2
import time
import subprocess
import numpy as np
import pandas as pd
from pathlib import Path


print("=" * 70)
print("PHASE 3 — FINAL CFR TRACKED VIDEO")
print("=" * 70)


# ============================================================
# PATHS
# ============================================================

PHASE3_TRACKED_VIDEO_PATH = (
    PHASE3_DIR /
    "tracked_video_diagonal_counting_CFR.mp4"
)

FFMPEG_LOG_PATH = (
    PHASE3_DIR /
    "ffmpeg_render.log"
)


# Remove previous failed output
if PHASE3_TRACKED_VIDEO_PATH.exists():

    PHASE3_TRACKED_VIDEO_PATH.unlink()


if FFMPEG_LOG_PATH.exists():

    FFMPEG_LOG_PATH.unlink()


print(
    f"Output video : "
    f"{PHASE3_TRACKED_VIDEO_PATH}"
)

print(
    f"FFmpeg log   : "
    f"{FFMPEG_LOG_PATH}"
)


# ============================================================
# OPEN SOURCE VIDEO
# ============================================================

cap = cv2.VideoCapture(
    str(VIDEO_PATH)
)


if not cap.isOpened():

    raise RuntimeError(
        f"Cannot open video:\n"
        f"{VIDEO_PATH}"
    )


# ============================================================
# READ VIDEO METADATA
# ============================================================

FPS = cap.get(
    cv2.CAP_PROP_FPS
)

FRAME_WIDTH = int(
    cap.get(
        cv2.CAP_PROP_FRAME_WIDTH
    )
)

FRAME_HEIGHT = int(
    cap.get(
        cv2.CAP_PROP_FRAME_HEIGHT
    )
)

TOTAL_FRAMES = int(
    cap.get(
        cv2.CAP_PROP_FRAME_COUNT
    )
)


print("\nVIDEO")
print("-" * 70)

print(
    f"Resolution : "
    f"{FRAME_WIDTH} x {FRAME_HEIGHT}"
)

print(
    f"FPS        : "
    f"{FPS:.12f}"
)

print(
    f"Frames     : "
    f"{TOTAL_FRAMES:,}"
)


# ============================================================
# COUNTING LINE
#
# Diagonal line:
#
# Upper-right
#       \
#        \
#         \
#          \
#           Lower-left
# ============================================================

COUNT_LINE = {

    "orientation": "diagonal",

    "x1": int(
        FRAME_WIDTH * 0.95
    ),

    "y1": int(
        FRAME_HEIGHT * 0.20
    ),

    "x2": int(
        FRAME_WIDTH * 0.05
    ),

    "y2": int(
        FRAME_HEIGHT * 0.95
    ),
}


print("\nCOUNTING LINE")
print("-" * 70)

print(
    COUNT_LINE
)


# ============================================================
# PREPARE CUMULATIVE COUNT ARRAY
# ============================================================

if isinstance(
    cumulative_count_by_frame,
    pd.Series
):

    cumulative_count_array = (
        cumulative_count_by_frame
        .reset_index(drop=True)
        .to_numpy(
            dtype=np.int64
        )
    )

else:

    cumulative_count_array = np.asarray(
        cumulative_count_by_frame,
        dtype=np.int64
    )


# ============================================================
# HANDLE POSSIBLE 1-BASED COUNT ARRAY
# ============================================================

if len(
    cumulative_count_array
) == TOTAL_FRAMES + 1:

    cumulative_count_array = (
        cumulative_count_array[1:]
    )


# ============================================================
# COUNT ARRAY VALIDATION
# ============================================================

if len(
    cumulative_count_array
) != TOTAL_FRAMES:

    cap.release()

    raise ValueError(

        "Cumulative count length mismatch.\n\n"

        f"Video frames : "
        f"{TOTAL_FRAMES:,}\n"

        f"Count array  : "
        f"{len(cumulative_count_array):,}"
    )


if np.any(
    cumulative_count_array < 0
):

    cap.release()

    raise ValueError(
        "Negative cumulative count detected."
    )


if np.any(
    np.diff(
        cumulative_count_array
    ) < 0
):

    cap.release()

    raise ValueError(
        "Cumulative count is not monotonic."
    )


print("\nCOUNT DATA")
print("-" * 70)

print(
    f"Count array length : "
    f"{len(cumulative_count_array):,}"
)

print(
    f"Final vehicle count: "
    f"{int(cumulative_count_array[-1])}"
)


# ============================================================
# FFMPEG COMMAND
#
# IMPORTANT:
# ------------------------------------------------------------
# FFmpeg 4.4.2 does NOT support:
#
#     -fps_mode cfr
#
# Therefore we use:
#
#     -vsync cfr
#
# which is supported by FFmpeg 4.4.x.
# ============================================================

ffmpeg_cmd = [

    "ffmpeg",

    "-y",

    "-hide_banner",

    "-loglevel",
    "error",

    # --------------------------------------------------------
    # RAWVIDEO INPUT
    # --------------------------------------------------------

    "-f",
    "rawvideo",

    "-pixel_format",
    "bgr24",

    "-video_size",
    f"{FRAME_WIDTH}x{FRAME_HEIGHT}",

    "-framerate",
    f"{FPS:.12f}",

    "-i",
    "pipe:0",

    # --------------------------------------------------------
    # VIDEO ENCODER
    # --------------------------------------------------------

    "-an",

    "-c:v",
    "libx264",

    "-preset",
    "medium",

    "-crf",
    "18",

    "-pix_fmt",
    "yuv420p",

    # --------------------------------------------------------
    # CFR
    #
    # Compatible with FFmpeg 4.4.2
    # --------------------------------------------------------

    "-vsync",
    "cfr",

    # --------------------------------------------------------
    # MP4
    # --------------------------------------------------------

    "-movflags",
    "+faststart",

    str(
        PHASE3_TRACKED_VIDEO_PATH
    ),
]


print("\nFFMPEG COMMAND")
print("-" * 70)

print(
    " ".join(
        ffmpeg_cmd
    )
)


# ============================================================
# START FFMPEG
# ============================================================

process = subprocess.Popen(

    ffmpeg_cmd,

    stdin=subprocess.PIPE,

    stdout=subprocess.DEVNULL,

    stderr=subprocess.PIPE,

    bufsize=0,
)


frame_id = 0

start_time = time.time()

render_error = None


# ============================================================
# FRAME RENDER LOOP
# ============================================================

try:

    while True:

        success, frame = (
            cap.read()
        )


        if not success:

            break


        frame_id += 1


        # ====================================================
        # FRAME VALIDATION
        # ====================================================

        if frame.shape != (
            FRAME_HEIGHT,
            FRAME_WIDTH,
            3
        ):

            raise RuntimeError(

                "Unexpected frame shape.\n"

                f"Frame ID : {frame_id}\n"

                f"Expected : "
                f"({FRAME_HEIGHT}, "
                f"{FRAME_WIDTH}, 3)\n"

                f"Actual   : "
                f"{frame.shape}"
            )


        # ====================================================
        # DRAW COUNTING LINE
        # ====================================================

        frame = draw_counting_line(
            frame
        )


        # ====================================================
        # DRAW TRACKS
        #
        # IMPORTANT:
        #
        # track_class
        # = Phase 2 track-level class
        #
        # NEVER use raw class_name here.
        # ====================================================

        group = (
            frame_groups.get(
                frame_id
            )
        )


        if group is not None:

            for row in group.itertuples(
                index=False
            ):

                row_data = {

                    "x1": float(
                        row.x1
                    ),

                    "y1": float(
                        row.y1
                    ),

                    "x2": float(
                        row.x2
                    ),

                    "y2": float(
                        row.y2
                    ),

                    "track_id": int(
                        row.track_id
                    ),

                    "track_class": str(
                        row.track_class
                    ),

                    "track_class_ratio": float(
                        row.track_class_ratio
                    ),

                    "class_ambiguous": bool(
                        row.class_ambiguous
                    ),
                }


                frame = draw_track(
                    frame,
                    row_data
                )


        # ====================================================
        # CURRENT CUMULATIVE VEHICLE COUNT
        # ====================================================

        current_count = int(

            cumulative_count_array[
                frame_id - 1
            ]

        )


        # ====================================================
        # COUNT PANEL
        # ====================================================

        cv2.rectangle(

            frame,

            (
                15,
                10
            ),

            (
                370,
                82
            ),

            (0, 0, 0),

            -1
        )


        cv2.putText(

            frame,

            (
                f"VEHICLE COUNT: "
                f"{current_count}"
            ),

            (
                25,
                40
            ),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.65,

            (255, 255, 255),

            2,

            cv2.LINE_AA
        )


        cv2.putText(

            frame,

            "MOTORCYCLE + RIDER = 1",

            (
                25,
                65
            ),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.38,

            (200, 200, 200),

            1,

            cv2.LINE_AA
        )


        # ====================================================
        # CROSSING EVENTS
        #
        # These events have already been calculated
        # by Phase 3.
        # ====================================================

        events = (
            crossing_frame_groups.get(
                frame_id
            )
        )


        if events is not None:

            max_events = max(

                1,

                int(
                    (
                        FRAME_HEIGHT - 40
                    ) / 25
                )

            )


            for offset, event in enumerate(

                events.itertuples(
                    index=False
                )

            ):

                if offset >= max_events:

                    break


                event_text = (

                    f"COUNTED: "

                    f"{str(event.track_class).upper()} "

                    f"| ID {int(event.track_id)} "

                    f"| {event.direction}"

                )


                y = (

                    FRAME_HEIGHT
                    -
                    25
                    -
                    (
                        offset * 24
                    )

                )


                cv2.putText(

                    frame,

                    event_text,

                    (
                        20,
                        y
                    ),

                    cv2.FONT_HERSHEY_SIMPLEX,

                    0.48,

                    (0, 255, 255),

                    2,

                    cv2.LINE_AA
                )


        # ====================================================
        # FRAME NUMBER
        # ====================================================

        cv2.putText(

            frame,

            (
                f"Frame: "
                f"{frame_id:,}"
            ),

            (
                FRAME_WIDTH - 180,
                30
            ),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.45,

            (255, 255, 255),

            1,

            cv2.LINE_AA
        )


        # ====================================================
        # SEND EXACTLY ONE FRAME TO FFMPEG
        # ====================================================

        try:

            process.stdin.write(
                frame.tobytes()
            )

        except BrokenPipeError:

            render_error = (
                f"FFmpeg closed pipe "
                f"at frame {frame_id:,}."
            )

            break


finally:

    cap.release()


# ============================================================
# CLOSE PIPE
# ============================================================

try:

    process.stdin.close()

except Exception:

    pass


# ============================================================
# READ FFMPEG STDERR
# ============================================================

ffmpeg_stderr = (
    process.stderr
    .read()
    .decode(
        "utf-8",
        errors="replace"
    )
)


# ============================================================
# WAIT FOR PROCESS
# ============================================================

return_code = (
    process.wait()
)


elapsed = (
    time.time()
    -
    start_time
)


# ============================================================
# SAVE FFMPEG LOG
# ============================================================

FFMPEG_LOG_PATH.write_text(
    ffmpeg_stderr
)


# ============================================================
# HANDLE FFMPEG ERROR
# ============================================================

if return_code != 0:

    print("\n" + "=" * 70)
    print("FFMPEG FAILED")
    print("=" * 70)

    print(
        f"Return code : "
        f"{return_code}"
    )

    print(
        f"Last frame  : "
        f"{frame_id:,}"
    )

    print(
        "\nFFmpeg error:"
    )

    print(
        ffmpeg_stderr
    )

    raise RuntimeError(
        "FFmpeg rendering failed."
    )


if render_error is not None:

    raise RuntimeError(
        render_error
    )


# ============================================================
# FRAME COUNT VALIDATION
# ============================================================

if frame_id != TOTAL_FRAMES:

    raise RuntimeError(

        "Source video was not fully rendered.\n"

        f"Rendered frames : "
        f"{frame_id:,}\n"

        f"Expected frames : "
        f"{TOTAL_FRAMES:,}"
    )


# ============================================================
# OUTPUT VALIDATION
# ============================================================

if not (
    PHASE3_TRACKED_VIDEO_PATH.exists()
):

    raise RuntimeError(
        "FFmpeg finished but output "
        "video does not exist."
    )


output_size_mb = (

    PHASE3_TRACKED_VIDEO_PATH
    .stat()
    .st_size
    /
    (1024 ** 2)

)


if output_size_mb <= 0:

    raise RuntimeError(
        "Output video is empty."
    )


# ============================================================
# SUCCESS
# ============================================================

print("\n" + "=" * 70)
print("VIDEO GENERATION COMPLETE")
print("=" * 70)

print(
    f"Frames rendered : "
    f"{frame_id:,}"
)

print(
    f"Expected frames : "
    f"{TOTAL_FRAMES:,}"
)

print(
    f"Frame validation: "
    f"{frame_id == TOTAL_FRAMES}"
)

print(
    f"FPS             : "
    f"{FPS:.12f}"
)

print(
    f"Elapsed         : "
    f"{elapsed:.2f} sec"
)

print(
    f"Render FPS      : "
    f"{frame_id / elapsed:.2f}"
)

print(
    f"Final count     : "
    f"{int(cumulative_count_array[-1])}"
)

print(
    f"Output          : "
    f"{PHASE3_TRACKED_VIDEO_PATH}"
)

print(
    f"File size       : "
    f"{output_size_mb:.2f} MB"
)
# ============================================================
# PHASE 3 — FFmpeg ERROR DIAGNOSTIC
# ============================================================

from pathlib import Path

FFMPEG_LOG_PATH = (
    PHASE3_DIR /
    "ffmpeg_render.log"
)

print("=" * 70)
print("FFMPEG DIAGNOSTIC")
print("=" * 70)

print(
    "Log path:",
    FFMPEG_LOG_PATH
)

if not FFMPEG_LOG_PATH.exists():

    raise FileNotFoundError(
        f"FFmpeg log not found:\n"
        f"{FFMPEG_LOG_PATH}"
    )


log_text = FFMPEG_LOG_PATH.read_text(
    errors="replace"
)

print("\n" + "=" * 70)
print("LAST 100 LINES")
print("=" * 70)

lines = log_text.splitlines()

for line in lines[-100:]:
    print(line)
# ============================================================
# PHASE 3 — SOURCE vs OUTPUT VIDEO VALIDATION
# ============================================================

def probe_video(path):

    cap = cv2.VideoCapture(
        str(path)
    )

    if not cap.isOpened():

        raise RuntimeError(
            f"Cannot open: {path}"
        )

    metadata = {

        "path": str(path),

        "width": int(
            cap.get(
                cv2.CAP_PROP_FRAME_WIDTH
            )
        ),

        "height": int(
            cap.get(
                cv2.CAP_PROP_FRAME_HEIGHT
            )
        ),

        "fps": cap.get(
            cv2.CAP_PROP_FPS
        ),

        "frame_count": int(
            cap.get(
                cv2.CAP_PROP_FRAME_COUNT
            )
        ),

    }

    if metadata["fps"] > 0:

        metadata["duration_sec"] = (
            metadata["frame_count"]
            /
            metadata["fps"]
        )

    else:

        metadata["duration_sec"] = None

    cap.release()

    return metadata


source_info = probe_video(
    VIDEO_PATH
)

output_info = probe_video(
    PHASE3_TRACKED_VIDEO_PATH
)


validation = pd.DataFrame(

    [
        source_info,
        output_info
    ],

    index=[
        "SOURCE",
        "OUTPUT"
    ]

)


display(
    validation
)
# ============================================================
# FINAL VALIDATION
# ============================================================

print("=" * 70)
print("PHASE 3 VIDEO VALIDATION")
print("=" * 70)


checks = {

    "width":
        source_info["width"]
        ==
        output_info["width"],

    "height":
        source_info["height"]
        ==
        output_info["height"],

    "frame_count":
        source_info["frame_count"]
        ==
        output_info["frame_count"],

    "fps":
        abs(
            source_info["fps"]
            -
            output_info["fps"]
        ) < 0.01,

    "duration":
        abs(
            source_info["duration_sec"]
            -
            output_info["duration_sec"]
        ) < 0.10,
}


for name, passed in checks.items():

    print(
        f"{name:15s}: "
        f"{'PASS' if passed else 'FAIL'}"
    )


print("\nOverall:")

if all(checks.values()):

    print(
        "PASS — SOURCE AND OUTPUT VIDEO "
        "TIMELINE ARE CONSISTENT."
    )

else:

    print(
        "WARNING — VIDEO METADATA "
        "DIFFERS."
    )
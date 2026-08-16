# PHASE 1

## Configuration
# !pip install -q -U ultralytics opencv-python-headless psutil
from pathlib import Path
from collections import defaultdict
import json
import math
import os
import subprocess
import time
import warnings

import cv2
import numpy as np
import pandas as pd
import torch

from ultralytics import YOLO

warnings.filterwarnings("ignore")


# ============================================================
# PATHS
# ============================================================

VIDEO_PATH = Path(
    "/kaggle/input/datasets/chrisbiran/traffic-tracker-videos/TOR3-PAGI.mp4"
)

PROJECT_DIR = Path(
    "/kaggle/working/traffic_counting"
)

PHASE1_DIR = PROJECT_DIR / "phase1"
PHASE2_DIR = PROJECT_DIR / "phase2"
PHASE3_DIR = PROJECT_DIR / "phase3"

for directory in [
    PHASE1_DIR,
    PHASE2_DIR,
    PHASE3_DIR,
]:
    directory.mkdir(
        parents=True,
        exist_ok=True
    )


# ============================================================
# MODEL
# ============================================================

MODEL_NAME = "yolo26m.pt"

TRACKER = "botsort.yaml"

DEVICE = 0 if torch.cuda.is_available() else "cpu"


# ============================================================
# DETECTION CONFIG
# ============================================================

IMG_SIZE = 960

CONF_THRESHOLD = 0.25

IOU_THRESHOLD = 0.50


# ============================================================
# TARGET CLASSES
# ============================================================

TARGET_CLASSES = {
    "person",
    "motorcycle",
    "car",
    "truck",
    "bus",
}


# ============================================================
# VEHICLE CLASSES
# ============================================================

VEHICLE_CLASSES = {
    "motorcycle",
    "car",
    "truck",
    "bus",
}


# ============================================================
# DIAGONAL COUNTING LINE
#
# Upper point
# Lower point
# ============================================================

COUNT_LINE = {
    "orientation": "diagonal",

    "x1": None,
    "y1": None,

    "x2": None,
    "y2": None,
}


# ============================================================
# COUNTING PARAMETERS
# ============================================================

# Distance from line where a point is considered "near line".
# This prevents tiny jitter around the line from creating
# false crossing events.
LINE_DEADBAND_PX = 8.0


# Minimum number of trajectory observations required.
MIN_TRACK_OBSERVATIONS = 5


# Maximum allowed time gap between consecutive observations
# when analyzing trajectory.
MAX_TRAJECTORY_GAP_SEC = 1.5


# ============================================================
# MOTORCYCLE FRAGMENTATION DEDUP
#
# Conservative on purpose.
# We only suppress motorcycle events when BOTH:
#   1. crossing events are close in time
#   2. crossing points are close in space
#   3. direction is identical
#
# This is NOT aggressive duplicate removal.
# ============================================================

MOTO_DEDUP_TIME_SEC = 1.50

MOTO_DEDUP_DISTANCE_PX = 80.0


# ============================================================
# OUTPUTS
# ============================================================

PHASE1_TRACKS_PATH = (
    PHASE1_DIR /
    "tracks_raw.csv"
)

PHASE2_TRACKS_PATH = (
    PHASE2_DIR /
    "tracks_with_track_class.csv"
)

PHASE2_SUMMARY_PATH = (
    PHASE2_DIR /
    "track_quality_summary.csv"
)

PHASE3_CROSSINGS_PATH = (
    PHASE3_DIR /
    "crossing_events.csv"
)

PHASE3_FINAL_COUNTS_PATH = (
    PHASE3_DIR /
    "final_vehicle_counts.csv"
)

PHASE3_TRACKED_VIDEO_PATH = (
    PHASE3_DIR /
    "tracked_video_diagonal_counting_CFR.mp4"
)


print("=" * 70)
print("PROJECT CONFIGURATION")
print("=" * 70)

print("Video       :", VIDEO_PATH)
print("Model       :", MODEL_NAME)
print("Tracker     :", TRACKER)
print("Device      :", DEVICE)
print("Image size  :", IMG_SIZE)
print("Confidence  :", CONF_THRESHOLD)

# ============================================================
# ENVIRONMENT CHECK
# ============================================================

import ultralytics

print("=" * 70)
print("ENVIRONMENT")
print("=" * 70)

print("Python          :", os.sys.version.split()[0])
print("PyTorch         :", torch.__version__)
print("Ultralytics    :", ultralytics.__version__)
print("CUDA available  :", torch.cuda.is_available())
print("CUDA version    :", torch.version.cuda)

if torch.cuda.is_available():

    print(
        "GPU count       :",
        torch.cuda.device_count()
    )

    for i in range(torch.cuda.device_count()):

        print(
            f"GPU {i}          :",
            torch.cuda.get_device_name(i)
        )

# ============================================================
# VIDEO METADATA
# ============================================================

def read_video_metadata(video_path):

    cap = cv2.VideoCapture(
        str(video_path)
    )

    if not cap.isOpened():
        raise RuntimeError(
            f"Cannot open video: {video_path}"
        )

    fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    frame_count = int(
        cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    width = int(
        cap.get(
            cv2.CAP_PROP_FRAME_WIDTH
        )
    )

    height = int(
        cap.get(
            cv2.CAP_PROP_FRAME_HEIGHT
        )
    )

    cap.release()

    duration = (
        frame_count / fps
        if fps > 0
        else None
    )

    return {
        "path": str(video_path),
        "filename": video_path.name,
        "width": width,
        "height": height,
        "fps": fps,
        "frame_count": frame_count,
        "duration_sec": duration,
        "duration_min": (
            duration / 60
            if duration
            else None
        ),
    }


VIDEO_META = read_video_metadata(
    VIDEO_PATH
)


FRAME_WIDTH = VIDEO_META["width"]
FRAME_HEIGHT = VIDEO_META["height"]
FPS = VIDEO_META["fps"]
TOTAL_FRAMES = VIDEO_META["frame_count"]


print("=" * 70)
print("VIDEO")
print("=" * 70)

for key, value in VIDEO_META.items():
    print(f"{key:20s}: {value}")

# ============================================================
# DIAGONAL COUNTING LINE
# ============================================================

COUNT_LINE = {
    "orientation": "diagonal",

    # Upper point
    "x1": int(FRAME_WIDTH * 0.95),
    "y1": int(FRAME_HEIGHT * 0.20),

    # Lower point
    "x2": int(FRAME_WIDTH * 0.05),
    "y2": int(FRAME_HEIGHT * 0.95),
}


print("=" * 70)
print("COUNTING LINE")
print("=" * 70)

print(
    f"Point 1: "
    f"({COUNT_LINE['x1']}, {COUNT_LINE['y1']})"
)

print(
    f"Point 2: "
    f"({COUNT_LINE['x2']}, {COUNT_LINE['y2']})"
)

# ============================================================
# LOAD MODEL
# ============================================================

print("=" * 70)
print("LOADING MODEL")
print("=" * 70)

model = YOLO(
    MODEL_NAME
)

print(
    "Model loaded:",
    MODEL_NAME
)

print(
    "Classes:",
    len(model.names)
)

# ============================================================
# CLASS MAPPING
# ============================================================

CLASS_NAMES = model.names

TARGET_CLASS_IDS = {
    class_id
    for class_id, class_name
    in CLASS_NAMES.items()
    if class_name in TARGET_CLASSES
}


print("=" * 70)
print("TARGET CLASSES")
print("=" * 70)

for class_id in sorted(
    TARGET_CLASS_IDS
):

    print(
        class_id,
        "->",
        CLASS_NAMES[class_id]
    )

print("=" * 70)
print("PHASE 1 — YOLO26m + BoT-SORT")
print("=" * 70)

start_time = time.time()

track_records = []

results_stream = model.track(
    source=str(VIDEO_PATH),

    tracker=TRACKER,

    imgsz=IMG_SIZE,

    conf=CONF_THRESHOLD,

    iou=IOU_THRESHOLD,

    device=DEVICE,

    stream=True,

    persist=True,

    verbose=False,
)


for frame_id, result in enumerate(
    results_stream,
    start=1
):

    boxes = result.boxes

    if (
        boxes is None
        or len(boxes) == 0
    ):
        continue


    # --------------------------------------------------------
    # Move tensors to CPU once
    # --------------------------------------------------------

    xyxy = (
        boxes.xyxy
        .cpu()
        .numpy()
    )

    conf = (
        boxes.conf
        .cpu()
        .numpy()
    )

    cls = (
        boxes.cls
        .cpu()
        .numpy()
        .astype(np.int16)
    )


    # --------------------------------------------------------
    # Track IDs
    # --------------------------------------------------------

    if boxes.id is None:
        continue

    track_ids = (
        boxes.id
        .cpu()
        .numpy()
        .astype(np.int32)
    )


    # --------------------------------------------------------
    # Filter target classes
    # --------------------------------------------------------

    mask = np.isin(
        cls,
        list(TARGET_CLASS_IDS)
    )

    if not np.any(mask):
        continue


    xyxy = xyxy[mask]
    conf = conf[mask]
    cls = cls[mask]
    track_ids = track_ids[mask]


    # --------------------------------------------------------
    # Vectorized bbox extraction
    # --------------------------------------------------------

    x1 = xyxy[:, 0]
    y1 = xyxy[:, 1]
    x2 = xyxy[:, 2]
    y2 = xyxy[:, 3]


    # --------------------------------------------------------
    # Bottom-center
    #
    # THIS is the trajectory point used by Phase 3.
    # --------------------------------------------------------

    bottom_center_x = (
        (x1 + x2) / 2.0
    )

    bottom_center_y = y2


    timestamp = (
        (frame_id - 1)
        / FPS
    )


    # --------------------------------------------------------
    # Build frame dataframe
    # --------------------------------------------------------

    frame_df = pd.DataFrame({

        "frame_id": frame_id,

        "timestamp_sec": timestamp,

        "track_id": track_ids,

        "class_id": cls,

        "class_name": [
            CLASS_NAMES[int(c)]
            for c in cls
        ],

        "confidence": conf,

        "x1": x1,

        "y1": y1,

        "x2": x2,

        "y2": y2,

        "bottom_center_x": bottom_center_x,

        "bottom_center_y": bottom_center_y,

    })


    track_records.append(
        frame_df
    )


# ------------------------------------------------------------
# Concatenate once
# ------------------------------------------------------------

tracks_raw = pd.concat(
    track_records,
    ignore_index=True
)


# ------------------------------------------------------------
# Sort
# ------------------------------------------------------------

tracks_raw.sort_values(
    [
        "track_id",
        "frame_id"
    ],
    inplace=True
)


tracks_raw.reset_index(
    drop=True,
    inplace=True
)


# ------------------------------------------------------------
# Save
# ------------------------------------------------------------

tracks_raw.to_csv(
    PHASE1_TRACKS_PATH,
    index=False
)


elapsed = (
    time.time()
    -
    start_time
)


print("=" * 70)
print("PHASE 1 COMPLETE")
print("=" * 70)

print(
    f"Frames processed : {TOTAL_FRAMES:,}"
)

print(
    f"Tracking records : {len(tracks_raw):,}"
)

print(
    f"Unique tracks    : "
    f"{tracks_raw['track_id'].nunique():,}"
)

print(
    f"Elapsed          : "
    f"{elapsed:.2f} sec"
)

print(
    f"Processing FPS   : "
    f"{TOTAL_FRAMES / elapsed:.2f}"
)

print(
    "Output           :",
    PHASE1_TRACKS_PATH
)

# ============================================================
# LOAD PHASE 1 DATA
# ============================================================

tracks_raw = pd.read_csv(
    PHASE1_TRACKS_PATH
)

tracks_raw["track_id"] = (
    tracks_raw["track_id"]
    .astype(np.int32)
)

tracks_raw["frame_id"] = (
    tracks_raw["frame_id"]
    .astype(np.int32)
)

tracks_raw["class_name"] = (
    tracks_raw["class_name"]
    .astype("category")
)

print(
    tracks_raw.shape
)

display(
    tracks_raw.head()
)
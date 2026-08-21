from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


# ============================================================
# DETECTION CONFIG
# ============================================================

@dataclass(frozen=True)
class DetectionConfig:
    """
    YOLO26m + BoT-SORT configuration.

    The robust branch uses:
        YOLO26m TensorRT FP16
        BoT-SORT
        vid_stride=2
    """

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model_name: str = (
        "models/yolo26m_512_fp16.engine"
    )

    tracker: str = (
        "botsort.yaml"
    )

    # --------------------------------------------------------
    # Inference
    # --------------------------------------------------------

    imgsz: int = 512

    conf_threshold: float = 0.20

    iou_threshold: float = 0.70

    # --------------------------------------------------------
    # Process every 2nd source frame.
    #
    # IMPORTANT:
    # detector_tracker.py expects this parameter.
    #
    # Source frame IDs are preserved, so:
    #
    # 1, 3, 5, 7, ...
    #
    # rather than:
    #
    # 1, 2, 3, 4, ...
    # --------------------------------------------------------

    vid_stride: int = 2

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    device: str = "auto"


# ============================================================
# COUNTING CONFIG
# ============================================================

@dataclass(frozen=True)
class CountingConfig:
    """
    Robust Phase 3 counting configuration.

    Counting line
        ↓
    Crossing geometry
        ↓
    Crossing identity
        ↓
    Physical vehicle counting
    """

    # ========================================================
    # COUNTING LINE
    # ========================================================

    # Current project line:
    #
    # (1216, 144)
    #      ↓
    # (64, 684)
    #
    # for a 1280x720 video.

    line_x1_ratio: float = 0.95

    line_y1_ratio: float = 0.20

    line_x2_ratio: float = 0.05

    line_y2_ratio: float = 0.95

    # --------------------------------------------------------
    # Deadband around counting line.
    # --------------------------------------------------------

    line_deadband_px: float = 8.0

    # ========================================================
    # TRAJECTORY
    # ========================================================

    # Because vid_stride=2:
    #
    # two observations are approximately:
    #
    # 2 / FPS seconds apart.
    #
    # 1.5 sec is intentionally generous for temporary
    # detector/tracker gaps.

    max_trajectory_gap_sec: float = 1.50

    # ========================================================
    # MOTORCYCLE FRAGMENTATION
    # ========================================================

    # Legacy compatibility.
    #
    # These are still passed into TrafficCounter.
    #
    # The robust identity engine should be responsible for
    # physical identity matching.

    moto_dedup_time_sec: float = 1.20

    moto_dedup_distance_px: float = 90.0

    # ========================================================
    # CROSSING IDENTITY
    # ========================================================

    # --------------------------------------------------------
    # Distance at which a track becomes relevant to crossing.
    #
    # This is NOT the counting corridor.
    #
    # It is used for identity/reconnection logic.
    # --------------------------------------------------------

    pre_crossing_distance_px: float = 120.0

    # --------------------------------------------------------
    # Maximum time gap between fragments that may belong to
    # the same physical vehicle.
    # --------------------------------------------------------

    max_identity_reconnect_gap_sec: float = 0.75

    # --------------------------------------------------------
    # Maximum spatial distance between the end of an old
    # fragment and the beginning of a new fragment.
    # --------------------------------------------------------

    max_identity_reconnect_distance_px: float = 75.0

    # --------------------------------------------------------
    # Identity matching score threshold.
    # --------------------------------------------------------

    identity_match_threshold: float = 0.82

    # --------------------------------------------------------
    # Required margin between best and second-best identity.
    #
    # This is VERY important for your:
    #
    # "2 motorcycles enter simultaneously"
    #
    # scenario.
    #
    # If two identities have similar scores, the new fragment
    # should NOT automatically be merged.
    # --------------------------------------------------------

    identity_match_margin: float = 0.08

    # --------------------------------------------------------
    # Velocity gate.
    #
    # Maximum expected position discrepancy per source frame.
    # --------------------------------------------------------

    velocity_gate_px_per_frame: float = 30.0

    # --------------------------------------------------------
    # Minimum observations before considering a fragment a
    # reliable pre-crossing identity.
    # --------------------------------------------------------

    min_pre_crossing_observations: int = 3

    # ========================================================
    # ROBUST CROSSING GEOMETRY
    # ========================================================

    # Additional corridor around the counting line.

    crossing_corridor_px: float = 45.0

    # Minimum trajectory displacement required to infer
    # direction from motion.

    min_direction_displacement_px: float = 8.0

    # Number of observations around crossing used for
    # directional validation.

    direction_window: int = 3

    # ========================================================
    # FINAL DUPLICATE SUPPRESSION
    # ========================================================

    # IMPORTANT:
    #
    # Keep these conservative.
    #
    # Two motorcycles crossing at approximately the same time
    # must NOT accidentally collapse into one.
    #
    # Identity matching should happen before final counting.

    duplicate_time_sec: float = 0.50

    duplicate_distance_px: float = 35.0


# ============================================================
# APPLICATION CONFIG
# ============================================================

@dataclass(frozen=True)
class AppConfig:

    # --------------------------------------------------------
    # Output
    # --------------------------------------------------------

    output_dir: str = "outputs"

    # --------------------------------------------------------
    # Nested configs
    # --------------------------------------------------------

    detection: DetectionConfig = field(
        default_factory=DetectionConfig
    )

    counting: CountingConfig = field(
        default_factory=CountingConfig
    )

    # ========================================================
    # TARGET CLASSES
    # ========================================================

    target_classes: tuple[str, ...] = (
        "person",
        "motorcycle",
        "car",
        "bus",
        "truck",
    )

    # ========================================================
    # VEHICLES
    # ========================================================

    vehicle_classes: tuple[str, ...] = (
        "motorcycle",
        "car",
        "truck",
        "bus",
    )


# ============================================================
# BUILD CONFIG
# ============================================================

def build_config() -> AppConfig:

    config = AppConfig()

    Path(
        config.output_dir
    ).mkdir(
        parents=True,
        exist_ok=True,
    )

    return config

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


# ============================================================
# DETECTION CONFIG
# ============================================================

@dataclass(frozen=True)
class DetectionConfig:
    """
    BASELINE detection configuration.

    IMPORTANT:
        Robust branch changes Phase 3 counting logic only.

        Phase 1 should remain aligned with baseline:
            YOLO26m .pt
            BoT-SORT
            existing inference settings
    """

    # --------------------------------------------------------
    # YOLO26m pretrained weights
    #
    # Do NOT use TensorRT on robust branch.
    # --------------------------------------------------------

    model_name: str = "yolo26m.pt"

    # --------------------------------------------------------
    # Tracker
    # --------------------------------------------------------

    tracker: str = "botsort.yaml"

    # --------------------------------------------------------
    # Baseline inference size
    # --------------------------------------------------------

    imgsz: int = 640

    # --------------------------------------------------------
    # Detection confidence
    # --------------------------------------------------------

    conf_threshold: float = 0.20

    # --------------------------------------------------------
    # IoU
    # --------------------------------------------------------

    iou_threshold: float = 0.70

    # --------------------------------------------------------
    # Baseline processes the source video normally.
    #
    # We are NOT changing temporal sampling in this branch.
    # --------------------------------------------------------

    vid_stride: int = 1

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
    Robust Phase 3 configuration.

    Phase 1 / Phase 2 remain baseline.

    Only Phase 3 crossing/counting is changed.
    """

    # ========================================================
    # COUNTING LINE
    # ========================================================

    # Current project line:
    # (1216, 144) -> (64, 684)
    #
    # Designed for 1280x720.

    line_x1_ratio: float = 0.95
    line_y1_ratio: float = 0.20

    line_x2_ratio: float = 0.05
    line_y2_ratio: float = 0.95

    # ========================================================
    # LINE DEADBAND
    # ========================================================

    line_deadband_px: float = 8.0

    # ========================================================
    # TRACK TRAJECTORY
    # ========================================================

    # Keep this reasonably permissive because temporary
    # tracking gaps can occur during occlusion.
    max_trajectory_gap_sec: float = 1.50

    # ========================================================
    # LEGACY MOTORCYCLE FRAGMENTATION
    # ========================================================

    # These remain for compatibility with TrafficCounter.
    moto_dedup_time_sec: float = 0.25
    moto_dedup_distance_px: float = 30.0

    # ========================================================
    # ROBUST CROSSING GEOMETRY
    # ========================================================

    # Wider corridor helps fast vehicles that may have sparse
    # observations around the line.
    crossing_corridor_px: float = 45.0

    # Minimum movement before direction is trusted.
    min_direction_displacement_px: float = 8.0

    # Observations around crossing used for direction estimate.
    direction_window: int = 3

    # ========================================================
    # FINAL DUPLICATE SUPPRESSION
    # ========================================================

    # VERY conservative.
    #
    # Goal:
    #   two real motorcycles close together = 2
    #
    # Fragmentation should primarily be resolved by track
    # continuity / identity, not by an enormous spatial window.

    duplicate_time_sec: float = 0.30
    duplicate_distance_px: float = 25.0


# ============================================================
# APPLICATION CONFIG
# ============================================================

@dataclass(frozen=True)
class AppConfig:

    output_dir: str = "outputs"

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

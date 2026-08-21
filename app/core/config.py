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
    # Temporal sampling
    #
    # Every 2nd source frame is processed.
    # Source frame IDs remain tied to the original video.
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
    Robust Phase 3 configuration.

    Detection
        ↓
    Tracking
        ↓
    Track-level class
        ↓
    Robust crossing
        ↓
    Final vehicle count
    """

    # ========================================================
    # COUNTING LINE
    # ========================================================

    # 1280x720:
    #
    # (1216, 144)
    #      ↓
    # (64, 684)

    line_x1_ratio: float = 0.95
    line_y1_ratio: float = 0.20

    line_x2_ratio: float = 0.05
    line_y2_ratio: float = 0.95

    # ========================================================
    # LINE DEADZONE
    # ========================================================

    line_deadband_px: float = 8.0

    # ========================================================
    # TRAJECTORY
    # ========================================================

    max_trajectory_gap_sec: float = 1.50

    # ========================================================
    # LEGACY MOTORCYCLE FRAGMENTATION
    # ========================================================

    moto_dedup_time_sec: float = 1.20

    moto_dedup_distance_px: float = 90.0

    # ========================================================
    # ROBUST CROSSING
    # ========================================================

    crossing_corridor_px: float = 45.0

    min_direction_displacement_px: float = 8.0

    direction_window: int = 3

    # ========================================================
    # FINAL DEDUP
    # ========================================================
    #
    # Conservative values.
    #
    # IMPORTANT:
    # Two real motorcycles crossing together must not collapse
    # into one.
    # ========================================================

    duplicate_time_sec: float = 0.50

    duplicate_distance_px: float = 35.0


# ============================================================
# APP CONFIG
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
    # VEHICLE CLASSES
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

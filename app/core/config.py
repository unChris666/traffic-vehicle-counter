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
    # Reserved for the tracker implementation that supports
    # vid_stride.
    #
    # IMPORTANT:
    # Current robust branch detector_tracker.py does NOT
    # accept this argument yet.
    #
    # Therefore engine.py intentionally does not pass it.
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

    Pipeline:

        counting line
            ↓
        robust crossing geometry
            ↓
        direction validation
            ↓
        duplicate protection
            ↓
        final vehicle count
    """

    # ========================================================
    # COUNTING LINE
    # ========================================================

    # Current project line for 1280x720:
    #
    # (1216, 144)
    #      ↓
    # (64, 684)

    line_x1_ratio: float = 0.95
    line_y1_ratio: float = 0.20

    line_x2_ratio: float = 0.05
    line_y2_ratio: float = 0.95

    # ========================================================
    # LINE GEOMETRY
    # ========================================================

    # Small deadband around the counting line.
    line_deadband_px: float = 8.0

    # ========================================================
    # TRACK TRAJECTORY
    # ========================================================

    # Maximum allowed gap between consecutive observations
    # used by the robust crossing detector.
    max_trajectory_gap_sec: float = 1.50

    # ========================================================
    # MOTORCYCLE FRAGMENTATION
    # ========================================================

    # Existing configuration retained for compatibility.
    moto_dedup_time_sec: float = 1.20
    moto_dedup_distance_px: float = 90.0

    # ========================================================
    # ROBUST CROSSING GEOMETRY
    # ========================================================

    # Corridor around counting line.
    #
    # This helps when a fast vehicle has only a few useful
    # observations around the line.
    crossing_corridor_px: float = 45.0

    # Minimum displacement required to infer direction from
    # motion.
    min_direction_displacement_px: float = 8.0

    # Number of observations before/after crossing used for
    # direction estimation.
    direction_window: int = 3

    # ========================================================
    # DUPLICATE SUPPRESSION
    # ========================================================

    # IMPORTANT:
    #
    # Keep these conservative.
    #
    # Two real motorcycles crossing together must NOT be
    # collapsed into one just because they are close in time.
    #
    # The robust branch therefore starts with a very small
    # final dedup window.

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
    # Nested configuration
    # --------------------------------------------------------

    detection: DetectionConfig = field(
        default_factory=DetectionConfig
    )

    counting: CountingConfig = field(
        default_factory=CountingConfig
    )

    # ========================================================
    # DETECTION TARGETS
    # ========================================================

    target_classes: tuple[str, ...] = (
        "person",
        "motorcycle",
        "car",
        "bus",
        "truck",
    )

    # ========================================================
    # FINAL VEHICLE CLASSES
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

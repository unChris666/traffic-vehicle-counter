from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


# ============================================================
# DETECTION CONFIG
# ============================================================

@dataclass(frozen=True)
class DetectionConfig:
    """
    Detection and tracking configuration.

    YOLO26m + BoT-SORT remain unchanged for Phase 1.
    """

    model_name: str = "yolo26m.pt"
    tracker: str = "botsort.yaml"

    imgsz: int = 640
    conf_threshold: float = 0.20
    iou_threshold: float = 0.70

    # Process every frame.
    vid_stride: int = 1

    device: str = "auto"


# ============================================================
# COUNTING CONFIG
# ============================================================

@dataclass(frozen=True)
class CountingConfig:
    """
    Robust counting configuration.

    This configuration intentionally contains both:

    1. Existing identity-management parameters required by
       TrafficCounter / CrossingIdentityEngine.

    2. New Phase 1 / Phase 2 trajectory and corridor parameters.
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

    line_deadband_px: float = 8.0


    # ========================================================
    # TRACK TRAJECTORY
    # ========================================================

    max_trajectory_gap_sec: float = 1.50


    # ========================================================
    # LEGACY MOTORCYCLE DEDUP
    # ========================================================

    # Preserved for API compatibility.
    # Generic time/distance dedup remains disabled in the
    # current identity architecture.

    moto_dedup_time_sec: float = 0.25
    moto_dedup_distance_px: float = 30.0


    # ========================================================
    # IDENTITY MANAGEMENT / FRAGMENT RECONNECT
    # ========================================================

    pre_crossing_distance_px: float = 100.0

    max_identity_reconnect_gap_sec: float = 1.0

    max_identity_reconnect_distance_px: float = 100.0

    identity_match_threshold: float = 0.82

    identity_match_margin: float = 0.08

    velocity_gate_px_per_frame: float = 30.0

    min_pre_crossing_observations: int = 2


    # ========================================================
    # ROBUST CROSSING GEOMETRY
    # ========================================================

    crossing_corridor_px: float = 45.0

    min_direction_displacement_px: float = 8.0

    direction_window: int = 3


    # ========================================================
    # PHASE 1 — TRAJECTORY ENGINE
    # ========================================================

    # Causal EMA smoothing.
    #
    # Important:
    # trajectory analysis may use smoothed coordinates,
    # while zone classification should use raw coordinates
    # to avoid smoothing latency.

    trajectory_smoothing_alpha: float = 0.35

    # Number of recent observations used for velocity analysis.

    trajectory_velocity_window: int = 5

    # Diagnostic upper bound for image-space speed.

    max_velocity_px_per_frame: float = 80.0


    # ========================================================
    # PHASE 2 — CROSSING CORRIDOR
    # ========================================================

    # Minimum evidence required in each zone.

    min_pre_zone_observations: int = 2

    min_corridor_observations: int = 1

    min_post_zone_observations: int = 1

    # Phase 2 PASS requires evidence after the line.

    require_post_zone: bool = True


    # ========================================================
    # FINAL DUPLICATE SUPPRESSION
    # ========================================================

    # Conservative values retained for compatibility.
    # Identity management remains the primary mechanism.

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
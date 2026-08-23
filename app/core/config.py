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
    # IDENTITY / FRAGMENT RECONNECT COMPATIBILITY
    # ========================================================

    # These fields are required by the existing TrafficCounter and
    # CrossingIdentityEngine interfaces. They remain available even
    # while Phase 1/2 development is being performed.

    pre_crossing_distance_px: float = 100.0
    max_identity_reconnect_gap_sec: float = 1.5
    max_identity_reconnect_distance_px: float = 140.0
    identity_match_threshold: float = 0.82
    identity_match_margin: float = 0.08
    velocity_gate_px_per_frame: float = 30.0
    min_pre_crossing_observations: int = 2

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
    # PHASE 1 — TRAJECTORY ENGINE
    # ========================================================

    # Causal EMA used for trajectory analysis. Zone membership still uses
    # the raw observed bbox position to avoid smoothing latency.
    trajectory_smoothing_alpha: float = 0.35

    # Velocity samples used internally by downstream phases when needed.
    trajectory_velocity_window: int = 5

    # Maximum image-space speed considered reasonable for trajectory
    # quality diagnostics. This is not a detector confidence threshold.
    max_velocity_px_per_frame: float = 80.0

    # ========================================================
    # PHASE 2 — CROSSING CORRIDOR
    # ========================================================

    min_pre_zone_observations: int = 2
    min_corridor_observations: int = 1
    min_post_zone_observations: int = 1

    # Require evidence that the object actually reaches the post-zone
    # before Phase 2 can be considered PASS.
    require_post_zone: bool = False

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

    # Additional Phase 1/2 controls
    corridor_exit_px: float = 60.0
    approach_distance_px: float = 120.0
    short_track_observation_threshold: int = 8
    class_evidence_window_frames: int = 8
    class_recency_decay: float = 0.18
    min_counting_class_confidence: float = 0.45
    zone_enter_confirm_observations: int = 2
    zone_exit_confirm_observations: int = 2
    gap_bridge_enabled: bool = True
    gap_bridge_max_frames: int = 12
    max_velocity_bridge_px_per_frame: float = 160.0
    fast_speed_multiplier: float = 0.80
    min_normal_velocity_px_per_frame: float = 1.0
    min_normal_displacement_px: float = 8.0

    # ========================================================
    # IDENTITY-GAP CROSSING
    # ========================================================
    identity_gap_crossing_enabled: bool = True
    candidate_generation_per_raw_track: bool = True
    canonical_candidate_id_by_event: bool = True
    identity_gap_max_frames: int = 8
    identity_gap_max_endpoint_distance_px: float = 140.0
    identity_gap_min_side_displacement_px: float = 12.0

    # ========================================================
    # CANONICAL CROSSING-CANDIDATE DUPLICATE AUDIT
    # ========================================================

    # Duplicate identity detection is intentionally trajectory-aware.
    # These parameters are NOT generic time-distance dedup thresholds.
    candidate_duplicate_max_frame_gap: int = 8
    candidate_duplicate_max_endpoint_distance_px: float = 55.0
    candidate_duplicate_max_crossing_distance_px: float = 55.0
    candidate_duplicate_min_direction_cosine: float = 0.75
    candidate_duplicate_require_non_overlapping_tracks: bool = True



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

# v9 raw-fragment fallback crossing
IDENTITY_GAP_FALLBACK_ENABLED = True
IDENTITY_GAP_FALLBACK_MIN_SCORE = 0.72

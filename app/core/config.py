from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class DetectionConfig:
    # Production model: TensorRT FP16 engine exported from YOLO26m at imgsz=512.
    model_name: str = "models/yolo26m.pt"
    tracker: str = "botsort.yaml"

    # Smaller inference size for speed.
    imgsz: int = 512

    # Keep enough low-confidence detections for tracking.
    # Do not push this too high: the tracker can recover weak observations.
    conf_threshold: float = 0.20
    iou_threshold: float = 0.70

    # Process every 2nd source frame.
    vid_stride: int = 2

    # Use the first CUDA GPU when available.
    device: str = "auto"


@dataclass(frozen=True)
class CountingConfig:
    line_x1_ratio: float = 0.95
    line_y1_ratio: float = 0.20
    line_x2_ratio: float = 0.05
    line_y2_ratio: float = 0.95
    line_deadband_px: float = 8.0
    max_trajectory_gap_sec: float = 1.50
    moto_dedup_time_sec: float = 1.20
    moto_dedup_distance_px: float = 90.0

    pre_crossing_distance_px: float = 100.0
    max_identity_reconnect_gap_sec: float = 1.0
    max_identity_reconnect_distance_px: float = 100.0
    identity_match_threshold: float = 0.82
    identity_match_margin: float = 0.08
    velocity_gate_px_per_frame: float = 30.0
    min_pre_crossing_observations: int = 2
    identity_same_side_near_line_block: bool = True
    identity_prediction_gate_max_px: float = 220.0
    identity_min_velocity_cosine: float = 0.35
    identity_min_normal_velocity_px_per_frame: float = 1.0

    crossing_corridor_px: float = 45.0
    min_direction_displacement_px: float = 8.0
    direction_window: int = 3
    trajectory_smoothing_alpha: float = 0.35
    trajectory_velocity_window: int = 5
    max_velocity_px_per_frame: float = 80.0
    min_pre_zone_observations: int = 2
    min_corridor_observations: int = 1
    min_post_zone_observations: int = 1
    require_post_zone: bool = True
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

    identity_gap_crossing_enabled: bool = True
    identity_gap_max_frames: int = 8
    identity_gap_max_endpoint_distance_px: float = 140.0
    identity_gap_min_side_displacement_px: float = 12.0
    candidate_duplicate_max_frame_gap: int = 8
    candidate_duplicate_max_endpoint_distance_px: float = 55.0
    candidate_duplicate_max_crossing_distance_px: float = 55.0
    candidate_duplicate_min_direction_cosine: float = 0.75
    candidate_duplicate_require_non_overlapping_tracks: bool = True
    multi_crossing_max_candidates_per_track: int = 32
    multi_crossing_min_separation_frames: int = 2

    concurrent_duplicate_enabled: bool = True
    concurrent_duplicate_min_overlap_frames: int = 3
    concurrent_duplicate_min_overlap_ratio: float = 0.50
    concurrent_duplicate_min_mean_iou: float = 0.65
    concurrent_duplicate_min_max_iou: float = 0.80
    concurrent_duplicate_max_center_distance_px: float = 25.0
    concurrent_duplicate_min_motion_cosine: float = 0.80
    concurrent_duplicate_max_motion_speed_ratio: float = 2.50
    concurrent_duplicate_allow_class_mismatch: bool = True

    identity_class_min_confidence: float = 0.45
    identity_class_stable_track_ratio: float = 0.70
    identity_class_ambiguous_penalty: float = 0.55
    identity_class_alias_bonus: float = 1.20

    state_min_confirmed_observations: int = 2
    state_min_direction_confidence: float = 0.45
    state_count_threshold: float = 0.62
    state_review_threshold: float = 0.45
    state_weight_geometry: float = 0.32
    state_weight_direction: float = 0.16
    state_weight_continuity: float = 0.14
    state_weight_class: float = 0.12
    state_weight_pre: float = 0.08
    state_weight_corridor: float = 0.05
    state_weight_post: float = 0.07
    state_weight_fast_sparse: float = 0.06
    state_fast_crossing_floor: float = 0.55
    state_short_crossing_floor: float = 0.52
    state_person_count_threshold: float = 0.55
    state_person_review_threshold: float = 0.42

    duplicate_time_sec: float = 0.30
    duplicate_distance_px: float = 25.0


@dataclass(frozen=True)
class AppConfig:
    output_dir: str = "outputs"

    detection: DetectionConfig = field(
        default_factory=DetectionConfig
    )
    counting: CountingConfig = field(
        default_factory=CountingConfig
    )

    target_classes: tuple[str, ...] = (
        "person",
        "motorcycle",
        "car",
        "bus",
        "truck",
    )

    vehicle_classes: tuple[str, ...] = (
        "motorcycle",
        "car",
        "truck",
        "bus",
    )


def build_config() -> AppConfig:
    config = AppConfig()

    Path(config.output_dir).mkdir(
        parents=True,
        exist_ok=True,
    )

    return config

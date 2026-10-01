from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class DetectionConfig:
    model_name: str = "yolo26m.pt"
    tracker: str = "configs/botsort_phone.yaml"
    imgsz: int = 1280
    conf_threshold: float = 0.20
    iou_threshold: float = 0.50

    # HARD REQUIREMENT FOR PHASE3NEW:
    # one inference iteration == one source video frame.
    vid_stride: int = 1

    device: str = "auto"


@dataclass(frozen=True)
class CountingConfig:
    # ---------------------------------------------------------
    # CROSSING LINE
    # ---------------------------------------------------------
    line_x1_ratio: float = 0.95
    line_y1_ratio: float = 0.20
    line_x2_ratio: float = 0.05
    line_y2_ratio: float = 0.95

    # ---------------------------------------------------------
    # SIDE A = vehicle approaching camera
    # SIDE B = vehicle moving away from camera
    #
    # Rectangles are expressed as normalized x/y ratios.
    # Tune these four values for each camera.
    # ---------------------------------------------------------
    side_a_x1_ratio: float = 0.00
    side_a_y1_ratio: float = 0.55
    side_a_x2_ratio: float = 1.00
    side_a_y2_ratio: float = 1.00

    side_b_x1_ratio: float = 0.00
    side_b_y1_ratio: float = 0.00
    side_b_x2_ratio: float = 1.00
    side_b_y2_ratio: float = 0.40

    # ---------------------------------------------------------
    # SPEED
    # ---------------------------------------------------------
    speed_window_frames: int = 5

    # Keep state alive while BoT-SORT temporarily loses a target.
    # This must be >= botsort track_buffer for safe state retention.
    state_max_missing_frames: int = 90

    # ---------------------------------------------------------
    # BUSINESS RULE
    # ---------------------------------------------------------
    vehicle_classes: tuple[str, ...] = (
        "motorcycle",
        "car",
        "truck",
        "bus",
    )


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

    if config.detection.vid_stride != 1:
        raise ValueError(
            "Phase3New requires detection.vid_stride == 1"
        )

    Path(config.output_dir).mkdir(
        parents=True,
        exist_ok=True,
    )

    return config

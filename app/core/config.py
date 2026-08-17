from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class DetectionConfig:
    # Production model: TensorRT FP16 engine exported from YOLO26m at imgsz=512.
    model_name: str = "models/yolo26m_512_fp16.engine"
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
    # Current project counting line:
    # (1216, 144) -> (64, 684) on 1280x720 video.
    line_x1_ratio: float = 0.95
    line_y1_ratio: float = 0.20
    line_x2_ratio: float = 0.05
    line_y2_ratio: float = 0.95

    line_deadband_px: float = 8.0

    # With vid_stride=2, two processed observations are ~2/fps apart.
    # 1.5 s leaves generous room for temporary detector gaps.
    max_trajectory_gap_sec: float = 1.5

    # Existing motorcycle fragmentation protection.
    moto_dedup_time_sec: float = 1.20
    moto_dedup_distance_px: float = 90.0


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

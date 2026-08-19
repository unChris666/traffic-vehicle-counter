# from dataclasses import dataclass, field
# from pathlib import Path


# @dataclass(frozen=True)
# class DetectionConfig:
#     model_name: str = "yolo26m.pt"
#     tracker: str = "botsort.yaml"
#     imgsz: int = 960
#     conf_threshold: float = 0.25
#     iou_threshold: float = 0.50
#     device: str = "auto"


# @dataclass(frozen=True)
# class CountingConfig:
#     line_x1_ratio: float = 0.95
#     line_y1_ratio: float = 0.20
#     line_x2_ratio: float = 0.05
#     line_y2_ratio: float = 0.95
#     line_deadband_px: float = 8.0
#     # Defined in notebook baseline.
#     # Currently diagnostic/config metadata only;
#     # not used to exclude tracks from counting.
#     min_track_observations: int = 5
#     max_trajectory_gap_sec: float = 1.5
#     moto_dedup_time_sec: float = 1.50
#     moto_dedup_distance_px: float = 80.0


# @dataclass(frozen=True)
# class AppConfig:
#     detection: DetectionConfig = field(default_factory=DetectionConfig)
#     counting: CountingConfig = field(default_factory=CountingConfig)
#     target_classes: tuple[str, ...] = (
#         "person", "motorcycle", "car", "truck", "bus"
#     )
#     vehicle_classes: tuple[str, ...] = (
#         "motorcycle", "car", "truck", "bus"
#     )
#     output_dir: Path = Path("outputs")


# def build_config(output_dir: str | Path = "outputs") -> AppConfig:
#     return AppConfig(output_dir=Path(output_dir))



from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class DetectionConfig:
    model_name: str = "yolo26m.pt"

    tracker: str = "configs/botsort_phone.yaml"

    # Native-width inference
    imgsz: int = 1280

    # Recall-first
    conf_threshold: float = 0.15

    iou_threshold: float = 0.50

    device: str = "auto"

@dataclass(frozen=True)
class CountingConfig:
    line_x1_ratio: float = 0.95
    line_y1_ratio: float = 0.20
    line_x2_ratio: float = 0.05
    line_y2_ratio: float = 0.95

    # Larger tolerance for trajectory jitter
    line_deadband_px: float = 20.0

    # Track quality
    min_track_observations: int = 8
    max_trajectory_gap_sec: float = 1.0

    # Majority voting
    majority_vote_min_observations: int = 5
    majority_vote_ratio: float = 0.60

    # Confidence
    min_track_confidence: float = 0.30

    # Dedup fallback
    moto_dedup_time_sec: float = 1.0
    moto_dedup_distance_px: float = 50.0


@dataclass(frozen=True)
class AppConfig:
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
        "truck",
        "bus",
    )

    vehicle_classes: tuple[str, ...] = (
        "motorcycle",
        "car",
        "truck",
        "bus",
    )

    output_dir: Path = Path("outputs")


def build_config(
    output_dir: str | Path = "outputs"
) -> AppConfig:

    return AppConfig(
        output_dir=Path(output_dir)
    )

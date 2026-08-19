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

    # ------------------------------------------------------
    # Counting line
    # ------------------------------------------------------

    line_x1_ratio: float = 0.95
    line_y1_ratio: float = 0.20

    line_x2_ratio: float = 0.05
    line_y2_ratio: float = 0.95

    line_deadband_px: float = 20.0

    # ------------------------------------------------------
    # Existing track diagnostics
    # ------------------------------------------------------

    min_track_observations: int = 5

    max_trajectory_gap_sec: float = 1.0

    # ------------------------------------------------------
    # Phase 1: Crossing Identity
    # ------------------------------------------------------

    pre_crossing_distance_px: float = 100.0

    max_identity_reconnect_gap_sec: float = 1.0

    max_identity_reconnect_distance_px: float = 100.0

    identity_match_threshold: float = 0.82

    identity_match_margin: float = 0.08

    velocity_gate_px_per_frame: float = 30.0

    min_pre_crossing_observations: int = 3

    # ------------------------------------------------------
    # Deprecated for Phase 1.
    # Keep for backward compatibility.
    # ------------------------------------------------------

    moto_dedup_time_sec: float = 1.50

    moto_dedup_distance_px: float = 80.0


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

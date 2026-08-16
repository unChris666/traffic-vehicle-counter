from dataclasses import dataclass
from pathlib import Path

import cv2


@dataclass(frozen=True)
class VideoMetadata:
    path: Path
    filename: str
    width: int
    height: int
    fps: float
    frame_count: int
    duration_sec: float | None


class VideoValidationError(RuntimeError):
    pass


def read_video_metadata(video_path: str | Path) -> VideoMetadata:
    path = Path(video_path)
    if not path.exists():
        raise VideoValidationError(f"Video does not exist: {path}")

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise VideoValidationError(f"Cannot open video: {path}")

    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    finally:
        cap.release()

    duration = frame_count / fps if fps > 0 else None

    if width <= 0 or height <= 0:
        raise VideoValidationError("Invalid video resolution")
    if fps <= 0:
        raise VideoValidationError("Invalid video FPS")
    if frame_count <= 0:
        raise VideoValidationError("Video contains no frames")

    return VideoMetadata(
        path=path,
        filename=path.name,
        width=width,
        height=height,
        fps=fps,
        frame_count=frame_count,
        duration_sec=duration,
    )

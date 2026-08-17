from __future__ import annotations

import shutil
import subprocess
import time
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
import pandas as pd


class VideoRenderer:
    """
    Render Phase 3 tracking/counting result into annotated CFR MP4.

    No YOLO.
    No tracking.
    No counting.
    """

    def __init__(
        self,
        *,
        line_x1: float,
        line_y1: float,
        line_x2: float,
        line_y2: float,
    ) -> None:
        self.line_x1 = int(round(line_x1))
        self.line_y1 = int(round(line_y1))
        self.line_x2 = int(round(line_x2))
        self.line_y2 = int(round(line_y2))

    @staticmethod
    def _require_ffmpeg() -> str:
        ffmpeg = shutil.which("ffmpeg")

        if ffmpeg is None:
            raise RuntimeError(
                "FFmpeg was not found in PATH."
            )

        return ffmpeg

    def render(
        self,
        *,
        input_path: str | Path,
        output_path: str | Path,
        fps: float,
        width: int,
        height: int,
        total_frames: int,
        tracks_phase2: pd.DataFrame,
        final_crossings: pd.DataFrame,
        progress_callback: Callable[[float, str], None]
        | None = None,
    ) -> Path:

        ffmpeg = self._require_ffmpeg()

        input_path = Path(input_path)
        output_path = Path(output_path)

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        render_df = tracks_phase2[
            [
                "frame_id",
                "track_id",
                "x1",
                "y1",
                "x2",
                "y2",
                "track_class",
                "track_class_ratio",
                "class_ambiguous",
            ]
        ].copy()

        render_df["frame_id"] = (
            render_df["frame_id"].astype(np.int32)
        )

        frame_groups = {
            frame_id: group
            for frame_id, group
            in render_df.groupby(
                "frame_id",
                sort=False,
            )
        }

        crossing_frame_groups = {
            frame_id: group
            for frame_id, group
            in final_crossings.groupby(
                "crossing_frame",
                sort=False,
            )
        }

        crossing_frame_counts = (
            final_crossings["crossing_frame"]
            .value_counts()
            .sort_index()
        )

        cumulative_count = (
            crossing_frame_counts
            .reindex(
                range(1, total_frames + 1),
                fill_value=0,
            )
            .cumsum()
            .astype(int)
            .to_numpy()
        )

        ffmpeg_cmd = [
            ffmpeg,
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",

            "-f",
            "rawvideo",

            "-pixel_format",
            "bgr24",

            "-video_size",
            f"{width}x{height}",

            "-framerate",
            f"{fps:.12f}",

            "-i",
            "pipe:0",

            "-an",

            "-c:v",
            "libx264",

            "-preset",
            "medium",

            "-crf",
            "18",

            "-pix_fmt",
            "yuv420p",

            "-vsync",
            "cfr",

            "-movflags",
            "+faststart",

            str(output_path),
        ]

        process = subprocess.Popen(
            ffmpeg_cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            bufsize=0,
        )

        cap = cv2.VideoCapture(
            str(input_path)
        )

        if not cap.isOpened():
            process.kill()
            raise RuntimeError(
                f"Cannot open input video: {input_path}"
            )

        frame_id = 0
        render_error: str | None = None

        start = time.perf_counter()

        try:
            while True:
                success, frame = cap.read()

                if not success:
                    break

                frame_id += 1

                if frame.shape != (
                    height,
                    width,
                    3,
                ):
                    raise RuntimeError(
                        f"Unexpected frame shape at "
                        f"frame {frame_id}: {frame.shape}"
                    )

                # Counting line.
                cv2.line(
                    frame,
                    (
                        self.line_x1,
                        self.line_y1,
                    ),
                    (
                        self.line_x2,
                        self.line_y2,
                    ),
                    (0, 255, 255),
                    4,
                    cv2.LINE_AA,
                )

                group = frame_groups.get(
                    frame_id
                )

                if group is not None:
                    for row in group.itertuples(
                        index=False
                    ):
                        x1 = int(row.x1)
                        y1 = int(row.y1)
                        x2 = int(row.x2)
                        y2 = int(row.y2)

                        label = (
                            f"ID {int(row.track_id)} | "
                            f"{str(row.track_class).upper()} | "
                            f"{float(row.track_class_ratio):.0%}"
                        )

                        if bool(row.class_ambiguous):
                            label += " | AMBIG"

                        cv2.rectangle(
                            frame,
                            (x1, y1),
                            (x2, y2),
                            (0, 255, 0),
                            2,
                        )

                        cv2.putText(
                            frame,
                            label,
                            (
                                x1,
                                max(y1 - 8, 18),
                            ),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.45,
                            (255, 255, 255),
                            1,
                            cv2.LINE_AA,
                        )

                # Cumulative final vehicle count.
                current_count = int(
                    cumulative_count[
                        frame_id - 1
                    ]
                )

                cv2.rectangle(
                    frame,
                    (15, 10),
                    (370, 80),
                    (0, 0, 0),
                    -1,
                )

                cv2.putText(
                    frame,
                    f"VEHICLE COUNT: {current_count}",
                    (25, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

                cv2.putText(
                    frame,
                    "MOTORCYCLE + RIDER = 1",
                    (25, 65),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.38,
                    (200, 200, 200),
                    1,
                    cv2.LINE_AA,
                )

                # Crossing events.
                events = crossing_frame_groups.get(
                    frame_id
                )

                if events is not None:
                    max_events = max(
                        1,
                        int(
                            (height - 40) / 25
                        ),
                    )

                    for offset, event in enumerate(
                        events.itertuples(
                            index=False
                        )
                    ):
                        if offset >= max_events:
                            break

                        event_text = (
                            "COUNTED: "
                            f"{str(event.track_class).upper()} "
                            f"| ID {int(event.track_id)} "
                            f"| {event.direction}"
                        )

                        cv2.putText(
                            frame,
                            event_text,
                            (
                                20,
                                height
                                - 25
                                - offset * 24,
                            ),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.48,
                            (0, 255, 255),
                            2,
                            cv2.LINE_AA,
                        )

                cv2.putText(
                    frame,
                    f"Frame: {frame_id:,}",
                    (
                        max(
                            10,
                            width - 180,
                        ),
                        30,
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

                if process.stdin is None:
                    raise RuntimeError(
                        "FFmpeg stdin is unavailable."
                    )

                process.stdin.write(
                    frame.tobytes()
                )

                if (
                    progress_callback is not None
                    and (
                        frame_id == 1
                        or frame_id % 30 == 0
                        or frame_id == total_frames
                    )
                ):
                    progress_callback(
                        frame_id / total_frames,
                        f"Rendering video "
                        f"{frame_id:,}/{total_frames:,}",
                    )

        except (
            BrokenPipeError,
            OSError,
            RuntimeError,
        ) as exc:
            render_error = str(exc)

        finally:
            cap.release()

            if process.stdin is not None:
                try:
                    process.stdin.close()
                except Exception:
                    pass

        stderr = (
            process.stderr.read().decode(
                "utf-8",
                errors="replace",
            )
            if process.stderr is not None
            else ""
        )

        return_code = process.wait()

        if render_error is not None:
            raise RuntimeError(
                f"Video rendering failed: "
                f"{render_error}"
            )

        if return_code != 0:
            raise RuntimeError(
                "FFmpeg rendering failed:\n"
                f"{stderr}"
            )

        if frame_id != total_frames:
            raise RuntimeError(
                "Rendered frame count mismatch: "
                f"{frame_id}/{total_frames}"
            )

        if (
            not output_path.exists()
            or output_path.stat().st_size <= 0
        ):
            raise RuntimeError(
                "Rendered output video is missing or empty."
            )

        if progress_callback is not None:
            elapsed = (
                time.perf_counter() - start
            )

            progress_callback(
                1.0,
                f"Rendering complete "
                f"({elapsed:.1f}s)",
            )

        return output_path
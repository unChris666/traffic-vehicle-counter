from __future__ import annotations

import subprocess
from collections import defaultdict, deque
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


class VideoRenderer:

    def __init__(
        self,
        *,
        line_x1: float,
        line_y1: float,
        line_x2: float,
        line_y2: float,
        trajectory_length: int = 45,
    ) -> None:

        self.line_x1 = int(round(line_x1))
        self.line_y1 = int(round(line_y1))
        self.line_x2 = int(round(line_x2))
        self.line_y2 = int(round(line_y2))

        self.trajectory_length = trajectory_length

    @staticmethod
    def _draw_label(
        frame,
        text: str,
        x: int,
        y: int,
    ) -> None:

        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.45
        thickness = 1

        (tw, th), baseline = cv2.getTextSize(
            text,
            font,
            scale,
            thickness,
        )

        y1 = max(0, y - th - baseline - 4)

        cv2.rectangle(
            frame,
            (x, y1),
            (x + tw + 6, y),
            (0, 0, 0),
            -1,
        )

        cv2.putText(
            frame,
            text,
            (x + 3, y - 3),
            font,
            scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )

    @staticmethod
    def _color_from_track_id(track_id: int):
        rng = np.random.default_rng(track_id)

        return (
            int(rng.integers(60, 255)),
            int(rng.integers(60, 255)),
            int(rng.integers(60, 255)),
        )

    @staticmethod
    def _encode_h264(
        input_path: Path,
        output_path: Path,
        fps: float,
    ) -> None:

        command = [
            "ffmpeg",
            "-y",

            "-i",
            str(input_path),

            # H.264
            "-c:v",
            "libx264",

            # Browser compatible pixel format
            "-pix_fmt",
            "yuv420p",

            # Force constant frame rate
            "-r",
            f"{fps:.6f}",

            # Good visual quality
            "-crf",
            "20",

            # Reasonable Kaggle processing speed
            "-preset",
            "fast",

            # Important for browser streaming
            "-movflags",
            "+faststart",

            # Tracking output has no audio
            "-an",

            str(output_path),
        ]

        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        if result.returncode != 0:
            raise RuntimeError(
                "FFmpeg H.264 encoding failed:\n"
                + result.stderr[-4000:]
            )

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
        progress_callback=None,
    ) -> Path:

        input_path = Path(input_path)
        output_path = Path(output_path)

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        required_columns = {
            "frame_id",
            "track_id",
            "x1",
            "y1",
            "x2",
            "y2",
            "bottom_center_x",
            "bottom_center_y",
            "track_class",
            "confidence",
        }

        missing = required_columns - set(
            tracks_phase2.columns
        )

        if missing:
            raise ValueError(
                f"tracks_phase2 missing required columns: "
                f"{sorted(missing)}"
            )

        # ------------------------------------------
        # Frame lookup
        # ------------------------------------------

        frame_groups = {
            frame_id: group
            for frame_id, group
            in tracks_phase2.groupby("frame_id")
        }

        track_history = defaultdict(
            lambda: deque(
                maxlen=self.trajectory_length
            )
        )

        # ------------------------------------------
        # Input video
        # ------------------------------------------

        cap = cv2.VideoCapture(
            str(input_path)
        )

        if not cap.isOpened():
            raise RuntimeError(
                f"Unable to open video: {input_path}"
            )

        # ------------------------------------------
        # Temporary rendering file
        # ------------------------------------------

        temp_path = (
            output_path.parent
            / f"{output_path.stem}_opencv_temp.mp4"
        )

        # OpenCV -> MPEG-4 temporary
        fourcc = cv2.VideoWriter_fourcc(
            *"mp4v"
        )

        writer = cv2.VideoWriter(
            str(temp_path),
            fourcc,
            fps,
            (width, height),
        )

        if not writer.isOpened():
            cap.release()

            raise RuntimeError(
                f"Unable to create temporary video: "
                f"{temp_path}"
            )

        frame_id = 0

        try:

            while True:

                success, frame = cap.read()

                if not success:
                    break

                frame_id += 1

                # ----------------------------------
                # Counting line
                # ----------------------------------

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
                    3,
                    cv2.LINE_AA,
                )

                # ----------------------------------
                # Current frame tracks
                # ----------------------------------

                current_tracks = frame_groups.get(
                    frame_id
                )

                if current_tracks is not None:

                    for _, row in current_tracks.iterrows():

                        track_id = int(
                            row["track_id"]
                        )

                        x1 = int(
                            round(row["x1"])
                        )
                        y1 = int(
                            round(row["y1"])
                        )
                        x2 = int(
                            round(row["x2"])
                        )
                        y2 = int(
                            round(row["y2"])
                        )

                        cx = int(
                            round(
                                row[
                                    "bottom_center_x"
                                ]
                            )
                        )

                        cy = int(
                            round(
                                row[
                                    "bottom_center_y"
                                ]
                            )
                        )

                        cls_name = str(
                            row["track_class"]
                        )

                        confidence = float(
                            row["confidence"]
                        )

                        color = (
                            self._color_from_track_id(
                                track_id
                            )
                        )

                        # ----------------------------------
                        # Track history
                        # ----------------------------------

                        track_history[
                            track_id
                        ].append(
                            (cx, cy)
                        )

                        points = np.array(
                            track_history[
                                track_id
                            ],
                            dtype=np.int32,
                        )

                        # ----------------------------------
                        # Trajectory
                        # ----------------------------------

                        if len(points) >= 2:

                            cv2.polylines(
                                frame,
                                [points],
                                False,
                                color,
                                2,
                                cv2.LINE_AA,
                            )

                        # ----------------------------------
                        # Bounding box
                        # ----------------------------------

                        cv2.rectangle(
                            frame,
                            (x1, y1),
                            (x2, y2),
                            color,
                            2,
                        )

                        # ----------------------------------
                        # Bottom-center point
                        # ----------------------------------

                        cv2.circle(
                            frame,
                            (cx, cy),
                            6,
                            color,
                            -1,
                            cv2.LINE_AA,
                        )

                        cv2.circle(
                            frame,
                            (cx, cy),
                            8,
                            (255, 255, 255),
                            1,
                            cv2.LINE_AA,
                        )

                        # ----------------------------------
                        # Label
                        # ----------------------------------

                        label = (
                            f"ID {track_id} | "
                            f"{cls_name.upper()} | "
                            f"{confidence:.0%}"
                        )

                        self._draw_label(
                            frame,
                            label,
                            x1,
                            y1,
                        )

                # ----------------------------------
                # Frame number
                # ----------------------------------

                cv2.putText(
                    frame,
                    f"Frame: {frame_id:,}",
                    (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

                writer.write(frame)

                if progress_callback is not None:

                    if (
                        frame_id % 30 == 0
                        or frame_id == total_frames
                    ):

                        progress_callback(
                            min(
                                1.0,
                                frame_id
                                / max(
                                    total_frames,
                                    1,
                                ),
                            ),
                            (
                                f"Rendering "
                                f"{frame_id:,}/"
                                f"{total_frames:,}"
                            ),
                        )

        finally:

            cap.release()
            writer.release()

        # ------------------------------------------
        # H.264 conversion
        # ------------------------------------------

        self._encode_h264(
            input_path=temp_path,
            output_path=output_path,
            fps=fps,
        )

        # ------------------------------------------
        # Remove temporary file
        # ------------------------------------------

        if temp_path.exists():
            temp_path.unlink()

        return output_path

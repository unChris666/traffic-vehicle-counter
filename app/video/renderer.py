from __future__ import annotations

import subprocess
from collections import defaultdict, deque
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


class VideoRenderer:
    """
    Render annotated traffic-counting video.

    Visualization only:
    - Does NOT change detection
    - Does NOT change tracking
    - Does NOT change counting
    - Uses Phase 2 `track_class`
    - Uses final_crossings for cumulative direction counts
    """

    # ==========================================================
    # CLASS COLORS
    #
    # OpenCV uses BGR format.
    # ==========================================================

    CLASS_COLORS = {
        "person": (255, 0, 255),      # Magenta
        "motorcycle": (0, 165, 255),  # Orange
        "car": (255, 0, 0),           # Blue
        "truck": (0, 0, 255),         # Red
        "bus": (0, 255, 255),         # Yellow
    }

    DEFAULT_COLOR = (255, 255, 255)

    # ==========================================================
    # DISPLAY NAMES
    # ==========================================================

    CLASS_DISPLAY_NAMES = {
        "person": "Pejalan Kaki",
        "motorcycle": "Motor",
        "car": "Mobil",
        "truck": "Truck",
        "bus": "Bus",
    }

    # ==========================================================
    # DIRECTION DISPLAY
    # ==========================================================

    DIRECTION_DISPLAY_NAMES = {
        "side_+1_to_-1": "Kanan-Kiri",
        "side_-1_to_+1": "Kiri-Kanan",
    }

    # ==========================================================
    # VEHICLE ORDER
    # ==========================================================

    DISPLAY_CLASSES = [
        "person",
        "motorcycle",
        "car",
        "truck",
        "bus",
    ]

    DISPLAY_DIRECTIONS = [
        "side_+1_to_-1",
        "side_-1_to_+1",
    ]

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

    # ==========================================================
    # COLOR
    # ==========================================================

    @classmethod
    def _color_from_class(cls, class_name: str):
        """
        Return a consistent color based on track_class.

        IMPORTANT:
        Color is based on track_class, NOT track_id.
        """

        class_name = str(class_name).lower().strip()

        return cls.CLASS_COLORS.get(
            class_name,
            cls.DEFAULT_COLOR,
        )

    # ==========================================================
    # LABEL
    # ==========================================================

    @staticmethod
    def _draw_label(
        frame,
        text: str,
        x: int,
        y: int,
        color,
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

        y1 = max(
            0,
            y - th - baseline - 4,
        )

        # Black background
        cv2.rectangle(
            frame,
            (x, y1),
            (x + tw + 8, y),
            (0, 0, 0),
            -1,
        )

        # Small class-color indicator
        cv2.rectangle(
            frame,
            (x, y1),
            (x + 4, y),
            color,
            -1,
        )

        cv2.putText(
            frame,
            text,
            (x + 7, y - 3),
            font,
            scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )

    # ==========================================================
    # TRANSPARENT OVERLAY
    # ==========================================================

    @staticmethod
    def _draw_transparent_panel(
        frame,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        alpha: float = 0.72,
    ):
        """
        Draw a semi-transparent black panel.

        Returns overlay so text can be drawn on top.
        """

        overlay = frame.copy()

        cv2.rectangle(
            overlay,
            (x1, y1),
            (x2, y2),
            (0, 0, 0),
            -1,
        )

        cv2.addWeighted(
            overlay,
            alpha,
            frame,
            1 - alpha,
            0,
            frame,
        )

    # ==========================================================
    # COUNTING STATE
    # ==========================================================

    @classmethod
    def _build_empty_counts(cls):
        """
        Create:

        {
            "person": {
                "side_+1_to_-1": 0,
                "side_-1_to_+1": 0
            },
            ...
        }
        """

        return {
            class_name: {
                direction: 0
                for direction in cls.DISPLAY_DIRECTIONS
            }
            for class_name in cls.DISPLAY_CLASSES
        }

    # ==========================================================
    # NORMALIZE DIRECTION
    # ==========================================================

    @staticmethod
    def _normalize_direction(value) -> str | None:

        if value is None:
            return None

        value = str(value).strip()

        if value in (
            "side_+1_to_-1",
            "side_-1_to_+1",
        ):
            return value

        return None

    # ==========================================================
    # UPDATE COUNTING PANEL
    # ==========================================================

    @classmethod
    def _update_counts_from_crossings(
        cls,
        counts,
        final_crossings: pd.DataFrame,
        frame_id: int,
        counted_track_ids: set,
    ) -> None:
        """
        Add crossings that have happened up to current frame.

        Each track is counted only once.
        """

        if final_crossings is None:
            return

        if final_crossings.empty:
            return

        required_columns = {
            "track_id",
            "track_class",
            "direction",
        }

        if not required_columns.issubset(
            final_crossings.columns
        ):
            return

        # Try to identify the crossing frame.
        frame_column = None

        for candidate in [
            "crossing_frame",
            "frame_id",
            "cross_frame",
        ]:
            if candidate in final_crossings.columns:
                frame_column = candidate
                break

        if frame_column is None:
            return

        eligible = final_crossings[
            final_crossings[frame_column] <= frame_id
        ]

        for _, row in eligible.iterrows():

            track_id = int(row["track_id"])

            if track_id in counted_track_ids:
                continue

            class_name = (
                str(row["track_class"])
                .lower()
                .strip()
            )

            direction = (
                cls._normalize_direction(
                    row["direction"]
                )
            )

            if (
                class_name not in counts
                or direction not in cls.DISPLAY_DIRECTIONS
            ):
                continue

            counts[class_name][direction] += 1

            counted_track_ids.add(track_id)

    # ==========================================================
    # DRAW COUNT PANEL
    # ==========================================================

    @classmethod
    def _draw_count_panel(
        cls,
        frame,
        counts,
        width: int,
        height: int,
    ) -> None:

        # ------------------------------------------------------
        # Panel geometry
        # ------------------------------------------------------

        margin = max(
            12,
            int(width * 0.015),
        )

        panel_width = min(
            int(width * 0.46),
            560,
        )

        row_height = max(
            24,
            int(height * 0.045),
        )

        header_height = row_height * 2

        num_rows = len(cls.DISPLAY_CLASSES)

        panel_height = (
            header_height
            + num_rows * row_height
            + margin
        )

        x1 = margin
        y1 = margin

        x2 = min(
            width - margin,
            x1 + panel_width,
        )

        y2 = min(
            height - margin,
            y1 + panel_height,
        )

        # ------------------------------------------------------
        # Transparent black background
        # ------------------------------------------------------

        cls._draw_transparent_panel(
            frame,
            x1,
            y1,
            x2,
            y2,
            alpha=0.72,
        )

        # ------------------------------------------------------
        # Header
        # ------------------------------------------------------

        font = cv2.FONT_HERSHEY_SIMPLEX

        title_scale = max(
            0.48,
            min(
                0.68,
                width / 1800,
            ),
        )

        cv2.putText(
            frame,
            "TRAFFIC COUNT",
            (
                x1 + 14,
                y1 + 25,
            ),
            font,
            title_scale,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        # ------------------------------------------------------
        # Column positions
        # ------------------------------------------------------

        class_x = x1 + 14

        right_left_x = x1 + int(
            panel_width * 0.48
        )

        left_right_x = x1 + int(
            panel_width * 0.75
        )

        header_y = y1 + 50

        small_scale = max(
            0.32,
            min(
                0.48,
                width / 2200,
            ),
        )

        cv2.putText(
            frame,
            "KENDARAAN",
            (
                class_x,
                header_y,
            ),
            font,
            small_scale,
            (210, 210, 210),
            1,
            cv2.LINE_AA,
        )

        cv2.putText(
            frame,
            "KANAN-KIRI",
            (
                right_left_x,
                header_y,
            ),
            font,
            small_scale,
            (210, 210, 210),
            1,
            cv2.LINE_AA,
        )

        cv2.putText(
            frame,
            "KIRI-KANAN",
            (
                left_right_x,
                header_y,
            ),
            font,
            small_scale,
            (210, 210, 210),
            1,
            cv2.LINE_AA,
        )

        # ------------------------------------------------------
        # Rows
        # ------------------------------------------------------

        for index, class_name in enumerate(
            cls.DISPLAY_CLASSES
        ):

            y = (
                y1
                + header_height
                + index * row_height
            )

            color = cls._color_from_class(
                class_name
            )

            # Color indicator
            cv2.rectangle(
                frame,
                (
                    class_x,
                    y + 5,
                ),
                (
                    class_x + 10,
                    y + row_height - 5,
                ),
                color,
                -1,
            )

            display_name = (
                cls.CLASS_DISPLAY_NAMES.get(
                    class_name,
                    class_name.title(),
                )
            )

            cv2.putText(
                frame,
                display_name,
                (
                    class_x + 18,
                    y + row_height - 8,
                ),
                font,
                small_scale,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

            # Right → Left
            right_left_count = counts[
                class_name
            ][
                "side_+1_to_-1"
            ]

            cv2.putText(
                frame,
                str(right_left_count),
                (
                    right_left_x,
                    y + row_height - 8,
                ),
                font,
                small_scale,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            # Left → Right
            left_right_count = counts[
                class_name
            ][
                "side_-1_to_+1"
            ]

            cv2.putText(
                frame,
                str(left_right_count),
                (
                    left_right_x,
                    y + row_height - 8,
                ),
                font,
                small_scale,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

        # ------------------------------------------------------
        # Panel border
        # ------------------------------------------------------

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (180, 180, 180),
            1,
            cv2.LINE_AA,
        )

    # ==========================================================
    # ENCODE H264
    # ==========================================================

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

            "-c:v",
            "libx264",

            "-pix_fmt",
            "yuv420p",

            "-r",
            f"{fps:.6f}",

            "-crf",
            "20",

            "-preset",
            "fast",

            "-movflags",
            "+faststart",

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

    # ==========================================================
    # MAIN RENDER
    # ==========================================================

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

        # ======================================================
        # VALIDATE TRACK DATA
        # ======================================================

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

        missing = (
            required_columns
            - set(tracks_phase2.columns)
        )

        if missing:
            raise ValueError(
                "tracks_phase2 missing required "
                f"columns: {sorted(missing)}"
            )

        # ======================================================
        # FRAME LOOKUP
        # ======================================================

        frame_groups = {
            frame_id: group
            for frame_id, group
            in tracks_phase2.groupby(
                "frame_id"
            )
        }

        # ======================================================
        # TRACK HISTORY
        # ======================================================

        track_history = defaultdict(
            lambda: deque(
                maxlen=self.trajectory_length
            )
        )

        # ======================================================
        # COUNTING STATE
        # ======================================================

        counts = self._build_empty_counts()

        counted_track_ids = set()

        # ======================================================
        # OPEN INPUT VIDEO
        # ======================================================

        cap = cv2.VideoCapture(
            str(input_path)
        )

        if not cap.isOpened():
            raise RuntimeError(
                f"Unable to open video: {input_path}"
            )

        # ======================================================
        # TEMPORARY OUTPUT
        # ======================================================

        temp_path = (
            output_path.parent
            / f"{output_path.stem}_opencv_temp.mp4"
        )

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
                "Unable to create temporary "
                f"video: {temp_path}"
            )

        frame_id = 0

        try:

            while True:

                success, frame = cap.read()

                if not success:
                    break

                frame_id += 1

                # ==================================================
                # COUNTING LINE
                # ==================================================

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

                # ==================================================
                # UPDATE COUNTING PANEL
                # ==================================================

                self._update_counts_from_crossings(
                    counts=counts,
                    final_crossings=final_crossings,
                    frame_id=frame_id,
                    counted_track_ids=counted_track_ids,
                )

                # ==================================================
                # CURRENT FRAME TRACKS
                # ==================================================

                current_tracks = (
                    frame_groups.get(
                        frame_id
                    )
                )

                if current_tracks is not None:

                    for _, row in (
                        current_tracks.iterrows()
                    ):

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

                        # IMPORTANT:
                        # Use Phase 2 track_class.
                        cls_name = (
                            str(
                                row["track_class"]
                            )
                            .lower()
                            .strip()
                        )

                        confidence = float(
                            row["confidence"]
                        )

                        # ==================================================
                        # CLASS COLOR
                        # ==================================================

                        color = (
                            self._color_from_class(
                                cls_name
                            )
                        )

                        # ==================================================
                        # TRACK HISTORY
                        # ==================================================

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

                        # ==================================================
                        # TRAJECTORY
                        # ==================================================

                        if len(points) >= 2:

                            cv2.polylines(
                                frame,
                                [points],
                                False,
                                color,
                                2,
                                cv2.LINE_AA,
                            )

                        # ==================================================
                        # BOUNDING BOX
                        # ==================================================

                        cv2.rectangle(
                            frame,
                            (x1, y1),
                            (x2, y2),
                            color,
                            2,
                        )

                        # ==================================================
                        # BOTTOM CENTER
                        # ==================================================

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

                        # ==================================================
                        # LABEL
                        # ==================================================

                        display_name = (
                            self.CLASS_DISPLAY_NAMES.get(
                                cls_name,
                                cls_name.upper(),
                            )
                        )

                        label = (
                            f"ID {track_id} | "
                            f"{display_name} | "
                            f"{confidence:.0%}"
                        )

                        self._draw_label(
                            frame,
                            label,
                            x1,
                            y1,
                            color,
                        )

                # ==================================================
                # COUNT PANEL
                # ==================================================

                self._draw_count_panel(
                    frame=frame,
                    counts=counts,
                    width=width,
                    height=height,
                )

                # ==================================================
                # FRAME NUMBER
                # ==================================================

                cv2.putText(
                    frame,
                    f"Frame: {frame_id:,}",
                    (20, height - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

                # ==================================================
                # WRITE FRAME
                # ==================================================

                writer.write(frame)

                # ==================================================
                # PROGRESS
                # ==================================================

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

        # ======================================================
        # H264 CONVERSION
        # ======================================================

        self._encode_h264(
            input_path=temp_path,
            output_path=output_path,
            fps=fps,
        )

        # ======================================================
        # REMOVE TEMP FILE
        # ======================================================

        if temp_path.exists():
            temp_path.unlink()

        return output_path

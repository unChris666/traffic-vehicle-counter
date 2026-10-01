from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable

import cv2
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Polygon:
    points: tuple[tuple[float, float], ...]

    def contains(self, x: float, y: float) -> bool:
        pts = np.asarray(self.points, dtype=np.float32)
        return cv2.pointPolygonTest(pts, (float(x), float(y)), False) >= 0


@dataclass(frozen=True)
class Line:
    x1: float
    y1: float
    x2: float
    y2: float

    def signed_distance(self, x: float, y: float) -> float:
        # Positive/negative is the mathematical side of the directed line.
        return (
            (self.x2 - self.x1) * (y - self.y1)
            - (self.y2 - self.y1) * (x - self.x1)
        )

    def crossed(self, prev_x: float, prev_y: float, x: float, y: float) -> bool:
        a = self.signed_distance(prev_x, prev_y)
        b = self.signed_distance(x, y)
        return (a < 0 <= b) or (a > 0 >= b)


@dataclass
class TrackState:
    track_id: int
    vehicle_category: str
    first_frame: int
    last_frame: int
    last_x: float
    last_y: float
    last_bbox: tuple[float, float, float, float]
    last_confidence: float

    speed_px_s: float = 0.0
    direction: str = "UNKNOWN"

    current_side: str = "OUTSIDE"
    previous_side: str = "OUTSIDE"

    entered_side_a: bool = False
    entered_side_b: bool = False
    crossed_line: bool = False
    crossing_direction: str = ""
    counted: bool = False
    exited_box: bool = False

    # The category is deliberately locked for the life of a tracker ID.
    class_evidence: dict[str, float] = field(default_factory=dict)

    previous_frame: int | None = None
    previous_x: float | None = None
    previous_y: float | None = None


class SideBoxVehicleCounter:
    """
    Frame-by-frame state engine for Phase3New.

    Identity source of truth:
        BoT-SORT track_id.

    For every observation the same track_id carries:
        bbox
        center
        vehicle_category
        speed
        direction
        current side
        crossing state

    A vehicle is counted exactly once, on the first valid transition:
        SIDE_A -> CROSSING -> SIDE_B
    or:
        SIDE_B -> CROSSING -> SIDE_A

    A track is never assigned a new identity by this class.
    """

    OUTPUT_COLUMNS = [
        "frame_id",
        "timestamp_sec",
        "track_id",
        "vehicle_category",
        "confidence",
        "x1", "y1", "x2", "y2",
        "center_x", "center_y",
        "speed_px_s",
        "direction",
        "previous_side",
        "current_side",
        "crossing_detected",
        "crossing_direction",
        "entered_side_a",
        "entered_side_b",
        "counted",
        "exited_box",
    ]

    def __init__(
        self,
        *,
        side_a: Polygon,
        side_b: Polygon,
        crossing_line: Line,
        fps: float,
        vehicle_classes: Iterable[str],
        speed_window: int = 5,
        max_missing_frames: int = 90,
    ) -> None:
        if fps <= 0:
            raise ValueError("fps must be > 0")

        self.side_a = side_a
        self.side_b = side_b
        self.crossing_line = crossing_line
        self.fps = float(fps)
        self.vehicle_classes = set(vehicle_classes)
        self.speed_window = max(1, int(speed_window))
        self.max_missing_frames = max(1, int(max_missing_frames))

        self.states: dict[int, TrackState] = {}
        self.positions: dict[int, list[tuple[int, float, float]]] = {}
        self.events: list[dict] = []
        self._counted_ids: set[int] = set()

    @staticmethod
    def _bbox_center(row: pd.Series) -> tuple[float, float]:
        # Geometric center is used for Side A/B and crossing geometry.
        # bottom-center remains available in the raw tracking output.
        return (
            (float(row["x1"]) + float(row["x2"])) / 2.0,
            (float(row["y1"]) + float(row["y2"])) / 2.0,
        )

    def _side(self, x: float, y: float) -> str:
        if self.side_a.contains(x, y):
            return "A"
        if self.side_b.contains(x, y):
            return "B"
        return "OUTSIDE"

    def _stable_class(self, state: TrackState, observed: str, confidence: float) -> str:
        if observed in self.vehicle_classes:
            state.class_evidence[observed] = (
                state.class_evidence.get(observed, 0.0) + max(0.01, confidence)
            )

        # Once a vehicle has a category, keep it fixed. For the first observation
        # use the observed vehicle class; subsequent observations can only replace
        # it before the first class is established.
        if state.vehicle_category in self.vehicle_classes:
            return state.vehicle_category

        if state.class_evidence:
            return max(state.class_evidence, key=state.class_evidence.get)

        return observed

    def _speed_direction(
        self,
        track_id: int,
        frame_id: int,
        x: float,
        y: float,
    ) -> tuple[float, str]:
        history = self.positions.setdefault(track_id, [])
        history.append((frame_id, x, y))
        if len(history) > self.speed_window:
            del history[:-self.speed_window]

        if len(history) < 2:
            return 0.0, "UNKNOWN"

        f0, x0, y0 = history[0]
        f1, x1, y1 = history[-1]
        dt = (f1 - f0) / self.fps
        if dt <= 0:
            return 0.0, "UNKNOWN"

        dx = x1 - x0
        dy = y1 - y0
        distance = float(np.hypot(dx, dy))
        speed = distance / dt

        # Direction is expressed in image-space terms as well as movement
        # relative to the configured crossing line.
        if distance < 2.0:
            direction = "STATIONARY"
        else:
            direction = np.degrees(np.arctan2(dy, dx))
            direction = f"{direction:.1f}deg"

        return speed, direction

    def process_frame(self, frame_id: int, timestamp_sec: float, frame_tracks: pd.DataFrame) -> pd.DataFrame:
        """
        Process ONE video frame.

        frame_tracks must contain all tracker observations for this frame.
        """
        output_rows: list[dict] = []
        seen_ids: set[int] = set()

        if frame_tracks is None or frame_tracks.empty:
            self._expire_missing(frame_id)
            return pd.DataFrame(columns=self.OUTPUT_COLUMNS)

        for _, row in frame_tracks.sort_values("track_id").iterrows():
            track_id = int(row["track_id"])
            category = str(row["class_name"])
            confidence = float(row.get("confidence", 0.0))

            # Only vehicles participate in the crossing counter.
            if category not in self.vehicle_classes:
                continue

            seen_ids.add(track_id)

            x1, y1, x2, y2 = (
                float(row["x1"]),
                float(row["y1"]),
                float(row["x2"]),
                float(row["y2"]),
            )
            cx, cy = self._bbox_center(row)
            current_side = self._side(cx, cy)

            state = self.states.get(track_id)
            if state is None:
                state = TrackState(
                    track_id=track_id,
                    vehicle_category=category,
                    first_frame=frame_id,
                    last_frame=frame_id,
                    last_x=cx,
                    last_y=cy,
                    last_bbox=(x1, y1, x2, y2),
                    last_confidence=confidence,
                    previous_frame=None,
                    previous_x=None,
                    previous_y=None,
                )
                self.states[track_id] = state

            # Identity/category are tied to the tracker ID.
            stable_category = self._stable_class(
                state,
                category,
                confidence,
            )
            state.vehicle_category = stable_category

            speed, direction = self._speed_direction(
                track_id,
                frame_id,
                cx,
                cy,
            )

            previous_side = state.current_side
            previous_x = state.last_x
            previous_y = state.last_y
            had_side_a = state.entered_side_a
            had_side_b = state.entered_side_b

            state.previous_side = previous_side
            state.current_side = current_side
            state.last_frame = frame_id
            state.last_x = cx
            state.last_y = cy
            state.last_bbox = (x1, y1, x2, y2)
            state.last_confidence = confidence
            state.speed_px_s = speed
            state.direction = direction

            crossing_detected = False
            crossing_direction = state.crossing_direction

            if (
                previous_x is not None
                and previous_y is not None
                and self.crossing_line.crossed(
                    previous_x,
                    previous_y,
                    cx,
                    cy,
                )
            ):
                # The first confirmed line crossing after entering Side A
                # defines A -> B. Likewise Side B defines B -> A.
                #
                # The vehicle does not need to be inside the destination box
                # on the exact crossing frame; the two boxes and line are
                # independent spatial states.
                if (
                    had_side_a
                    and not had_side_b
                    and not state.crossed_line
                ):
                    state.crossed_line = True
                    crossing_direction = "A_TO_B"
                    state.crossing_direction = crossing_direction
                    crossing_detected = True

                elif (
                    had_side_b
                    and not had_side_a
                    and not state.crossed_line
                ):
                    state.crossed_line = True
                    crossing_direction = "B_TO_A"
                    state.crossing_direction = crossing_direction
                    crossing_detected = True

            # Side entry is committed after crossing evaluation so the
            # previous-side information remains authoritative.
            if current_side == "A":
                state.entered_side_a = True
            elif current_side == "B":
                state.entered_side_b = True

            # More robust than requiring the two boxes to touch:
            # once the line has been crossed, arrival in the opposite box
            # completes the event.
            if state.crossed_line and not state.counted:
                valid_ab = (
                    state.crossing_direction == "A_TO_B"
                    and state.entered_side_a
                    and current_side == "B"
                )
                valid_ba = (
                    state.crossing_direction == "B_TO_A"
                    and state.entered_side_b
                    and current_side == "A"
                )

                if valid_ab or valid_ba:
                    state.counted = True
                    self._counted_ids.add(track_id)

                    self.events.append(
                        {
                            "frame_id": frame_id,
                            "timestamp_sec": timestamp_sec,
                            "track_id": track_id,
                            "vehicle_category": state.vehicle_category,
                            "direction": state.crossing_direction,
                            "speed_px_s": state.speed_px_s,
                        }
                    )

            # "Exited" means the track was last observed outside both boxes
            # after it had completed a crossing.
            state.exited_box = (
                state.counted
                and current_side == "OUTSIDE"
            )

            output_rows.append(
                {
                    "frame_id": frame_id,
                    "timestamp_sec": timestamp_sec,
                    "track_id": track_id,
                    "vehicle_category": state.vehicle_category,
                    "confidence": confidence,
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "center_x": cx,
                    "center_y": cy,
                    "speed_px_s": state.speed_px_s,
                    "direction": state.direction,
                    "previous_side": previous_side,
                    "current_side": current_side,
                    "crossing_detected": crossing_detected,
                    "crossing_direction": state.crossing_direction,
                    "entered_side_a": state.entered_side_a,
                    "entered_side_b": state.entered_side_b,
                    "counted": state.counted,
                    "exited_box": state.exited_box,
                }
            )

        self._expire_missing(frame_id)
        return pd.DataFrame(output_rows, columns=self.OUTPUT_COLUMNS)

    def _expire_missing(self, frame_id: int) -> None:
        # Do NOT delete the state immediately. BoT-SORT may temporarily lose
        # an object during occlusion. Keep the state alive for track_buffer
        # frames so a returning observation retains its original ID/state.
        for state in self.states.values():
            if frame_id - state.last_frame > self.max_missing_frames:
                state.exited_box = True

    def finalize(self) -> pd.DataFrame:
        if not self.events:
            return pd.DataFrame(
                columns=[
                    "frame_id",
                    "timestamp_sec",
                    "track_id",
                    "vehicle_category",
                    "direction",
                    "speed_px_s",
                ]
            )
        return pd.DataFrame(self.events).drop_duplicates(
            subset=["track_id"],
            keep="first",
        )

    def summary(self) -> dict:
        events = self.finalize()
        counts = (
            events["vehicle_category"].value_counts().to_dict()
            if not events.empty
            else {}
        )
        direction_counts = (
            events.groupby(
                ["vehicle_category", "direction"]
            )
            .size()
            .reset_index(name="count")
            .to_dict("records")
            if not events.empty
            else []
        )

        return {
            "counts": {str(k): int(v) for k, v in counts.items()},
            "direction_counts": direction_counts,
            "total": int(len(events)),
            "tracked_vehicle_ids": int(
                len(
                    [
                        state
                        for state in self.states.values()
                        if state.vehicle_category in self.vehicle_classes
                    ]
                )
            ),
            "counted_vehicle_ids": int(len(self._counted_ids)),
        }

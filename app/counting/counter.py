from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CountingResult:
    counts: dict[str, int]
    total: int
    trajectory: pd.DataFrame
    crossing_candidates: pd.DataFrame
    crossing_events: pd.DataFrame
    crossing_vehicle: pd.DataFrame
    crossing_person: pd.DataFrame
    vehicle_events: pd.DataFrame
    final_crossings: pd.DataFrame
    audit: dict[str, int]


class TrafficCounter:
    """Phase 3 traffic counting logic."""

    def __init__(self, *, line_x1: float, line_y1: float, line_x2: float, line_y2: float,
                 line_deadband_px: float, max_trajectory_gap_sec: float,
                 moto_dedup_time_sec: float, moto_dedup_distance_px: float,
                 vehicle_classes: set[str], fps: float) -> None:
        if fps <= 0:
            raise ValueError(f"fps must be > 0, got {fps}")

        self.x1 = float(line_x1)
        self.y1 = float(line_y1)
        self.x2 = float(line_x2)
        self.y2 = float(line_y2)
        self.line_dx = self.x2 - self.x1
        self.line_dy = self.y2 - self.y1
        self.line_length = math.hypot(self.line_dx, self.line_dy)
        if self.line_length <= 0:
            raise ValueError("Counting line cannot have zero length")

        self.line_deadband_px = float(line_deadband_px)
        self.max_trajectory_gap_sec = float(max_trajectory_gap_sec)
        self.moto_dedup_time_sec = float(moto_dedup_time_sec)
        self.moto_dedup_distance_px = float(moto_dedup_distance_px)
        self.vehicle_classes = set(vehicle_classes)
        self.fps = float(fps)

    def _build_trajectory(self, tracks_phase2: pd.DataFrame) -> pd.DataFrame:
        required = {
            "track_id", "frame_id", "timestamp_sec", "bottom_center_x",
            "bottom_center_y", "track_class", "track_class_ratio", "class_ambiguous",
        }
        missing = required - set(tracks_phase2.columns)
        if missing:
            raise ValueError(f"tracks_phase2 missing required columns: {sorted(missing)}")

        trajectory = (
            tracks_phase2[list(required)]
            .sort_values(["track_id", "frame_id"])
            .copy()
        )
        # Preserve notebook column order.
        trajectory = trajectory[[
            "track_id", "frame_id", "timestamp_sec", "bottom_center_x",
            "bottom_center_y", "track_class", "track_class_ratio", "class_ambiguous",
        ]]
        trajectory["dx"] = trajectory.groupby("track_id")["bottom_center_x"].diff()
        trajectory["dy"] = trajectory.groupby("track_id")["bottom_center_y"].diff()
        trajectory["frame_delta"] = trajectory.groupby("track_id")["frame_id"].diff()
        trajectory["time_delta_sec"] = trajectory["frame_delta"] / self.fps
        return trajectory

    def _apply_line_geometry(self, trajectory: pd.DataFrame) -> pd.DataFrame:
        trajectory = trajectory.copy()
        trajectory["line_value"] = (
            self.line_dx * (trajectory["bottom_center_y"] - self.y1)
            - self.line_dy * (trajectory["bottom_center_x"] - self.x1)
        )
        trajectory["line_distance_px"] = (
            np.abs(trajectory["line_value"]) / self.line_length
        )
        trajectory["side"] = np.select(
            [
                trajectory["line_distance_px"] <= self.line_deadband_px,
                trajectory["line_value"] > 0,
            ],
            [0, 1],
            default=-1,
        )
        return trajectory

    def _detect_crossings(self, trajectory: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        stable = trajectory[trajectory["side"] != 0].copy()
        stable["previous_side"] = stable.groupby("track_id")["side"].shift(1)
        stable["previous_frame"] = stable.groupby("track_id")["frame_id"].shift(1)
        stable["frame_gap"] = stable["frame_id"] - stable["previous_frame"]
        stable["valid_temporal_transition"] = (
            stable["frame_gap"] <= self.max_trajectory_gap_sec * self.fps
        )
        stable["crossed"] = (
            stable["valid_temporal_transition"]
            & stable["previous_side"].notna()
            & (stable["side"] != stable["previous_side"])
        )

        crossing_candidates = stable[stable["crossed"]].copy()
        crossing_candidates["direction"] = np.where(
            crossing_candidates["previous_side"] < 0,
            "side_-1_to_+1",
            "side_+1_to_-1",
        )

        # Keep the first valid crossing per track, exactly as the notebook.
        crossing_events = (
            crossing_candidates
            .sort_values(["track_id", "frame_id"])
            .drop_duplicates("track_id", keep="first")
            [[
                "track_id", "frame_id", "timestamp_sec", "bottom_center_x",
                "bottom_center_y", "direction", "track_class", "track_class_ratio",
                "class_ambiguous", "line_distance_px", "previous_side", "frame_gap",
            ]]
            .rename(columns={
                "frame_id": "crossing_frame",
                "timestamp_sec": "crossing_time_sec",
                "bottom_center_x": "crossing_x",
                "bottom_center_y": "crossing_y",
            })
            .reset_index(drop=True)
        )
        return crossing_candidates, crossing_events

    def _split_vehicle_person(self, crossing_events: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        crossing_vehicle = crossing_events[
            crossing_events["track_class"].isin(self.vehicle_classes)
        ].copy()
        crossing_person = crossing_events[
            crossing_events["track_class"] == "person"
        ].copy()
        return crossing_vehicle, crossing_person

    def _deduplicate_motorcycles(self, crossing_vehicle: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        vehicle_events = (
            crossing_vehicle.sort_values("crossing_time_sec")
            .reset_index(drop=True)
            .copy()
        )
        vehicle_events["duplicate_of_track_id"] = pd.NA
        vehicle_events["dedup_reason"] = ""

        motorcycle_idx = np.flatnonzero(
            (vehicle_events["track_class"] == "motorcycle").to_numpy()
        )
        accepted_motorcycles: list[int] = []

        for idx in motorcycle_idx:
            current = vehicle_events.iloc[idx]
            duplicate_found = False
            duplicate_track_id = None

            for accepted_idx in accepted_motorcycles:
                previous = vehicle_events.iloc[accepted_idx]

                if current["direction"] != previous["direction"]:
                    continue

                dt = abs(
                    current["crossing_time_sec"] - previous["crossing_time_sec"]
                )
                if dt > self.moto_dedup_time_sec:
                    continue

                distance = math.hypot(
                    current["crossing_x"] - previous["crossing_x"],
                    current["crossing_y"] - previous["crossing_y"],
                )
                if distance > self.moto_dedup_distance_px:
                    continue

                duplicate_found = True
                duplicate_track_id = previous["track_id"]
                break

            if duplicate_found:
                vehicle_events.at[idx, "duplicate_of_track_id"] = duplicate_track_id
                vehicle_events.at[idx, "dedup_reason"] = "motorcycle_fragmentation"
            else:
                accepted_motorcycles.append(idx)

        vehicle_events["is_duplicate"] = (
            vehicle_events["duplicate_of_track_id"].notna()
        )
        final_crossings = vehicle_events[~vehicle_events["is_duplicate"]].copy()
        return vehicle_events, final_crossings

    def _aggregate_counts(self, final_crossings: pd.DataFrame) -> tuple[dict[str, int], int]:
        final_counts = (
            final_crossings.groupby("track_class").size()
            .reindex(["motorcycle", "car", "truck", "bus"], fill_value=0)
            .astype(int)
        )
        counts = {
            "motorcycle": int(final_counts["motorcycle"]),
            "car": int(final_counts["car"]),
            "truck": int(final_counts["truck"]),
            "bus": int(final_counts["bus"]),
        }
        return counts, int(sum(counts.values()))

    def count(self, tracks_phase2: pd.DataFrame) -> CountingResult:
        trajectory = self._apply_line_geometry(self._build_trajectory(tracks_phase2))
        crossing_candidates, crossing_events = self._detect_crossings(trajectory)
        crossing_vehicle, crossing_person = self._split_vehicle_person(crossing_events)
        vehicle_events, final_crossings = self._deduplicate_motorcycles(crossing_vehicle)
        counts, total = self._aggregate_counts(final_crossings)

        audit = {
            "all_crossing_events": len(crossing_events),
            "person_crossings_excluded": len(crossing_person),
            "vehicle_crossings_before_dedup": len(vehicle_events),
            "motorcycle_duplicates_removed": int(vehicle_events["is_duplicate"].sum()),
            "final_vehicle_crossings": len(final_crossings),
            "final_vehicle_count": total,
        }

        return CountingResult(
            counts=counts,
            total=total,
            trajectory=trajectory,
            crossing_candidates=crossing_candidates,
            crossing_events=crossing_events,
            crossing_vehicle=crossing_vehicle,
            crossing_person=crossing_person,
            vehicle_events=vehicle_events,
            final_crossings=final_crossings,
            audit=audit,
        )
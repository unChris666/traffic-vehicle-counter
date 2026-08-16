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
    """
    Phase 3 traffic counting logic.

    This implementation intentionally preserves the notebook behavior.
    It does not perform detection, tracking, class assignment, or rendering.
    """

    def __init__(
        self,
        *,
        line_x1: float,
        line_y1: float,
        line_x2: float,
        line_y2: float,
        line_deadband_px: float,
        max_trajectory_gap_sec: float,
        moto_dedup_time_sec: float,
        moto_dedup_distance_px: float,
        vehicle_classes: set[str],
        fps: float,
    ) -> None:

        if fps <= 0:
            raise ValueError(f"fps must be > 0, got {fps}")

        self.x1 = float(line_x1)
        self.y1 = float(line_y1)
        self.x2 = float(line_x2)
        self.y2 = float(line_y2)

        self.line_dx = self.x2 - self.x1
        self.line_dy = self.y2 - self.y1

        self.line_length = math.hypot(
            self.line_dx,
            self.line_dy,
        )

        if self.line_length <= 0:
            raise ValueError("Counting line cannot have zero length")

        self.line_deadband_px = float(line_deadband_px)
        self.max_trajectory_gap_sec = float(
            max_trajectory_gap_sec
        )

        self.moto_dedup_time_sec = float(
            moto_dedup_time_sec
        )
        self.moto_dedup_distance_px = float(
            moto_dedup_distance_px
        )

        self.vehicle_classes = set(vehicle_classes)
        self.fps = float(fps)

    # ============================================================
    # PHASE 3 — TRAJECTORY
    # ============================================================

    def _build_trajectory(
        self,
        tracks_phase2: pd.DataFrame,
    ) -> pd.DataFrame:

        required_columns = {
            "track_id",
            "frame_id",
            "timestamp_sec",
            "bottom_center_x",
            "bottom_center_y",
            "track_class",
            "track_class_ratio",
            "class_ambiguous",
        }

        missing = (
            required_columns
            - set(tracks_phase2.columns)
        )

        if missing:
            raise ValueError(
                "tracks_phase2 missing required columns: "
                f"{sorted(missing)}"
            )

        trajectory = (
            tracks_phase2[
                [
                    "track_id",
                    "frame_id",
                    "timestamp_sec",
                    "bottom_center_x",
                    "bottom_center_y",
                    "track_class",
                    "track_class_ratio",
                    "class_ambiguous",
                ]
            ]
            .sort_values(
                ["track_id", "frame_id"]
            )
            .copy()
        )

        trajectory["dx"] = (
            trajectory
            .groupby("track_id")[
                "bottom_center_x"
            ]
            .diff()
        )

        trajectory["dy"] = (
            trajectory
            .groupby("track_id")[
                "bottom_center_y"
            ]
            .diff()
        )

        trajectory["frame_delta"] = (
            trajectory
            .groupby("track_id")[
                "frame_id"
            ]
            .diff()
        )

        trajectory["time_delta_sec"] = (
            trajectory["frame_delta"]
            / self.fps
        )

        return trajectory

    # ============================================================
    # LINE GEOMETRY
    # ============================================================

    def _apply_line_geometry(
        self,
        trajectory: pd.DataFrame,
    ) -> pd.DataFrame:

        trajectory = trajectory.copy()

        trajectory["line_value"] = (
            self.line_dx
            * (
                trajectory["bottom_center_y"]
                - self.y1
            )
            -
            self.line_dy
            * (
                trajectory["bottom_center_x"]
                - self.x1
            )
        )

        trajectory["line_distance_px"] = (
            np.abs(trajectory["line_value"])
            / self.line_length
        )

        # Exact notebook behavior:
        #
        # distance <= deadband → 0
        # positive             → +1
        # otherwise            → -1
        trajectory["side"] = np.select(
            [
                trajectory["line_distance_px"]
                <= self.line_deadband_px,

                trajectory["line_value"] > 0,
            ],
            [
                0,
                1,
            ],
            default=-1,
        )

        return trajectory

    # ============================================================
    # STABLE SIDE + CROSSING
    # ============================================================

    def _detect_crossings(
        self,
        trajectory: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:

        trajectory_stable = trajectory[
            trajectory["side"] != 0
        ].copy()

        trajectory_stable["previous_side"] = (
            trajectory_stable
            .groupby("track_id")["side"]
            .shift(1)
        )

        trajectory_stable["previous_frame"] = (
            trajectory_stable
            .groupby("track_id")["frame_id"]
            .shift(1)
        )

        trajectory_stable["frame_gap"] = (
            trajectory_stable["frame_id"]
            - trajectory_stable["previous_frame"]
        )

        trajectory_stable[
            "valid_temporal_transition"
        ] = (
            trajectory_stable["frame_gap"]
            <= (
                self.max_trajectory_gap_sec
                * self.fps
            )
        )

        trajectory_stable["crossed"] = (
            trajectory_stable[
                "valid_temporal_transition"
            ]
            &
            trajectory_stable[
                "previous_side"
            ].notna()
            &
            (
                trajectory_stable["side"]
                !=
                trajectory_stable["previous_side"]
            )
        )

        crossing_candidates = (
            trajectory_stable[
                trajectory_stable["crossed"]
            ]
            .copy()
        )

        crossing_candidates["direction"] = np.where(
            crossing_candidates["previous_side"] < 0,
            "side_-1_to_+1",
            "side_+1_to_-1",
        )

        # Exact notebook behavior:
        # one crossing event per track,
        # first valid crossing only.
        crossing_events = (
            crossing_candidates
            .sort_values(
                ["track_id", "frame_id"]
            )
            .drop_duplicates(
                subset=["track_id"],
                keep="first",
            )
            [
                [
                    "track_id",
                    "frame_id",
                    "timestamp_sec",
                    "bottom_center_x",
                    "bottom_center_y",
                    "direction",
                    "track_class",
                    "track_class_ratio",
                    "class_ambiguous",
                ]
            ]
            .rename(
                columns={
                    "frame_id": "crossing_frame",
                    "timestamp_sec": "crossing_time_sec",
                    "bottom_center_x": "crossing_x",
                    "bottom_center_y": "crossing_y",
                }
            )
            .reset_index(drop=True)
        )

        return (
            crossing_candidates,
            crossing_events,
        )

    # ============================================================
    # PERSON EXCLUSION
    # ============================================================

    def _split_vehicle_person(
        self,
        crossing_events: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:

        crossing_vehicle = crossing_events[
            crossing_events["track_class"].isin(
                self.vehicle_classes
            )
        ].copy()

        crossing_person = crossing_events[
            crossing_events["track_class"]
            == "person"
        ].copy()

        return (
            crossing_vehicle,
            crossing_person,
        )

    # ============================================================
    # MOTORCYCLE FRAGMENTATION DEDUP
    # ============================================================

    def _deduplicate_motorcycles(
        self,
        crossing_vehicle: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:

        vehicle_events = (
            crossing_vehicle
            .sort_values(
                "crossing_time_sec"
            )
            .reset_index(drop=True)
            .copy()
        )

        vehicle_events[
            "duplicate_of_track_id"
        ] = pd.NA

        vehicle_events[
            "dedup_reason"
        ] = ""

        motorcycle_idx = np.flatnonzero(
            (
                vehicle_events["track_class"]
                == "motorcycle"
            ).to_numpy()
        )

        accepted_motorcycles: list[int] = []

        for idx in motorcycle_idx:

            current = vehicle_events.iloc[idx]

            duplicate_found = False
            duplicate_track_id = None

            # Compare ONLY against previously accepted
            # motorcycle events, exactly like notebook.
            for accepted_idx in accepted_motorcycles:

                previous = (
                    vehicle_events
                    .iloc[accepted_idx]
                )

                # 1. Same direction
                if (
                    current["direction"]
                    != previous["direction"]
                ):
                    continue

                # 2. Temporal proximity
                dt = abs(
                    current["crossing_time_sec"]
                    -
                    previous["crossing_time_sec"]
                )

                if dt > self.moto_dedup_time_sec:
                    continue

                # 3. Spatial proximity
                distance = math.hypot(
                    current["crossing_x"]
                    -
                    previous["crossing_x"],
                    current["crossing_y"]
                    -
                    previous["crossing_y"],
                )

                if (
                    distance
                    > self.moto_dedup_distance_px
                ):
                    continue

                duplicate_found = True

                duplicate_track_id = (
                    previous["track_id"]
                )

                break

            if duplicate_found:

                vehicle_events.at[
                    idx,
                    "duplicate_of_track_id",
                ] = duplicate_track_id

                vehicle_events.at[
                    idx,
                    "dedup_reason",
                ] = (
                    "motorcycle_fragmentation"
                )

            else:
                accepted_motorcycles.append(idx)

        vehicle_events["is_duplicate"] = (
            vehicle_events[
                "duplicate_of_track_id"
            ].notna()
        )

        final_crossings = vehicle_events[
            ~vehicle_events["is_duplicate"]
        ].copy()

        return (
            vehicle_events,
            final_crossings,
        )

    # ============================================================
    # FINAL COUNT
    # ============================================================

    def _aggregate_counts(
        self,
        final_crossings: pd.DataFrame,
    ) -> tuple[dict[str, int], int]:

        final_counts = (
            final_crossings
            .groupby("track_class")
            .size()
            .reindex(
                [
                    "motorcycle",
                    "car",
                    "truck",
                    "bus",
                ],
                fill_value=0,
            )
            .astype(int)
        )

        counts = {
            "motorcycle": int(
                final_counts["motorcycle"]
            ),
            "car": int(
                final_counts["car"]
            ),
            "truck": int(
                final_counts["truck"]
            ),
            "bus": int(
                final_counts["bus"]
            ),
        }

        total = int(
            sum(counts.values())
        )

        return counts, total

    # ============================================================
    # PUBLIC API
    # ============================================================

    def count(
        self,
        tracks_phase2: pd.DataFrame,
    ) -> CountingResult:

        trajectory = self._build_trajectory(
            tracks_phase2
        )

        trajectory = self._apply_line_geometry(
            trajectory
        )

        (
            crossing_candidates,
            crossing_events,
        ) = self._detect_crossings(
            trajectory
        )

        (
            crossing_vehicle,
            crossing_person,
        ) = self._split_vehicle_person(
            crossing_events
        )

        (
            vehicle_events,
            final_crossings,
        ) = self._deduplicate_motorcycles(
            crossing_vehicle
        )

        (
            counts,
            total,
        ) = self._aggregate_counts(
            final_crossings
        )

        audit = {
            "all_crossing_events":
                len(crossing_events),

            "person_crossings_excluded":
                len(crossing_person),

            "vehicle_crossings_before_dedup":
                len(vehicle_events),

            "motorcycle_duplicates_removed":
                int(
                    vehicle_events[
                        "is_duplicate"
                    ].sum()
                ),

            "final_vehicle_crossings":
                len(final_crossings),

            "final_vehicle_count":
                total,
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
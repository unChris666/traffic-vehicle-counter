from __future__ import annotations

import pandas as pd
from dataclasses import dataclass

from app.counting.crossing_identity import (
    CrossingIdentityEngine,
)


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
    Phase 3 counting.

    Phase 1 upgrade:
        track_id
            ↓
        crossing_id
            ↓
        crossing state machine
            ↓
        counted crossing identities

    Motorcycle-specific deduplication is intentionally
    disabled in this phase.
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

        # ------------------------------------------
        # NEW: crossing identity
        # ------------------------------------------

        pre_crossing_distance_px: float = 100.0,

        max_identity_reconnect_gap_sec: float = 1.0,

        max_identity_reconnect_distance_px: float = 100.0,

        identity_match_threshold: float = 0.82,

        identity_match_margin: float = 0.08,

        velocity_gate_px_per_frame: float = 30.0,

        min_pre_crossing_observations: int = 3,

    ) -> None:

        if fps <= 0:
            raise ValueError(
                f"fps must be > 0, got {fps}"
            )

        self.vehicle_classes = set(
            vehicle_classes
        )

        self.fps = float(fps)

        # Kept for backward compatibility.
        self.moto_dedup_time_sec = float(
            moto_dedup_time_sec
        )

        self.moto_dedup_distance_px = float(
            moto_dedup_distance_px
        )

        self.identity_engine = (
            CrossingIdentityEngine(
                fps=fps,

                line_x1=line_x1,
                line_y1=line_y1,
                line_x2=line_x2,
                line_y2=line_y2,

                line_deadband_px=(
                    line_deadband_px
                ),

                pre_crossing_distance_px=(
                    pre_crossing_distance_px
                ),

                max_reconnect_gap_sec=(
                    max_identity_reconnect_gap_sec
                ),

                max_reconnect_distance_px=(
                    max_identity_reconnect_distance_px
                ),

                identity_match_threshold=(
                    identity_match_threshold
                ),

                identity_match_margin=(
                    identity_match_margin
                ),

                velocity_gate_px_per_frame=(
                    velocity_gate_px_per_frame
                ),

                min_pre_crossing_observations=(
                    min_pre_crossing_observations
                ),

                max_crossing_gap_sec=(
                    max_trajectory_gap_sec
                ),
            )
        )

    # =========================================================
    # SPLIT VEHICLE / PERSON
    # =========================================================

    def _split_vehicle_person(
        self,
        crossing_events: pd.DataFrame,
    ) -> tuple[
        pd.DataFrame,
        pd.DataFrame,
    ]:

        crossing_vehicle = (
            crossing_events[
                crossing_events[
                    "track_class"
                ].isin(
                    self.vehicle_classes
                )
            ]
            .copy()
        )

        crossing_person = (
            crossing_events[
                crossing_events[
                    "track_class"
                ]
                == "person"
            ]
            .copy()
        )

        return (
            crossing_vehicle,
            crossing_person,
        )

    # =========================================================
    # AGGREGATE
    # =========================================================

    @staticmethod
    def _aggregate_counts(
        final_crossings: pd.DataFrame,
    ) -> tuple[
        dict[str, int],
        int,
    ]:

        if final_crossings.empty:

            return (
                {
                    "motorcycle": 0,
                    "car": 0,
                    "truck": 0,
                    "bus": 0,
                },
                0,
            )

        counts_series = (
            final_crossings[
                "track_class"
            ]
            .value_counts()
        )

        counts = {
            "motorcycle": int(
                counts_series.get(
                    "motorcycle",
                    0,
                )
            ),

            "car": int(
                counts_series.get(
                    "car",
                    0,
                )
            ),

            "truck": int(
                counts_series.get(
                    "truck",
                    0,
                )
            ),

            "bus": int(
                counts_series.get(
                    "bus",
                    0,
                )
            ),
        }

        return (
            counts,
            int(
                sum(
                    counts.values()
                )
            ),
        )

    # =========================================================
    # MAIN
    # =========================================================

    def count(
        self,
        tracks_phase2: pd.DataFrame,
    ) -> CountingResult:

        (
            trajectory,
            crossing_events,
            identity_audit,
        ) = self.identity_engine.run(
            tracks_phase2
        )

        # Phase 1:
        # crossing events are already unique by crossing_id.
        crossing_candidates = (
            crossing_events.copy()
        )

        (
            crossing_vehicle,
            crossing_person,
        ) = self._split_vehicle_person(
            crossing_events
        )

        # -----------------------------------------------------
        # IMPORTANT:
        # No motorcycle-specific dedup yet.
        # crossing_id is already the primary identity.
        # -----------------------------------------------------

        vehicle_events = (
            crossing_vehicle.copy()
        )

        if not vehicle_events.empty:

            vehicle_events[
                "duplicate_of_crossing_id"
            ] = pd.NA

            vehicle_events[
                "is_duplicate"
            ] = False

            vehicle_events[
                "dedup_reason"
            ] = ""

        final_crossings = (
            vehicle_events.copy()
        )

        (
            counts,
            total,
        ) = self._aggregate_counts(
            final_crossings
        )

        # -----------------------------------------------------
        # Audit
        # -----------------------------------------------------

        audit = {
            **identity_audit,

            "all_crossing_events": int(
                len(crossing_events)
            ),

            "person_crossings_excluded": int(
                len(crossing_person)
            ),

            "vehicle_crossings_before_dedup": int(
                len(vehicle_events)
            ),

            "crossing_event_duplicates_removed": 0,

            "final_vehicle_crossings": int(
                len(final_crossings)
            ),

            "final_vehicle_count": int(
                total
            ),
        }

        return CountingResult(
            counts=counts,

            total=total,

            trajectory=trajectory,

            crossing_candidates=(
                crossing_candidates
            ),

            crossing_events=(
                crossing_events
            ),

            crossing_vehicle=(
                crossing_vehicle
            ),

            crossing_person=(
                crossing_person
            ),

            vehicle_events=(
                vehicle_events
            ),

            final_crossings=(
                final_crossings
            ),

            audit=audit,
        )

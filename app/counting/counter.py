from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from app.counting.crossing_identity import (
    CrossingIdentity,
    CrossingIdentityEngine,
)
from app.counting.robust_crossing import (
    CrossingConfig,
    RobustCrossingEngine,
)


@dataclass(frozen=True)
class CountingResult:
    counts: dict[str, int]
    total: int

    trajectory: pd.DataFrame

    # Phase 1/2 diagnostics. These are not used to calculate counts;
    # they are exposed so notebook/Gradio users can audit trajectory and
    # corridor behavior before moving to the state-machine phase.
    phase12_trajectory: pd.DataFrame
    phase12_audit: pd.DataFrame

    crossing_candidates: pd.DataFrame
    crossing_events: pd.DataFrame

    crossing_vehicle: pd.DataFrame
    crossing_person: pd.DataFrame

    vehicle_events: pd.DataFrame
    final_crossings: pd.DataFrame

    audit: dict[str, int]

    track_audit: pd.DataFrame


class TrafficCounter:
    """
    Production Phase 3 counter.

    Architecture:
        tracks_phase2
            ↓
        CrossingIdentityEngine
            ↓
        physical crossing_id
            ↓
        RobustCrossingEngine
            ↓
        one crossing event / physical identity
            ↓
        vehicle filtering
            ↓
        final counts

    Important:
        We do NOT perform generic time+distance deduplication after
        identity matching. That old approach can merge two real
        motorcycles entering together.

        Fragmentation is handled by CrossingIdentityEngine instead.
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

        # Identity parameters already used by engine.py
        pre_crossing_distance_px: float = 100.0,
        max_identity_reconnect_gap_sec: float = 1.0,
        max_identity_reconnect_distance_px: float = 100.0,
        identity_match_threshold: float = 0.82,
        identity_match_margin: float = 0.08,
        velocity_gate_px_per_frame: float = 30.0,
        min_pre_crossing_observations: int = 2,

        # Robust geometry parameters
        crossing_corridor_px: float = 45.0,
        min_direction_displacement_px: float = 8.0,
        direction_window: int = 3,

        # Phase 1 — trajectory engine
        trajectory_smoothing_alpha: float = 0.35,
        trajectory_velocity_window: int = 5,
        max_velocity_px_per_frame: float = 80.0,

        # Phase 2 — crossing corridor
        min_pre_zone_observations: int = 2,
        min_corridor_observations: int = 1,
        min_post_zone_observations: int = 1,
        require_post_zone: bool = True,

    ) -> None:

        if fps <= 0:
            raise ValueError(
                f"fps must be > 0, got {fps}"
            )

        self.line_x1 = float(line_x1)
        self.line_y1 = float(line_y1)
        self.line_x2 = float(line_x2)
        self.line_y2 = float(line_y2)

        self.line_deadband_px = float(
            line_deadband_px
        )

        self.max_trajectory_gap_sec = float(
            max_trajectory_gap_sec
        )

        # Preserved for API compatibility.
        self.moto_dedup_time_sec = float(
            moto_dedup_time_sec
        )

        self.moto_dedup_distance_px = float(
            moto_dedup_distance_px
        )

        self.vehicle_classes = set(
            vehicle_classes
        )

        self.fps = float(fps)

        self.crossing_corridor_px = float(
            crossing_corridor_px
        )

        self.min_direction_displacement_px = float(
            min_direction_displacement_px
        )

        self.direction_window = max(
            1,
            int(direction_window),
        )

        # ----------------------------------------------------------
        # Identity engine.
        #
        # Fragment reconnect remains conservative.
        # ----------------------------------------------------------

        self.identity_engine = (
            CrossingIdentityEngine(
                fps=self.fps,
                line_x1=self.line_x1,
                line_y1=self.line_y1,
                line_x2=self.line_x2,
                line_y2=self.line_y2,
                line_deadband_px=(
                    self.line_deadband_px
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

        # ----------------------------------------------------------
        # Geometric crossing engine.
        # ----------------------------------------------------------

        self.crossing_engine = (
            RobustCrossingEngine(
                line_x1=self.line_x1,
                line_y1=self.line_y1,
                line_x2=self.line_x2,
                line_y2=self.line_y2,
                fps=self.fps,
                config=CrossingConfig(
                    line_deadband_px=(
                        self.line_deadband_px
                    ),
                    corridor_px=(
                        self.crossing_corridor_px
                    ),
                    max_trajectory_gap_sec=(
                        self.max_trajectory_gap_sec
                    ),
                    min_direction_displacement_px=(
                        self.min_direction_displacement_px
                    ),
                    direction_window=(
                        self.direction_window
                    ),
                    min_track_observations=2,
                    smoothing_alpha=(
                        trajectory_smoothing_alpha
                    ),
                    velocity_window=(
                        trajectory_velocity_window
                    ),
                    max_velocity_px_per_frame=(
                        max_velocity_px_per_frame
                    ),
                    min_pre_zone_observations=(
                        min_pre_zone_observations
                    ),
                    min_corridor_observations=(
                        min_corridor_observations
                    ),
                    min_post_zone_observations=(
                        min_post_zone_observations
                    ),
                    require_post_zone=(
                        require_post_zone
                    ),
                    vehicle_classes=tuple(
                        sorted(self.vehicle_classes)
                    ),
                ),
            )
        )

    # ------------------------------------------------------------------
    # Geometry helpers for trajectory output
    # ------------------------------------------------------------------

    def _apply_line_geometry(
        self,
        trajectory: pd.DataFrame,
    ) -> pd.DataFrame:

        trajectory = trajectory.copy()

        line_dx = (
            self.line_x2
            -
            self.line_x1
        )

        line_dy = (
            self.line_y2
            -
            self.line_y1
        )

        line_length = np.hypot(
            line_dx,
            line_dy,
        )

        trajectory[
            "line_value"
        ] = (
            line_dx
            *
            (
                trajectory[
                    "bottom_center_y"
                ]
                -
                self.line_y1
            )
            -
            line_dy
            *
            (
                trajectory[
                    "bottom_center_x"
                ]
                -
                self.line_x1
            )
        )

        trajectory[
            "line_distance_px"
        ] = (
            trajectory["line_value"].abs()
            /
            line_length
        )

        trajectory[
            "side"
        ] = np.select(
            [
                trajectory[
                    "line_distance_px"
                ]
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

    # ------------------------------------------------------------------
    # Empty DataFrames
    # ------------------------------------------------------------------

    @staticmethod
    def _empty_crossing_events() -> pd.DataFrame:
        return pd.DataFrame(
            columns=[
                "crossing_id",
                "track_id",
                "track_ids",
                "first_frame",
                "last_frame",
                "crossing_frame",
                "crossing_time_sec",
                "crossing_x",
                "crossing_y",
                "direction",
                "side_transition",
                "track_class",
                "track_class_ratio",
                "class_ambiguous",
                "line_distance_px",
                "previous_side",
                "current_side",
                "frame_gap",
                "crossing_method",
                "track_observations",
                "counted",
            ]
        )

    @staticmethod
    def _empty_track_audit() -> pd.DataFrame:
        return pd.DataFrame(
            columns=[
                "crossing_id",
                "track_ids",
                "track_class",
                "first_frame",
                "last_frame",
                "crossing_frame",
                "direction",
                "counted",
                "crossing_method",
                "track_observations",
                "duplicate_of_track_id",
                "dedup_reason",
            ]
        )

    # ------------------------------------------------------------------
    # Build audit
    # ------------------------------------------------------------------

    @staticmethod
    def _build_track_audit(
        trajectory: pd.DataFrame,
        events: pd.DataFrame,
        identity_map: dict[int, CrossingIdentity],
    ) -> pd.DataFrame:

        rows: list[dict] = []

        event_map = {}

        if not events.empty:
            for _, event in events.iterrows():
                event_map[
                    int(event["crossing_id"])
                ] = event

        for crossing_id, identity in identity_map.items():

            event = event_map.get(
                int(crossing_id)
            )

            identity_rows = trajectory[
                trajectory["crossing_id"]
                ==
                crossing_id
            ]

            if identity_rows.empty:
                continue

            first_frame = int(
                identity_rows["frame_id"].min()
            )

            last_frame = int(
                identity_rows["frame_id"].max()
            )

            track_ids = ",".join(
                str(int(v))
                for v in (
                    identity_rows[
                        "track_id"
                    ]
                    .drop_duplicates()
                    .tolist()
                )
            )

            if event is None:

                rows.append(
                    {
                        "crossing_id": int(
                            crossing_id
                        ),
                        "track_ids": track_ids,
                        "track_class": (
                            identity.vehicle_class
                        ),
                        "first_frame": first_frame,
                        "last_frame": last_frame,
                        "crossing_frame": pd.NA,
                        "direction": (
                            "UNKNOWN"
                        ),
                        "counted": False,
                        "crossing_method": "",
                        "track_observations": int(
                            len(identity_rows)
                        ),
                        "duplicate_of_track_id": pd.NA,
                        "dedup_reason": "",
                    }
                )

            else:

                rows.append(
                    {
                        "crossing_id": int(
                            crossing_id
                        ),
                        "track_ids": track_ids,
                        "track_class": (
                            event[
                                "track_class"
                            ]
                        ),
                        "first_frame": first_frame,
                        "last_frame": last_frame,
                        "crossing_frame": (
                            event[
                                "crossing_frame"
                            ]
                        ),
                        "direction": (
                            event["direction"]
                        ),
                        "counted": bool(
                            event["counted"]
                        ),
                        "crossing_method": (
                            event[
                                "crossing_method"
                            ]
                        ),
                        "track_observations": int(
                            len(identity_rows)
                        ),
                        "duplicate_of_track_id": pd.NA,
                        "dedup_reason": "",
                    }
                )

        if not rows:
            return TrafficCounter._empty_track_audit()

        return (
            pd.DataFrame(rows)
            .sort_values(
                "crossing_id"
            )
            .reset_index(drop=True)
        )

    # ------------------------------------------------------------------
    # Main count
    # ------------------------------------------------------------------

    def count(
        self,
        tracks_phase2: pd.DataFrame,
    ) -> CountingResult:

        required = {
            "track_id",
            "frame_id",
            "timestamp_sec",
            "bottom_center_x",
            "bottom_center_y",
            "track_class",
        }

        missing = (
            required
            -
            set(tracks_phase2.columns)
        )

        if missing:
            raise ValueError(
                "tracks_phase2 missing required "
                f"columns: {sorted(missing)}"
            )

        if tracks_phase2.empty:
            empty = self._empty_crossing_events()

            return CountingResult(
                counts={
                    "motorcycle": 0,
                    "car": 0,
                    "truck": 0,
                    "bus": 0,
                },
                total=0,
                trajectory=tracks_phase2.copy(),
                phase12_trajectory=tracks_phase2.copy(),
                phase12_audit=self._empty_track_audit(),
                crossing_candidates=empty.copy(),
                crossing_events=empty.copy(),
                crossing_vehicle=empty.copy(),
                crossing_person=empty.copy(),
                vehicle_events=empty.copy(),
                final_crossings=empty.copy(),
                audit={
                    "all_tracks_analyzed": 0,
                    "unique_physical_identities": 0,
                    "track_reconnections": 0,
                    "person_crossings_excluded": 0,
                    "vehicle_crossings_before_filter": 0,
                    "final_vehicle_crossings": 0,
                    "final_vehicle_count": 0,
                },
                track_audit=self._empty_track_audit(),
            )

        # ==========================================================
        # 1. PHYSICAL IDENTITY
        # ==========================================================

        (
            trajectory,
            identity_map,
            track_to_identity,
            identity_audit,
        ) = self.identity_engine.run(
            tracks_phase2
        )

        # ==========================================================
        # 2. LINE GEOMETRY
        # ==========================================================

        trajectory = self._apply_line_geometry(
            trajectory
        )

        # ==========================================================
        # 3. GEOMETRIC CROSSING
        # ==========================================================
        #
        # Use crossing_id as the grouping identity.
        #
        # This is the critical separation:
        #
        # raw track_id != physical vehicle identity
        #
        # Therefore fragmentation can reconnect without
        # causing simultaneous independent motorcycles to merge.
        # ==========================================================

        (
            events_df,
            phase12_audit,
            phase12_trajectory,
        ) = self.crossing_engine.process(
            trajectory,
            identity_column="crossing_id",
            return_diagnostics=True,
        )

        # Print a human-readable Phase 1/2 report in the Kaggle/Gradio
        # backend logs. This does not modify the count.
        self.crossing_engine.print_phase_report(
            events_df,
            phase12_audit,
        )

        if events_df.empty:
            crossing_events = (
                self._empty_crossing_events()
            )
        else:
            crossing_events = (
                events_df.copy()
                .sort_values(
                    [
                        "crossing_frame",
                        "crossing_id",
                    ],
                    na_position="last",
                )
                .reset_index(drop=True)
            )

        # ==========================================================
        # 4. VEHICLE / PERSON
        # ==========================================================

        crossing_vehicle = (
            crossing_events[
                crossing_events[
                    "track_class"
                ].isin(
                    self.vehicle_classes
                )
                &
                crossing_events[
                    "counted"
                ].astype(bool)
            ]
            .copy()
        )

        crossing_person = (
            crossing_events[
                (
                    crossing_events[
                        "track_class"
                    ]
                    ==
                    "person"
                )
                &
                crossing_events[
                    "counted"
                ].astype(bool)
            ]
            .copy()
        )

        # ==========================================================
        # 5. FINAL VEHICLE EVENTS
        # ==========================================================
        #
        # No generic time/distance dedup.
        #
        # Each physical crossing_id is already unique.
        #
        # This is intentionally different from the old motorcycle
        # dedup implementation, which could merge two true
        # motorcycles crossing close together.
        # ==========================================================

        vehicle_events = (
            crossing_vehicle.copy()
        )

        if vehicle_events.empty:
            final_crossings = (
                self._empty_crossing_events()
            )
        else:
            final_crossings = (
                vehicle_events[
                    [
                        "crossing_id",
                        "track_id",
                        "track_ids",
                        "first_frame",
                        "last_frame",
                        "crossing_frame",
                        "crossing_time_sec",
                        "crossing_x",
                        "crossing_y",
                        "direction",
                        "side_transition",
                        "track_class",
                        "track_class_ratio",
                        "class_ambiguous",
                        "line_distance_px",
                        "previous_side",
                        "current_side",
                        "frame_gap",
                        "crossing_method",
                        "track_observations",
                        "counted",
                    ]
                ]
                .drop_duplicates(
                    "crossing_id",
                    keep="first",
                )
                .sort_values(
                    "crossing_frame",
                    na_position="last",
                )
                .reset_index(drop=True)
            )

        # ==========================================================
        # 6. COUNTS
        # ==========================================================

        counts_series = (
            final_crossings[
                final_crossings[
                    "counted"
                ].astype(bool)
            ]
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
                counts_series[
                    "motorcycle"
                ]
            ),
            "car": int(
                counts_series["car"]
            ),
            "truck": int(
                counts_series["truck"]
            ),
            "bus": int(
                counts_series["bus"]
            ),
        }

        total = int(
            sum(counts.values())
        )

        # ==========================================================
        # 7. AUDIT
        # ==========================================================

        track_audit = self._build_track_audit(
            trajectory,
            crossing_events,
            identity_map,
        )

        # Counted status after vehicle/person filtering.
        if not track_audit.empty:
            non_vehicle_crossings = set(
                crossing_events[
                    ~crossing_events[
                        "track_class"
                    ].isin(
                        self.vehicle_classes
                    )
                ]["crossing_id"].dropna().astype(int)
            )

            for crossing_id in non_vehicle_crossings:
                track_audit.loc[
                    track_audit[
                        "crossing_id"
                    ]
                    ==
                    crossing_id,
                    "counted",
                ] = False

        audit = {
            "all_tracks_analyzed": int(
                trajectory[
                    "track_id"
                ].nunique()
            ),
            "unique_physical_identities": int(
                len(identity_map)
            ),
            "track_reconnections": int(
                identity_audit[
                    "track_reconnections"
                ]
            ),
            "fragmented_identities": int(
                identity_audit[
                    "fragmented_identities"
                ]
            ),
            "all_crossing_events": int(
                len(crossing_events)
            ),
            "person_crossings_excluded": int(
                len(crossing_person)
            ),
            "vehicle_crossings_before_filter": int(
                len(crossing_vehicle)
            ),
            "final_vehicle_crossings": int(
                len(final_crossings)
            ),
            "final_vehicle_count": int(
                total
            ),

            # Explicitly record that broad duplicate suppression
            # is disabled by design.
            "generic_time_distance_duplicates_removed": 0,
        }

        return CountingResult(
            counts=counts,
            total=total,
            trajectory=trajectory,
            phase12_trajectory=phase12_trajectory,
            phase12_audit=phase12_audit,
            crossing_candidates=(
                crossing_events.copy()
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
            track_audit=track_audit,
        )

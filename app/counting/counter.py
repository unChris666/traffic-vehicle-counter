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
        short_track_observation_threshold: int = 8,
        class_evidence_window_frames: int = 8,
        class_recency_decay: float = 0.18,
        min_counting_class_confidence: float = 0.45,
        zone_enter_confirm_observations: int = 2,
        zone_exit_confirm_observations: int = 2,
        gap_bridge_enabled: bool = True,
        gap_bridge_max_frames: int = 12,
        max_velocity_bridge_px_per_frame: float = 160.0,
        fast_speed_multiplier: float = 0.80,
        min_normal_velocity_px_per_frame: float = 1.0,
        min_normal_displacement_px: float = 8.0,

        # Identity-gap crossing / duplicate candidate audit.
        identity_gap_crossing_enabled: bool = True,
        identity_gap_max_frames: int = 8,
        identity_gap_max_endpoint_distance_px: float = 140.0,
        identity_gap_min_side_displacement_px: float = 12.0,
        candidate_duplicate_max_frame_gap: int = 8,
        candidate_duplicate_max_endpoint_distance_px: float = 55.0,
        candidate_duplicate_max_crossing_distance_px: float = 55.0,
        candidate_duplicate_min_direction_cosine: float = 0.75,
        candidate_duplicate_require_non_overlapping_tracks: bool = True,

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
                    short_track_observation_threshold=(
                        short_track_observation_threshold
                    ),
                    class_evidence_window_frames=(
                        class_evidence_window_frames
                    ),
                    class_recency_decay=(
                        class_recency_decay
                    ),
                    min_counting_class_confidence=(
                        min_counting_class_confidence
                    ),
                    zone_enter_confirm_observations=(
                        zone_enter_confirm_observations
                    ),
                    zone_exit_confirm_observations=(
                        zone_exit_confirm_observations
                    ),
                    gap_bridge_enabled=gap_bridge_enabled,
                    gap_bridge_max_frames=gap_bridge_max_frames,
                    max_velocity_bridge_px_per_frame=max_velocity_bridge_px_per_frame,
                    fast_speed_multiplier=fast_speed_multiplier,
                    min_normal_velocity_px_per_frame=min_normal_velocity_px_per_frame,
                    min_normal_displacement_px=min_normal_displacement_px,
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
                "detector_track_class",
                "track_class_ratio",
                "class_ambiguous",
                "counting_class",
                "counting_class_confidence",
                "class_transition",
                "class_evidence",
                "line_distance_px",
                "previous_side",
                "current_side",
                "frame_gap",
                "crossing_method",
                "track_observations",
                "counted",
                "candidate_quality",
                "geometry_crossing",
                "gap_count",
                "identity_gap_side_transition",
                "identity_gap_frames",
                "candidate_duplicate_of",
                "candidate_duplicate_confidence",
                "candidate_duplicate_reason",
                "candidate_duplicate_suppressed",
            ]
        )

    @staticmethod
    def _empty_track_audit() -> pd.DataFrame:
        return pd.DataFrame(
            columns=[
                "crossing_id",
                "track_ids",
                "track_class",
                "detector_track_class",
                "counting_class",
                "counting_class_confidence",
                "class_transition",
                "class_evidence",
                "first_frame",
                "last_frame",
                "crossing_frame",
                "direction",
                "normal_direction",
                "counted",
                "counted_vehicle",
                "crossing_candidate_class",
                "geometry_crossing",
                "crossing_method",
                "candidate_quality",
                "track_observations",
                "gap_count",
                "short_track",
                "fast_crossing",
                "sparse_crossing",
                "zone_path",
                "zone_chatter_count",
                "pre_zone_evidence",
                "corridor_evidence",
                "post_zone_evidence",
                "failure_reason",
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
                        "counting_class": event.get("counting_class", event.get("track_class", "unknown")),
                        "counting_class_confidence": float(event.get("counting_class_confidence", 1.0)),
                        "class_transition": event.get("class_transition", ""),
                        "class_evidence": event.get("class_evidence", ""),
                        "short_track": bool(event.get("short_track", False)),
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
        """Consume canonical CrossingCandidate events.

        IMPORTANT:
            This method NEVER performs geometric crossing detection.
            RobustCrossingEngine is the sole producer of CrossingCandidate.
            Counter only consumes, classifies, and aggregates those candidates.

        Consequence:
            Two independent crossing_id values at the same frame are always
            counted independently. There is no same-frame suppression.
        """
        required = {
            "track_id",
            "frame_id",
            "timestamp_sec",
            "bottom_center_x",
            "bottom_center_y",
            "track_class",
        }
        missing = required - set(tracks_phase2.columns)
        if missing:
            raise ValueError(
                "tracks_phase2 missing required "
                f"columns: {sorted(missing)}"
            )

        if tracks_phase2.empty:
            empty = self._empty_crossing_events()
            return CountingResult(
                counts={"motorcycle": 0, "car": 0, "truck": 0, "bus": 0},
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
                    "canonical_crossing_candidates": 0,
                    "count_eligible_candidates": 0,
                    "person_crossings": 0,
                    "vehicle_crossings_before_filter": 0,
                    "final_vehicle_crossings": 0,
                    "final_vehicle_count": 0,
                    "same_frame_max_crossings": 0,
                },
                track_audit=self._empty_track_audit(),
            )

        # ----------------------------------------------------------
        # 1. PHYSICAL IDENTITY
        # ----------------------------------------------------------
        trajectory, identity_map, track_to_identity, identity_audit = (
            self.identity_engine.run(tracks_phase2)
        )

        # ----------------------------------------------------------
        # 2. GEOMETRY + CANDIDATE PRODUCTION
        # ----------------------------------------------------------
        # One and only one module performs crossing detection.
        # This is the canonical CrossingCandidate dataframe.
        candidates_df, phase12_audit, prepared = (
            self.crossing_engine.process(
                trajectory,
                identity_column="crossing_id",
                return_diagnostics=True,
            )
        )

        crossing_candidates = candidates_df.copy()
        if crossing_candidates.empty:
            crossing_events = crossing_candidates.copy()
        else:
            crossing_events = (
                crossing_candidates[
                    crossing_candidates["geometry_crossing"].astype(bool)
                ]
                .copy()
                .sort_values(["crossing_frame", "crossing_id"])
                .reset_index(drop=True)
            )

        # ----------------------------------------------------------
        # 3. COUNTER CONSUMES CANDIDATES — NO RE-DETECTION
        # ----------------------------------------------------------
        eligible = crossing_events[
            crossing_events["count_eligibility"].astype(bool)
            & ~crossing_events.get(
                "candidate_duplicate_suppressed",
                pd.Series(False, index=crossing_events.index),
            ).fillna(False).astype(bool)
        ].copy()

        # Keep exactly one event per physical identity. There is deliberately
        # NO dedup by frame, class, distance, or direction. Simultaneous
        # motorcycle + car therefore remain two independent events.
        eligible = (
            eligible
            .drop_duplicates("crossing_id", keep="first")
            .reset_index(drop=True)
        )

        if "counting_class" not in eligible.columns:
            eligible["counting_class"] = eligible["track_class"]

        eligible["counting_class"] = (
            eligible["counting_class"]
            .astype(str)
            .str.lower()
            .str.strip()
        )

        crossing_vehicle = eligible[
            eligible["counting_class"].isin(self.vehicle_classes)
        ].copy()

        crossing_person = eligible[
            eligible["counting_class"].eq("person")
        ].copy()

        # Canonical class becomes the counter-facing track_class for
        # backward-compatible downstream modules.
        if not crossing_vehicle.empty:
            crossing_vehicle["track_class"] = crossing_vehicle["counting_class"]
        if not crossing_person.empty:
            crossing_person["track_class"] = crossing_person["counting_class"]

        vehicle_events = crossing_vehicle.copy()
        final_crossings = (
            vehicle_events
            .drop_duplicates("crossing_id", keep="first")
            .sort_values("crossing_frame")
            .reset_index(drop=True)
            if not vehicle_events.empty
            else self._empty_crossing_events()
        )

        # ----------------------------------------------------------
        # 4. FINAL COUNTS
        # ----------------------------------------------------------
        counts_series = (
            final_crossings.groupby("counting_class")
            .size()
            .reindex(
                ["motorcycle", "car", "truck", "bus"],
                fill_value=0,
            )
            .astype(int)
            if not final_crossings.empty
            else pd.Series(
                {"motorcycle": 0, "car": 0, "truck": 0, "bus": 0},
                dtype=int,
            )
        )

        counts = {
            "motorcycle": int(counts_series["motorcycle"]),
            "car": int(counts_series["car"]),
            "truck": int(counts_series["truck"]),
            "bus": int(counts_series["bus"]),
        }
        total = int(sum(counts.values()))

        # ----------------------------------------------------------
        # 5. AUDIT
        # ----------------------------------------------------------
        track_audit = phase12_audit.copy()
        if not track_audit.empty:
            counted_ids = set(final_crossings["crossing_id"].astype(int).tolist()) if not final_crossings.empty else set()
            track_audit["counted_vehicle"] = track_audit["crossing_id"].isin(counted_ids)
            track_audit["counter_class"] = track_audit["counting_class"]
            track_audit["duplicate_identity_suppressed"] = track_audit.get(
                "candidate_duplicate_suppressed",
                False,
            )

        same_frame_max = 0
        if not crossing_events.empty and "crossing_frame" in crossing_events.columns:
            frame_counts = crossing_events.groupby("crossing_frame").size()
            same_frame_max = int(frame_counts.max()) if not frame_counts.empty else 0

        audit = {
            "all_tracks_analyzed": int(tracks_phase2["track_id"].nunique()),
            "unique_physical_identities": int(len(identity_map)),
            "track_reconnections": int(identity_audit.get("track_reconnections", 0)),
            "class_conflict_rejections": int(identity_audit.get("class_conflict_rejections", 0)),
            "direction_conflict_rejections": int(identity_audit.get("direction_conflict_rejections", 0)),
            "canonical_crossing_candidates": int(len(crossing_events)),
            "count_eligible_candidates": int(len(eligible)),
            "person_crossings": int(len(crossing_person)),
            "vehicle_crossings_before_filter": int(len(eligible)),
            "final_vehicle_crossings": int(len(final_crossings)),
            "final_vehicle_count": int(total),
            "same_frame_max_crossings": int(same_frame_max),
            "identity_gap_crossing_candidates": int(
                crossing_events.get(
                    "identity_gap_side_transition",
                    pd.Series(False, index=crossing_events.index),
                ).fillna(False).astype(bool).sum()
            ),
            "trajectory_duplicate_candidates": int(
                crossing_events.get(
                    "candidate_duplicate_of",
                    pd.Series(pd.NA, index=crossing_events.index),
                ).notna().sum()
            ),
            "trajectory_duplicate_suppressed": int(
                crossing_events.get(
                    "candidate_duplicate_suppressed",
                    pd.Series(False, index=crossing_events.index),
                ).fillna(False).astype(bool).sum()
            ),
        }

        return CountingResult(
            counts=counts,
            total=total,
            trajectory=prepared,
            phase12_trajectory=prepared,
            phase12_audit=phase12_audit,
            crossing_candidates=crossing_candidates,
            crossing_events=crossing_events,
            crossing_vehicle=crossing_vehicle,
            crossing_person=crossing_person,
            vehicle_events=vehicle_events,
            final_crossings=final_crossings,
            audit=audit,
            track_audit=track_audit,
        )


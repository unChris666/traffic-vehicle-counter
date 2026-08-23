# Traffic Counter Phase 1/2 v6 — Simultaneous Crossing / Identity Safety

This release fixes a failure mode where two vehicles cross the counting line at nearly the same time, especially when one object becomes fragmented or briefly occluded.

## Core architecture

TRACK
  ↓
TRAJECTORY
  ↓
CROSSING CANDIDATE
  ↓
CLASS EVIDENCE MERGE
  ↓
AUDIT
  ↓
COUNTER

`RobustCrossingEngine` remains the only crossing detector.
`TrafficCounter` only consumes canonical crossing candidates.

## Main fixes

1. Vehicle-class reconnect safety
   - `person ↔ motorcycle` reconnect remains allowed.
   - `car ↔ motorcycle`, `car ↔ truck`, `car ↔ bus`, etc. are hard-rejected during identity reconnect.
   - This prevents a fragmented vehicle from attaching to a different vehicle identity with a different vehicle class.

2. Direction-aware reconnect safety
   - Fragment reconnect now uses velocity normal to the counting line.
   - Opposite-direction fragments are rejected when the normal-motion evidence conflicts.
   - This is designed for two vehicles meeting near the line from opposite directions.

3. Simultaneous crossing
   - No deduplication by crossing frame.
   - No deduplication by class.
   - No generic time-distance deduplication.
   - Each `crossing_id` is counted once.
   - Two different `crossing_id`s at the same `crossing_frame` remain two events.

4. Audit additions
   - `same_frame_max_crossings`
   - `class_conflict_rejections`
   - `direction_conflict_rejections`
   - `canonical_crossing_candidates`
   - `count_eligible_candidates`

## Runtime contract

All existing CountingConfig compatibility parameters are retained, including:

- pre_crossing_distance_px
- max_identity_reconnect_gap_sec
- max_identity_reconnect_distance_px
- identity_match_threshold
- identity_match_margin
- velocity_gate_px_per_frame
- min_pre_crossing_observations
- crossing_corridor_px
- min_direction_displacement_px
- direction_window
- trajectory_smoothing_alpha
- trajectory_velocity_window
- max_velocity_px_per_frame
- min_pre_zone_observations
- min_corridor_observations
- min_post_zone_observations
- require_post_zone

## Validation performed

- All replacement Python modules compile successfully.
- Synthetic simultaneous motorcycle + car crossing: PASS (2 counts).
- Synthetic simultaneous opposite-direction cars: PASS (2 counts; same_frame_max_crossings = 2).
- Synthetic fragmented car vs motorcycle: PASS (1 motorcycle + 1 car).

## Important interpretation

If the real video still produces only one count for two visible crossing objects after this release, inspect:

`canonical_crossing_candidates.csv`

and compare `crossing_id`, `track_ids`, `crossing_frame`, `counting_class`, and `crossing_method` for the two physical objects.

That will tell us whether the loss happens in tracker identity continuity or in candidate generation, rather than hiding it inside final aggregation.

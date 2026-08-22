# Phase 1-2 v4: Class Evidence + Short Track + Zone Stability

Architecture:

TRACK
  -> TRAJECTORY
  -> CROSSING CANDIDATE
  -> AUDIT
  -> COUNTER

This release does NOT implement Phase 3 State Machine.

## 1. Class evidence

`track_class` from the earlier track-level majority classifier is preserved as `detector_track_class`.
The crossing engine computes `counting_class` using raw `class_name` observations near the crossing, confidence-weighted and recency-weighted.

Example:

person -> person -> motorcycle -> motorcycle -> motorcycle

can become:

counting_class = motorcycle
class_transition = person->motorcycle

The event still exposes `track_class` as `counting_class` for backward compatibility with the existing counter.

## 2. Short-track eligibility

A short track is no longer an automatic invalid object.

`short_track=True` is diagnostic. A geometric crossing can remain `count_eligibility=True` when direction and counting-class evidence are strong enough.

The engine records `short_track_crossing` and reasons in the audit.

## 3. Zone stability

Raw geometry continues to use raw bbox coordinates.
Zone labels use temporal hysteresis:

- `zone_enter_confirm_observations`
- `zone_exit_confirm_observations`

`PRE -> NEAR_LINE -> CORRIDOR` is normal progression and is NOT counted as chatter.
Chatter means actual backtracking such as:

`PRE -> NEAR_LINE -> PRE`

or

`CORRIDOR -> NEAR_LINE -> CORRIDOR`.

## 4. Counter

`TrafficCounter` uses `counting_class` exposed through `track_class` by `RobustCrossingEngine`, so the existing vehicle filter remains compatible.

Additional class evidence columns are preserved in `final_crossings`.

## 5. Required replacement files

- app/counting/robust_crossing.py
- app/counting/counter.py
- app/core/config.py
- app/inference/engine.py

No changes to YOLO26 / BoT-SORT are required.

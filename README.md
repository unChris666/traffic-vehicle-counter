Phase 1/2 v2 — Pre-Phase-3 release

Files:

robust_crossing.py: full replacement for Phase 1 Trajectory Engine + Phase 2 Crossing Corridor.

engine.py: full replacement of the console Phase 1/2 audit report.

Main changes:

NOT_CROSSING is no longer treated as a failure.

Crossing direction is derived from signed distance to the line (normal direction), not raw X motion.

Zone assignment uses hysteresis: corridor enter at corridor_px, corridor exit at corridor_exit_px.

Crossing geometry uses raw observations, preventing EMA lag from suppressing fast crossings.

A crossing that jumps from PRE to POST without an observed corridor bbox is labeled FAST_CROSSING and audited separately.

Audit reports NOT_CROSSING, NEAR_LINE, TRUE_CROSSING, and FAST_CROSSING separately.

Audit reports direction coverage, zone chatter, fast-crossing speed, and review reasons.

Important:

This release does not implement the Phase 3 state machine.

This release does not alter YOLO26, BoT-SORT, or identity reconnect logic.

FAST_CROSSING can still remain REVIEW when there is insufficient PRE/POST evidence. That is deliberate; the next phase will decide whether such evidence can be safely accepted.
Phase 1-2 v3 — Candidate-Preserving Trajectory + Crossing Audit

This release implements the requested architecture before State Machine Phase 3:

TRACK
-> TRAJECTORY
-> ZONE CONTEXT
-> CROSSING DETECTOR
-> CROSSING EVIDENCE
-> AUDIT

Main changes

NO_CROSSING, APPROACHING, NEAR_LINE, and CROSSING_CANDIDATE are separate concepts.

PRE/POST/CORRIDOR evidence does not delete a geometric crossing candidate.

Stable-side transition handles +1 -> 0 -> -1 and -1 -> 0 -> +1.

Raw segment intersection is the primary geometric crossing signal.

Signed-distance sign change is an additional crossing signal.

Sparse observation gaps can use a velocity/sign bridge.

Fast crossings can be valid even with zero observed corridor frames.

Normal velocity is computed relative to the counting line, not from raw X only.

Zone context uses spatial hysteresis.

Every track stays in the audit output, including non-crossing tracks.

Multiple geometric crossing candidates are detected internally, while one primary event is exposed for backward compatibility.

Final counting remains intentionally outside this module; Phase 3 State Machine should own final count decisions.

Drop-in file

Replace:

app/counting/robust_crossing.py

with:

robust_crossing_phase12_v3.py

The public interface remains:

CrossingConfig

RobustCrossingEngine.process(trajectory, identity_column='crossing_id', return_diagnostics=False)

Important

This release intentionally does not implement the State Machine. Do not tune the final count from counted yet; treat count_eligibility and the audit fields as candidate-level evidence until Phase 3 is implemented.
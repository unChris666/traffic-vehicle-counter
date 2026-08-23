# Traffic Counter Phase 1–2 v7 — Identity-Gap Crossing + Duplicate Identity Audit

This release is the final pre-State-Machine refactor for the current architecture.

## Canonical flow

TRACK
→ TRAJECTORY
→ CROSSING CANDIDATE
→ CLASS EVIDENCE MERGE
→ AUDIT
→ COUNTER

`TrafficCounter` does not run geometric crossing detection. `RobustCrossingEngine` is the single source of `CrossingCandidate` events.

## New capability 1 — identity_gap_side_transition

A physical identity can now be counted as crossing when:

- the identity has a stable side before a tracking gap;
- the same physical identity reappears on the opposite side;
- the temporal gap is within `identity_gap_max_frames`;
- endpoint continuity is plausible;
- the transition is sufficiently close to the counting boundary.

The candidate is tagged with:

- `identity_gap_side_transition`
- `identity_gap_frames`
- `crossing_method += identity_gap_side_transition`

No bbox observation inside the corridor is required for this case.

## New capability 2 — trajectory-aware duplicate identity audit

A candidate is flagged as a duplicate identity only when all of the following are consistent:

- same counting class;
- same direction;
- different crossing IDs;
- non-overlapping raw track intervals when configured;
- small gap between identities;
- close crossing points;
- close identity endpoints;
- compatible motion direction.

Same-frame simultaneous vehicles are explicitly not treated as duplicates.

The candidate stores:

- `candidate_duplicate_of`
- `candidate_duplicate_confidence`
- `candidate_duplicate_reason`
- `candidate_duplicate_suppressed`

Suppressed duplicates have `count_eligibility=False`.

## New artifacts

`crossing_candidates_canonical.csv` is the canonical candidate table.

`trajectory_duplicate_identity_audit.csv` is a filtered duplicate view.

`phase12_crossing_corridor_audit.csv` contains the per-identity audit.

## Compatibility

The legacy counting parameters remain available in `CountingConfig` and the `TrafficCounter` constructor. `engine.py` uses safe config lookups so stale config objects do not trigger missing-field `AttributeError` failures.

## Expected audit interpretation

- `identity_gap_side_transition=True` means a same-identity observation gap itself supplied crossing evidence.
- `candidate_duplicate_suppressed=True` means the candidate was strongly consistent with a prior physical identity and is not counted.
- Two different `crossing_id` values on the same `crossing_frame` remain independent candidates and are not frame-deduplicated.

# Phase 1–2 v9

V9 fixes the remaining candidate-generation failure by decoupling direct crossing detection from physical identity grouping.

Pipeline:
TRACK → raw-track TRAJECTORY → direct CrossingCandidate per `track_id` → physical identity annotation → identity-gap candidate → fallback unlinked fragment-gap candidate → class evidence → duplicate audit → counter.

Why: if identity management accidentally merges two simultaneous tracks, v8 can still lose a candidate if crossing detection groups by physical identity. V9 never does that. Every raw tracker track gets its own direct crossing opportunity.

Additional fallback: when two sequential raw track fragments have a strong side-A→gap→side-B continuity but the identity engine failed to reconnect them, v9 can generate `fragment_gap_side_transition`.

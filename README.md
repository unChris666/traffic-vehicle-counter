# Traffic Counter

Production refactor of the Kaggle notebook for YOLO26m + BoT-SORT traffic counting.

## Current step

Phase 6 foundation: convert notebook assumptions into a Python package with:

- centralized configuration
- video validation
- production inference entry point
- CLI entry point
- separation between CV, video, API, and configuration layers

## Current interface

```python
from app.inference.engine import TrafficCountingEngine

engine = TrafficCountingEngine()
metadata = engine.validate("traffic.mp4")
result = engine.process("traffic.mp4")
```

`process()` is intentionally not implemented yet. The next development step is to move the existing Phase 1-3 pipeline behind this interface without changing its counting logic.

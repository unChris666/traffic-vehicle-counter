from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent


def load_module(name: str):
    package = sys.modules.setdefault("app", type(sys)("app"))
    package.__path__ = [str(ROOT)]

    counting = sys.modules.setdefault(
        "app.counting",
        type(sys)("app.counting"),
    )
    counting.__path__ = [str(ROOT)]

    spec = importlib.util.spec_from_file_location(
        f"app.counting.{name}",
        ROOT / f"{name}.py",
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[f"app.counting.{name}"] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


load_module("robust_crossing")
load_module("crossing_identity")
counter_mod = load_module("counter")

TrafficCounter = counter_mod.TrafficCounter


def make_row(track_id, frame_id, x, y):
    return {
        "track_id": track_id,
        "frame_id": frame_id,
        "timestamp_sec": frame_id / 30.0,
        "bottom_center_x": x,
        "bottom_center_y": y,
        "track_class": "motorcycle",
        "track_class_ratio": 1.0,
        "class_ambiguous": False,
        "class_name": "motorcycle",
        "confidence": 0.9,
    }


def make_counter():
    return TrafficCounter(
        line_x1=50,
        line_y1=0,
        line_x2=50,
        line_y2=300,
        line_deadband_px=5,
        max_trajectory_gap_sec=1.5,
        moto_dedup_time_sec=1.5,
        moto_dedup_distance_px=100,
        vehicle_classes={
            "motorcycle",
            "car",
            "truck",
            "bus",
        },
        fps=30,
        pre_crossing_distance_px=100,
        max_identity_reconnect_gap_sec=1.0,
        max_identity_reconnect_distance_px=100,
        identity_match_threshold=0.82,
        identity_match_margin=0.08,
        velocity_gate_px_per_frame=30,
        min_pre_crossing_observations=2,
    )


def test_two_simultaneous_motorcycles():
    rows = []

    for track_id, y in [
        (1, 100),
        (2, 130),
    ]:
        rows.extend([
            make_row(track_id, 100, 40, y),
            make_row(track_id, 101, 60, y),
        ])

    result = make_counter().count(
        pd.DataFrame(rows)
    )

    assert result.counts["motorcycle"] == 2
    assert result.total == 2
    assert len(result.final_crossings) == 2


def test_fragment_reconnection():
    rows = [
        make_row(1, 1, 20, 100),
        make_row(1, 2, 30, 100),
        make_row(1, 3, 40, 100),
        make_row(1, 10, 45, 100),
        make_row(3, 12, 60, 100),
        make_row(3, 13, 70, 100),
    ]

    result = make_counter().count(
        pd.DataFrame(rows)
    )

    assert result.counts["motorcycle"] == 1
    assert result.total == 1
    assert result.audit["track_reconnections"] == 1


if __name__ == "__main__":
    test_two_simultaneous_motorcycles()
    test_fragment_reconnection()
    print("ALL TESTS PASSED")

import pandas as pd

from app.counting.counter import TrafficCounter


def make_counter():
    return TrafficCounter(
        line_x1=100,
        line_y1=0,
        line_x2=0,
        line_y2=100,
        line_deadband_px=8,
        max_trajectory_gap_sec=1.5,
        moto_dedup_time_sec=1.5,
        moto_dedup_distance_px=80,
        vehicle_classes={
            "motorcycle",
            "car",
            "truck",
            "bus",
        },
        fps=30.0,
    )


def test_vehicle_crossing_is_counted():

    tracks_phase2 = pd.DataFrame(
        {
            "track_id": [1, 1],
            "frame_id": [1, 2],
            "timestamp_sec": [0.0, 0.5],

            # x + y = 80  -> one side
            # x + y = 120 -> opposite side
            "bottom_center_x": [
                80,
                20,
            ],
            "bottom_center_y": [
                0,
                100,
            ],

            "track_class": [
                "car",
                "car",
            ],
            "track_class_ratio": [
                1.0,
                1.0,
            ],
            "class_ambiguous": [
                False,
                False,
            ],
        }
    )

    result = make_counter().count(
        tracks_phase2
    )

    assert result.total == 1
    assert result.counts["car"] == 1
    assert result.counts["motorcycle"] == 0
    assert result.counts["truck"] == 0
    assert result.counts["bus"] == 0


def test_person_is_excluded():

    tracks_phase2 = pd.DataFrame(
        {
            "track_id": [1, 1],
            "frame_id": [1, 2],
            "timestamp_sec": [0.0, 0.5],

            "bottom_center_x": [
                80,
                20,
            ],
            "bottom_center_y": [
                0,
                100,
            ],

            "track_class": [
                "person",
                "person",
            ],
            "track_class_ratio": [
                1.0,
                1.0,
            ],
            "class_ambiguous": [
                False,
                False,
            ],
        }
    )

    result = make_counter().count(
        tracks_phase2
    )

    assert result.total == 0
    assert len(result.crossing_person) == 1
    assert (
        result.audit[
            "person_crossings_excluded"
        ]
        == 1
    )


def test_motorcycle_fragmentation_is_deduplicated():

    tracks_phase2 = pd.DataFrame(
        {
            "track_id": [
                1, 1,
                2, 2,
            ],
            "frame_id": [
                1, 2,
                3, 4,
            ],
            "timestamp_sec": [
                0.0,
                0.5,
                1.0,
                1.5,
            ],

            # Track 1:
            # 80 + 0 = 80
            # 40 + 80 = 120
            #
            # Track 2:
            # 82 + 0 = 82
            # 42 + 78 = 120
            #
            # Both cross in the same direction.
            "bottom_center_x": [
                80, 40,
                82, 42,
            ],
            "bottom_center_y": [
                0, 80,
                0, 78,
            ],

            "track_class": [
                "motorcycle",
                "motorcycle",
                "motorcycle",
                "motorcycle",
            ],
            "track_class_ratio": [
                1.0,
                1.0,
                1.0,
                1.0,
            ],
            "class_ambiguous": [
                False,
                False,
                False,
                False,
            ],
        }
    )

    result = make_counter().count(
        tracks_phase2
    )

    assert result.total == 1
    assert result.counts["motorcycle"] == 1
    assert (
        result.audit[
            "motorcycle_duplicates_removed"
        ]
        == 1
    )
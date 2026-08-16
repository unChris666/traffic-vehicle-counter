import pandas as pd

from app.counting.track_classifier import build_track_level


def test_majority_class_and_quality():

    tracks_raw = pd.DataFrame(
        {
            "track_id": [
                1, 1, 1,
                2, 2,
            ],
            "frame_id": [
                1, 2, 3,
                1, 2,
            ],
            "class_name": [
                "car",
                "car",
                "truck",
                "motorcycle",
                "motorcycle",
            ],
            "confidence": [
                0.9,
                0.8,
                0.95,
                0.7,
                0.8,
            ],
        }
    )

    track_level, tracks_phase2 = build_track_level(
        tracks_raw,
        fps=30.0,
    )

    track_1 = track_level[
        track_level["track_id"] == 1
    ].iloc[0]

    track_2 = track_level[
        track_level["track_id"] == 2
    ].iloc[0]

    assert track_1["track_class"] == "car"
    assert track_1["track_class_ratio"] == 2 / 3
    assert bool(track_1["class_ambiguous"]) is True

    assert track_2["track_class"] == "motorcycle"
    assert track_2["track_class_ratio"] == 1.0
    assert bool(track_2["class_ambiguous"]) is False

    assert len(tracks_phase2) == len(tracks_raw)
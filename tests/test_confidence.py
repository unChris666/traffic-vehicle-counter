import pandas as pd
import pytest

from app.inference.confidence import ConfidenceEngine


def test_track_confidence():

    track_level = pd.DataFrame(
        {
            "track_id": [1, 2],
            "track_class": [
                "car",
                "motorcycle",
            ],
            "track_class_ratio": [
                1.0,
                0.60,
            ],
            "observation_ratio": [
                1.0,
                0.50,
            ],
            "mean_confidence": [
                0.90,
                0.50,
            ],
        }
    )

    engine = ConfidenceEngine()

    result = engine.build_track_confidence(
        track_level
    )

    assert len(result) == 2

    track_1_confidence = (
        result.loc[
            result["track_id"] == 1,
            "track_confidence",
        ].iloc[0]
    )

    assert track_1_confidence == pytest.approx(
        0.96
    )

    assert (
        result.loc[
            result["track_id"] == 1,
            "track_confidence_flag",
        ].iloc[0]
        == "HIGH"
    )
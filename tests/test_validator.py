from app.video.validator import VideoValidationError, read_video_metadata


def test_missing_video_rejected(tmp_path):
    missing = tmp_path / "missing.mp4"
    try:
        read_video_metadata(missing)
    except VideoValidationError:
        return
    raise AssertionError("Missing video should be rejected")

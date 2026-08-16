import argparse

from app.inference.engine import TrafficCountingEngine


def main() -> None:
    parser = argparse.ArgumentParser(description="Traffic counting inference")
    parser.add_argument("input", help="Path to input video")
    parser.add_argument("--output", default="outputs", help="Output directory")
    args = parser.parse_args()

    engine = TrafficCountingEngine()
    metadata = engine.validate(args.input)
    print("VIDEO VALIDATION: PASS")
    print(f"Resolution : {metadata.width}x{metadata.height}")
    print(f"FPS        : {metadata.fps:.3f}")
    print(f"Frames     : {metadata.frame_count:,}")
    print(f"Duration   : {metadata.duration_sec:.2f}s")
    print("Phase 6A inference implementation: next step")


if __name__ == "__main__":
    main()

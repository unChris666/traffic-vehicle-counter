from pathlib import Path

from ultralytics import YOLO


ROOT = Path(__file__).resolve().parent

PT_MODEL = (
    ROOT
    / "models"
    / "yolo26m.pt"
)

OUTPUT_ENGINE = (
    ROOT
    / "models"
    / "yolo26m_512_fp16.engine"
)


def main() -> None:
    if not PT_MODEL.exists():
        raise FileNotFoundError(
            f"Missing source model: {PT_MODEL}"
        )

    model = YOLO(
        str(PT_MODEL),
        task="detect",
    )

    print(
        "Exporting YOLO26m -> TensorRT FP16..."
    )

    engine_path = model.export(
        format="engine",
        imgsz=512,
        dynamic=False,
        batch=1,
        quantize=16,
        device=0,
        verbose=True,
    )

    print(
        "\nTensorRT export completed:"
    )
    print(engine_path)

    print(
        "\nProduction config should point to:"
    )
    print(OUTPUT_ENGINE)


if __name__ == "__main__":
    main()

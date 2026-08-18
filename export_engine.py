from pathlib import Path
import shutil
import time

from ultralytics import YOLO


ROOT = Path(__file__).resolve().parent

PT_MODEL = ROOT / "models" / "yolo26m.pt"
OUTPUT_ENGINE = ROOT / "models" / "yolo26m_512_fp16.engine"


def main() -> None:
    # ---------------------------------------------------------
    # Validate source model
    # ---------------------------------------------------------
    if not PT_MODEL.exists():
        raise FileNotFoundError(
            f"Source model not found:\n{PT_MODEL}"
        )

    # ---------------------------------------------------------
    # Skip if engine already exists
    # ---------------------------------------------------------
    if OUTPUT_ENGINE.exists():
        print(f"TensorRT engine already exists:")
        print(OUTPUT_ENGINE)
        return

    print("Building YOLO26m TensorRT FP16 engine...")
    print(f"Input : {PT_MODEL}")
    print(f"Output: {OUTPUT_ENGINE}")
    print("Config: imgsz=512 | FP16 | batch=1 | static shape")

    start = time.perf_counter()

    # ---------------------------------------------------------
    # Load YOLO26m
    # ---------------------------------------------------------
    model = YOLO(
        str(PT_MODEL),
        task="detect",
    )

    # ---------------------------------------------------------
    # Export TensorRT
    # ---------------------------------------------------------
    # exported_path = model.export(
    #     format="engine",
    #     imgsz=512,
    #     dynamic=False,
    #     batch=1,
    #     quantize=16,
    #     device=0,
    #     verbose=False,
    # )

    exported_path = model.export(
        format="engine",
        imgsz=512,
        dynamic=False,
        batch=1,
        half=True,
        device=0,
        verbose=False,
    )

    # ---------------------------------------------------------
    # Normalize output filename
    # ---------------------------------------------------------
    exported_path = Path(exported_path)

    if exported_path != OUTPUT_ENGINE:
        if not exported_path.exists():
            raise FileNotFoundError(
                f"Ultralytics reported export path, "
                f"but file does not exist:\n{exported_path}"
            )

        shutil.move(
            str(exported_path),
            str(OUTPUT_ENGINE),
        )

    elapsed = time.perf_counter() - start

    # ---------------------------------------------------------
    # Final validation
    # ---------------------------------------------------------
    if not OUTPUT_ENGINE.exists():
        raise FileNotFoundError(
            "TensorRT export failed. "
            f"Expected:\n{OUTPUT_ENGINE}"
        )

    print("\nTensorRT export completed.")
    print(f"Engine : {OUTPUT_ENGINE}")
    print(
        f"Size   : "
        f"{OUTPUT_ENGINE.stat().st_size / 1024**2:.2f} MB"
    )
    print(
        f"Time   : "
        f"{elapsed / 60:.2f} minutes"
    )


if __name__ == "__main__":
    main()

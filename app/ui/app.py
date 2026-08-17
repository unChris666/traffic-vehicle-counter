from __future__ import annotations

from pathlib import Path

import gradio as gr
import pandas as pd

from app.inference.engine import TrafficCountingEngine


engine = TrafficCountingEngine()


def process_video(
    video_path: str | None,
    progress=gr.Progress(),
):
    if not video_path:
        raise gr.Error(
            "Please upload a video first."
        )

    progress(
        0.01,
        desc="Validating video...",
    )

    result = engine.process(
        video_path,
        render_video=True,
        progress_callback=progress,
    )

    rows = [
        {
            "CLASS": class_name.title(),
            "QUANTITY": quantity,
        }
        for class_name, quantity
        in result.counts.items()
    ]

    result_table = pd.DataFrame(
        rows
    )

    json_path = (
        Path(engine.config.output_dir)
        / Path(video_path).stem
        / "result.json"
    )

    csv_path = (
        Path(engine.config.output_dir)
        / Path(video_path).stem
        / "final_vehicle_counts.csv"
    )

    video_output = (
        Path(engine.config.output_dir)
        / Path(video_path).stem
        / "annotated_video.mp4"
    )

    confidence_rows = pd.DataFrame(
        result.count_confidence
    )

    return (
        result_table,
        confidence_rows,
        result.total,
        result.overall_confidence["confidence"],
        result.overall_confidence["flag"],
        str(video_output),
        str(csv_path),
        str(json_path),
    )


with gr.Blocks(
    title="Traffic Counter"
) as demo:

    gr.Markdown(
        """
        # Traffic Counter

        Upload a fixed-camera traffic video to count
        vehicles crossing the configured counting line.
        """
    )

    with gr.Row():

        video_input = gr.Video(
            label="Traffic Video",
            sources=["upload"],
            format="mp4",
        )

        video_output = gr.Video(
            label="Annotated Video",
            format="mp4",
        )

    process_button = gr.Button(
        "Process Video",
        variant="primary",
    )

    with gr.Row():

        total_output = gr.Number(
            label="Total Vehicles"
        )

        overall_confidence = gr.Number(
            label="Overall Confidence"
        )

        overall_flag = gr.Textbox(
            label="Confidence Flag"
        )

    result_table = gr.Dataframe(
        label="Vehicle Count",
        headers=[
            "CLASS",
            "QUANTITY",
        ],
        interactive=False,
    )

    confidence_table = gr.Dataframe(
        label="Count Confidence",
        interactive=False,
    )

    with gr.Row():

        csv_output = gr.File(
            label="Download CSV"
        )

        json_output = gr.File(
            label="Download JSON"
        )

    process_button.click(
        fn=process_video,
        inputs=[
            video_input,
        ],
        outputs=[
            result_table,
            confidence_table,
            total_output,
            overall_confidence,
            overall_flag,
            video_output,
            csv_output,
            json_output,
        ],
    )


if __name__ == "__main__":
    demo.queue().launch()
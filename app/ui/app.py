from __future__ import annotations

import json
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
            "Please upload a traffic video."
        )

    try:
        result = engine.process(
            video_path,
            render_video=True,
            progress_callback=progress,
        )

        counts_df = pd.DataFrame(
            [
                {
                    "CLASS": class_name,
                    "QUANTITY": quantity,
                }
                for class_name, quantity
                in result.counts.items()
            ]
        )

        confidence_df = pd.DataFrame(
            result.count_confidence
        )

        output_dir = (
            Path(engine.config.output_dir)
            / Path(video_path).stem
        )

        result_json = (
            output_dir / "result.json"
        )

        csv_path = (
            output_dir
            / "final_vehicle_counts.csv"
        )

        annotated_video = (
            output_dir
            / "annotated_video.mp4"
        )

        return (
            counts_df,
            confidence_df,
            result.total,
            result.overall_confidence[
                "confidence"
            ],
            result.overall_confidence[
                "flag"
            ],
            str(annotated_video),
            str(csv_path),
            str(result_json),
        )

    except Exception as exc:
        raise gr.Error(
            f"Processing failed: {exc}"
        ) from exc


with gr.Blocks(
    title="Traffic Counter",
) as demo:

    gr.Markdown(
        """
        # Traffic Counter

        Upload a fixed-camera traffic video
        to count vehicles crossing the configured
        counting line.
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
            label="Total Vehicles",
        )

        confidence_output = gr.Number(
            label="Overall Confidence",
        )

        confidence_flag = gr.Textbox(
            label="Confidence Flag",
        )

    result_table = gr.Dataframe(
        label="Vehicle Count",
        headers=[
            "CLASS",
            "QUANTITY",
        ],
        datatype=[
            "str",
            "number",
        ],
        interactive=False,
    )

    confidence_table = gr.Dataframe(
        label="Count Confidence",
        interactive=False,
    )

    with gr.Row():

        csv_output = gr.File(
            label="Download CSV",
        )

        json_output = gr.File(
            label="Download JSON",
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
            confidence_output,
            confidence_flag,
            video_output,
            csv_output,
            json_output,
        ],
    )


if __name__ == "__main__":
    demo.queue().launch()
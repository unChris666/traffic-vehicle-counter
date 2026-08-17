from __future__ import annotations

from pathlib import Path

import gradio as gr
import pandas as pd

from app.inference.engine import TrafficCountingEngine


engine = TrafficCountingEngine()


def process_video(
    video_path: str | None,
    mode: str,
    progress=gr.Progress(),
):
    if not video_path:
        raise gr.Error(
            "Please upload a traffic video."
        )

    render_video = (
        mode == "debug"
    )

    try:
        result = engine.process(
            video_path,
            render_video=render_video,
            progress_callback=progress,
        )

        confidence_lookup = {
            row["class"]: row
            for row in result.count_confidence
        }

        counts_df = pd.DataFrame(
            [
                {
                    "CLASS": class_name,
                    "QUANTITY": quantity,
                    "CONFIDENCE": (
                        confidence_lookup[
                            class_name
                        ]["confidence"]
                    ),
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

        video_value = (
            str(annotated_video)
            if render_video
            and annotated_video.exists()
            else None
        )

        return (
            counts_df,
            confidence_df,
            result.total,
            result.overall_confidence["confidence"],
            result.overall_confidence["flag"],
            video_value,
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

        Upload a fixed-camera traffic video and choose a processing mode.

        **Fast Production Mode**
        runs detection, BoT-SORT, track-level class voting,
        line crossing and exports JSON/CSV without rendering a video.

        **Debug Mode**
        runs the same counting pipeline and additionally renders
        an annotated MP4 for visual inspection.
        """
    )

    video_input = gr.Video(
        label="Traffic Video",
        sources=["upload"],
        format="mp4",
    )

    with gr.Row():
        fast_button = gr.Button(
            "Fast Production Mode",
            variant="primary",
        )

        debug_button = gr.Button(
            "Debug Mode",
            variant="secondary",
        )

    mode_state = gr.State("fast")

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
            "CONFIDENCE",
        ],
        datatype=[
            "str",
            "number",
            "number",
        ],
        interactive=False,
    )

    confidence_table = gr.Dataframe(
        label="Detailed Count Confidence",
        interactive=False,
    )

    video_output = gr.Video(
        label="Annotated Video (Debug Mode Only)",
        format="mp4",
    )

    with gr.Row():
        csv_output = gr.File(
            label="Download CSV",
        )

        json_output = gr.File(
            label="Download JSON",
        )

    shared_outputs = [
        result_table,
        confidence_table,
        total_output,
        confidence_output,
        confidence_flag,
        video_output,
        csv_output,
        json_output,
    ]

    def process_fast(
        video_path: str | None,
        progress=gr.Progress(),
    ):
        return process_video(
            video_path,
            "fast",
            progress,
        )

    def process_debug(
        video_path: str | None,
        progress=gr.Progress(),
    ):
        return process_video(
            video_path,
            "debug",
            progress,
        )

    fast_button.click(
        fn=process_fast,
        inputs=[video_input],
        outputs=shared_outputs,
    )

    debug_button.click(
        fn=process_debug,
        inputs=[video_input],
        outputs=shared_outputs,
    )


if __name__ == "__main__":
    demo.queue().launch()

"""
This tool convets all the iterations of a recorded agent loop session in a local directory.
It accepts a directory which contains an `iterations` dir.
The `iterations` dir may have multiple iterations as a subdirectories, and each subdirectory contains the streamed chat completion chunks for that iteration.
For each iteration, it merges the stream_chunks into a single chat completion response JSON, and saves it to the same subdirectory.
It call `merge_chat_completion_chunks(stream_chunks_dir: Path, response_json_path: Path)` from `merge_chat_completions_stream_chunks.py` for each iteration.
"""

import argparse
from pathlib import Path

from scripts.lib.merge_chat_completions_stream_chunks import merge_chat_completion_chunks


def merge_chat_completions_stream_chunks_for_recorded_iterations(recorded_iterations_dir: Path) -> int:
    if not recorded_iterations_dir.exists() or not recorded_iterations_dir.is_dir():
        raise SystemExit(
            f"recorded_iterations_dir is not a directory: {recorded_iterations_dir}")

    # append iterations if the recorded_iterations_dir does not end with "iterations"
    if recorded_iterations_dir.name != "iterations":
        recorded_iterations_dir = recorded_iterations_dir / "iterations"

    for iteration_dir in recorded_iterations_dir.iterdir():
        if iteration_dir.is_dir():
            print(
                f"Merging chat completion chunks for iteration: {iteration_dir.name}")
            stream_chunks_dir = iteration_dir / "stream_chunks"
            response_json_path = iteration_dir / "response.json"

            if not stream_chunks_dir.exists() or not stream_chunks_dir.is_dir():
                print(
                    f"Skipping iteration {iteration_dir.name}: stream_chunks directory not found")
                continue
            try:
                merge_chat_completion_chunks(
                    stream_chunks_dir, response_json_path)
            except Exception as e:
                print(
                    f"Error merging chunks for iteration {iteration_dir.name}: {e}")

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Merge streamed OpenAI-compatible chat completion chunks into a single "
            "non-stream response JSON file for each iteration in a recorded agent loop session."
        )
    )
    parser.add_argument("recorded_iterations_dir", type=Path,
                        help="Directory containing subdirectories for each recorded iteration")
    args = parser.parse_args()
    recorded_iterations_dir = args.recorded_iterations_dir.expanduser().resolve()
    return merge_chat_completions_stream_chunks_for_recorded_iterations(recorded_iterations_dir)


if __name__ == "__main__":
    raise SystemExit(main())

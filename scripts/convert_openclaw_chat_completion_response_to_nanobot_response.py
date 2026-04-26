"""
This tool converts all chat completion responses (with stream chunks) of a 
recorded agent loop session in a local directory from OpenClaw-compatible format to Nanobot-compatible format.

It accepts a directory which contains an `iterations` dir.

The `iterations` dir may have multiple iterations as a subdirectories.
For each iteration, it converts the chat completion response JSON from OpenClaw-compatible format to Nanobot-compatible format, and saves it to the same subdirectory with the same name `response.json` (overwriting the original file).

Converted targets:
- Replace string based on REPLACE_TOOL_CALL_ARGUMENTS_IN_STRING
- For tool_call function read:
    - change the `name` from `read` to `read_file`
- For tool_call function exec:
    - rename argument `workdir` to `working_dir` if exists in the tool_call `arguments` JSON
    - keep these arguments only if they exist: `command`, `working_dir`, `timeout`
"""

REPLACE_TOOL_CALL_ARGUMENTS_IN_STRING = [
    {
        "before": "~/.nvm/versions/node/v22.22.1/lib/node_modules/openclaw/skills",
        "after": "~/work/nanobot/nanobot/skills",
    },
]

import argparse
import json
from pathlib import Path


def convert_response_from_openclaw_to_nanobot(response_json_path: Path) -> None:
    with response_json_path.open("r", encoding="utf-8") as file:
        response = json.load(file)

    # convert the response in-place
    _convert_response_from_openclaw_to_nanobot_impl(response)

    with response_json_path.open("w", encoding="utf-8") as file:
        json.dump(response, file, ensure_ascii=False, indent=2)
        file.write("\n")


def _convert_response_from_openclaw_to_nanobot_impl(response: dict) -> None:
    choices = response.get("choices", [])
    for choice in choices:
        message = choice.get("message", {})
        tool_calls = message.get("tool_calls", [])
        for tool_call in tool_calls:
            if tool_call.get("type") == "function":
                function = tool_call.get("function", {})
                arguments = function.get("arguments", "")
                if isinstance(arguments, str) and arguments:
                    for replacement in REPLACE_TOOL_CALL_ARGUMENTS_IN_STRING:
                        arguments = arguments.replace(
                            replacement["before"], replacement["after"])
                    function["arguments"] = arguments

                if function.get("name") == "read":
                    function["name"] = "read_file"
                elif function.get("name") == "exec":
                    arguments = function.get("arguments")
                    if isinstance(arguments, str) and arguments:
                        try:
                            arguments = json.loads(arguments)
                        except json.JSONDecodeError:
                            continue
                    if "workdir" in arguments:
                        arguments["working_dir"] = arguments.pop("workdir")
                    allowed_keys = {"command", "working_dir", "timeout"}
                    function["arguments"] = {
                        k: v for k, v in arguments.items() if k in allowed_keys}
                    function["arguments"] = json.dumps(function["arguments"])


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Convert all chat completion responses (with stream chunks) of a recorded agent loop session in a local directory from OpenClaw-compatible format to Nanobot-compatible format."
        )
    )
    parser.add_argument("recorded_iterations_dir", type=Path,
                        help="Directory containing subdirectories for each recorded iteration")
    args = parser.parse_args()
    recorded_iterations_dir = args.recorded_iterations_dir.expanduser().resolve()

    if not recorded_iterations_dir.exists() or not recorded_iterations_dir.is_dir():
        raise SystemExit(
            f"recorded_iterations_dir is not a directory: {recorded_iterations_dir}")

    # append iterations if the recorded_iterations_dir does not end with "iterations"
    if recorded_iterations_dir.name != "iterations":
        recorded_iterations_dir = recorded_iterations_dir / "iterations"

    for iteration_dir in recorded_iterations_dir.iterdir():
        if iteration_dir.is_dir():
            print(
                f"Converting chat completion response for iteration: {iteration_dir.name}")
            response_json_path = iteration_dir / "response.json"
            if not response_json_path.exists() or not response_json_path.is_file():
                print(
                    f"Skipping iteration {iteration_dir.name}: response.json file not found")
                continue
            try:
                convert_response_from_openclaw_to_nanobot(response_json_path)
            except Exception as e:
                print(
                    f"Error converting response for iteration {iteration_dir.name}: {e}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

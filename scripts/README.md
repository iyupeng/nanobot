# Scripts to merge and convert recorded OpenClaw sessions

## Merge stream chunks from chat.completions into response JSON files
``` bash
python3 -m scripts.merge_chat_completions_stream_chunks_for_recorded_iterations <recorded_iterations_dir>
```

## Convert tool name and arguments
```bash
python3 -m scripts.convert_openclaw_chat_completion_response_to_nanobot_response <recorded_iterations_dir>
```

Note: Modify `REPLACE_TOOL_CALL_ARGUMENTS_IN_STRING` to replace strings in tool call arguments, e.g., path to skills.

import asyncio
import json
from pathlib import Path

import modal


async def stream_log_file(log_file_path: Path, modal_volume: modal.Volume):
    """Stream log file from Modal volume, reloading volume to get latest changes."""
    log_file = Path(log_file_path) / "logs/log.txt"
    max_wait = 60  # sec
    waited = 0

    # Wait for log file to be created
    while not log_file.exists() and waited < max_wait:
        modal_volume.reload()
        await asyncio.sleep(0.5)
        waited += 0.5

    if not log_file.exists():
        yield json.dumps(
            {
                "type": "error",
                "message": f"Log file not created at {log_file} (parent exists: {log_file.parent.exists()})",
            }
        )
        return

    yield json.dumps(
        {
            "type": "log",
            "message": f"Streaming from {log_file} (size: {log_file.stat().st_size} bytes)",
        }
    )

    lines_buffer = set()  # Use set for O(1) lookup
    last_position = 0

    while True:
        modal_volume.reload()

        try:
            with open(log_file, "r", encoding="utf-8") as f:
                # Seek to last known position
                f.seek(last_position)

                # Read all new lines
                new_lines = f.readlines()

                if new_lines:
                    # Update position for next read
                    last_position = f.tell()

                    for line in new_lines:
                        line = line.rstrip("\n\r")
                        if not line:
                            continue

                        # Check if we've seen this line before
                        if line not in lines_buffer:
                            lines_buffer.add(line)

                            if "Training completed" in line:
                                yield json.dumps(
                                    {"type": "log", "message": "Training completed"}
                                )
                                return

                            # Parse and yield the log message
                            try:
                                if "INFO" in line and ":" in line:
                                    parts = line.split(":", 4)
                                    if len(parts) >= 5:
                                        message = parts[4].strip()
                                    else:
                                        message = line
                                else:
                                    message = line

                                yield json.dumps(
                                    {
                                        "type": "log",
                                        "message": message,
                                    }
                                )
                            except Exception:
                                # If parsing fails, just yield the raw line
                                yield json.dumps(
                                    {
                                        "type": "log",
                                        "message": line,
                                    }
                                )
        except FileNotFoundError:
            # File might have been deleted or moved, wait and retry
            await asyncio.sleep(0.5)
            continue
        except Exception as e:
            yield json.dumps(
                {"type": "error", "message": f"Error reading log file: {str(e)}"}
            )
            await asyncio.sleep(0.5)
            continue

        await asyncio.sleep(0.5)

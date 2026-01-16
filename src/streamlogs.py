import asyncio
import json
from pathlib import Path


async def stream_log_file(log_file_path: Path):
    log_file = Path(log_file_path) / "logs/log.txt"
    max_wait = 30  # sec
    waited = 0

    while not log_file.exists() and waited < max_wait:
        await asyncio.sleep(0.5)
        waited += 0.5

    if not log_file.exists():
        yield json.dumps({"type": "error", "message": "Log file not created"})
        return

    yield json.dumps({"type": "log", "message": f"Streaming from {log_file}"})

    lines_buffer = []

    with open(log_file, "r") as f:
        f.seek(0, 2)
        while True:
            line = f.readline()
            if not line:
                await asyncio.sleep(0.5)
                continue
            elif "Training completed" in line:
                yield json.dumps({"type": "log", "message": "Training completed"})
                break
            elif line not in lines_buffer:
                lines_buffer.append(line)
                yield json.dumps(
                    {
                        "type": "log",
                        "message": line.split(":", 4)[4].strip()
                        if "INFO" in line
                        else line,
                    }
                )

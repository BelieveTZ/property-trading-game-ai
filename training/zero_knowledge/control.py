from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


def gpu_status() -> dict[str, object] | None:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=temperature.gpu,power.draw,memory.used,memory.total,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=4,
        )
        temperature, power, used, total, utilization = [value.strip() for value in result.stdout.splitlines()[0].split(",")]
        return {
            "temperature_c": int(temperature),
            "power_w": float(power),
            "memory_used_mib": int(used),
            "memory_total_mib": int(total),
            "utilization_percent": int(utilization),
        }
    except Exception:
        return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("status", "pause", "clear-pause"))
    parser.add_argument("--run-dir", default="training/runs/zero-knowledge-main")
    args = parser.parse_args()
    run_dir = Path(args.run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    pause = run_dir / "pause.request"
    status = run_dir / "status.json"
    if args.command == "pause":
        pause.write_text("pause after the current PPO update\n", encoding="utf-8")
        print(json.dumps({"pause_requested": True, "run_dir": str(run_dir)}, ensure_ascii=False))
        return
    if args.command == "clear-pause":
        pause.unlink(missing_ok=True)
        print(json.dumps({"pause_requested": False, "run_dir": str(run_dir)}, ensure_ascii=False))
        return
    payload: dict[str, object] = {"state": "not_started"}
    if status.exists():
        payload = json.loads(status.read_text(encoding="utf-8"))
    payload["pause_requested"] = pause.exists()
    payload["gpu"] = gpu_status()
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

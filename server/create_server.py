import os
import subprocess

from utils.load_config import MLFlowConfig


def start_mlflow_server(config: MLFlowConfig) -> None:
    tracking_dir = config.tracking_url
    artifact_dir = config.artifact_root

    os.makedirs(tracking_dir, exist_ok=True)
    os.makedirs(artifact_dir, exist_ok=True)

    host = config.host
    port = config.port

    print(f"[INFO] Starting MLFlow server...")
    print(f"[INFO] Tracking URI: {tracking_dir}")
    print(f"[INFO] Artifact Root: {artifact_dir}")
    print(f"[INFO] Host: {host}")
    print(f"[INFO] Port: {port}")

    command = [
        "mlflow", "server",
        "--backend-store-uri", tracking_dir,
        "--default-artifact-root", artifact_dir,
        "--host", host,
        "--port", str(port)
    ]

    if not config.ui_enabled:
        command.append("--gunicorn-opts")
        command.append("disable-mlflow-ui=true")

    try:
        subprocess.run(command, check=True)
    except KeyboardInterrupt:
        print("[INFO] MLFlow server terminated manually.")
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] MLFlow server failed to start: {e}")
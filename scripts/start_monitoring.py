"""Start local monitoring services (Phoenix + MLflow) for the RAG system.

Usage:
    python scripts/start_monitoring.py              # Start both
    python scripts/start_monitoring.py --phoenix     # Phoenix only
    python scripts/start_monitoring.py --mlflow      # MLflow only
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.resolve()


def check_package(name: str) -> bool:
    """Check if a Python package is installed.

    Args:
        name: Package import name.

    Returns:
        True if the package is importable.
    """
    try:
        __import__(name)
        return True
    except ImportError:
        return False


def start_phoenix(port: int = 6006) -> subprocess.Popen:
    """Start the Arize Phoenix server.

    Args:
        port: Port number for the Phoenix UI.

    Returns:
        Subprocess handle.
    """
    if not check_package("phoenix"):
        print("  ERROR: arize-phoenix not installed.")
        print("  Run: pip install arize-phoenix")
        return None

    print(f"  Starting Phoenix on http://localhost:{port} ...")
    env = os.environ.copy()
    env["PHOENIX_PORT"] = str(port)
    proc = subprocess.Popen(
        [sys.executable, "-m", "phoenix.server.main", "serve"],
        cwd=str(PROJECT_ROOT),
        env=env,
    )
    time.sleep(3)
    print(f"  Phoenix UI: http://localhost:{port}")
    return proc


def start_mlflow(host: str = "127.0.0.1", port: int = 5000) -> subprocess.Popen:
    """Start the MLflow tracking server.

    Args:
        host: Host to bind the MLflow server.
        port: Port number for the MLflow UI.

    Returns:
        Subprocess handle.
    """
    if not check_package("mlflow"):
        print("  ERROR: mlflow not installed.")
        print("  Run: pip install mlflow")
        return None

    mlruns_dir = PROJECT_ROOT / "mlruns"
    mlruns_dir.mkdir(exist_ok=True)
    mlartifacts_dir = PROJECT_ROOT / "mlartifacts"
    mlartifacts_dir.mkdir(exist_ok=True)

    # Convert Windows paths to file:/// URIs to avoid scheme-parsing errors
    backend_uri = mlruns_dir.as_uri()
    artifact_uri = mlartifacts_dir.as_uri()

    print(f"  Starting MLflow on http://{host}:{port} ...")
    proc = subprocess.Popen(
        [
            sys.executable, "-m", "mlflow", "server",
            "--host", host,
            "--port", str(port),
            "--backend-store-uri", backend_uri,
            "--default-artifact-root", artifact_uri,
        ],
        cwd=str(PROJECT_ROOT),
    )
    time.sleep(3)
    print(f"  MLflow UI: http://{host}:{port}")
    return proc


def main() -> None:
    """Start monitoring services."""
    parser = argparse.ArgumentParser(description="Start monitoring services")
    parser.add_argument("--phoenix", action="store_true", help="Start Phoenix only")
    parser.add_argument("--mlflow", action="store_true", help="Start MLflow only")
    parser.add_argument("--phoenix-port", type=int, default=6006, help="Phoenix port")
    parser.add_argument("--mlflow-port", type=int, default=5000, help="MLflow port")
    args = parser.parse_args()

    start_both = not args.phoenix and not args.mlflow

    print("\n" + "=" * 50)
    print("  RAG Monitoring Services")
    print("=" * 50)

    processes = []

    if start_both or args.phoenix:
        print("\n[Phoenix] RAG Trace Visualization")
        proc = start_phoenix(port=args.phoenix_port)
        if proc:
            processes.append(("Phoenix", proc))

    if start_both or args.mlflow:
        print("\n[MLflow] Experiment Tracking")
        proc = start_mlflow(port=args.mlflow_port)
        if proc:
            processes.append(("MLflow", proc))

    if not processes:
        print("\nNo services started. Install required packages:")
        print("  pip install arize-phoenix mlflow")
        return

    print("\n" + "=" * 50)
    print("  Services running. Press Ctrl+C to stop.")
    print("=" * 50 + "\n")

    try:
        for name, proc in processes:
            proc.wait()
    except KeyboardInterrupt:
        print("\n\nShutting down monitoring services...")
        for name, proc in processes:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
            print(f"  {name} stopped")
        print("All services stopped.")


if __name__ == "__main__":
    main()

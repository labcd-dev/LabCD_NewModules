"""
================================================================================
run_agent_mpc_ui.py -- one-command launcher for the AgentMPC Streamlit UI
================================================================================
Invokes Streamlit as ``python -m streamlit`` so PATH is not required.

Run from repository root:
    python frontend_streamlit/run_agent_mpc_ui.py
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
import webbrowser
from pathlib import Path

PORT = 8501
# This file lives in frontend_streamlit/; repo root is parent
FRONTEND_DIR = Path(__file__).resolve().parent
REPO_ROOT = FRONTEND_DIR.parent
APP_PATH = FRONTEND_DIR / "agent_mpc_app.py"


def ensure_labcd_agents_installed() -> None:
    """Editable-installs the vendored labcd_agents package with the [all] extra."""
    pkg_dir = REPO_ROOT / "packages" / "labcd_agents"
    if not pkg_dir.exists():
        return
    print(f"Ensuring labcd_agents is installed (editable, [all] extra) for: {sys.executable}")
    try:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q", "-e", f"{pkg_dir}[all]"]
        )
    except subprocess.CalledProcessError as e:
        print(
            f"WARNING: pip install -e \"packages/labcd_agents[all]\" failed ({e}). "
            f"Continuing anyway -- if the app then complains about labcd_agents, run: "
            f'{sys.executable} -m pip install -e "packages/labcd_agents[all]"'
        )


def ensure_requirements_installed() -> None:
    """Installs packages from the repo-root requirements.txt with this interpreter."""
    req = REPO_ROOT / "requirements.txt"
    if not req.exists():
        return
    print(f"Ensuring requirements from {req} for: {sys.executable}")
    try:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q", "-r", str(req)]
        )
    except subprocess.CalledProcessError as e:
        print(f"WARNING: pip install -r requirements.txt failed ({e}). Continuing anyway.")


def main() -> None:
    ensure_labcd_agents_installed()
    ensure_requirements_installed()

    env = os.environ.copy()
    # Ensure repo root is importable (backend_core.AgentMPC)
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        str(REPO_ROOT) if not existing else f"{REPO_ROOT}{os.pathsep}{existing}"
    )

    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(APP_PATH),
        "--server.port",
        str(PORT),
        "--server.headless",
        "true",
    ]
    print(f"Launching: {' '.join(cmd)}")
    print(f"PYTHONPATH={env['PYTHONPATH']}")
    proc = subprocess.Popen(cmd, cwd=str(REPO_ROOT), env=env)
    time.sleep(2.5)
    try:
        webbrowser.open(f"http://localhost:{PORT}")
    except Exception:
        pass
    try:
        proc.wait()
    except KeyboardInterrupt:
        proc.terminate()


if __name__ == "__main__":
    main()

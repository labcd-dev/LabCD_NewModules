"""Extract artifact python_code into artifacts/dynamic_models/*.py.

Copies source verbatim without importing or executing the plant code.
All sources are checked before any files are written. Matching output files
are replaced on reruns; JSON artifacts and unrelated files are never changed.
"""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path

ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"


def extract_dynamics(
    artifacts_dir: Path = ARTIFACTS_DIR,
    output_dir: Path | None = None,
) -> list[Path]:
    artifacts_dir = Path(artifacts_dir).resolve()
    output_dir = (
        Path(output_dir).resolve() if output_dir is not None
        else artifacts_dir / "dynamic_models"
    )
    artifacts = sorted(artifacts_dir.glob("*.json"))
    if not artifacts:
        raise FileNotFoundError(f"No JSON artifacts found in {artifacts_dir}.")

    sources = []
    for artifact_path in artifacts:
        artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
        code = artifact.get("python_code") if isinstance(artifact, dict) else None
        if not isinstance(code, str) or not code.strip():
            raise ValueError(f"{artifact_path.name} must contain a nonempty python_code string.")
        ast.parse(code, filename=f"{artifact_path.name}:python_code")
        sources.append((artifact_path.stem, code))

    output_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for name, code in sources:
        destination = output_dir / f"{name}.py"
        destination.write_text(code, encoding="utf-8")
        written.append(destination)
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract Python dynamics from MPC artifacts.")
    parser.add_argument("--artifacts-dir", type=Path, default=ARTIFACTS_DIR)
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Default: dynamic_models/ inside the selected artifacts directory.",
    )
    args = parser.parse_args()
    written = extract_dynamics(args.artifacts_dir, args.output_dir)
    for path in written:
        print(f"Extracted: {path.name}")
    print(f"Saved {len(written)} dynamics files in {written[0].parent}")


if __name__ == "__main__":
    main()

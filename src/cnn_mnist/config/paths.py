from __future__ import annotations
from pathlib import Path


def get_root_dir():
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    raise FileNotFoundError(
        "Could not find project root: no pyproject.toml found in any parent directory."
    )


def get_paths(root_dir: Path) -> dict:
    paths = {
        "outputs": root_dir / "outputs",
        "figures": root_dir / "outputs" / "figures",
        "models": root_dir / "outputs" / "models",
        "data_raw": root_dir / "data" / "raw",
        "data_processed": root_dir / "data" / "processed",
    }

    for _, path in paths.items():
        path.mkdir(parents=True, exist_ok=True)
    return paths

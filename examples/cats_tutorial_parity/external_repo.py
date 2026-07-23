"""Locate and verify the pinned external tutorial repository.

The simulated CATS data and R syntax from "Multiple imputation for
longitudinal data: A tutorial" (doi:10.1002/sim.10274) are used as an
external cross-language validation oracle. The repository has no explicit
license, so it is never copied into longmi; it is an external checkout
located via LONGMI_TUTORIAL_REPO and pinned by commit in
validation/external_repositories.toml.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PIN_FILE = REPO_ROOT / "validation" / "external_repositories.toml"


def pin() -> dict:
    with PIN_FILE.open("rb") as fh:
        return tomllib.load(fh)["wijesuriya_tutorial"]


def tutorial_repo() -> Path:
    config = pin()
    raw = os.environ.get(config["path_environment_variable"])
    if not raw:
        raise RuntimeError(
            f"Set {config['path_environment_variable']} to the cloned "
            "tutorial repository "
            "(github.com/rushwije/Longitudinal_multiple_imputation_tutorial)."
        )
    path = Path(raw).expanduser().resolve()
    missing = [name for name in config["required_files"] if not (path / name).exists()]
    if missing:
        raise FileNotFoundError(
            f"Tutorial repository at {path} is missing required files: {missing}"
        )
    return path


def current_commit(path: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
    ).strip()


def verify() -> Path:
    """Return the repo path after checking the pinned commit; warn on drift."""
    path = tutorial_repo()
    expected = pin()["expected_commit"]
    actual = current_commit(path)
    if actual != expected:
        print(
            f"warning: tutorial repo at commit {actual[:12]}, pinned "
            f"{expected[:12]}; parity results may not be comparable",
            file=sys.stderr,
        )
    return path


if __name__ == "__main__":
    path = verify()
    print(f"tutorial repository OK: {path} @ {current_commit(path)[:12]}")

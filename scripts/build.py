"""Build script for Meridian — runs PyInstaller with the project spec file."""

import subprocess
import sys
from pathlib import Path


def main():
    root = Path(__file__).resolve().parent.parent
    spec = root / "meridian.spec"

    if not spec.exists():
        print(f"ERROR: {spec} not found")
        sys.exit(1)

    cmd = [
        sys.executable, "-m", "PyInstaller",
        "--clean",
        "--noconfirm",
        str(spec),
    ]

    print(f"Building Meridian from {spec}")
    print(f"Command: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=str(root))
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()

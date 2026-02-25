"""Runtime hook: ensure cvxpy directory exists before import.

In a PyInstaller bundle, pure Python modules live inside the PYZ archive.
cvxpy/utilities/warn.py (Python 3.12+) calls os.listdir() on the cvxpy
package directory at import time. If the directory doesn't exist on disk,
the import fails with FileNotFoundError.

This runtime hook creates the directory (and key subdirectories) before
any cvxpy module is imported, so os.listdir() succeeds.
"""

import os
import sys

if getattr(sys, "frozen", False):
    base = getattr(sys, "_MEIPASS", os.path.dirname(sys.executable))
    cvxpy_dir = os.path.join(base, "cvxpy")
    # Create the cvxpy package directory and its subdirectories.
    # os.listdir() on the package dir returns these entries, which are
    # used to build _CVXPY_SKIP_PREFIXES for warning stack-level detection.
    for subdir in [
        "", "atoms", "constraints", "cvxcore", "expressions",
        "interface", "lin_ops", "problems", "reductions",
        "transforms", "utilities",
    ]:
        os.makedirs(os.path.join(cvxpy_dir, subdir), exist_ok=True)

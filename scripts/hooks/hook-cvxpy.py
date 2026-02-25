"""PyInstaller hook for cvxpy — collect all submodules and data files.

cvxpy/utilities/warn.py calls os.listdir() on the package directory at
import time (Python 3.12+). Without this hook, the directory structure
doesn't exist in the PyInstaller bundle and the import crashes with
FileNotFoundError.
"""

from PyInstaller.utils.hooks import collect_data_files, collect_submodules

datas = collect_data_files("cvxpy")
hiddenimports = collect_submodules("cvxpy")

# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for Meridian Portfolio Terminal.

Build with:
    pyinstaller meridian.spec

Or via the build script:
    python scripts/build.py
"""

import sys
from pathlib import Path

block_cipher = None

# Project root
ROOT = Path(SPECPATH)
SRC = ROOT / "src"

a = Analysis(
    [str(SRC / "portopt" / "app.py")],
    pathex=[str(SRC)],
    binaries=[],
    datas=[
        (str(SRC / "portopt" / "assets"), "portopt/assets"),
        (str(SRC / "portopt" / "samples"), "portopt/samples"),
    ],
    hiddenimports=[
        "portopt.engine.optimization.mean_variance",
        "portopt.engine.optimization.hrp",
        "portopt.engine.optimization.herc",
        "portopt.engine.optimization.black_litterman",
        "portopt.engine.optimization.tic",
        "portopt.engine.factors",
        "portopt.engine.regime",
        "portopt.engine.risk_budgeting",
        "portopt.engine.tax_harvest",
        "portopt.engine.stress",
        "portopt.engine.rolling",
        "portopt.data.providers.yfinance_provider",
        "portopt.data.providers.alphavantage_provider",
        "portopt.data.providers.tiingo_provider",
        "portopt.data.providers.fred_provider",
        "portopt.data.importers.fidelity_csv",
        "portopt.data.importers.schwab_csv",
        "portopt.data.importers.robinhood_csv",
        "portopt.data.importers.generic_csv",
        "portopt.samples",
        "jinja2",
        "anthropic",
        "openpyxl",
        "hmmlearn",
        "hmmlearn.hmm",
        "sklearn.utils._cython_blas",
        "sklearn.utils._typedefs",
        "scipy.special._cdflib",
        "cvxpy",
        "cvxpy.atoms",
        "cvxpy.constraints",
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "tkinter",
        "matplotlib.backends.backend_tkagg",
        "IPython",
        "jupyter",
        "notebook",
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="Meridian",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,  # windowed app
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=str(SRC / "portopt" / "assets" / "icon.ico"),
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="Meridian",
)

"""Sample portfolios bundled with Meridian."""

from pathlib import Path

SAMPLES_DIR = Path(__file__).parent

SAMPLE_PORTFOLIOS = {
    "Balanced 60/40": SAMPLES_DIR / "balanced_60_40.csv",
    "Tech Growth": SAMPLES_DIR / "tech_growth.csv",
    "Dividend Income": SAMPLES_DIR / "dividend_income.csv",
}

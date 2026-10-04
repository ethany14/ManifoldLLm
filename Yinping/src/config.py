from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
REPORTS_DIR = ROOT / "reports"

PDC_RAW = RAW_DIR / "pdc"
WCPR14_RAW = RAW_DIR / "wcpr14"

PDC_CLEAN = PROCESSED_DIR / "pdc_model_input.csv"
WCPR14_CLEAN = PROCESSED_DIR / "wcpr14_model_input.csv"
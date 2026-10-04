from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
EMBEDDINGS_DIR = ROOT / "data" / "embeddings"
OUTPUTS_DIR = ROOT / "outputs"
REPORTS_DIR = ROOT / "reports"

PDC_RAW = RAW_DIR / "pdc"
WCPR14_RAW = RAW_DIR / "wcpr14"

PDC_CLEAN = PROCESSED_DIR / "pdc_model_input.csv"
WCPR14_CLEAN = PROCESSED_DIR / "wcpr14_model_input.csv"

PDC_EMBEDDINGS = EMBEDDINGS_DIR / "pdc_embeddings.npz"
WCPR14_EMBEDDINGS = EMBEDDINGS_DIR / "wcpr14_embeddings.npz"

# Fast, light model that outputs 384-dimensional sentence embeddings.
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

RANDOM_STATE = 42

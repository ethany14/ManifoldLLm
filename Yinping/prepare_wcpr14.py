import pandas as pd
from src.config import WCPR14_RAW, WCPR14_CLEAN, REPORTS_DIR

PERSONALITY_COLUMNS = [
    "extraversion",
    "agreableness",
    "conscientiousness",
    "stability",
    "openness",
]

def main():
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    WCPR14_CLEAN.parent.mkdir(parents=True, exist_ok=True)

    train = pd.read_csv(
        WCPR14_RAW / "vlogs-wcpr14-training.csv"
    )
    test = pd.read_csv(
        WCPR14_RAW / "vlogs-wcpr14-test.csv"
    )

    train["split"] = "train"
    test["split"] = "test"

    df = pd.concat([train, test], ignore_index=True)

    if "transcript" not in df.columns:
        raise KeyError(
            "WCPR14 file does not contain expected 'transcript' column."
        )

    for col in PERSONALITY_COLUMNS:
        if col not in df.columns:
            raise KeyError(
                f"WCPR14 file does not contain expected personality column: {col}"
            )

    id_col = "vlogId" if "vlogId" in df.columns else (
        "labelID" if "labelID" in df.columns else None
    )

    if id_col is None:
        df["person_id"] = [f"vlogger_{i:04d}" for i in range(len(df))]
        id_col = "person_id"

    clean = df[
        [id_col, "transcript", *PERSONALITY_COLUMNS, "split"]
    ].copy()

    clean = clean.rename(
        columns={
            id_col: "person_id",
            "transcript": "text",
            "agreableness": "agreeableness",
        }
    )

    clean["text"] = clean["text"].fillna("").astype(str).str.strip()
    clean = clean[clean["text"].str.len() > 0].reset_index(drop=True)

    clean.to_csv(WCPR14_CLEAN, index=False)

    summary = clean[
        [
            "extraversion",
            "agreeableness",
            "conscientiousness",
            "stability",
            "openness",
        ]
    ].describe().T

    summary.to_csv(
        REPORTS_DIR / "wcpr14_personality_summary.csv"
    )

    print("WCPR14 cleaned shape:", clean.shape)
    print("\nPersonality summary:")
    print(summary.to_string())
    print(f"\nSaved: {WCPR14_CLEAN}")

if __name__ == "__main__":
    main()

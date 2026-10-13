import pandas as pd
from src.config import PDC_RAW, PDC_CLEAN, REPORTS_DIR

def main():
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    PDC_CLEAN.parent.mkdir(parents=True, exist_ok=True)

    main_df = pd.read_json(PDC_RAW / "main.jsonl", lines=True)
    speakers_df = pd.read_json(PDC_RAW / "speakers.jsonl", lines=True)
    videos_df = pd.read_json(PDC_RAW / "videos.jsonl", lines=True)

    print("PDC raw shapes:")
    print(" main:", main_df.shape)
    print(" speakers:", speakers_df.shape)
    print(" videos:", videos_df.shape)

    speaker_cols = [c for c in ["speaker", "name", "domain", "n_videos"] if c in speakers_df.columns]
    video_cols = [c for c in ["video_id", "title", "channel", "upload_date", "duration"] if c in videos_df.columns]

    merged = main_df.merge(
        speakers_df[speaker_cols],
        on="speaker",
        how="left",
    )

    if "video_id" in merged.columns and "video_id" in video_cols:
        merged = merged.merge(
            videos_df[video_cols],
            on="video_id",
            how="left",
        )

    required = ["speaker", "sentence", "valence"]
    missing = [c for c in required if c not in merged.columns]
    if missing:
        raise KeyError(f"PDC is missing expected columns: {missing}")

    if "name" not in merged.columns:
        merged["name"] = merged["speaker"].astype(str)

    if "upload_date" not in merged.columns:
        merged["upload_date"] = pd.NA

    if "modality" not in merged.columns:
        merged["modality"] = pd.NA

    if "domain" not in merged.columns:
        merged["domain"] = pd.NA

    if "video_id" not in merged.columns:
        merged["video_id"] = pd.NA

    clean = merged[
        [
            "speaker",
            "name",
            "domain",
            "video_id",
            "upload_date",
            "sentence",
            "valence",
            "modality",
        ]
    ].copy()

    clean["sentence"] = clean["sentence"].fillna("").astype(str).str.strip()
    clean = clean[clean["sentence"].str.len() > 0].copy()
    clean = clean.drop_duplicates(
        subset=["speaker", "video_id", "sentence"]
    ).reset_index(drop=True)

    clean["upload_date"] = pd.to_datetime(
        clean["upload_date"],
        errors="coerce",
    )

    model_data = clean.rename(
        columns={
            "name": "person",
            "upload_date": "time",
            "sentence": "text",
            "valence": "emotion",
        }
    )[
        [
            "person",
            "time",
            "text",
            "emotion",
            "modality",
            "domain",
            "speaker",
            "video_id",
        ]
    ]

    model_data.to_csv(PDC_CLEAN, index=False)

    # Reports
    person_coverage = (
        model_data.groupby("person", dropna=False)
        .agg(
            sentence_count=("text", "count"),
            video_count=("video_id", "nunique"),
            first_time=("time", "min"),
            last_time=("time", "max"),
        )
        .reset_index()
        .sort_values(
            ["sentence_count", "video_count"],
            ascending=[False, False],
        )
    )
    person_coverage.to_csv(
        REPORTS_DIR / "pdc_person_coverage.csv",
        index=False,
    )

    emotion_distribution = (
        model_data["emotion"]
        .value_counts(dropna=False)
        .rename_axis("emotion")
        .reset_index(name="count")
    )
    emotion_distribution["percent"] = (
        emotion_distribution["count"]
        / emotion_distribution["count"].sum()
    )
    emotion_distribution.to_csv(
        REPORTS_DIR / "pdc_emotion_distribution.csv",
        index=False,
    )

    print("\nTop 20 documented people:")
    print(person_coverage.head(20).to_string(index=False))

    print("\nPDC emotion distribution:")
    print(emotion_distribution.to_string(index=False))

    print(f"\nSaved: {PDC_CLEAN}")

if __name__ == "__main__":
    main()

from pathlib import Path
from huggingface_hub import hf_hub_download
from src.config import PDC_RAW, WCPR14_RAW

def copy_hf_file(repo_id: str, filename: str, target: Path):
    target.parent.mkdir(parents=True, exist_ok=True)

    if target.exists():
        print(f"[skip] {target.name} already exists")
        return

    print(f"[download] {repo_id}/{filename}")
    tmp = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        repo_type="dataset",
    )
    target.write_bytes(Path(tmp).read_bytes())
    print(f"[ok] {target}")

def main():
    print("=" * 70)
    print("Downloading Public Discourse Corpus (PDC)")
    print("=" * 70)

    pdc_repo = "ictchenbo/public-discourse-corpus"
    for filename in ["main.jsonl", "speakers.jsonl", "videos.jsonl"]:
        copy_hf_file(
            pdc_repo,
            filename,
            PDC_RAW / filename,
        )

    print("\n" + "=" * 70)
    print("Downloading WCPR14 personality dataset")
    print("=" * 70)

    wcpr_repo = "facells/youtube-vlog-personality-recognition-wcpr14"

    copy_hf_file(
        wcpr_repo,
        "vlogs-wcpr14-training.csv",
        WCPR14_RAW / "vlogs-wcpr14-training.csv",
    )
    copy_hf_file(
        wcpr_repo,
        "vlogs-wcpr14-test.csv",
        WCPR14_RAW / "vlogs-wcpr14-test.csv",
    )

    print("\nAll core datasets downloaded.")

if __name__ == "__main__":
    main()

import shutil
import subprocess
from pathlib import Path

RAW = Path(__file__).resolve().parent / "raw"
SE_DIR = RAW / "senti4sd"
HEALTH_DIR = RAW / "druglib"

SENTI4SD_REPO = "https://github.com/collab-uniba/Senti4SD.git"



# 1. Senti4SD  (StackOverflow, ~4,400 posts, human-annotated)
def download_senti4sd():
    if SE_DIR.exists():
        print("[SE] already downloaded, skipping")
        return

    if shutil.which("git") is None:
        print("[SE] ERROR: git not found. Install Git first.")
        return

    print("[SE] cloning Senti4SD (this is large, be patient)...")
    SE_DIR.parent.mkdir(parents=True, exist_ok=True)

    subprocess.run(
        ["git", "clone", "--depth", "1", SENTI4SD_REPO, str(SE_DIR)],
        check=True,
    )

    print("[SE] done ->", SE_DIR)
    print("[SE] CSV files found:")
    for f in sorted(SE_DIR.rglob("*.csv")):
        print("     ", f.relative_to(SE_DIR))


# 2. Druglib  (UCI id 461, ~4,100 patient drug reviews)
def download_druglib():
    train_file = HEALTH_DIR / "druglib_train.csv"
    test_file = HEALTH_DIR / "druglib_test.csv"

    if train_file.exists() and test_file.exists():
        print("[HEALTH] already downloaded, skipping")
        return

    from ucimlrepo import fetch_ucirepo

    print("[HEALTH] fetching UCI dataset 461...")
    HEALTH_DIR.mkdir(parents=True, exist_ok=True)

    data = fetch_ucirepo(id=461)
    df = data.data.original 

    # UCI ships a 80/20 split;
    split = int(len(df) * 0.8)
    df.iloc[:split].to_csv(train_file, index=False)
    df.iloc[split:].to_csv(test_file, index=False)

    print(f"[HEALTH] done -> {HEALTH_DIR}  ({len(df)} rows total)")
    print("[HEALTH] columns:", list(df.columns))


def main():
    RAW.mkdir(parents=True, exist_ok=True)
    download_senti4sd()
    print()
    download_druglib()
    print("\nAll done.")
    print("REMINDER: add data/raw/druglib/ to .gitignore -- the licence")
    print("forbids redistribution. Commit this script instead of the data.")


if __name__ == "__main__":
    main()

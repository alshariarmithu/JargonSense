"""
    Stage 1 -- fetch the raw corpus and the pretrained embeddings.
    python -m src.acquisition.download
"""

import pathlib
import shutil
import subprocess

from src.paths import RAW, RAW_SENTI4SD as SE_DIR

SENTI4SD_REPO = "https://github.com/collab-uniba/Senti4SD.git"

# gensim's name for glove.6B.100d -- the same 400k vectors trained on
# Wikipedia + Gigaword, distributed in a form gensim can load directly.
GLOVE_MODEL = "glove-wiki-gigaword-100"



# Senti4SD  (StackOverflow, ~4,400 posts, human-annotated)
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


def main():
    RAW.mkdir(parents=True, exist_ok=True)
    download_senti4sd()
    print("\nAll done.")


if __name__ == "__main__":
    main()

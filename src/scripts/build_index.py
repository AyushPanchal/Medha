# src/scripts/build_index.py
import os
from pathlib import Path
from src.utils.logging import get_logger

logger = get_logger("build_index")

DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
INDEX_DIR = DATA_DIR / "indexes" / "faiss"

def ensure_dirs():
    for d in [RAW_DIR, PROCESSED_DIR, INDEX_DIR]:
        d.mkdir(parents=True, exist_ok=True)
        logger.info(f"Ensured directory: {d}")

def main():
    logger.info("Starting index build (skeleton)")
    ensure_dirs()
    # Placeholder: next step will parse processed text and build FAISS.
    artifacts = {
        "raw_dir": str(RAW_DIR.resolve()),
        "processed_dir": str(PROCESSED_DIR.resolve()),
        "index_dir": str(INDEX_DIR.resolve()),
    }
    logger.info(f"Artifacts ready: {artifacts}")
    logger.info("Done (no-op). Next step: scraping and processing.")

if __name__ == "__main__":
    main()

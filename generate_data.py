"""
generate_data.py  (updated)
─────────────────────────────────────────────────────────────────────────────
Entry point for data preparation.

Priority order
──────────────
1. Real public dataset via Kaggle API
   → okiasstephanie/e-commerce-user-behaviour-data  (6 019 rows)
2. Synthetic fallback  (500 rows, same distribution as the original)

Run:
    python data/generate_data.py

If you have Kaggle credentials (~/.kaggle/kaggle.json  or the
KAGGLE_USERNAME / KAGGLE_KEY env-vars), the real dataset is downloaded
automatically.  Otherwise the synthetic fallback is used silently.
"""

import os, sys

# Make sure the project root (where load_external_dataset.py lives) is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from load_external_dataset import build_dataset

if __name__ == "__main__":
    build_dataset()

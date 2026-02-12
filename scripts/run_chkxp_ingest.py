"""Ingest all Check Expert CSV files into parquet.

Usage:
    uv run python scripts/run_chkxp_ingest.py            # idempotent
    uv run python scripts/run_chkxp_ingest.py --force     # force rebuild
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

from utils.chkxp_ingest import open as chkxp_open

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Target files (everything under data/raw/chkxp/ EXCEPT stock_lob/)
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = _PROJECT_ROOT / "data" / "raw" / "chkxp"

TARGETS: list[Path] = [
    RAW_DIR / "chkxp_etf(kodex200)_(1m)_ohlcvNAV.csv",
    RAW_DIR / "chkxp_etf(kodex200)_(10s)_ohlcvNAVlob.csv",
    RAW_DIR / "kp200_(fut)(mini)(v)_(1m)_from(20250101)_to(20260207).csv",
    RAW_DIR / "ktb_(3)(10)_(fut)(spread)(2nd)_(1m)_from(20200101)_to(20260207).csv",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest Check Expert CSVs → Parquet")
    parser.add_argument(
        "--force", action="store_true", help="Force rebuild even if output exists"
    )
    args = parser.parse_args()

    total_t0 = time.perf_counter()

    for csv_path in TARGETS:
        if not csv_path.exists():
            log.warning("SKIP  %s  (file not found)", csv_path.name)
            continue

        log.info("─" * 60)
        log.info("FILE  %s", csv_path.name)
        t0 = time.perf_counter()

        ds = chkxp_open(str(csv_path), force=args.force)

        elapsed = time.perf_counter() - t0
        info = ds.describe()

        log.info("  format     : %s", info.format_name)
        log.info("  frequency  : %s", info.frequency)
        log.info("  shape      : %s rows × %s cols", *info.shape)
        log.info("  entities   : %s", info.entities)
        log.info("  items      : %s", info.items)
        log.info("  date range : %s → %s", *info.date_range)
        log.info("  output     : %s", ds.output_dir)
        log.info("  elapsed    : %.2fs", elapsed)

    total_elapsed = time.perf_counter() - total_t0
    log.info("─" * 60)
    log.info("ALL DONE  (%.2fs total)", total_elapsed)


if __name__ == "__main__":
    main()

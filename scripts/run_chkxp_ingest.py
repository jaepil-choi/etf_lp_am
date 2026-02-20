"""Ingest all Check Expert CSV files into parquet.

Usage:
    uv run python -m scripts.run_chkxp_ingest            # idempotent
    uv run python -m scripts.run_chkxp_ingest --force     # force rebuild
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

from utils.chkxp_ingest import open as chkxp_open
from utils.chkxp_ingest import open_etf_daily

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
    RAW_DIR / "chkxp_etc(tiger반도체top10)_(1m)_ohlcvNAV.csv",
    RAW_DIR / "chkxp_etc(tiger국채3년)_(1m)_ohlcvNAV.csv",
    RAW_DIR / "kp200_(fut)(mini)(v)_(1m)_from(20250101)_to(20260207).csv",
    RAW_DIR / "ktb_(3)(10)_(fut)(spread)(2nd)_(1m)_from(20200101)_to(20260207).csv",
]

ETF_DAILY_FILES: list[Path] = [
    RAW_DIR / "chkxp_etf(ACE)(KIWOOM)(PLUS)_from(20201010)_to(20260219).csv",
    RAW_DIR / "chkxp_etf(KODEX)_from(20201010)_to(20260219).csv",
    RAW_DIR / "chkxp_etf(RISE)(SOL)_from(20201010)_to(20260219).csv",
    RAW_DIR / "chkxp_etf(TIGER)_from(20201010)_to(20260219).csv",
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

    # -- Daily ETF (multi-file merge) -------------------------------------
    missing = [p for p in ETF_DAILY_FILES if not p.exists()]
    if missing:
        for p in missing:
            log.warning("SKIP  %s  (file not found)", p.name)
    if len(missing) < len(ETF_DAILY_FILES):
        present = [p for p in ETF_DAILY_FILES if p.exists()]
        log.info("─" * 60)
        log.info("ETF DAILY  (%d files)", len(present))
        t0 = time.perf_counter()

        ds_etf = open_etf_daily(
            present,
            output_dir=_PROJECT_ROOT / "data" / "db" / "chkxp" / "etf_daily",
            force=args.force,
        )

        elapsed = time.perf_counter() - t0
        info = ds_etf.describe()

        log.info("  format     : %s", info.format_name)
        log.info("  frequency  : %s", info.frequency)
        log.info("  shape      : %s rows × %s cols", *info.shape)
        log.info("  entities   : %d total", len(info.entities))
        log.info("  items      : %s", info.items)
        log.info("  date range : %s → %s", *info.date_range)
        log.info("  output     : %s", ds_etf.output_dir)
        log.info("  elapsed    : %.2fs", elapsed)

    total_elapsed = time.perf_counter() - total_t0
    log.info("─" * 60)
    log.info("ALL DONE  (%.2fs total)", total_elapsed)


if __name__ == "__main__":
    main()

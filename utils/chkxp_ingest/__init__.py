"""Check Expert CSV data ingestion module.

Mirrors the fn-dg6-ingest interface: idempotent ``open()``, single-file
parquet output with metadata, and filtered ``load()`` with predicate
pushdown.

Quick start (intraday)::

    from utils.chkxp_ingest import open

    ds = open("data/raw/chkxp/kp200_(fut)(mini)(v)_(1m)_from(20250101)_to(20260207).csv")
    df = ds.load()
    df = ds.load(entities=["KOSPI200 선물 2603"], items=["Intra시가", "Intra종가"])
    df = ds.load(date_from="2025-06-01", date_to="2025-12-31")

Quick start (daily ETF)::

    from utils.chkxp_ingest import open_etf_daily
    from pathlib import Path

    RAW = Path("data/raw/chkxp")
    ds = open_etf_daily(
        [
            RAW / "chkxp_etf(ACE)(KIWOOM)(PLUS)_from(20201010)_to(20260219).csv",
            RAW / "chkxp_etf(KODEX)_from(20201010)_to(20260219).csv",
            RAW / "chkxp_etf(RISE)(SOL)_from(20201010)_to(20260219).csv",
            RAW / "chkxp_etf(TIGER)_from(20201010)_to(20260219).csv",
        ],
        output_dir="data/db/chkxp/etf_daily",
    )
    df = ds.load(entities=["RISE 200"], date_from="2023-01-01")
"""

from __future__ import annotations

import logging
from pathlib import Path

from utils.chkxp_ingest.config import ChkxpConfig
from utils.chkxp_ingest.dataset import Dataset, DatasetInfo

__all__ = ["open", "open_etf_daily", "Dataset", "DatasetInfo", "ChkxpConfig"]

logger = logging.getLogger(__name__)

# Default base directories for parquet output.  Relative → resolves against
# the caller's CWD.  Scripts / notebooks should always pass an explicit
# ``output_dir`` to avoid surprises.
_DEFAULT_DB_DIR      = Path("data/db/chkxp")
_DEFAULT_ETF_DAILY   = _DEFAULT_DB_DIR / "etf_daily"


def open(
    input_path: str | Path,
    output_dir: str | Path | None = None,
    *,
    force: bool = False,
    run_immediately: bool = True,
) -> Dataset:
    """Single entry-point for Check Expert datasets.

    Behaviour is **polymorphic** on *input_path*:

    * **CSV file** -- first call: parse header → generate config → build
      parquet.  Subsequent calls skip the build if output already exists
      (idempotent).
    * **YAML config** -- load the saved config and return a ``Dataset``
      handle (no pipeline execution).

    Parameters
    ----------
    input_path : str | Path
        Path to a raw Check Expert CSV **or** a previously-generated
        YAML config file.
    output_dir : str | Path | None
        Where to write parquet output.  Defaults to
        ``data/db/chkxp/<filename_stem>/``.
    force : bool
        If ``True``, rebuild even when output already exists.
    run_immediately : bool
        If ``False``, create the config without building parquet.
        Call ``ds.ingest()`` later to trigger the build.

    Returns
    -------
    Dataset
    """
    input_path = Path(input_path)

    if input_path.suffix.lower() in (".yaml", ".yml"):
        return _open_from_yaml(input_path)

    if input_path.suffix.lower() == ".csv":
        return _open_from_csv(
            input_path,
            output_dir=Path(output_dir) if output_dir else None,
            force=force,
            run_immediately=run_immediately,
        )

    raise ValueError(
        f"Unsupported file type: {input_path.suffix}. "
        "Expected .csv or .yaml/.yml."
    )


# ---------------------------------------------------------------------------
# Internal dispatch — single-file (intraday)
# ---------------------------------------------------------------------------


def _open_from_yaml(yaml_path: Path) -> Dataset:
    """Load an existing YAML config and return a Dataset handle."""
    config = ChkxpConfig.from_yaml(yaml_path)
    return Dataset(config=config, config_path=yaml_path)


def _open_from_csv(
    csv_path: Path,
    output_dir: Path | None,
    force: bool,
    run_immediately: bool,
) -> Dataset:
    """Parse a raw CSV, generate config/parquet, return Dataset."""
    from utils.chkxp_ingest.parser import build_config, parse_header

    if output_dir is None:
        output_dir = _DEFAULT_DB_DIR / csv_path.stem

    config_path = output_dir.parent / f"{output_dir.name}.yaml"

    # Fast path: if output exists and not forcing, reuse config.
    if not force and config_path.exists():
        config = ChkxpConfig.from_yaml(config_path)
        ds = Dataset(config=config, config_path=config_path)
        if ds.is_built:
            logger.info("Output already exists, skipping build: %s", output_dir)
            return ds

    # Parse header and build config from scratch.
    header = parse_header(csv_path)
    config = build_config(csv_path, header, output_dir)

    ds = Dataset(config=config, config_path=config_path)
    ds.save_config()

    if run_immediately:
        ds.ingest()

    return ds


# ---------------------------------------------------------------------------
# open_etf_daily — multi-file daily ETF entry-point
# ---------------------------------------------------------------------------


def open_etf_daily(
    csv_paths: list[str | Path],
    output_dir: str | Path | None = None,
    *,
    force: bool = False,
    run_immediately: bool = True,
) -> Dataset:
    """Parse and merge all daily ETF CSVs into one hive-partitioned dataset.

    Steps
    -----
    1. Parse the header of each source CSV to discover entity groups and
       item codes.
    2. Compute the **intersection** of item code sets across all source files.
       This naturally excludes KODEX-only codes (e.g. ``12506`` 입회일) while
       keeping the 12 codes common to all four files.
    3. Build a merged :class:`ChkxpConfig` covering all entities and common
       items, with ``partition_by=["year"]``.
    4. Call ``ds.ingest()`` which reads all files, filters to common codes,
       concatenates, and writes ``year=YYYY/part-0.parquet`` under
       ``output_dir``.

    Parameters
    ----------
    csv_paths : list[str | Path]
        Paths to the four source ETF daily CSVs.
    output_dir : str | Path | None
        Root of the hive-partitioned output.  Defaults to
        ``data/db/chkxp/etf_daily/``.
    force : bool
        Rebuild even when output already exists.
    run_immediately : bool
        If ``False``, return the Dataset without running ``ingest()``.

    Returns
    -------
    Dataset
    """
    from utils.chkxp_ingest.parser import (
        build_etf_daily_config,
        parse_header,
    )

    csv_paths_resolved = [Path(p) for p in csv_paths]
    for p in csv_paths_resolved:
        if not p.exists():
            raise FileNotFoundError(f"ETF daily source CSV not found: {p}")

    if output_dir is None:
        output_dir = _DEFAULT_ETF_DAILY
    output_dir = Path(output_dir)

    config_path = output_dir.parent / f"{output_dir.name}.yaml"

    # Fast path: reuse existing config if output is present and not forcing.
    if not force and config_path.exists():
        config = ChkxpConfig.from_yaml(config_path)
        ds = Dataset(config=config, config_path=config_path)
        if ds.is_built:
            logger.info(
                "ETF daily output already exists, skipping build: %s", output_dir
            )
            return ds

    # -- Parse all headers -------------------------------------------------
    logger.info("Parsing headers for %d ETF daily CSV files …", len(csv_paths_resolved))
    headers = [parse_header(p) for p in csv_paths_resolved]

    # -- Compute item code intersection ------------------------------------
    # Each header's unique_item_codes comes from its first entity group.
    # We need the intersection across *all* entity groups across all files.
    per_file_code_sets = [
        set(eg_code for eg in h.entity_groups for eg_code in
            [fc.lstrip("F") for fc in eg.item_field_codes])
        for h in headers
    ]
    common_codes_set: set[str] = per_file_code_sets[0]
    for s in per_file_code_sets[1:]:
        common_codes_set &= s

    # Preserve the ordering from the first file's first entity group.
    first_eg = headers[0].entity_groups[0]
    common_item_codes       = [fc.lstrip("F") for fc in first_eg.item_field_codes
                               if fc.lstrip("F") in common_codes_set]
    common_item_field_codes = [fc for fc in first_eg.item_field_codes
                               if fc.lstrip("F") in common_codes_set]
    common_item_names       = [name for fc, name in zip(first_eg.item_field_codes,
                                                         first_eg.item_names)
                               if fc.lstrip("F") in common_codes_set]

    logger.info(
        "Common item codes (%d): %s", len(common_item_codes), common_item_codes
    )

    # -- Build merged config -----------------------------------------------
    config = build_etf_daily_config(
        csv_paths=csv_paths_resolved,
        headers=headers,
        common_item_codes=common_item_codes,
        common_item_field_codes=common_item_field_codes,
        common_item_names=common_item_names,
        output_dir=output_dir,
    )

    ds = Dataset(config=config, config_path=config_path)
    ds.save_config()

    if run_immediately:
        ds.ingest()

    return ds

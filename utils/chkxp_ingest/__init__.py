"""Check Expert CSV data ingestion module.

Mirrors the fn-dg6-ingest interface: idempotent ``open()``, single-file
parquet output with metadata, and filtered ``load()`` with predicate
pushdown.

Quick start::

    from utils.chkxp_ingest import open

    ds = open("data/raw/chkxp/kp200_(fut)(mini)(v)_(1m)_from(20250101)_to(20260207).csv")
    df = ds.load()
    df = ds.load(entities=["KOSPI200 선물 2603"], items=["Intra시가", "Intra종가"])
    df = ds.load(date_from="2025-06-01", date_to="2025-12-31")

    info = ds.describe()
    meta = ds.load_meta()
"""

from __future__ import annotations

import logging
from pathlib import Path

from utils.chkxp_ingest.config import ChkxpConfig
from utils.chkxp_ingest.dataset import Dataset, DatasetInfo

__all__ = ["open", "Dataset", "DatasetInfo", "ChkxpConfig"]

logger = logging.getLogger(__name__)

# Default base directory for parquet output (relative to project root).
_DEFAULT_DB_DIR = Path("data/db/chkxp")


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
# Internal dispatch
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

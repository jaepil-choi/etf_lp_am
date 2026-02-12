"""Dataset handle for ingested Check Expert data.

Provides filtered reads via PyArrow predicate-pushdown / column-pruning,
quick metadata inspection (``describe``), and idempotent rebuild
(``ingest``).  Mirrors the ``fn_dg6_ingest.Dataset`` interface.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import NamedTuple

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from utils.chkxp_ingest.config import ChkxpConfig

logger = logging.getLogger(__name__)

# Names of the parquet files inside the output directory.
_DATA_FILE = "default.parquet"
_META_FILE = "_meta.parquet"

# Columns that are always present in the data parquet.
_INDEX_COLS = ["datetime", "entity", "entity_code"]


# ---------------------------------------------------------------------------
# DatasetInfo (returned by describe())
# ---------------------------------------------------------------------------


class DatasetInfo(NamedTuple):
    """Lightweight summary read from parquet footer — no data scan."""

    shape: tuple[int, int]
    entities: list[str]
    items: list[str]
    date_range: tuple[str, str]
    frequency: str
    format_name: str


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class Dataset:
    """Handle to an ingested Check Expert dataset.

    Parameters
    ----------
    config : ChkxpConfig
        Parsed configuration (source, metadata, output, entities, items).
    config_path : Path
        Path to the YAML config file on disk.
    """

    def __init__(self, config: ChkxpConfig, config_path: Path) -> None:
        self.config = config
        self.config_path = Path(config_path)
        self.output_dir = Path(config.output.output_dir)

    # -- public properties --------------------------------------------------

    @property
    def data_path(self) -> Path:
        return self.output_dir / _DATA_FILE

    @property
    def meta_path(self) -> Path:
        return self.output_dir / _META_FILE

    @property
    def is_built(self) -> bool:
        """True if the output parquet files already exist."""
        return self.data_path.exists() and self.meta_path.exists()

    # -- load() -------------------------------------------------------------

    def load(
        self,
        entities: list[str] | None = None,
        items: list[str] | None = None,
        date_from: str | None = None,
        date_to: str | None = None,
    ) -> pd.DataFrame:
        """Read data with optional filtering via PyArrow pushdown.

        Parameters
        ----------
        entities : list[str], optional
            Filter rows where ``entity`` is in this list.
        items : list[str], optional
            Select only these value columns (``datetime``, ``entity``,
            ``entity_code`` are always included).
        date_from, date_to : str, optional
            ISO-format date strings for inclusive range filtering on
            ``datetime``.

        Returns
        -------
        pd.DataFrame
        """
        if not self.data_path.exists():
            raise FileNotFoundError(
                f"Data file not found: {self.data_path}. Run ingest() first."
            )

        # -- Column pruning ------------------------------------------------
        columns: list[str] | None = None
        if items is not None:
            columns = list(dict.fromkeys(_INDEX_COLS + items))

        # -- Row filters (PyArrow expressions) -----------------------------
        filters: list[pa.compute.Expression] = []
        if entities is not None:
            filters.append(pa.compute.field("entity").isin(entities))
        if date_from is not None:
            ts_from = pd.Timestamp(date_from)
            filters.append(pa.compute.field("datetime") >= pa.scalar(ts_from))
        if date_to is not None:
            ts_to = pd.Timestamp(date_to)
            filters.append(pa.compute.field("datetime") <= pa.scalar(ts_to))

        combined_filter = None
        for f in filters:
            combined_filter = f if combined_filter is None else (combined_filter & f)

        table = pq.read_table(
            self.data_path,
            columns=columns,
            filters=combined_filter,
        )
        return table.to_pandas()

    # -- load_meta() --------------------------------------------------------

    def load_meta(self) -> pd.DataFrame:
        """Read the ``_meta.parquet`` lineage table."""
        if not self.meta_path.exists():
            raise FileNotFoundError(
                f"Meta file not found: {self.meta_path}. Run ingest() first."
            )
        return pd.read_parquet(self.meta_path)

    # -- describe() ---------------------------------------------------------

    def describe(self) -> DatasetInfo:
        """Return a quick summary by reading only the parquet footer.

        No data is scanned — schema and row-count come from metadata.
        """
        if not self.data_path.exists():
            raise FileNotFoundError(
                f"Data file not found: {self.data_path}. Run ingest() first."
            )
        pf = pq.ParquetFile(self.data_path)
        schema = pf.schema_arrow
        num_rows = pf.metadata.num_rows
        num_cols = len(schema)

        # Item columns = everything except the fixed index columns.
        all_cols = [f.name for f in schema]
        item_cols = [c for c in all_cols if c not in _INDEX_COLS]

        # Entity and date info from config.
        entity_names = [e.name for e in self.config.entities]
        date_range = (
            self.config.metadata.period_start,
            self.config.metadata.period_end,
        )

        return DatasetInfo(
            shape=(num_rows, num_cols),
            entities=entity_names,
            items=item_cols,
            date_range=date_range,
            frequency=self.config.metadata.frequency,
            format_name=self.config.source.detected_format,
        )

    # -- ingest() -----------------------------------------------------------

    def ingest(self) -> None:
        """Parse the source CSV, write ``default.parquet`` + ``_meta.parquet``.

        Re-reads the raw CSV referenced in ``self.config.source.input_path``.
        """
        from utils.chkxp_ingest.parser import (
            parse_header,
            read_and_transform,
        )

        csv_path = Path(self.config.source.input_path)
        if not csv_path.exists():
            raise FileNotFoundError(f"Source CSV not found: {csv_path}")

        logger.info("Parsing header: %s", csv_path)
        header = parse_header(csv_path)

        logger.info("Reading and transforming data …")
        df = read_and_transform(csv_path, header)

        # -- Write data parquet --------------------------------------------
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Store lightweight metadata in the parquet file-level key-value
        # metadata so downstream readers can inspect without loading _meta.
        file_meta = {
            "chkxp.frequency": self.config.metadata.frequency,
            "chkxp.period_start": self.config.metadata.period_start,
            "chkxp.period_end": self.config.metadata.period_end,
            "chkxp.detected_format": self.config.source.detected_format,
            "chkxp.entities": json.dumps(
                [e.name for e in self.config.entities], ensure_ascii=False
            ),
        }

        table = pa.Table.from_pandas(df, preserve_index=False)
        # Merge our custom metadata with the existing Arrow/pandas metadata.
        existing_meta = table.schema.metadata or {}
        merged_meta = {
            **existing_meta,
            **{k.encode(): v.encode() for k, v in file_meta.items()},
        }
        table = table.replace_schema_metadata(merged_meta)

        pq.write_table(table, self.data_path)
        logger.info(
            "Wrote %s  (%d rows x %d cols)",
            self.data_path,
            len(df),
            len(df.columns),
        )

        # -- Write _meta parquet -------------------------------------------
        meta_df = self._build_meta_df()
        meta_df.to_parquet(self.meta_path, index=False)
        logger.info("Wrote %s", self.meta_path)

    # -- save_config() ------------------------------------------------------

    def save_config(self) -> None:
        """Persist the current config to YAML."""
        self.config.to_yaml(self.config_path)
        logger.info("Saved config: %s", self.config_path)

    # -- internals ----------------------------------------------------------

    def _build_meta_df(self) -> pd.DataFrame:
        """Build the ``_meta`` lineage table (one row per item × entity)."""
        rows: list[dict] = []
        now = datetime.now(tz=timezone.utc).isoformat()

        for entity in self.config.entities:
            for item in self.config.items:
                rows.append(
                    {
                        "source_file": Path(self.config.source.input_path).name,
                        "source_hash": self.config.source.source_hash,
                        "detected_format": self.config.source.detected_format,
                        "entity_name": entity.name,
                        "entity_code": entity.code,
                        "item_code": item.code,
                        "field_code": item.field_code,
                        "item_name": item.name,
                        "frequency": self.config.metadata.frequency,
                        "period_start": self.config.metadata.period_start,
                        "period_end": self.config.metadata.period_end,
                        "processed_at": now,
                    }
                )

        return pd.DataFrame(rows)

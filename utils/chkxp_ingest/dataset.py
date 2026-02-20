"""Dataset handle for ingested Check Expert data.

Provides filtered reads via PyArrow predicate-pushdown / column-pruning,
quick metadata inspection (``describe``), and idempotent rebuild
(``ingest``).  Mirrors the ``fn_dg6_ingest.Dataset`` interface.

Two storage layouts are supported:

* **single-file** (intraday): ``output_dir/default.parquet`` +
  ``output_dir/_meta.parquet``.  Used when ``config.output.partition_by``
  is empty.
* **hive-partitioned** (daily ETF): ``output_dir/year=YYYY/part-0.parquet``
  directories + ``output_dir/_meta.parquet``.  Used when
  ``config.output.partition_by == ["year"]``.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import NamedTuple

import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as pad
import pyarrow.parquet as pq

from utils.chkxp_ingest.config import ChkxpConfig

logger = logging.getLogger(__name__)

# File names inside the output directory.
_DATA_FILE = "default.parquet"
_META_FILE = "_meta.parquet"

# Fixed index columns for each format.
_INTRADAY_INDEX_COLS = ["datetime", "entity", "entity_code"]
_DAILY_INDEX_COLS    = ["date", "entity", "entity_code", "year"]


# ---------------------------------------------------------------------------
# DatasetInfo (returned by describe())
# ---------------------------------------------------------------------------


class DatasetInfo(NamedTuple):
    """Lightweight summary — no full data scan."""

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
        self.config      = config
        self.config_path = Path(config_path)
        self.output_dir  = Path(config.output.output_dir)

    # -- public properties --------------------------------------------------

    @property
    def _is_partitioned(self) -> bool:
        return bool(self.config.output.partition_by)

    @property
    def _index_cols(self) -> list[str]:
        return _DAILY_INDEX_COLS if self._is_partitioned else _INTRADAY_INDEX_COLS

    @property
    def data_path(self) -> Path:
        """Single-file path — valid only for non-partitioned datasets."""
        return self.output_dir / _DATA_FILE

    @property
    def meta_path(self) -> Path:
        return self.output_dir / _META_FILE

    @property
    def is_built(self) -> bool:
        """True if the output parquet artefacts already exist."""
        if self._is_partitioned:
            return any(self.output_dir.glob("year=*/")) and self.meta_path.exists()
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
            Select only these value columns (index columns are always included).
        date_from, date_to : str, optional
            ISO-format date strings for inclusive range filtering on the time
            axis (``datetime`` for intraday; ``date`` for daily).

        Returns
        -------
        pd.DataFrame
        """
        if not self.is_built:
            raise FileNotFoundError(
                f"Output not found at {self.output_dir}. Run ingest() first."
            )

        index_cols = self._index_cols
        time_field = "date" if self._is_partitioned else "datetime"

        # -- Column pruning ------------------------------------------------
        columns: list[str] | None = None
        if items is not None:
            columns = list(dict.fromkeys(index_cols + items))

        # -- Row filters ---------------------------------------------------
        filters: list[pa.compute.Expression] = []
        if entities is not None:
            filters.append(pc.field("entity").isin(entities))
        if date_from is not None:
            ts_from = pd.Timestamp(date_from)
            filters.append(pc.field(time_field) >= pa.scalar(ts_from))
        if date_to is not None:
            ts_to = pd.Timestamp(date_to)
            filters.append(pc.field(time_field) <= pa.scalar(ts_to))

        combined_filter = None
        for f in filters:
            combined_filter = f if combined_filter is None else (combined_filter & f)

        if self._is_partitioned:
            ds = pad.dataset(
                str(self.output_dir),
                format="parquet",
                partitioning="hive",
            )
            table = ds.to_table(columns=columns, filter=combined_filter)
        else:
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
        """Return a quick summary without a full data scan."""
        if not self.is_built:
            raise FileNotFoundError(
                f"Output not found at {self.output_dir}. Run ingest() first."
            )

        index_cols = self._index_cols

        if self._is_partitioned:
            ds = pad.dataset(
                str(self.output_dir),
                format="parquet",
                partitioning="hive",
            )
            schema   = ds.schema
            num_rows = sum(
                f.metadata.num_rows
                for f in ds.get_fragments()
            )
            num_cols = len(schema)
            all_cols = schema.names
        else:
            pf       = pq.ParquetFile(self.data_path)
            schema   = pf.schema_arrow
            num_rows = pf.metadata.num_rows
            num_cols = len(schema)
            all_cols = [f.name for f in schema]

        item_cols    = [c for c in all_cols if c not in index_cols]
        entity_names = [e.name for e in self.config.entities]
        date_range   = (
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
        """Parse source CSV(s), write parquet output + ``_meta.parquet``.

        For **daily_etf** format: reads all files listed in
        ``config.source.source_files``, filters to the common item codes
        stored in ``config.items``, concatenates, and writes a
        hive-partitioned dataset (``year=YYYY/part-0.parquet``).

        For **intraday** formats: reads ``config.source.input_path`` and
        writes a single ``default.parquet``.
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)

        if self._is_partitioned:
            self._ingest_daily()
        else:
            self._ingest_intraday()

        meta_df = self._build_meta_df()
        meta_df.to_parquet(self.meta_path, index=False)
        logger.info("Wrote %s", self.meta_path)

    def _ingest_intraday(self) -> None:
        from utils.chkxp_ingest.parser import parse_header, read_and_transform

        csv_path = Path(self.config.source.input_path)
        if not csv_path.exists():
            raise FileNotFoundError(f"Source CSV not found: {csv_path}")

        logger.info("Parsing header: %s", csv_path)
        header = parse_header(csv_path)

        logger.info("Reading and transforming data …")
        df = read_and_transform(csv_path, header)

        file_meta = {
            "chkxp.frequency":       self.config.metadata.frequency,
            "chkxp.period_start":    self.config.metadata.period_start,
            "chkxp.period_end":      self.config.metadata.period_end,
            "chkxp.detected_format": self.config.source.detected_format,
            "chkxp.entities": json.dumps(
                [e.name for e in self.config.entities], ensure_ascii=False
            ),
        }

        table = pa.Table.from_pandas(df, preserve_index=False)
        existing_meta = table.schema.metadata or {}
        merged_meta = {
            **existing_meta,
            **{k.encode(): v.encode() for k, v in file_meta.items()},
        }
        table = table.replace_schema_metadata(merged_meta)

        pq.write_table(table, self.data_path)
        logger.info(
            "Wrote %s  (%d rows × %d cols)",
            self.data_path, len(df), len(df.columns),
        )

    def _ingest_daily(self) -> None:
        from utils.chkxp_ingest.parser import parse_header, read_daily_data

        source_files = self.config.source.source_files
        if not source_files:
            raise ValueError(
                "config.source.source_files is empty for a daily_etf dataset."
            )

        common_codes = {item.code for item in self.config.items}
        logger.info(
            "Ingesting %d daily ETF files with %d common item codes …",
            len(source_files), len(common_codes),
        )

        frames: list[pd.DataFrame] = []
        for path_str in source_files:
            csv_path = Path(path_str)
            if not csv_path.exists():
                raise FileNotFoundError(f"Source CSV not found: {csv_path}")
            logger.info("  Reading %s", csv_path.name)
            header = parse_header(csv_path)
            df_part = read_daily_data(csv_path, header, common_item_codes=common_codes)
            frames.append(df_part)
            logger.info("    → %d rows, %d entities", len(df_part),
                        df_part["entity_code"].nunique())

        df = pd.concat(frames, ignore_index=True)
        df = df.sort_values(["entity_code", "date"]).reset_index(drop=True)

        logger.info(
            "Total: %d rows × %d cols, %d unique entities",
            len(df), len(df.columns), df["entity_code"].nunique(),
        )

        table = pa.Table.from_pandas(df, preserve_index=False)

        pad.write_dataset(
            table,
            base_dir=str(self.output_dir),
            format="parquet",
            partitioning=pad.partitioning(
                pa.schema([("year", pa.int16())]),
                flavor="hive",
            ),
            existing_data_behavior="overwrite_or_ignore",
        )
        logger.info("Wrote hive-partitioned dataset → %s", self.output_dir)

    # -- save_config() ------------------------------------------------------

    def save_config(self) -> None:
        """Persist the current config to YAML."""
        self.config.to_yaml(self.config_path)
        logger.info("Saved config: %s", self.config_path)

    # -- internals ----------------------------------------------------------

    def _build_meta_df(self) -> pd.DataFrame:
        """Build the ``_meta`` lineage table."""
        rows: list[dict] = []
        now = datetime.now(tz=timezone.utc).isoformat()

        # For multi-file daily datasets, emit one source_file per source.
        source_files = self.config.source.source_files or [
            self.config.source.input_path
        ]

        for src in source_files:
            for entity in self.config.entities:
                for item in self.config.items:
                    rows.append(
                        {
                            "source_file":     Path(src).name,
                            "source_hash":     self.config.source.source_hash,
                            "detected_format": self.config.source.detected_format,
                            "entity_name":     entity.name,
                            "entity_code":     entity.code,
                            "item_code":       item.code,
                            "field_code":      item.field_code,
                            "item_name":       item.name,
                            "frequency":       self.config.metadata.frequency,
                            "period_start":    self.config.metadata.period_start,
                            "period_end":      self.config.metadata.period_end,
                            "processed_at":    now,
                        }
                    )

        return pd.DataFrame(rows)

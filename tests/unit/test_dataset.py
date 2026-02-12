"""Unit tests for the Dataset class.

Tests load() filtering (column pruning, entity filter, date range)
and describe() against synthetic parquet files built from fixture CSVs.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from utils.chkxp_ingest import open as chkxp_open
from utils.chkxp_ingest.dataset import Dataset


# ---------------------------------------------------------------------------
# Helper: build a Dataset from a fixture CSV (runs ingest in tmp_path)
# ---------------------------------------------------------------------------


@pytest.fixture()
def single_ds(sample_single_entity_csv: Path, output_dir: Path) -> Dataset:
    """Ingest the single-entity fixture and return a Dataset handle."""
    return chkxp_open(sample_single_entity_csv, output_dir=output_dir)


@pytest.fixture()
def multi_ds(sample_multi_entity_csv: Path, output_dir: Path) -> Dataset:
    """Ingest the multi-entity fixture and return a Dataset handle."""
    return chkxp_open(sample_multi_entity_csv, output_dir=output_dir)


# ---------------------------------------------------------------------------
# load() — column pruning
# ---------------------------------------------------------------------------


class TestLoadColumnPruning:
    def test_items_filter_single(self, single_ds: Dataset) -> None:
        df = single_ds.load(items=["Intra시가"])
        assert set(df.columns) == {"datetime", "entity", "entity_code", "Intra시가"}

    def test_items_filter_multi(self, multi_ds: Dataset) -> None:
        df = multi_ds.load(items=["Intra시가"])
        assert "Intra종가" not in df.columns
        assert "Intra시가" in df.columns

    def test_load_all_columns(self, single_ds: Dataset) -> None:
        df = single_ds.load()
        assert set(df.columns) == {
            "datetime",
            "entity",
            "entity_code",
            "Intra시가",
            "Intra고가",
            "Intra저가",
        }


# ---------------------------------------------------------------------------
# load() — entity filter
# ---------------------------------------------------------------------------


class TestLoadEntityFilter:
    def test_filter_one_entity(self, multi_ds: Dataset) -> None:
        df = multi_ds.load(entities=["KOSPI200 선물 2603"])
        assert set(df["entity"].unique()) == {"KOSPI200 선물 2603"}
        # 3 data rows for this entity
        assert len(df) == 3

    def test_filter_nonexistent_entity(self, multi_ds: Dataset) -> None:
        df = multi_ds.load(entities=["DOES NOT EXIST"])
        assert len(df) == 0


# ---------------------------------------------------------------------------
# load() — date range filter
# ---------------------------------------------------------------------------


class TestLoadDateRange:
    def test_date_from(self, single_ds: Dataset) -> None:
        df = single_ds.load(date_from="2025-01-03")
        # Only 2025-01-03 rows should remain (2 rows)
        assert len(df) == 2
        assert (df["datetime"] >= pd.Timestamp("2025-01-03")).all()

    def test_date_to(self, single_ds: Dataset) -> None:
        df = single_ds.load(date_to="2025-01-02 09:02:00")
        # Only 09:01 and 09:02 on 2025-01-02
        assert len(df) == 2
        assert (df["datetime"] <= pd.Timestamp("2025-01-02 09:02:00")).all()

    def test_date_range_combined(self, single_ds: Dataset) -> None:
        df = single_ds.load(
            date_from="2025-01-02 09:02:00",
            date_to="2025-01-03 09:01:00",
        )
        # 09:02, 09:03 on Jan 2 + 09:01 on Jan 3 = 3 rows
        assert len(df) == 3


# ---------------------------------------------------------------------------
# describe()
# ---------------------------------------------------------------------------


class TestDescribe:
    def test_shape(self, single_ds: Dataset) -> None:
        info = single_ds.describe()
        assert info.shape == (5, 6)  # 5 rows, 6 cols

    def test_items(self, single_ds: Dataset) -> None:
        info = single_ds.describe()
        assert info.items == ["Intra시가", "Intra고가", "Intra저가"]

    def test_entities(self, multi_ds: Dataset) -> None:
        info = multi_ds.describe()
        assert set(info.entities) == {"KOSPI200 선물 2603", "K200 스프레드 6366"}

    def test_frequency(self, single_ds: Dataset) -> None:
        info = single_ds.describe()
        assert info.frequency == "1M"

    def test_format_name(self, multi_ds: Dataset) -> None:
        info = multi_ds.describe()
        assert info.format_name == "multi_entity"


# ---------------------------------------------------------------------------
# load_meta()
# ---------------------------------------------------------------------------


class TestLoadMeta:
    def test_meta_not_empty(self, single_ds: Dataset) -> None:
        meta = single_ds.load_meta()
        assert len(meta) > 0

    def test_meta_columns(self, single_ds: Dataset) -> None:
        meta = single_ds.load_meta()
        expected = {
            "source_file",
            "source_hash",
            "detected_format",
            "entity_name",
            "entity_code",
            "item_code",
            "field_code",
            "item_name",
            "frequency",
            "period_start",
            "period_end",
            "processed_at",
        }
        assert set(meta.columns) == expected

    def test_meta_row_count(self, multi_ds: Dataset) -> None:
        """2 entities × 2 items = 4 rows."""
        meta = multi_ds.load_meta()
        assert len(meta) == 4


# ---------------------------------------------------------------------------
# is_built / FileNotFoundError guard
# ---------------------------------------------------------------------------


class TestGuards:
    def test_load_before_ingest_raises(self, tmp_path: Path) -> None:
        from utils.chkxp_ingest.config import ChkxpConfig, OutputConfig

        cfg = ChkxpConfig(output=OutputConfig(output_dir=str(tmp_path / "empty")))
        ds = Dataset(config=cfg, config_path=tmp_path / "dummy.yaml")
        assert not ds.is_built
        with pytest.raises(FileNotFoundError):
            ds.load()

    def test_describe_before_ingest_raises(self, tmp_path: Path) -> None:
        from utils.chkxp_ingest.config import ChkxpConfig, OutputConfig

        cfg = ChkxpConfig(output=OutputConfig(output_dir=str(tmp_path / "empty")))
        ds = Dataset(config=cfg, config_path=tmp_path / "dummy.yaml")
        with pytest.raises(FileNotFoundError):
            ds.describe()

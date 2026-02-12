"""Integration tests -- end-to-end ingestion of real Check Expert CSVs.

These tests require the actual data files in ``data/raw/chkxp/``.
Mark: ``@pytest.mark.integration`` so they can be run separately::

    uv run pytest tests/integration/ -v -m integration
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd
import pytest

from utils.chkxp_ingest import open as chkxp_open
from utils.chkxp_ingest.dataset import Dataset

# ---------------------------------------------------------------------------
# Paths to real CSV files (relative to project root)
# ---------------------------------------------------------------------------

_RAW_DIR = Path("data/raw/chkxp")

_SINGLE_ETF_1M = _RAW_DIR / "chkxp_etf(kodex200)_(1m)_ohlcvNAV.csv"
_SINGLE_ETF_10S = _RAW_DIR / "chkxp_etf(kodex200)_(10s)_ohlcvNAVlob.csv"
_MULTI_KP200 = (
    _RAW_DIR / "kp200_(fut)(mini)(v)_(1m)_from(20250101)_to(20260207).csv"
)
_MULTI_KTB = (
    _RAW_DIR
    / "ktb_(3)(10)_(fut)(spread)(2nd)_(1m)_from(20200101)_to(20260207).csv"
)

# Skip entire module if real data is not available.
pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not _SINGLE_ETF_1M.exists(),
        reason="Real Check Expert CSV files not found in data/raw/chkxp/",
    ),
]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def int_output_dir(tmp_path: Path) -> Path:
    """Temp output dir for integration test artifacts."""
    return tmp_path / "db_chkxp"


# ---------------------------------------------------------------------------
# Single-entity tests
# ---------------------------------------------------------------------------


class TestSingleEntityETF1M:
    """End-to-end: 1-minute KODEX 200 ETF data."""

    def test_open_creates_parquet(self, int_output_dir: Path) -> None:
        out = int_output_dir / "etf_1m"
        ds = chkxp_open(str(_SINGLE_ETF_1M), output_dir=str(out))
        assert ds.is_built
        assert (out / "default.parquet").exists()
        assert (out / "_meta.parquet").exists()

    def test_config_yaml_created(self, int_output_dir: Path) -> None:
        out = int_output_dir / "etf_1m"
        ds = chkxp_open(str(_SINGLE_ETF_1M), output_dir=str(out))
        assert ds.config_path.exists()
        assert ds.config_path.suffix == ".yaml"

    def test_load_returns_dataframe(self, int_output_dir: Path) -> None:
        out = int_output_dir / "etf_1m"
        ds = chkxp_open(str(_SINGLE_ETF_1M), output_dir=str(out))
        df = ds.load()
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        assert "datetime" in df.columns
        assert "entity" in df.columns

    def test_single_entity_name(self, int_output_dir: Path) -> None:
        out = int_output_dir / "etf_1m"
        ds = chkxp_open(str(_SINGLE_ETF_1M), output_dir=str(out))
        df = ds.load()
        assert df["entity"].nunique() == 1
        assert df["entity"].iloc[0] == "KODEX 200"

    def test_describe(self, int_output_dir: Path) -> None:
        out = int_output_dir / "etf_1m"
        ds = chkxp_open(str(_SINGLE_ETF_1M), output_dir=str(out))
        info = ds.describe()
        assert info.frequency == "1M"
        assert info.format_name == "single_entity"
        assert len(info.items) > 0


class TestSingleEntityETF10S:
    """End-to-end: 10-second KODEX 200 ETF data."""

    def test_open_creates_parquet(self, int_output_dir: Path) -> None:
        out = int_output_dir / "etf_10s"
        ds = chkxp_open(str(_SINGLE_ETF_10S), output_dir=str(out))
        assert ds.is_built

    def test_describe_frequency(self, int_output_dir: Path) -> None:
        out = int_output_dir / "etf_10s"
        ds = chkxp_open(str(_SINGLE_ETF_10S), output_dir=str(out))
        info = ds.describe()
        assert info.frequency == "10S"


# ---------------------------------------------------------------------------
# Multi-entity tests
# ---------------------------------------------------------------------------


class TestMultiEntityKP200:
    """End-to-end: KP200 futures (7 entities)."""

    def test_open_creates_parquet(self, int_output_dir: Path) -> None:
        out = int_output_dir / "kp200"
        ds = chkxp_open(str(_MULTI_KP200), output_dir=str(out))
        assert ds.is_built

    def test_entity_count(self, int_output_dir: Path) -> None:
        out = int_output_dir / "kp200"
        ds = chkxp_open(str(_MULTI_KP200), output_dir=str(out))
        info = ds.describe()
        assert len(info.entities) == 7

    def test_load_filtered_entity(self, int_output_dir: Path) -> None:
        out = int_output_dir / "kp200"
        ds = chkxp_open(str(_MULTI_KP200), output_dir=str(out))
        df = ds.load(entities=["KOSPI200 선물 2603"])
        assert df["entity"].nunique() == 1
        assert len(df) > 0

    def test_load_filtered_items(self, int_output_dir: Path) -> None:
        out = int_output_dir / "kp200"
        ds = chkxp_open(str(_MULTI_KP200), output_dir=str(out))
        df = ds.load(items=["Intra시가", "Intra종가"])
        value_cols = [c for c in df.columns if c not in ("datetime", "entity", "entity_code")]
        assert set(value_cols) == {"Intra시가", "Intra종가"}


class TestMultiEntityKTB:
    """End-to-end: KTB futures (6 entities)."""

    def test_open_creates_parquet(self, int_output_dir: Path) -> None:
        out = int_output_dir / "ktb"
        ds = chkxp_open(str(_MULTI_KTB), output_dir=str(out))
        assert ds.is_built

    def test_entity_count(self, int_output_dir: Path) -> None:
        out = int_output_dir / "ktb"
        ds = chkxp_open(str(_MULTI_KTB), output_dir=str(out))
        info = ds.describe()
        assert len(info.entities) == 6


# ---------------------------------------------------------------------------
# Idempotency and rebuild
# ---------------------------------------------------------------------------


class TestIdempotency:
    """Verify open() skips rebuild when output exists."""

    def test_idempotent_reopen(self, int_output_dir: Path) -> None:
        out = int_output_dir / "etf_1m"
        ds1 = chkxp_open(str(_SINGLE_ETF_1M), output_dir=str(out))
        mtime_before = (out / "default.parquet").stat().st_mtime

        # Second open — should NOT rebuild.
        ds2 = chkxp_open(str(_SINGLE_ETF_1M), output_dir=str(out))
        mtime_after = (out / "default.parquet").stat().st_mtime

        assert mtime_before == mtime_after
        assert ds2.is_built

    def test_force_rebuild(self, int_output_dir: Path) -> None:
        out = int_output_dir / "etf_1m"
        ds1 = chkxp_open(str(_SINGLE_ETF_1M), output_dir=str(out))
        mtime_before = (out / "default.parquet").stat().st_mtime

        # Force rebuild — file mtime must change.
        ds2 = chkxp_open(str(_SINGLE_ETF_1M), output_dir=str(out), force=True)
        mtime_after = (out / "default.parquet").stat().st_mtime

        assert mtime_after > mtime_before


# ---------------------------------------------------------------------------
# Open from YAML
# ---------------------------------------------------------------------------


class TestOpenFromYAML:
    """Verify that opening from a generated YAML config works."""

    def test_open_from_yaml(self, int_output_dir: Path) -> None:
        out = int_output_dir / "etf_1m"
        ds1 = chkxp_open(str(_SINGLE_ETF_1M), output_dir=str(out))

        # Re-open via the YAML config.
        ds2 = chkxp_open(str(ds1.config_path))
        assert ds2.is_built

        df1 = ds1.load()
        df2 = ds2.load()
        pd.testing.assert_frame_equal(df1, df2)


# ---------------------------------------------------------------------------
# Load with date range on real data
# ---------------------------------------------------------------------------


class TestLoadDateRangeReal:
    def test_date_range_filter(self, int_output_dir: Path) -> None:
        out = int_output_dir / "etf_1m"
        ds = chkxp_open(str(_SINGLE_ETF_1M), output_dir=str(out))
        df_all = ds.load()
        df_sub = ds.load(date_from="2025-12-01", date_to="2025-12-31")
        assert len(df_sub) <= len(df_all)
        if len(df_sub) > 0:
            assert (df_sub["datetime"] >= pd.Timestamp("2025-12-01")).all()
            assert (df_sub["datetime"] <= pd.Timestamp("2025-12-31")).all()

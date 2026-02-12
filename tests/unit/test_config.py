"""Unit tests for the config module.

Tests Pydantic model validation and YAML round-trip serialization.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from utils.chkxp_ingest.config import (
    ChkxpConfig,
    EntityInfo,
    ItemInfo,
    MetadataConfig,
    OutputConfig,
    SourceConfig,
)


def _make_sample_config() -> ChkxpConfig:
    """Build a representative config object for testing."""
    return ChkxpConfig(
        source=SourceConfig(
            input_path="data/raw/chkxp/test.csv",
            source_hash="abc123",
            detected_format="multi_entity",
        ),
        metadata=MetadataConfig(
            instrument_codes=["K2FA020*005", "K2FS020*005"],
            item_codes=["20005", "20008"],
            frequency="1M",
            period_start="20250101",
            period_end="20260207",
            start_time="0900",
            end_time="1515",
        ),
        output=OutputConfig(
            output_dir="data/db/chkxp/test",
            output_format="parquet",
        ),
        entities=[
            EntityInfo(name="KOSPI200 선물 2603", code="K2FA020*005"),
            EntityInfo(name="K200 스프레드 6366", code="K2FS020*005"),
        ],
        items=[
            ItemInfo(code="20005", field_code="F20005", name="Intra시가"),
            ItemInfo(code="20008", field_code="F20008", name="Intra종가"),
        ],
    )


class TestChkxpConfig:
    def test_round_trip_yaml(self, tmp_path: Path) -> None:
        """Serialize → deserialize → assert equality."""
        original = _make_sample_config()
        yaml_path = tmp_path / "config.yaml"

        original.to_yaml(yaml_path)
        loaded = ChkxpConfig.from_yaml(yaml_path)

        assert loaded == original

    def test_yaml_file_created(self, tmp_path: Path) -> None:
        cfg = _make_sample_config()
        yaml_path = tmp_path / "sub" / "nested" / "config.yaml"
        cfg.to_yaml(yaml_path)
        assert yaml_path.exists()

    def test_yaml_unicode_preserved(self, tmp_path: Path) -> None:
        """Korean entity names must survive the YAML round-trip."""
        cfg = _make_sample_config()
        yaml_path = tmp_path / "config.yaml"
        cfg.to_yaml(yaml_path)

        text = yaml_path.read_text(encoding="utf-8")
        assert "KOSPI200 선물 2603" in text
        assert "Intra시가" in text

    def test_default_values(self) -> None:
        """Minimal config with defaults should not raise."""
        cfg = ChkxpConfig()
        assert cfg.source.detected_format == "single_entity"
        assert cfg.output.output_format == "parquet"
        assert cfg.entities == []
        assert cfg.items == []

    def test_invalid_format_rejected(self) -> None:
        """detected_format must be 'single_entity' or 'multi_entity'."""
        with pytest.raises(ValidationError):
            SourceConfig(
                input_path="x.csv",
                detected_format="unknown_format",  # type: ignore[arg-type]
            )

    def test_invalid_output_format_rejected(self) -> None:
        """output_format must be 'parquet'."""
        with pytest.raises(ValidationError):
            OutputConfig(
                output_dir="some/dir",
                output_format="csv",  # type: ignore[arg-type]
            )

"""ChkxpConfig pydantic model + YAML I/O.

Stores all metadata extracted from a Check Expert CSV header,
entity/item mappings, and output configuration. Serialized to YAML
for config-first workflows (mirroring fn-dg6-ingest).
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Sub-models
# ---------------------------------------------------------------------------


class SourceConfig(BaseModel):
    """Provenance of the raw CSV file."""

    input_path: str = ""
    source_hash: str = ""
    detected_format: Literal["single_entity", "multi_entity"] = "single_entity"


class MetadataConfig(BaseModel):
    """Header-level metadata parsed from rows 1-9."""

    instrument_codes: list[str] = Field(default_factory=list)
    item_codes: list[str] = Field(default_factory=list)
    frequency: str = ""  # e.g. "1M", "10S"
    period_start: str = ""
    period_end: str = ""
    start_time: str = "0900"
    end_time: str = "1515"


class OutputConfig(BaseModel):
    """Where and how the parquet output is written."""

    output_dir: str = ""
    output_format: Literal["parquet"] = "parquet"


class EntityInfo(BaseModel):
    """One entity (instrument) found in the CSV."""

    name: str
    code: str


class ItemInfo(BaseModel):
    """One data item (column) found in the CSV."""

    code: str  # raw item code, e.g. "20005"
    field_code: str  # prefixed code from row 10, e.g. "F20005"
    name: str  # human-readable name from row 11, e.g. "Intra시가"


# ---------------------------------------------------------------------------
# Root config
# ---------------------------------------------------------------------------


class ChkxpConfig(BaseModel):
    """Root configuration for a single Check Expert dataset."""

    source: SourceConfig = Field(default_factory=SourceConfig)
    metadata: MetadataConfig = Field(default_factory=MetadataConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    entities: list[EntityInfo] = Field(default_factory=list)
    items: list[ItemInfo] = Field(default_factory=list)

    # -- YAML I/O ----------------------------------------------------------

    def to_yaml(self, path: str | Path) -> None:
        """Serialize config to a YAML file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(
                self.model_dump(),
                f,
                default_flow_style=False,
                allow_unicode=True,
                sort_keys=False,
            )

    @classmethod
    def from_yaml(cls, path: str | Path) -> ChkxpConfig:
        """Deserialize config from a YAML file."""
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return cls.model_validate(data)

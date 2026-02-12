"""Header + data parsing for Check Expert CSV files.

Handles two sub-formats exported by Check Expert (체크전문가):

* **single_entity** -- one instrument with many items (e.g. ETF OHLCV + iNAV)
* **multi_entity** -- multiple instruments, each sharing the same item set
  (e.g. KP200 futures front/spread/back months)

CSV layout (all files):

    Row  0-8   key-value metadata (종목코드, 항목코드, 주기, …)
    Row  9     entity names interleaved with F-prefixed field codes
    Row 10     human-readable column names
    Row 11+    data
"""

from __future__ import annotations

import csv
import hashlib
import re
from dataclasses import dataclass, field as dc_field
from pathlib import Path

import pandas as pd

from utils.chkxp_ingest.config import (
    ChkxpConfig,
    EntityInfo,
    ItemInfo,
    MetadataConfig,
    OutputConfig,
    SourceConfig,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

HEADER_ROW_COUNT = 9  # rows 0-8 are key-value metadata
ENTITY_ROW_IDX = 9  # row 9: entity names + field codes
COLNAME_ROW_IDX = 10  # row 10: human-readable column names
DATA_START_ROW = 11  # data begins here

_FIELD_CODE_RE = re.compile(r"^F\d+$")


# ---------------------------------------------------------------------------
# Dataclasses returned by the parser
# ---------------------------------------------------------------------------


@dataclass
class EntityGroup:
    """One instrument block parsed from row 9/10 of the CSV."""

    name: str  # human-readable entity name (e.g. "KOSPI200 선물 2603")
    code: str  # instrument code from row 0 (e.g. "K2FA020*005")
    name_col: int  # column index of entity-name cell / 체결Intra생성시간
    data_cols: list[int] = dc_field(default_factory=list)  # indices of F-code cols
    item_field_codes: list[str] = dc_field(default_factory=list)  # F20005, …
    item_names: list[str] = dc_field(default_factory=list)  # Intra시가, …


@dataclass
class ParsedHeader:
    """Everything extracted from the first 11 rows of a Check Expert CSV."""

    raw_metadata: dict[str, str | list[str]]
    entity_groups: list[EntityGroup]
    detected_format: str  # "single_entity" | "multi_entity"

    # Unique items derived from the *first* entity group (representative).
    unique_item_codes: list[str]  # ["20005", "20006", …]
    unique_item_field_codes: list[str]  # ["F20005", "F20006", …]
    unique_item_names: list[str]  # ["Intra시가", "Intra고가", …]


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------


def _read_csv_rows(csv_path: Path, n_rows: int) -> list[list[str]]:
    """Read the first *n_rows* rows from a CSV using the stdlib csv reader."""
    rows: list[list[str]] = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for i, row in enumerate(reader):
            if i >= n_rows:
                break
            rows.append(row)
    return rows


def compute_file_hash(path: Path, algorithm: str = "sha256") -> str:
    """Return the hex-digest hash of a file (default SHA-256)."""
    h = hashlib.new(algorithm)
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Header parsing
# ---------------------------------------------------------------------------


def parse_header(csv_path: Path | str) -> ParsedHeader:
    """Parse rows 0-10 of a Check Expert CSV and return structured metadata.

    Steps:
    1. Rows 0-8 → key-value metadata dict.
    2. Row 9 → detect entity boundaries (non-F-code, non-empty cells)
       and map each entity to its column range + field codes.
    3. Row 10 → map field-code positions to human-readable item names.
    4. Format detection: >1 entity boundary → multi_entity, else single_entity.
    """
    csv_path = Path(csv_path)
    rows = _read_csv_rows(csv_path, COLNAME_ROW_IDX + 1)  # rows 0-10

    # -- 1. Metadata (rows 0-8) -------------------------------------------
    raw_metadata: dict[str, str | list[str]] = {}
    for row in rows[:HEADER_ROW_COUNT]:
        key = row[0].strip()
        values = [v.strip() for v in row[1:] if v.strip()]
        if not values:
            continue
        raw_metadata[key] = values[0] if len(values) == 1 else values

    # -- 2. Entity groups (row 9) -----------------------------------------
    row_entity = rows[ENTITY_ROW_IDX]
    row_colname = rows[COLNAME_ROW_IDX]

    # Instrument codes from row 0 (종목코드)
    instrument_codes = [v.strip() for v in rows[0][1:] if v.strip()]

    # Find entity-name positions: non-empty cells in row 9 that are NOT
    # F-prefixed field codes (and not col 0 which is a row label).
    entity_starts: list[tuple[int, str]] = []
    for i, cell in enumerate(row_entity):
        cell_stripped = cell.strip()
        if i == 0 or not cell_stripped:
            continue
        if not _FIELD_CODE_RE.match(cell_stripped):
            entity_starts.append((i, cell_stripped))

    detected_format = "multi_entity" if len(entity_starts) > 1 else "single_entity"

    # -- 3. Build EntityGroup objects --------------------------------------
    entity_groups: list[EntityGroup] = []
    for idx, (start_col, entity_name) in enumerate(entity_starts):
        # Data columns run from start_col+1 to the next entity (or row end).
        end_col = (
            entity_starts[idx + 1][0]
            if idx + 1 < len(entity_starts)
            else len(row_entity)
        )

        data_cols: list[int] = []
        item_field_codes: list[str] = []
        item_names: list[str] = []
        for col in range(start_col + 1, end_col):
            fc = row_entity[col].strip()
            if _FIELD_CODE_RE.match(fc):
                data_cols.append(col)
                item_field_codes.append(fc)
                name = row_colname[col].strip() if col < len(row_colname) else fc
                item_names.append(name)

        code = instrument_codes[idx] if idx < len(instrument_codes) else ""

        entity_groups.append(
            EntityGroup(
                name=entity_name,
                code=code,
                name_col=start_col,
                data_cols=data_cols,
                item_field_codes=item_field_codes,
                item_names=item_names,
            )
        )

    # -- 4. Unique item catalogue (from first entity group) ---------------
    first = entity_groups[0]
    unique_item_codes = [fc.lstrip("F") for fc in first.item_field_codes]

    return ParsedHeader(
        raw_metadata=raw_metadata,
        entity_groups=entity_groups,
        detected_format=detected_format,
        unique_item_codes=unique_item_codes,
        unique_item_field_codes=list(first.item_field_codes),
        unique_item_names=list(first.item_names),
    )


# ---------------------------------------------------------------------------
# Data transformation
# ---------------------------------------------------------------------------


def read_and_transform(csv_path: Path | str, header: ParsedHeader) -> pd.DataFrame:
    """Read data rows from the CSV and produce a tidy (long) DataFrame.

    Output columns: ``datetime | entity | entity_code | <item_1> | … | <item_N>``

    For **multi_entity** files the wide column groups are stacked vertically.
    For **single_entity** files the result has one entity value throughout.

    The redundant ``체결Intra생성시간`` column (time-only duplicate of the
    datetime index) is always dropped.
    """
    csv_path = Path(csv_path)

    raw = pd.read_csv(
        csv_path,
        skiprows=DATA_START_ROW,
        header=None,
        thousands=",",
        na_values=["", " "],
        low_memory=False,
    )

    # Column 0 is the full datetime timestamp.
    datetime_series = pd.to_datetime(raw.iloc[:, 0])

    frames: list[pd.DataFrame] = []

    for eg in header.entity_groups:
        # Slice only the data-item columns for this entity.
        sub = raw.iloc[:, eg.data_cols].copy()
        sub.columns = eg.item_names

        # Coerce every value column to numeric (safety net for mixed types).
        for col_name in sub.columns:
            sub[col_name] = pd.to_numeric(sub[col_name], errors="coerce")

        sub.insert(0, "datetime", datetime_series.values)
        sub.insert(1, "entity", eg.name)
        sub.insert(2, "entity_code", eg.code)

        frames.append(sub)

    df = pd.concat(frames, ignore_index=True)

    # Deterministic ordering: time-major, entity-minor.
    df = df.sort_values(["datetime", "entity"]).reset_index(drop=True)

    return df


# ---------------------------------------------------------------------------
# Config builder
# ---------------------------------------------------------------------------


def build_config(
    csv_path: Path | str,
    header: ParsedHeader,
    output_dir: Path | str,
) -> ChkxpConfig:
    """Construct a :class:`ChkxpConfig` from a parsed header."""
    csv_path = Path(csv_path)
    output_dir = Path(output_dir)
    meta = header.raw_metadata

    raw_item_codes = meta.get("항목코드", [])
    if isinstance(raw_item_codes, str):
        raw_item_codes = [raw_item_codes]

    return ChkxpConfig(
        source=SourceConfig(
            input_path=str(csv_path),
            source_hash=compute_file_hash(csv_path),
            detected_format=header.detected_format,
        ),
        metadata=MetadataConfig(
            instrument_codes=[eg.code for eg in header.entity_groups],
            item_codes=raw_item_codes,
            frequency=str(meta.get("주기", "")),
            period_start=str(meta.get("시작일자", "")),
            period_end=str(meta.get("종료일자", "")),
            start_time=str(meta.get("시작시간", "0900")),
            end_time=str(meta.get("종료시간", "1515")),
        ),
        output=OutputConfig(
            output_dir=str(output_dir),
            output_format="parquet",
        ),
        entities=[
            EntityInfo(name=eg.name, code=eg.code)
            for eg in header.entity_groups
        ],
        items=[
            ItemInfo(code=code, field_code=fc, name=name)
            for code, fc, name in zip(
                header.unique_item_codes,
                header.unique_item_field_codes,
                header.unique_item_names,
                strict=True,
            )
        ],
    )

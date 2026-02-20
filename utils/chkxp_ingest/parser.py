"""Header + data parsing for Check Expert CSV files.

Handles three sub-formats exported by Check Expert (체크전문가):

* **single_entity** -- one instrument with many items (e.g. ETF OHLCV + iNAV)
* **multi_entity** -- multiple instruments, each sharing the same item set
  (e.g. KP200 futures front/spread/back months)
* **daily_etf** -- 일별시계열 multi-block layout: N independent horizontal blocks
  in the same CSV, each with its own date axis, entity groups, and item set

Intraday CSV layout (single_entity / multi_entity):

    Row  0-8   key-value metadata (종목코드, 항목코드, 주기, …)
    Row  9     entity names interleaved with F-prefixed field codes
    Row 10     human-readable column names
    Row 11+    data

Daily ETF CSV layout (daily_etf):

    Row 0      종목코드 | codes… | [종목코드 block-2 marker] | …
    Row 1      항목코드 | codes… | [항목코드 block-2 marker] | …
    Row 2      주기     | D
    Row 3      시작일자 | YYYYMMDD
    Row 4      종료일자
    Row 5      데이터개수 | -1
    Row 6      정렬     | ASC
    Row 7      [entity_name] [F-code]… [entity_name] … | [시간/항목 block-2] …
    Row 8      시간/항목 | col-names … | 시간/항목 | col-names … | …
    Row 9+     data (each block has its own date column)
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
# Constants — intraday format (unchanged)
# ---------------------------------------------------------------------------

HEADER_ROW_COUNT = 9   # rows 0-8 are key-value metadata
ENTITY_ROW_IDX   = 9   # row 9: entity names + field codes
COLNAME_ROW_IDX  = 10  # row 10: human-readable column names
DATA_START_ROW   = 11  # data begins here

# Constants — daily ETF format
DAILY_ENTITY_ROW_IDX  = 7   # row 7: entity names + F-codes (per-block)
DAILY_COLNAME_ROW_IDX = 8   # row 8: "시간/항목" + human-readable names (per-block)
DAILY_DATA_START_ROW  = 9   # data begins here

_FIELD_CODE_RE = re.compile(r"^F\d+$")


# ---------------------------------------------------------------------------
# Dataclasses returned by the parser
# ---------------------------------------------------------------------------


@dataclass
class EntityGroup:
    """One instrument block parsed from the entity/colname rows of a CSV."""

    name: str               # human-readable entity name
    code: str               # instrument code from row 0
    name_col: int           # column index of entity-name cell
    time_col: int = 0       # daily format: absolute col of this block's 시간/항목
    data_cols: list[int]    = dc_field(default_factory=list)
    item_field_codes: list[str] = dc_field(default_factory=list)  # F20005, …
    item_names: list[str]   = dc_field(default_factory=list)


@dataclass
class ParsedHeader:
    """Everything extracted from the header rows of a Check Expert CSV."""

    raw_metadata: dict[str, str | list[str]]
    entity_groups: list[EntityGroup]
    detected_format: str  # "single_entity" | "multi_entity" | "daily_etf"

    # Unique items derived from the *first* entity group (representative).
    unique_item_codes: list[str]         # ["20005", "20006", …]
    unique_item_field_codes: list[str]   # ["F20005", "F20006", …]
    unique_item_names: list[str]         # ["Intra시가", "Intra고가", …]


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


def _safe_cell(rows: list[list[str]], row: int, col: int) -> str:
    """Return stripped cell value or '' if out of bounds."""
    if row >= len(rows) or col >= len(rows[row]):
        return ""
    return rows[row][col].strip()


# ---------------------------------------------------------------------------
# Format detection
# ---------------------------------------------------------------------------


def _is_daily_format(rows: list[list[str]]) -> bool:
    """Return True for 일별시계열 (daily) layout.

    Discriminator: row 2 is 주기 and its value is 'D'.  Intraday files use
    frequency strings like '1M' or '10S'.
    """
    if len(rows) < 3:
        return False
    return _safe_cell(rows, 2, 1) == "D"


# ---------------------------------------------------------------------------
# Header parsing — daily ETF format
# ---------------------------------------------------------------------------


def _parse_daily_header(rows: list[list[str]], csv_path: Path) -> ParsedHeader:
    """Parse the multi-block header of a 일별시계열 ETF CSV.

    Algorithm
    ---------
    1. Find block time-column positions: scan row 8 for '시간/항목'.
    2. For each block:
       a. Determine pattern: if row 1 at ``tc+1`` is '12506' → KODEX (skip
          the entity-name col in data; it holds the redundant 입회일 value).
          Otherwise → RISE/SOL/TIGER/ACE (entity-name col holds 현재가).
       b. Scan row 7 for entity names (non-empty, non-F-code cells after tc).
       c. For each entity, collect data_cols, item_field_codes, item_names.
    3. Extract metadata from rows 2-6 (주기, 시작일자, …).
    """
    row_colname = rows[DAILY_COLNAME_ROW_IDX]   # row 8

    # -- 1. Block boundaries: positions of '시간/항목' in row 8 --------------
    block_time_cols: list[int] = [
        i for i, cell in enumerate(row_colname)
        if cell.strip() == "시간/항목"
    ]
    if not block_time_cols:
        raise ValueError(
            f"No '시간/항목' found in row {DAILY_COLNAME_ROW_IDX} of {csv_path}. "
            "Is this really a daily ETF CSV?"
        )

    row_width = len(row_colname)
    block_ends = block_time_cols[1:] + [row_width]

    entity_groups: list[EntityGroup] = []

    for tc, tc_next in zip(block_time_cols, block_ends):
        # -- 2a. Detect KODEX pattern: first item code after tc in row 1 ----
        first_item_code = _safe_cell(rows, 1, tc + 1)
        is_kodex = (first_item_code == "12506")

        # -- 2b. Entity names in row 7 within this block --------------------
        entity_starts: list[tuple[int, str]] = []
        for c in range(tc + 1, tc_next):
            cell = _safe_cell(rows, DAILY_ENTITY_ROW_IDX, c)
            if cell and not _FIELD_CODE_RE.match(cell):
                entity_starts.append((c, cell))

        # -- 2c. Build EntityGroup per entity --------------------------------
        for eg_idx, (start_col, entity_name) in enumerate(entity_starts):
            eg_end = (
                entity_starts[eg_idx + 1][0]
                if eg_idx + 1 < len(entity_starts)
                else tc_next
            )

            if is_kodex:
                # Entity-name col holds 입회일 string data → skip it.
                # F-codes in row 7 at start_col+1 … eg_end are the real data.
                data_cols: list[int] = []
                item_field_codes: list[str] = []
                item_names_list: list[str] = []
                for c in range(start_col + 1, eg_end):
                    fc = _safe_cell(rows, DAILY_ENTITY_ROW_IDX, c)
                    if _FIELD_CODE_RE.match(fc):
                        data_cols.append(c)
                        item_field_codes.append(fc)
                        item_names_list.append(_safe_cell(rows, DAILY_COLNAME_ROW_IDX, c))
            else:
                # Entity-name col holds 현재가 (F15001) in actual data rows.
                # Include it; derive F-code from item codes list at that offset.
                col_name_at_start = _safe_cell(rows, DAILY_COLNAME_ROW_IDX, start_col)
                data_cols = [start_col]
                item_field_codes = ["F15001"]
                item_names_list = [col_name_at_start or "현재가"]
                for c in range(start_col + 1, eg_end):
                    fc = _safe_cell(rows, DAILY_ENTITY_ROW_IDX, c)
                    if _FIELD_CODE_RE.match(fc):
                        data_cols.append(c)
                        item_field_codes.append(fc)
                        item_names_list.append(_safe_cell(rows, DAILY_COLNAME_ROW_IDX, c))

            # Instrument code: row 0 at start_col holds the code for this entity.
            code = _safe_cell(rows, 0, start_col)
            # Fallback: scan row 0 within block range, pick by index.
            if not code or code in ("종목코드", "항목코드"):
                block_codes = [
                    _safe_cell(rows, 0, c)
                    for c in range(tc + 1, tc_next)
                    if _safe_cell(rows, 0, c) and _safe_cell(rows, 0, c) not in ("종목코드", "항목코드")
                ]
                code = block_codes[eg_idx] if eg_idx < len(block_codes) else ""

            entity_groups.append(
                EntityGroup(
                    name=entity_name,
                    code=code,
                    name_col=start_col,
                    time_col=tc,
                    data_cols=data_cols,
                    item_field_codes=item_field_codes,
                    item_names=item_names_list,
                )
            )

    # -- 3. Metadata from rows 2-6 (key in col 0, value in col 1) ----------
    raw_metadata: dict[str, str | list[str]] = {}
    for row in rows[2:7]:
        key = row[0].strip() if row else ""
        val = row[1].strip() if len(row) > 1 else ""
        if key and val:
            raw_metadata[key] = val

    first = entity_groups[0]
    unique_item_codes = [fc.lstrip("F") for fc in first.item_field_codes]

    return ParsedHeader(
        raw_metadata=raw_metadata,
        entity_groups=entity_groups,
        detected_format="daily_etf",
        unique_item_codes=unique_item_codes,
        unique_item_field_codes=list(first.item_field_codes),
        unique_item_names=list(first.item_names),
    )


# ---------------------------------------------------------------------------
# Header parsing — intraday format (single_entity / multi_entity)
# ---------------------------------------------------------------------------


def _parse_intraday_header(rows: list[list[str]]) -> ParsedHeader:
    """Parse rows 0-10 of an intraday Check Expert CSV."""
    # -- 1. Metadata (rows 0-8) -------------------------------------------
    raw_metadata: dict[str, str | list[str]] = {}
    for row in rows[:HEADER_ROW_COUNT]:
        key = row[0].strip()
        values = [v.strip() for v in row[1:] if v.strip()]
        if not values:
            continue
        raw_metadata[key] = values[0] if len(values) == 1 else values

    # -- 2. Entity groups (row 9) -----------------------------------------
    row_entity  = rows[ENTITY_ROW_IDX]
    row_colname = rows[COLNAME_ROW_IDX]

    instrument_codes = [v.strip() for v in rows[0][1:] if v.strip()]

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
                time_col=0,
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
# Public header parsing entry-point
# ---------------------------------------------------------------------------


def parse_header(csv_path: Path | str) -> ParsedHeader:
    """Parse the header of any Check Expert CSV and return structured metadata.

    Dispatches to the appropriate sub-parser based on the ``주기`` value in
    row 2: ``'D'`` → daily ETF multi-block format; anything else → intraday.
    """
    csv_path = Path(csv_path)
    # Read enough rows for both formats (intraday needs 11, daily needs 9).
    rows = _read_csv_rows(csv_path, COLNAME_ROW_IDX + 1)  # rows 0-10

    if _is_daily_format(rows):
        return _parse_daily_header(rows, csv_path)

    return _parse_intraday_header(rows)


# ---------------------------------------------------------------------------
# Data transformation — intraday
# ---------------------------------------------------------------------------


def read_and_transform(csv_path: Path | str, header: ParsedHeader) -> pd.DataFrame:
    """Read intraday data rows and produce a tidy (long) DataFrame.

    Output columns: ``datetime | entity | entity_code | <item_1> | … | <item_N>``

    For **multi_entity** files the wide column groups are stacked vertically.
    For **single_entity** files the result has one entity value throughout.
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

    datetime_series = pd.to_datetime(raw.iloc[:, 0])

    frames: list[pd.DataFrame] = []

    for eg in header.entity_groups:
        sub = raw.iloc[:, eg.data_cols].copy()
        sub.columns = eg.item_names

        for col_name in sub.columns:
            sub[col_name] = pd.to_numeric(sub[col_name], errors="coerce")

        sub.insert(0, "datetime", datetime_series.values)
        sub.insert(1, "entity", eg.name)
        sub.insert(2, "entity_code", eg.code)

        frames.append(sub)

    df = pd.concat(frames, ignore_index=True)
    df = df.sort_values(["datetime", "entity"]).reset_index(drop=True)

    return df


# ---------------------------------------------------------------------------
# Data transformation — daily ETF
# ---------------------------------------------------------------------------


def read_daily_data(
    csv_path: Path | str,
    header: ParsedHeader,
    common_item_codes: set[str] | None = None,
) -> pd.DataFrame:
    """Read 일별시계열 data rows and produce a tidy (long) DataFrame.

    Output columns: ``date | entity | entity_code | year | <item_1> | … | <item_N>``

    Each horizontal block in the CSV has its own date axis (``eg.time_col``).
    Rows where the date is NaT (entity not yet listed on that date) are dropped.

    Parameters
    ----------
    csv_path:
        Path to the raw daily ETF CSV.
    header:
        Parsed header from ``parse_header()``.
    common_item_codes:
        Optional set of raw item codes (e.g. ``{'15001', '15023', …}``).
        When provided, only columns whose ``item_field_code`` matches are kept.
        This is used during multi-file merging to align schemas.
    """
    csv_path = Path(csv_path)

    raw = pd.read_csv(
        csv_path,
        skiprows=DAILY_DATA_START_ROW,
        header=None,
        thousands=",",
        na_values=["", " "],
        low_memory=False,
    )

    frames: list[pd.DataFrame] = []

    for eg in header.entity_groups:
        # Filter to common item codes if specified.
        if common_item_codes is not None:
            triples = [
                (col, fc, name)
                for col, fc, name in zip(
                    eg.data_cols, eg.item_field_codes, eg.item_names
                )
                if fc.lstrip("F") in common_item_codes
            ]
        else:
            triples = list(zip(eg.data_cols, eg.item_field_codes, eg.item_names))

        if not triples:
            continue

        sel_cols  = [t[0] for t in triples]
        sel_names = [t[2] for t in triples]

        # Use this block's dedicated date column.
        date_series = pd.to_datetime(raw.iloc[:, eg.time_col], errors="coerce")

        sub = raw.iloc[:, sel_cols].copy()
        sub.columns = sel_names

        for col_name in sub.columns:
            sub[col_name] = pd.to_numeric(sub[col_name], errors="coerce")

        sub.insert(0, "date", date_series.dt.date)
        sub.insert(1, "entity", eg.name)
        sub.insert(2, "entity_code", eg.code)

        # Drop rows outside this entity's listing period (NaT date).
        sub = sub.dropna(subset=["date"])
        frames.append(sub)

    if not frames:
        raise ValueError(f"No data extracted from {csv_path}. Check item code filter.")

    df = pd.concat(frames, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"])
    df["year"] = df["date"].dt.year.astype("int16")

    return df.sort_values(["entity_code", "date"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Config builders
# ---------------------------------------------------------------------------


def build_config(
    csv_path: Path | str,
    header: ParsedHeader,
    output_dir: Path | str,
) -> ChkxpConfig:
    """Construct a :class:`ChkxpConfig` from a single-file parsed header."""
    csv_path   = Path(csv_path)
    output_dir = Path(output_dir)
    meta       = header.raw_metadata

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
            frequency_type="intraday",
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


def build_etf_daily_config(
    csv_paths: list[Path],
    headers: list[ParsedHeader],
    common_item_codes: list[str],
    common_item_field_codes: list[str],
    common_item_names: list[str],
    output_dir: Path,
) -> ChkxpConfig:
    """Construct a merged :class:`ChkxpConfig` for the daily ETF dataset.

    Parameters
    ----------
    csv_paths:
        All source CSV paths (4 files).
    headers:
        Parsed headers for each CSV (same order as csv_paths).
    common_item_codes, common_item_field_codes, common_item_names:
        Intersection of item codes across all source files, plus their
        F-code and human-readable name equivalents.
    output_dir:
        Where to write the hive-partitioned parquet dataset.
    """
    all_entities: list[EntityInfo] = []
    seen_codes: set[str] = set()
    for header in headers:
        for eg in header.entity_groups:
            if eg.code not in seen_codes:
                all_entities.append(EntityInfo(name=eg.name, code=eg.code))
                seen_codes.add(eg.code)

    # Use metadata from the first file (representative).
    meta = headers[0].raw_metadata

    period_starts = [h.raw_metadata.get("시작일자", "") for h in headers]
    period_ends   = [h.raw_metadata.get("종료일자", "") for h in headers]
    period_start  = min(s for s in period_starts if s) if any(period_starts) else ""
    period_end    = max(e for e in period_ends   if e) if any(period_ends)   else ""

    return ChkxpConfig(
        source=SourceConfig(
            input_path=str(csv_paths[0]),
            source_hash=compute_file_hash(csv_paths[0]),
            detected_format="daily_etf",
            source_files=[str(p) for p in csv_paths],
        ),
        metadata=MetadataConfig(
            instrument_codes=[e.code for e in all_entities],
            item_codes=common_item_codes,
            frequency=str(meta.get("주기", "D")),
            period_start=str(period_start),
            period_end=str(period_end),
            frequency_type="daily",
        ),
        output=OutputConfig(
            output_dir=str(output_dir),
            output_format="parquet",
            partition_by=["year"],
        ),
        entities=all_entities,
        items=[
            ItemInfo(code=code, field_code=fc, name=name)
            for code, fc, name in zip(
                common_item_codes,
                common_item_field_codes,
                common_item_names,
                strict=True,
            )
        ],
    )

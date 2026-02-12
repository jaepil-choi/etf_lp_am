"""Shared pytest fixtures for chkxp_ingest tests.

Provides factories that write minimal synthetic CSVs to ``tmp_path``,
matching the Check Expert header format exactly.  No real data files
are required for unit tests.
"""

from __future__ import annotations

from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Single-entity CSV (ETF-like, 1-minute, 3 items)
# ---------------------------------------------------------------------------

_SINGLE_ENTITY_HEADER = """\
종목코드,069500*001,,
항목코드,20004,20005,20006,20007
주기,1M,,
시작일자,20250101,,
시작시간,0900,,
종료일자,20260207,,
종료시간,1515,,
데이터 개수,-1,,
정렬,ASC,,
,KODEX 200,F20005,F20006,F20007
시간/항목,체결Intra생성시간,Intra시가,Intra고가,Intra저가"""

_SINGLE_ENTITY_ROWS = [
    '2025-01-02 09:01:00,09:01:00,"58,450","58,720","58,400"',
    '2025-01-02 09:02:00,09:02:00,"58,720","58,810","58,625"',
    '2025-01-02 09:03:00,09:03:00,"58,660","58,770","58,605"',
    '2025-01-03 09:01:00,09:01:00,"59,000","59,100","58,900"',
    '2025-01-03 09:02:00,09:02:00,"59,100","59,200","59,050"',
]


@pytest.fixture()
def sample_single_entity_csv(tmp_path: Path) -> Path:
    """Write a minimal single-entity CSV and return its path."""
    p = tmp_path / "single_entity.csv"
    content = _SINGLE_ENTITY_HEADER + "\n" + "\n".join(_SINGLE_ENTITY_ROWS) + "\n"
    p.write_text(content, encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# Multi-entity CSV (futures-like, 2 entities × 2 items)
# ---------------------------------------------------------------------------

_MULTI_ENTITY_HEADER = """\
종목코드,K2FA020*005,K2FS020*005,,,
항목코드,20004,20005,20008,,
주기,1M,,,
시작일자,20250101,,,
시작시간,0900,,,
종료일자,20260207,,,
종료시간,1515,,,
데이터 개수,-1,,,
정렬,ASC,,,
,KOSPI200 선물 2603,F20005,F20008,K200 스프레드 6366,F20005,F20008
시간/항목,체결Intra생성시간,Intra시가,Intra종가,체결Intra생성시간,Intra시가,Intra종가"""

_MULTI_ENTITY_ROWS = [
    "2025-01-02 09:46:00,09:46:00,317.70,318.45,09:46:00,-0.80,-0.85",
    "2025-01-02 09:47:00,09:47:00,318.40,318.00,09:47:00,-0.85,-0.85",
    '2025-01-02 09:48:00,09:48:00,318.05,317.40,09:48:00,-0.85,"1,234"',
]


@pytest.fixture()
def sample_multi_entity_csv(tmp_path: Path) -> Path:
    """Write a minimal multi-entity CSV and return its path."""
    p = tmp_path / "multi_entity.csv"
    content = _MULTI_ENTITY_HEADER + "\n" + "\n".join(_MULTI_ENTITY_ROWS) + "\n"
    p.write_text(content, encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# Convenience
# ---------------------------------------------------------------------------


@pytest.fixture()
def output_dir(tmp_path: Path) -> Path:
    """Temporary output directory for parquet artifacts."""
    d = tmp_path / "output"
    d.mkdir()
    return d

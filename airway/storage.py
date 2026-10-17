"""Anonymous SQLite record storage."""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

DB_PATH = Path(__file__).resolve().parent.parent / "airwayscope_records.sqlite3"


def save_record(case_id: str | None, measurements: dict, calibration_method: str, warnings: list[str]) -> str:
    """Store numeric measurements and warnings under an anonymous AW case code.

    Only an auto-generated case code is stored. No name, image, face, or identifying
    metadata is accepted. The database path can be overridden via AIRWAYSCOPE_DB.
    """
    import os
    db_path = Path(os.environ.get("AIRWAYSCOPE_DB", DB_PATH))
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as connection:
        connection.execute("""CREATE TABLE IF NOT EXISTS records (
            sequence INTEGER PRIMARY KEY AUTOINCREMENT, case_id TEXT UNIQUE NOT NULL,
            created_at TEXT NOT NULL, calibration_method TEXT NOT NULL,
            mouth_opening_mm REAL, mouth_width_mm REAL, jaw_width_mm REAL,
            face_height_mm REAL, mouth_opening_to_face_height_ratio REAL,
            warnings TEXT NOT NULL)""")
        generated = case_id or f"AW-{connection.execute('SELECT COALESCE(MAX(sequence), 0) + 1 FROM records').fetchone()[0]:04d}"
        connection.execute("""INSERT INTO records
            (case_id, created_at, calibration_method, mouth_opening_mm, mouth_width_mm,
             jaw_width_mm, face_height_mm, mouth_opening_to_face_height_ratio, warnings)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""", (
                generated, datetime.now(timezone.utc).isoformat(), calibration_method,
                measurements.get("mouth_opening_mm"), measurements.get("mouth_width_mm"),
                measurements.get("jaw_width_mm"), measurements.get("face_height_mm"),
                measurements.get("mouth_opening_to_face_height_ratio"), " | ".join(warnings),
            ))
    return generated


def load_records() -> pd.DataFrame:
    """Return all saved anonymous cases as a DataFrame, newest case first."""
    import os
    db_path = Path(os.environ.get("AIRWAYSCOPE_DB", DB_PATH))
    if not db_path.exists():
        return pd.DataFrame(columns=["case_id", "created_at", "calibration_method", "mouth_opening_mm", "mouth_width_mm", "jaw_width_mm", "face_height_mm", "mouth_opening_to_face_height_ratio", "warnings"])
    with sqlite3.connect(db_path) as connection:
        return pd.read_sql_query("SELECT case_id, created_at, calibration_method, mouth_opening_mm, mouth_width_mm, jaw_width_mm, face_height_mm, mouth_opening_to_face_height_ratio, warnings FROM records ORDER BY sequence DESC", connection)


def export_csv() -> bytes:
    """Serialize saved cases as UTF-8 CSV bytes suitable for download."""
    return load_records().to_csv(index=False).encode("utf-8")


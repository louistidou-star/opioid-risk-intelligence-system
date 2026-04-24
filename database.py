import json
import sqlite3
from datetime import datetime
import pandas as pd

DB_NAME = "opioid_risk.db"


def get_connection():
    return sqlite3.connect(DB_NAME, check_same_thread=False)


def ensure_column(cursor, table_name: str, column_name: str, column_def: str):
    cursor.execute(f"PRAGMA table_info({table_name})")
    existing_columns = [row[1] for row in cursor.fetchall()]
    if column_name not in existing_columns:
        cursor.execute(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_def}")


def init_db():
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS medication_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id TEXT NOT NULL,
            prescriber_id TEXT NOT NULL,
            drug TEXT NOT NULL,
            dosage_mg REAL NOT NULL,
            days_supply INTEGER NOT NULL,
            refill_count INTEGER NOT NULL,
            zip INTEGER NOT NULL,
            status TEXT NOT NULL,
            event_time TEXT NOT NULL
        )
    """)

    ensure_column(cursor, "medication_events", "scheduled_time", "TEXT")
    ensure_column(cursor, "medication_events", "taken_on_time", "INTEGER")
    ensure_column(cursor, "medication_events", "submitted_by", "TEXT")

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS audit_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            event_id INTEGER,
            patient_id TEXT NOT NULL,
            action TEXT NOT NULL,
            submitted_by TEXT,
            timestamp TEXT NOT NULL,
            previous_value TEXT,
            new_value TEXT
        )
    """)

    conn.commit()
    conn.close()


def insert_event(
    patient_id: str,
    prescriber_id: str,
    drug: str,
    dosage_mg: float,
    days_supply: int,
    refill_count: int,
    zip_code: int,
    status: str,
    scheduled_time: str,
    submitted_by: str = "unknown",
):
    conn = get_connection()
    cursor = conn.cursor()

    event_time = datetime.now()
    scheduled_dt = datetime.strptime(scheduled_time, "%Y-%m-%d %H:%M:%S")

    # "On time" = within 1 hour before/after scheduled time and status is Taken
    time_diff_minutes = abs((event_time - scheduled_dt).total_seconds()) / 60
    taken_on_time = 1 if status == "Taken" and time_diff_minutes <= 60 else 0

    cursor.execute("""
        INSERT INTO medication_events (
            patient_id,
            prescriber_id,
            drug,
            dosage_mg,
            days_supply,
            refill_count,
            zip,
            status,
            event_time,
            scheduled_time,
            taken_on_time,
            submitted_by
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        patient_id,
        prescriber_id,
        drug,
        dosage_mg,
        days_supply,
        refill_count,
        zip_code,
        status,
        event_time.strftime("%Y-%m-%d %H:%M:%S"),
        scheduled_time,
        taken_on_time,
        submitted_by,
    ))

    event_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return event_id


def log_audit(
    event_id: int,
    patient_id: str,
    action: str,
    submitted_by: str,
    previous_value: dict | None,
    new_value: dict | None,
):
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO audit_log (
            event_id,
            patient_id,
            action,
            submitted_by,
            timestamp,
            previous_value,
            new_value
        )
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (
        event_id,
        patient_id,
        action,
        submitted_by,
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        json.dumps(previous_value or {}),
        json.dumps(new_value or {}),
    ))

    conn.commit()
    conn.close()


def load_events() -> pd.DataFrame:
    conn = get_connection()

    query = """
        SELECT
            id,
            patient_id,
            prescriber_id,
            drug,
            dosage_mg,
            days_supply,
            refill_count,
            zip,
            status,
            event_time,
            scheduled_time,
            taken_on_time,
            submitted_by
        FROM medication_events
        ORDER BY event_time DESC
    """

    df = pd.read_sql_query(query, conn)
    conn.close()
    return df


def load_audit_log() -> pd.DataFrame:
    conn = get_connection()

    query = """
        SELECT
            id,
            event_id,
            patient_id,
            action,
            submitted_by,
            timestamp,
            previous_value,
            new_value
        FROM audit_log
        ORDER BY timestamp DESC, id DESC
    """

    df = pd.read_sql_query(query, conn)
    conn.close()
    return df

def update_event(
    event_id: int,
    patient_id: str,
    prescriber_id: str,
    drug: str,
    dosage_mg: float,
    days_supply: int,
    refill_count: int,
    zip_code: int,
    status: str,
    scheduled_time: str,
    submitted_by: str = "unknown",
):
    conn = get_connection()
    cursor = conn.cursor()

    now_dt = datetime.now()
    scheduled_dt = datetime.strptime(scheduled_time, "%Y-%m-%d %H:%M:%S")

    # "On time" = within 1 hour before/after scheduled time and status is Taken
    time_diff_minutes = abs((now_dt - scheduled_dt).total_seconds()) / 60
    taken_on_time = 1 if status == "Taken" and time_diff_minutes <= 60 else 0

    cursor.execute("""
        UPDATE medication_events
        SET
            patient_id = ?,
            prescriber_id = ?,
            drug = ?,
            dosage_mg = ?,
            days_supply = ?,
            refill_count = ?,
            zip = ?,
            status = ?,
            scheduled_time = ?,
            taken_on_time = ?,
            submitted_by = ?
        WHERE id = ?
    """, (
        patient_id,
        prescriber_id,
        drug,
        dosage_mg,
        days_supply,
        refill_count,
        zip_code,
        status,
        scheduled_time,
        taken_on_time,
        submitted_by,
        event_id,
    ))

    conn.commit()
    conn.close()
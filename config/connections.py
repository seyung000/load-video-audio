import os

import psycopg
from dotenv import load_dotenv

load_dotenv(override=True)

def _load_db_config() -> dict[str, str]:
    return {
        "host": os.getenv("POSTGRES_HOST", "localhost"),
        "port": os.getenv("POSTGRES_PORT", "5432"),
        "dbname": os.getenv("POSTGRES_DB", "postgres"),
        "user": os.getenv("POSTGRES_USER", "postgres"),
        "password": os.getenv("POSTGRES_PASSWORD", "postgres"),
    }


def connect_db() -> psycopg.Connection:
    """PostgreSQL 연결 반환. with 문으로 사용 (with connect_db() as conn:)"""
    return psycopg.connect(**_load_db_config())

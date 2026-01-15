from __future__ import annotations

from dataclasses import dataclass
import os

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    db_driver: str
    db_host: str
    db_name: str
    db_user: str
    db_password: str
    db_trust_cert: bool
    ollama_base_url: str
    ollama_model: str
    ollama_sql_model: str  # Specialized model for SQL generation (empty = use ollama_model)
    max_rows: int
    request_timeout: int
    join_rules_path: str

    @property
    def db_connection_string(self) -> str:
        trust = "yes" if self.db_trust_cert else "no"
        return (
            f"DRIVER={{{self.db_driver}}};"
            f"SERVER={self.db_host};"
            f"DATABASE={self.db_name};"
            f"UID={self.db_user};"
            f"PWD={self.db_password};"
            f"TrustServerCertificate={trust};"
        )


def get_settings() -> Settings:
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    default_rules = os.path.join(base_dir, "schema", "join_rules.yaml")
    return Settings(
        db_driver=os.getenv("DB_DRIVER", "ODBC Driver 17 for SQL Server"),
        db_host=os.getenv("DB_HOST", "localhost"),
        db_name=os.getenv("DB_NAME", "ConcordDb_v5"),
        db_user=os.getenv("DB_USER", "sa"),
        db_password=os.getenv("DB_PASSWORD", "1234"),
        db_trust_cert=os.getenv("DB_TRUST_CERT", "yes").lower() in ("1", "true", "yes", "y"),
        ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/"),
        ollama_model=os.getenv("OLLAMA_MODEL", "qwen2.5-coder:30b"),
        ollama_sql_model=os.getenv("OLLAMA_SQL_MODEL", ""),  # Empty = use ollama_model
        max_rows=int(os.getenv("MAX_ROWS", "200")),
        request_timeout=int(os.getenv("REQUEST_TIMEOUT", "90")),
        join_rules_path=os.getenv("JOIN_RULES_PATH", default_rules),
    )

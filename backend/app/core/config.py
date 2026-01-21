"""Application configuration with support for multiple databases."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import os
from typing import Optional

from dotenv import load_dotenv

load_dotenv(override=True)


class DatabaseType(str, Enum):
    """Available database connections."""
    LOCAL = "local"              # ConcordDb_V5 - main database
    IDENTITY = "identity"        # ConcordIdentityDb - identity/auth


@dataclass(frozen=True)
class DatabaseConnection:
    """Configuration for a single database connection."""
    name: str
    host: str
    database: str
    user: str = ""
    password: str = ""
    driver: str = "ODBC Driver 17 for SQL Server"
    trust_cert: bool = True
    timeout: int = 30
    encrypt: bool = False
    enabled: bool = True
    trusted_connection: bool = False  # Windows Authentication

    @property
    def connection_string(self) -> str:
        """Generate pyodbc connection string."""
        trust = "yes" if self.trust_cert else "no"
        base = (
            f"DRIVER={{{self.driver}}};"
            f"SERVER={self.host};"
            f"DATABASE={self.database};"
        )
        if self.trusted_connection:
            auth = "Trusted_Connection=yes;"
        else:
            auth = f"UID={self.user};PWD={self.password};"
        return base + auth + f"TrustServerCertificate={trust};Connection Timeout={self.timeout};"

    @property
    def ado_connection_string(self) -> str:
        """Generate ADO.NET-style connection string (for compatibility)."""
        return (
            f"Data Source={self.host};"
            f"Initial Catalog={self.database};"
            f"Integrated Security=False;"
            f"User ID={self.user};"
            f"Password={self.password};"
            f"Connect Timeout={self.timeout};"
            f"Encrypt={'True' if self.encrypt else 'False'};"
            f"TrustServerCertificate={'True' if self.trust_cert else 'False'};"
            f"ApplicationIntent=ReadWrite;"
            f"MultiSubnetFailover=False;"
        )


@dataclass(frozen=True)
class ConnectionStrings:
    """Collection of all database connections."""
    local: DatabaseConnection
    identity: Optional[DatabaseConnection] = None

    def get(self, db_type: DatabaseType) -> Optional[DatabaseConnection]:
        """Get connection by type."""
        mapping = {
            DatabaseType.LOCAL: self.local,
            DatabaseType.IDENTITY: self.identity,
        }
        return mapping.get(db_type)

    def get_enabled(self) -> list[tuple[DatabaseType, DatabaseConnection]]:
        """Get all enabled connections."""
        result = []
        for db_type in DatabaseType:
            conn = self.get(db_type)
            if conn and conn.enabled:
                result.append((db_type, conn))
        return result

    @property
    def available_databases(self) -> list[str]:
        """List of available database names."""
        return [db_type.value for db_type, _ in self.get_enabled()]


@dataclass(frozen=True)
class Settings:
    """Application settings."""
    # Database connections
    connections: ConnectionStrings
    default_database: DatabaseType

    # Legacy single-db properties (for backward compatibility)
    db_driver: str
    db_host: str
    db_name: str
    db_user: str
    db_password: str
    db_trust_cert: bool

    # Ollama settings
    ollama_base_url: str
    ollama_model: str
    ollama_sql_model: str

    # Query settings
    max_rows: int
    request_timeout: int

    # Schema and examples paths
    join_rules_path: str
    schema_vectors_path: str
    sql_examples_path: str
    sql_examples_extracted_path: str
    sql_examples_max_categories: int
    sql_examples_max_per_category: int
    sql_examples_max_total: int
    sql_index_path: str
    sql_index_enabled: bool
    table_selection_max_tables: int
    table_selection_vector_top_k: int
    table_selection_vector_min_similarity: float
    table_selection_min_score: float

    # Table selection scoring weights
    table_selection_vector_table_weight: float
    table_selection_vector_column_weight: float
    table_selection_vector_rel_weight: float
    table_selection_lexical_token_weight: float
    table_selection_lexical_exact_bonus: float

    @property
    def db_connection_string(self) -> str:
        """Legacy connection string property (uses default database)."""
        conn = self.connections.get(self.default_database)
        if conn:
            return conn.connection_string
        # Fallback to constructed string
        trust = "yes" if self.db_trust_cert else "no"
        return (
            f"DRIVER={{{self.db_driver}}};"
            f"SERVER={self.db_host};"
            f"DATABASE={self.db_name};"
            f"UID={self.db_user};"
            f"PWD={self.db_password};"
            f"TrustServerCertificate={trust};"
        )

    def get_connection_string(self, db_type: DatabaseType) -> Optional[str]:
        """Get connection string for a specific database."""
        conn = self.connections.get(db_type)
        return conn.connection_string if conn else None


def _parse_ado_connection_string(conn_str: str) -> dict:
    """Parse ADO.NET connection string into components."""
    result = {}
    if not conn_str:
        return result

    for part in conn_str.split(";"):
        if "=" in part:
            key, value = part.split("=", 1)
            key = key.strip().lower().replace(" ", "_")
            result[key] = value.strip()

    return result


def _create_connection_from_env(prefix: str, name: str) -> Optional[DatabaseConnection]:
    """Create a DatabaseConnection from environment variables.

    Supports both individual env vars and ADO.NET connection strings.
    """
    # Check for ADO.NET connection string first
    conn_str = os.getenv(f"{prefix}_CONNECTION_STRING", "")
    if conn_str:
        parsed = _parse_ado_connection_string(conn_str)
        integrated = parsed.get("integrated_security", "").lower() in ("true", "sspi", "yes")
        return DatabaseConnection(
            name=name,
            host=parsed.get("data_source", ""),
            database=parsed.get("initial_catalog", ""),
            user=parsed.get("user_id", ""),
            password=parsed.get("password", ""),
            timeout=int(parsed.get("connect_timeout", "30")),
            encrypt=parsed.get("encrypt", "").lower() == "true",
            trust_cert=parsed.get("trustservercertificate", "").lower() == "true",
            trusted_connection=integrated,
            enabled=bool(parsed.get("data_source")),
        )

    # Fall back to individual env vars
    host = os.getenv(f"{prefix}_HOST", "")
    if not host:
        return None

    trusted = os.getenv(f"{prefix}_TRUSTED_CONNECTION", "").lower() in ("1", "true", "yes", "y")
    return DatabaseConnection(
        name=name,
        host=host,
        database=os.getenv(f"{prefix}_DATABASE", ""),
        user=os.getenv(f"{prefix}_USER", "") if not trusted else "",
        password=os.getenv(f"{prefix}_PASSWORD", "") if not trusted else "",
        driver=os.getenv(f"{prefix}_DRIVER", "ODBC Driver 17 for SQL Server"),
        timeout=int(os.getenv(f"{prefix}_TIMEOUT", "30")),
        trust_cert=os.getenv(f"{prefix}_TRUST_CERT", "yes").lower() in ("1", "true", "yes", "y"),
        trusted_connection=trusted,
        enabled=True,
    )


def get_settings() -> Settings:
    """Load settings from environment variables."""
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    default_rules = os.path.join(base_dir, "schema", "join_rules.yaml")
    default_schema_vectors = os.path.join(base_dir, "schema", "schema_vectors.sqlite")
    default_examples = os.path.join(base_dir, "schema", "sql_examples.yaml")
    default_extracted = os.path.join(base_dir, "schema", "sql_examples_extracted.yaml")
    default_index = os.path.join(base_dir, "schema", "sql_index.sqlite")

    # Legacy single-database settings
    db_driver = os.getenv("DB_DRIVER", "ODBC Driver 17 for SQL Server")
    db_host = os.getenv("DB_HOST", "localhost")
    db_name = os.getenv("DB_NAME", "ConcordDb_V5")
    db_user = os.getenv("DB_USER", "sa")
    db_password = os.getenv("DB_PASSWORD", "")
    db_trust_cert = os.getenv("DB_TRUST_CERT", "yes").lower() in ("1", "true", "yes", "y")

    # Create local connection (required)
    local_conn = _create_connection_from_env("DB_LOCAL", "ConcordDb_V5")
    if not local_conn:
        # Fall back to legacy DB_* vars
        local_conn = DatabaseConnection(
            name="ConcordDb_V5",
            host=db_host,
            database=db_name,
            user=db_user,
            password=db_password,
            driver=db_driver,
            trust_cert=db_trust_cert,
            enabled=True,
        )

    # Create optional connections
    identity_conn = _create_connection_from_env("DB_IDENTITY", "ConcordIdentityDb")

    # Build ConnectionStrings
    connections = ConnectionStrings(
        local=local_conn,
        identity=identity_conn,
    )

    # Default database type
    default_db_str = os.getenv("DEFAULT_DATABASE", "local").lower()
    try:
        default_database = DatabaseType(default_db_str)
    except ValueError:
        default_database = DatabaseType.LOCAL

    return Settings(
        # Multi-database
        connections=connections,
        default_database=default_database,

        # Legacy (backward compatibility)
        db_driver=db_driver,
        db_host=db_host,
        db_name=db_name,
        db_user=db_user,
        db_password=db_password,
        db_trust_cert=db_trust_cert,

        # Ollama
        ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/"),
        ollama_model=os.getenv("OLLAMA_MODEL", "qwen2.5-coder:30b"),
        ollama_sql_model=os.getenv("OLLAMA_SQL_MODEL", ""),

        # Query settings
        max_rows=int(os.getenv("MAX_ROWS", "200")),
        request_timeout=int(os.getenv("REQUEST_TIMEOUT", "90")),

        # Paths
        join_rules_path=os.getenv("JOIN_RULES_PATH", default_rules),
        schema_vectors_path=os.getenv("SCHEMA_VECTORS_PATH", default_schema_vectors),
        sql_examples_path=os.getenv("SQL_EXAMPLES_PATH", default_examples),
        sql_examples_extracted_path=os.getenv("SQL_EXAMPLES_EXTRACTED_PATH", default_extracted),
        sql_examples_max_categories=int(os.getenv("SQL_EXAMPLES_MAX_CATEGORIES", "8")),
        sql_examples_max_per_category=int(os.getenv("SQL_EXAMPLES_MAX_PER_CATEGORY", "5")),
        sql_examples_max_total=int(os.getenv("SQL_EXAMPLES_MAX_TOTAL", "15")),
        sql_index_path=os.getenv("SQL_INDEX_PATH", default_index),
        sql_index_enabled=os.getenv("SQL_INDEX_ENABLED", "0").lower() in ("1", "true", "yes", "y"),
        table_selection_max_tables=int(os.getenv("TABLE_SELECTION_MAX_TABLES", "8")),
        table_selection_vector_top_k=int(os.getenv("TABLE_SELECTION_VECTOR_TOP_K", "50")),
        table_selection_vector_min_similarity=float(os.getenv("TABLE_SELECTION_VECTOR_MIN_SIMILARITY", "0.2")),
        table_selection_min_score=float(os.getenv("TABLE_SELECTION_MIN_SCORE", "0.25")),

        # Table selection scoring weights
        table_selection_vector_table_weight=float(os.getenv("TABLE_SELECTION_VECTOR_TABLE_WEIGHT", "1.0")),
        table_selection_vector_column_weight=float(os.getenv("TABLE_SELECTION_VECTOR_COLUMN_WEIGHT", "0.6")),
        table_selection_vector_rel_weight=float(os.getenv("TABLE_SELECTION_VECTOR_REL_WEIGHT", "0.4")),
        table_selection_lexical_token_weight=float(os.getenv("TABLE_SELECTION_LEXICAL_TOKEN_WEIGHT", "0.2")),
        table_selection_lexical_exact_bonus=float(os.getenv("TABLE_SELECTION_LEXICAL_EXACT_BONUS", "1.0")),
    )


# Singleton for caching settings
_settings_cache: Optional[Settings] = None


def get_cached_settings() -> Settings:
    """Get cached settings (loads once)."""
    global _settings_cache
    if _settings_cache is None:
        _settings_cache = get_settings()
    return _settings_cache


def clear_settings_cache() -> None:
    """Clear the settings cache (useful for testing)."""
    global _settings_cache
    _settings_cache = None

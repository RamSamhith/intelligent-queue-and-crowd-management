import sqlite3
import os
import logging

logger = logging.getLogger(__name__)

class Database:
    def __init__(self, db_path: str):
        self._db_path = db_path
        if db_path != ":memory:":
            dirname = os.path.dirname(os.path.abspath(db_path))
            if dirname:
                os.makedirs(dirname, exist_ok=True)
        
    def get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        # PRAGMAs for performance and safety
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA busy_timeout=5000")
        return conn

    def initialize(self):
        schema_path = os.path.join(os.path.dirname(__file__), "schema.sql")
        with open(schema_path, "r", encoding="utf-8") as f:
            schema = f.read()
            
        conn = self.get_connection()
        try:
            conn.executescript(schema)
            conn.commit()
            logger.info(f"Database initialized successfully at {self._db_path}")
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            raise
        finally:
            conn.close()

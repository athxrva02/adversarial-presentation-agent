"""
SQLite wrapper for structured data in the Storage Layer.

Provides RelationalStore with CRUD for:
- document_chunks
- session_records
- claim_records
- semantic_patterns  (+ pattern_evidence join table)
- common_ground

All methods accept/return Pydantic models from storage.schemas.
Table creation is idempotent (CREATE TABLE IF NOT EXISTS).

Design notes:
- Uses Python's stdlib sqlite3 – no ORM overhead.
- Row factory returns dicts for easy Pydantic coercion.
- JSON list columns (strengths, weaknesses, evidence) are serialised/deserialised
  transparently.
- All primary-key collisions on INSERT raise sqlite3.IntegrityError (the caller
  should upsert explicitly if needed).
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
from datetime import datetime
from typing import Any, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# DDL
# ---------------------------------------------------------------------------

_CREATE_TABLES_SQL = """
CREATE TABLE IF NOT EXISTS document_chunks (
    chunk_id TEXT PRIMARY KEY,
    slide_number INTEGER,
    chunk_type TEXT NOT NULL,
    text TEXT NOT NULL,
    position_in_pdf INTEGER,
    embedding_id TEXT,
    source_file TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS session_records (
    session_id TEXT PRIMARY KEY,
    timestamp TIMESTAMP NOT NULL,
    duration_seconds REAL,
    overall_score REAL,
    strengths TEXT,
    weaknesses TEXT,
    claims_count INTEGER,
    contradictions_detected INTEGER
);

CREATE TABLE IF NOT EXISTS claim_records (
    claim_id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    turn_number INTEGER NOT NULL,
    claim_text TEXT NOT NULL,
    alignment TEXT NOT NULL,
    mapped_to_slide INTEGER,
    prior_conflict TEXT,
    timestamp TIMESTAMP NOT NULL,
    FOREIGN KEY (session_id) REFERENCES session_records(session_id)
);

CREATE TABLE IF NOT EXISTS semantic_patterns (
    pattern_id TEXT PRIMARY KEY,
    category TEXT NOT NULL,
    text TEXT NOT NULL,
    confidence REAL NOT NULL,
    direction TEXT NOT NULL,
    first_seen TEXT NOT NULL,
    last_updated TEXT NOT NULL,
    session_count INTEGER NOT NULL,
    status TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS pattern_evidence (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    pattern_id TEXT NOT NULL,
    claim_id TEXT NOT NULL,
    session_id TEXT NOT NULL,
    FOREIGN KEY (pattern_id) REFERENCES semantic_patterns(pattern_id),
    FOREIGN KEY (claim_id) REFERENCES claim_records(claim_id)
);

CREATE TABLE IF NOT EXISTS common_ground (
    cg_id TEXT PRIMARY KEY,
    pdf_chunk_ref TEXT,
    original_text TEXT,
    negotiated_text TEXT NOT NULL,
    proposed_by TEXT NOT NULL,
    session_agreed TEXT NOT NULL,
    version INTEGER NOT NULL DEFAULT 1,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (pdf_chunk_ref) REFERENCES document_chunks(chunk_id)
);
"""


# ---------------------------------------------------------------------------
# Store class
# ---------------------------------------------------------------------------

class RelationalStore:
    """SQLite-backed structured storage."""

    def __init__(self, db_path: Optional[str] = None):
        try:
            from config import settings  # type: ignore
            self._db_path = db_path or settings.sqlite_path
        except Exception:
            self._db_path = db_path or "./data/db/agent.db"

        self._conn: Optional[sqlite3.Connection] = None
        self._init_db()

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------

    def _get_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            os.makedirs(os.path.dirname(os.path.abspath(self._db_path)), exist_ok=True)
            self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA journal_mode=WAL;")
            self._conn.execute("PRAGMA foreign_keys=ON;")
        return self._conn

    def _init_db(self) -> None:
        conn = self._get_conn()
        conn.executescript(_CREATE_TABLES_SQL)
        conn.commit()
        logger.info("SQLite database ready at '%s'.", self._db_path)

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None

    # ------------------------------------------------------------------
    # DocumentChunk
    # ------------------------------------------------------------------

    def insert_chunk(self, chunk) -> None:
        """Insert a DocumentChunk. Raises IntegrityError on duplicate chunk_id."""
        conn = self._get_conn()
        conn.execute(
            """
            INSERT INTO document_chunks
                (chunk_id, slide_number, chunk_type, text, position_in_pdf, embedding_id, source_file)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                chunk.chunk_id,
                chunk.slide_number,
                chunk.chunk_type,
                chunk.text,
                chunk.position_in_pdf,
                chunk.embedding_id,
                chunk.source_file,
            ),
        )
        conn.commit()

    def insert_chunks(self, chunks: list) -> None:
        """Batch insert DocumentChunk objects. Skips duplicates."""
        if not chunks:
            return
        conn = self._get_conn()
        conn.executemany(
            """
            INSERT OR IGNORE INTO document_chunks
                (chunk_id, slide_number, chunk_type, text, position_in_pdf, embedding_id, source_file)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (c.chunk_id, c.slide_number, c.chunk_type, c.text, c.position_in_pdf, c.embedding_id, c.source_file)
                for c in chunks
            ],
        )
        conn.commit()
        logger.debug("Batch-inserted %d chunks.", len(chunks))

    def update_chunk_embedding_id(self, chunk_id: str, embedding_id: str) -> None:
        conn = self._get_conn()
        conn.execute(
            "UPDATE document_chunks SET embedding_id = ? WHERE chunk_id = ?",
            (embedding_id, chunk_id),
        )
        conn.commit()

    def get_chunk(self, chunk_id: str) -> Optional[dict]:
        conn = self._get_conn()
        row = conn.execute(
            "SELECT * FROM document_chunks WHERE chunk_id = ?", (chunk_id,)
        ).fetchone()
        return dict(row) if row else None

    def get_all_chunks(self) -> list[dict]:
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT * FROM document_chunks ORDER BY position_in_pdf"
        ).fetchall()
        return [dict(r) for r in rows]

    def delete_all_chunks(self) -> None:
        """Remove all document chunks (e.g. when a new PDF is uploaded)."""
        conn = self._get_conn()
        conn.execute("DELETE FROM document_chunks")
        conn.commit()

    # ------------------------------------------------------------------
    # SessionRecord
    # ------------------------------------------------------------------

    def insert_session(self, session) -> None:
        conn = self._get_conn()
        conn.execute(
            """
            INSERT OR REPLACE INTO session_records
                (session_id, timestamp, duration_seconds, overall_score,
                 strengths, weaknesses, claims_count, contradictions_detected)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                session.session_id,
                _to_iso(session.timestamp),
                session.duration_seconds,
                session.overall_score,
                json.dumps(session.strengths),
                json.dumps(session.weaknesses),
                session.claims_count,
                session.contradictions_detected,
            ),
        )
        conn.commit()

    def get_session(self, session_id: str) -> Optional[dict]:
        conn = self._get_conn()
        row = conn.execute(
            "SELECT * FROM session_records WHERE session_id = ?", (session_id,)
        ).fetchone()
        if not row:
            return None
        d = dict(row)
        d["strengths"] = json.loads(d.get("strengths") or "[]")
        d["weaknesses"] = json.loads(d.get("weaknesses") or "[]")
        return d

    def get_all_sessions(self, limit: int = 50) -> list[dict]:
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT * FROM session_records ORDER BY timestamp DESC LIMIT ?", (limit,)
        ).fetchall()
        result = []
        for row in rows:
            d = dict(row)
            d["strengths"] = json.loads(d.get("strengths") or "[]")
            d["weaknesses"] = json.loads(d.get("weaknesses") or "[]")
            result.append(d)
        return result

    # ------------------------------------------------------------------
    # ClaimRecord
    # ------------------------------------------------------------------

    def insert_claim(self, claim) -> None:
        conn = self._get_conn()
        conn.execute(
            """
            INSERT OR IGNORE INTO claim_records
                (claim_id, session_id, turn_number, claim_text, alignment,
                 mapped_to_slide, prior_conflict, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                claim.claim_id,
                claim.session_id,
                claim.turn_number,
                claim.claim_text,
                str(claim.alignment.value if hasattr(claim.alignment, "value") else claim.alignment),
                claim.mapped_to_slide,
                claim.prior_conflict,
                _to_iso(claim.timestamp),
            ),
        )
        conn.commit()

    def get_claim(self, claim_id: str) -> Optional[dict]:
        conn = self._get_conn()
        row = conn.execute(
            "SELECT * FROM claim_records WHERE claim_id = ?", (claim_id,)
        ).fetchone()
        return dict(row) if row else None

    def get_claims_for_session(self, session_id: str) -> list[dict]:
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT * FROM claim_records WHERE session_id = ? ORDER BY turn_number",
            (session_id,),
        ).fetchall()
        return [dict(r) for r in rows]

    def get_recent_claims(self, limit: int = 50) -> list[dict]:
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT * FROM claim_records ORDER BY timestamp DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # SemanticPattern
    # ------------------------------------------------------------------

    def upsert_pattern(self, pattern) -> None:
        conn = self._get_conn()
        conn.execute(
            """
            INSERT OR REPLACE INTO semantic_patterns
                (pattern_id, category, text, confidence, direction,
                 first_seen, last_updated, session_count, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                pattern.pattern_id,
                pattern.category,
                pattern.text,
                pattern.confidence,
                pattern.direction,
                pattern.first_seen,
                pattern.last_updated,
                pattern.session_count,
                pattern.status,
            ),
        )
        # Insert evidence links (ignore duplicates)
        if hasattr(pattern, "evidence") and pattern.evidence:
            conn.executemany(
                """
                INSERT OR IGNORE INTO pattern_evidence (pattern_id, claim_id, session_id)
                VALUES (?, ?, ?)
                """,
                [(pattern.pattern_id, cid, pattern.last_updated) for cid in pattern.evidence],
            )
        conn.commit()

    def get_all_patterns(self, status: Optional[str] = "active") -> list[dict]:
        conn = self._get_conn()
        if status:
            rows = conn.execute(
                "SELECT * FROM semantic_patterns WHERE status = ?", (status,)
            ).fetchall()
        else:
            rows = conn.execute("SELECT * FROM semantic_patterns").fetchall()
        result = []
        for row in rows:
            d = dict(row)
            # Fetch evidence claim IDs
            evid_rows = conn.execute(
                "SELECT claim_id FROM pattern_evidence WHERE pattern_id = ?", (d["pattern_id"],)
            ).fetchall()
            d["evidence"] = [r["claim_id"] for r in evid_rows]
            result.append(d)
        return result

    # ------------------------------------------------------------------
    # CommonGroundEntry
    # ------------------------------------------------------------------

    def upsert_common_ground(self, entry) -> None:
        conn = self._get_conn()
        conn.execute(
            """
            INSERT OR REPLACE INTO common_ground
                (cg_id, pdf_chunk_ref, original_text, negotiated_text,
                 proposed_by, session_agreed, version, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                entry.cg_id,
                entry.pdf_chunk_ref,
                entry.original_text,
                entry.negotiated_text,
                entry.proposed_by,
                entry.session_agreed,
                entry.version,
                _to_iso(entry.timestamp),
            ),
        )
        conn.commit()

    def get_all_common_ground(self) -> list[dict]:
        conn = self._get_conn()
        rows = conn.execute("SELECT * FROM common_ground ORDER BY timestamp DESC").fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def execute_query(self, sql: str, params: tuple = ()) -> list[dict]:
        """Raw query for ad-hoc use in tests / dev."""
        conn = self._get_conn()
        rows = conn.execute(sql, params).fetchall()
        return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_iso(dt) -> str:
    if isinstance(dt, datetime):
        return dt.isoformat()
    return str(dt)


# Singleton
_default_store: Optional[RelationalStore] = None


def get_relational_store() -> RelationalStore:
    global _default_store
    if _default_store is None:
        _default_store = RelationalStore()
    return _default_store

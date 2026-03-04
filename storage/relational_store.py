"""SQLite wrapper for structured record storage."""
from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime

from config import settings
from storage.schemas import (
    ClaimAlignment,
    ClaimRecord,
    CommonGroundEntry,
    DocumentChunk,
    SemanticPattern,
    SessionRecord,
)

_SCHEMA_SQL = """\
CREATE TABLE IF NOT EXISTS document_chunks (
    chunk_id TEXT PRIMARY KEY,
    slide_number INTEGER,
    chunk_type TEXT NOT NULL,
    text TEXT NOT NULL,
    position_in_pdf INTEGER,
    embedding_id TEXT,
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
    FOREIGN KEY (pattern_id) REFERENCES semantic_patterns(pattern_id)
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


class RelationalStore:
    """Thin wrapper around SQLite for structured CRUD.

    Accepts an optional *db_path*; defaults to ``settings.sqlite_path``.
    Opens and closes a connection per method call for thread safety.
    """

    def __init__(self, *, db_path: str | None = None) -> None:
        self.db_path = db_path or settings.sqlite_path
        os.makedirs(os.path.dirname(os.path.abspath(self.db_path)), exist_ok=True)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA foreign_keys = ON")
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self) -> None:
        conn = self._connect()
        conn.executescript(_SCHEMA_SQL)
        conn.close()

    # ---- DocumentChunk ------------------------------------------------------

    def upsert_document_chunk(self, chunk: DocumentChunk) -> None:
        conn = self._connect()
        conn.execute(
            "INSERT OR REPLACE INTO document_chunks "
            "(chunk_id, slide_number, chunk_type, text, position_in_pdf, embedding_id) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                chunk.chunk_id,
                chunk.slide_number,
                chunk.chunk_type,
                chunk.text,
                chunk.position_in_pdf,
                chunk.embedding_id,
            ),
        )
        conn.commit()
        conn.close()

    def get_document_chunk(self, chunk_id: str) -> DocumentChunk | None:
        conn = self._connect()
        row = conn.execute(
            "SELECT * FROM document_chunks WHERE chunk_id = ?", (chunk_id,)
        ).fetchone()
        conn.close()
        if row is None:
            return None
        return DocumentChunk(
            chunk_id=row["chunk_id"],
            slide_number=row["slide_number"],
            chunk_type=row["chunk_type"],
            text=row["text"],
            position_in_pdf=row["position_in_pdf"],
            embedding_id=row["embedding_id"],
        )

    def get_all_document_chunks(self) -> list[DocumentChunk]:
        conn = self._connect()
        rows = conn.execute("SELECT * FROM document_chunks").fetchall()
        conn.close()
        return [
            DocumentChunk(
                chunk_id=r["chunk_id"],
                slide_number=r["slide_number"],
                chunk_type=r["chunk_type"],
                text=r["text"],
                position_in_pdf=r["position_in_pdf"],
                embedding_id=r["embedding_id"],
            )
            for r in rows
        ]

    # ---- SessionRecord ------------------------------------------------------

    def upsert_session(self, session: SessionRecord) -> None:
        conn = self._connect()
        conn.execute(
            "INSERT OR REPLACE INTO session_records "
            "(session_id, timestamp, duration_seconds, overall_score, "
            "strengths, weaknesses, claims_count, contradictions_detected) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                session.session_id,
                session.timestamp.isoformat(),
                session.duration_seconds,
                session.overall_score,
                json.dumps(session.strengths),
                json.dumps(session.weaknesses),
                session.claims_count,
                session.contradictions_detected,
            ),
        )
        conn.commit()
        conn.close()

    def _row_to_session(self, row: sqlite3.Row) -> SessionRecord:
        return SessionRecord(
            session_id=row["session_id"],
            timestamp=datetime.fromisoformat(row["timestamp"]),
            duration_seconds=row["duration_seconds"],
            overall_score=row["overall_score"],
            strengths=json.loads(row["strengths"]) if row["strengths"] else [],
            weaknesses=json.loads(row["weaknesses"]) if row["weaknesses"] else [],
            claims_count=row["claims_count"],
            contradictions_detected=row["contradictions_detected"],
        )

    def get_session(self, session_id: str) -> SessionRecord | None:
        conn = self._connect()
        row = conn.execute(
            "SELECT * FROM session_records WHERE session_id = ?", (session_id,)
        ).fetchone()
        conn.close()
        if row is None:
            return None
        return self._row_to_session(row)

    def get_all_sessions(self) -> list[SessionRecord]:
        conn = self._connect()
        rows = conn.execute(
            "SELECT * FROM session_records ORDER BY timestamp"
        ).fetchall()
        conn.close()
        return [self._row_to_session(r) for r in rows]

    # ---- ClaimRecord --------------------------------------------------------

    def insert_claim(self, claim: ClaimRecord) -> None:
        conn = self._connect()
        conn.execute(
            "INSERT OR REPLACE INTO claim_records "
            "(claim_id, session_id, turn_number, claim_text, alignment, "
            "mapped_to_slide, prior_conflict, timestamp) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                claim.claim_id,
                claim.session_id,
                claim.turn_number,
                claim.claim_text,
                claim.alignment.value,
                claim.mapped_to_slide,
                claim.prior_conflict,
                claim.timestamp.isoformat(),
            ),
        )
        conn.commit()
        conn.close()

    def _row_to_claim(self, row: sqlite3.Row) -> ClaimRecord:
        return ClaimRecord(
            claim_id=row["claim_id"],
            session_id=row["session_id"],
            turn_number=row["turn_number"],
            claim_text=row["claim_text"],
            alignment=ClaimAlignment(row["alignment"]),
            mapped_to_slide=row["mapped_to_slide"],
            prior_conflict=row["prior_conflict"],
            timestamp=datetime.fromisoformat(row["timestamp"]),
        )

    def get_claim(self, claim_id: str) -> ClaimRecord | None:
        conn = self._connect()
        row = conn.execute(
            "SELECT * FROM claim_records WHERE claim_id = ?", (claim_id,)
        ).fetchone()
        conn.close()
        if row is None:
            return None
        return self._row_to_claim(row)

    def get_claims_for_session(self, session_id: str) -> list[ClaimRecord]:
        conn = self._connect()
        rows = conn.execute(
            "SELECT * FROM claim_records WHERE session_id = ? ORDER BY turn_number",
            (session_id,),
        ).fetchall()
        conn.close()
        return [self._row_to_claim(r) for r in rows]

    def get_all_claims(self) -> list[ClaimRecord]:
        conn = self._connect()
        rows = conn.execute(
            "SELECT * FROM claim_records ORDER BY timestamp"
        ).fetchall()
        conn.close()
        return [self._row_to_claim(r) for r in rows]

    # ---- SemanticPattern ----------------------------------------------------

    def upsert_semantic_pattern(self, pattern: SemanticPattern) -> None:
        conn = self._connect()
        conn.execute(
            "INSERT OR REPLACE INTO semantic_patterns "
            "(pattern_id, category, text, confidence, direction, "
            "first_seen, last_updated, session_count, status) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
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
        # Replace evidence rows
        conn.execute(
            "DELETE FROM pattern_evidence WHERE pattern_id = ?",
            (pattern.pattern_id,),
        )
        for claim_id in pattern.evidence:
            conn.execute(
                "INSERT INTO pattern_evidence (pattern_id, claim_id) VALUES (?, ?)",
                (pattern.pattern_id, claim_id),
            )
        conn.commit()
        conn.close()

    def _row_to_pattern(self, row: sqlite3.Row, evidence: list[str]) -> SemanticPattern:
        return SemanticPattern(
            pattern_id=row["pattern_id"],
            category=row["category"],
            text=row["text"],
            confidence=row["confidence"],
            direction=row["direction"],
            first_seen=row["first_seen"],
            last_updated=row["last_updated"],
            session_count=row["session_count"],
            status=row["status"],
            evidence=evidence,
        )

    def get_pattern(self, pattern_id: str) -> SemanticPattern | None:
        conn = self._connect()
        row = conn.execute(
            "SELECT * FROM semantic_patterns WHERE pattern_id = ?", (pattern_id,)
        ).fetchone()
        if row is None:
            conn.close()
            return None
        evidence = [
            r["claim_id"]
            for r in conn.execute(
                "SELECT claim_id FROM pattern_evidence WHERE pattern_id = ?",
                (pattern_id,),
            ).fetchall()
        ]
        conn.close()
        return self._row_to_pattern(row, evidence)

    def get_semantic_patterns(
        self, status: str | None = None
    ) -> list[SemanticPattern]:
        conn = self._connect()
        if status:
            rows = conn.execute(
                "SELECT * FROM semantic_patterns WHERE status = ?", (status,)
            ).fetchall()
        else:
            rows = conn.execute("SELECT * FROM semantic_patterns").fetchall()

        patterns = []
        for row in rows:
            evidence = [
                r["claim_id"]
                for r in conn.execute(
                    "SELECT claim_id FROM pattern_evidence WHERE pattern_id = ?",
                    (row["pattern_id"],),
                ).fetchall()
            ]
            patterns.append(self._row_to_pattern(row, evidence))
        conn.close()
        return patterns

    # ---- CommonGround -------------------------------------------------------

    def upsert_common_ground(self, entry: CommonGroundEntry) -> None:
        conn = self._connect()
        conn.execute(
            "INSERT OR REPLACE INTO common_ground "
            "(cg_id, pdf_chunk_ref, original_text, negotiated_text, "
            "proposed_by, session_agreed, version, timestamp) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                entry.cg_id,
                entry.pdf_chunk_ref,
                entry.original_text,
                entry.negotiated_text,
                entry.proposed_by,
                entry.session_agreed,
                entry.version,
                entry.timestamp.isoformat(),
            ),
        )
        conn.commit()
        conn.close()

    def _row_to_cg(self, row: sqlite3.Row) -> CommonGroundEntry:
        return CommonGroundEntry(
            cg_id=row["cg_id"],
            pdf_chunk_ref=row["pdf_chunk_ref"],
            original_text=row["original_text"],
            negotiated_text=row["negotiated_text"],
            proposed_by=row["proposed_by"],
            session_agreed=row["session_agreed"],
            version=row["version"],
            timestamp=datetime.fromisoformat(row["timestamp"]),
        )

    def get_common_ground(self, cg_id: str) -> CommonGroundEntry | None:
        conn = self._connect()
        row = conn.execute(
            "SELECT * FROM common_ground WHERE cg_id = ?", (cg_id,)
        ).fetchone()
        conn.close()
        if row is None:
            return None
        return self._row_to_cg(row)

    def get_all_common_ground(self) -> list[CommonGroundEntry]:
        conn = self._connect()
        rows = conn.execute("SELECT * FROM common_ground").fetchall()
        conn.close()
        return [self._row_to_cg(r) for r in rows]

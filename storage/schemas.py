from pydantic import BaseModel
from typing import Optional
from enum import Enum
from datetime import datetime


class ClaimAlignment(str, Enum):
    SUPPORTED = "supported"
    CONTRADICTED = "contradicted"
    UNSUPPORTED = "unsupported"
    NOVEL = "novel"
    NEGOTIATED = "negotiated"


class ResponseClass(str, Enum):
    STRONG = "strong"
    WEAK = "weak"
    CONTRADICTION = "contradiction"
    EVASION = "evasion"


class ConflictStatus(str, Enum):
    TRUE_CONTRADICTION = "true_contradiction"
    NEEDS_CLARIFICATION = "needs_clarification"
    NO_CONFLICT = "no_conflict"


class ConflictAction(str, Enum):
    CLARIFY = "clarify"
    UPDATE = "update"
    IGNORE = "ignore"


class DocumentChunk(BaseModel):
    chunk_id: str
    slide_number: Optional[int]
    chunk_type: str                    # claim | definition | evidence | conclusion
    text: str
    position_in_pdf: int
    embedding_id: Optional[str] = None
    source_file: Optional[str] = None  # basename of the source PDF, used for re-upload detection


class ClaimRecord(BaseModel):
    claim_id: str
    session_id: str
    turn_number: int
    claim_text: str
    alignment: ClaimAlignment
    mapped_to_slide: Optional[int]
    prior_conflict: Optional[str] = None  # claim_id of conflicting prior claim
    timestamp: datetime


class SessionRecord(BaseModel):
    session_id: str
    timestamp: datetime
    duration_seconds: float
    overall_score: Optional[float]
    strengths: list[str]
    weaknesses: list[str]
    claims_count: int
    contradictions_detected: int


class SemanticPattern(BaseModel):
    pattern_id: str
    category: str                      # weakness | strength | style | contradiction
    text: str
    confidence: float
    direction: str                     # improving | declining | stable
    first_seen: str                    # session_id
    last_updated: str                  # session_id
    session_count: int
    status: str                        # active | resolved
    evidence: list[str]               # list of claim_ids


class CommonGroundEntry(BaseModel):
    cg_id: str
    pdf_chunk_ref: Optional[str]       # chunk_id it relates to
    original_text: Optional[str]
    negotiated_text: str
    proposed_by: str                   # "agent" | "user"
    session_agreed: str
    version: int
    timestamp: datetime


class MemoryBundle(BaseModel):
    """What the Memory Module returns to the Reasoning Layer."""
    document_context: list[DocumentChunk]
    episodic_claims: list[ClaimRecord]
    episodic_sessions: list[SessionRecord]
    semantic_patterns: list[SemanticPattern]
    common_ground: list[CommonGroundEntry]


class Classification(BaseModel):
    response_class: ResponseClass
    alignment: ClaimAlignment
    confidence: float
    reasoning: str


class ConflictResult(BaseModel):
    status: ConflictStatus
    action: ConflictAction
    current_claim: str
    prior_claim: Optional[str]
    explanation: str
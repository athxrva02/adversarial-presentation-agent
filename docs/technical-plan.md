# Adversarial Presentation Agent — Technical Implementation Plan

**Project:** Memory-Aware Conversational Agent for Adversarial Presentation Preparation
**Team:** 4 members | **Timeline:** 3 weeks | **Deployment:** Fully local (GDPR-compliant)

---

## 1. Architecture Overview

The system is structured as a four-layer pipeline. Each layer communicates through well-defined interfaces, allowing parallel development.

```
┌─────────────────────────────────────────────────────────┐
│  INTERACTION LAYER                                      │
│  Gradio UI ← → Whisper STT ← → PDF Parser              │
└──────────────────────┬──────────────────────────────────┘
                       │ structured text + document chunks
┌──────────────────────▼──────────────────────────────────┐
│  REASONING LAYER                                        │
│  LangGraph Session Graph (stateful, compiled)           │
│    ├── Classify Node                                    │
│    ├── Retrieve Node                                    │
│    ├── Detect Contradiction Node                        │
│    ├── Generate Question Node                           │
│    ├── Summarise Node                                   │
│    ├── Score Node                                       │
│    └── Negotiate Node                                   │
└──────────────────────┬──────────────────────────────────┘
                       │ memory requests / artifacts
┌──────────────────────▼──────────────────────────────────┐
│  MEMORY LAYER                                           │
│  Memory Module (routing, retrieval logic, promotion)    │
│    ├── Working Memory (in-prompt buffer)                │
│    ├── Document Memory                                  │
│    ├── Episodic Memory                                  │
│    ├── Semantic Memory                                  │
│    └── Common Ground Memory                             │
└──────────────────────┬──────────────────────────────────┘
                       │ store / retrieve / update
┌──────────────────────▼──────────────────────────────────┐
│  STORAGE LAYER                                          │
│  ChromaDB (embeddings) + SQLite (structured records)    │
└─────────────────────────────────────────────────────────┘
```

Data flows top-down during practice (user input → classify → retrieve → contradiction check → question generation) and bottom-up at session end (summarise → score → negotiate → memory writes → semantic promotion).

---

## 2. Tech Stack

| Component           | Technology                             | Version  | Purpose                                            |
|---------------------|----------------------------------------|----------|----------------------------------------------------|
| Language            | Python                                 | 3.11+    | All components                                     |
| LLM                 | Mistral 7B / LLaMA 3.1 8B              | latest   | Question generation, classification, summarisation |
| LLM Server          | Ollama                                 | latest   | Local model serving with OpenAI-compatible API     |
| LLM Framework       | LangChain + ChatOllama                 | latest   | Prompt management and LLM invocation               |
| Orchestration       | LangGraph (from LangChain)             | 0.2+     | Stateful session lifecycle as a compiled graph     |
| Speech-to-Text      | OpenAI Whisper (local)                 | large-v3 | Audio transcription                                |
| Vector Database     | ChromaDB                               | 0.5+     | Embedding storage and similarity retrieval         |
| Embedding Model     | sentence-transformers/all-MiniLM-L6-v2 | latest   | Local embedding generation                         |
| Relational Database | SQLite                                 | 3.x      | Structured records, metadata, versioning           |
| PDF Parsing         | PyMuPDF (fitz)                         | latest   | Argument-level PDF chunking                        |
| Frontend            | Gradio                                 | 4.x      | Chat UI, audio input, PDF upload                   |
| Testing             | pytest                                 | latest   | Unit and integration tests                         |
| Config              | pydantic-settings                      | latest   | Typed configuration management                     |

### Python Dependencies (`requirements.txt`)

```
ollama
langchain
langchain-community
langgraph
chromadb
sentence-transformers
openai-whisper
PyMuPDF
gradio
pydantic
pydantic-settings
pytest
```

### Local Services

```bash
# Start Ollama and pull model (one-time setup)
ollama serve
ollama pull mistral

# Everything else runs in-process via Python. No Docker required.
```

---

## 3. Project Structure

```
adversarial-agent/
│
├── README.md
├── requirements.txt
├── config.py                          # Pydantic settings (model name, DB paths, top-k, thresholds)
├── main.py                            # Entry point: launches Gradio + loads components
│
├── interaction/                       # INTERACTION LAYER
│   ├── __init__.py
│   ├── ui.py                          # Gradio interface: chat, audio, PDF upload, session controls
│   ├── stt.py                         # Whisper wrapper: audio file → text
│   └── pdf_parser.py                  # PDF → argument-level chunks with metadata
│
├── reasoning/                         # REASONING LAYER (LangGraph)
│   ├── __init__.py
│   ├── graph.py                       # LangGraph: defines nodes, edges, compiled session graph
│   ├── state.py                       # TypedDict defining the shared graph state
│   ├── edges.py                       # Conditional edge functions (route by phase + classification)
│   ├── llm.py                         # ChatOllama client wrapper via LangChain
│   ├── nodes/                         # One file per graph node
│   │   ├── __init__.py
│   │   ├── classify.py                # Node: classify user response
│   │   ├── retrieve.py                # Node: fetch memory bundle from Memory Module
│   │   ├── generate_question.py       # Node: produce adversarial follow-up
│   │   ├── detect_contradiction.py    # Node: LLM-as-judge contradiction check
│   │   ├── summarise.py               # Node: session summarisation at session end
│   │   ├── score.py                   # Node: composite performance scoring
│   │   └── negotiate.py               # Node: present items for user approval
│   └── prompts/                       # Prompt templates (one file per task)
│       ├── question_generation.py     # Adversarial question from context + claim
│       ├── classification.py          # Classify response: strong / weak / contradiction / evasion
│       ├── summarisation.py           # Session turns → structured session summary
│       ├── scoring.py                 # Session → composite performance score
│       └── contradiction_judge.py     # Compare two claims → conflict status + recommendation
│
├── memory/                            # MEMORY LAYER
│   ├── __init__.py
│   ├── module.py                      # Memory Module: routes requests, merges per-type results
│   ├── working.py                     # Working Memory: sliding window buffer (in-memory list)
│   ├── document.py                    # Document Memory: CRUD for PDF chunks
│   ├── episodic.py                    # Episodic Memory: session records + claim records
│   ├── semantic.py                    # Semantic Memory: pattern promotion, confidence updates
│   ├── common_ground.py               # Common Ground Memory: versioned negotiated entries
│   └── retrieval.py                   # Per-type top-k retrieval, merge, dedup, recency weighting
│
├── storage/                           # STORAGE LAYER
│   ├── __init__.py
│   ├── vector_store.py                # ChromaDB wrapper: embed, store, query, delete
│   ├── relational_store.py            # SQLite wrapper: schema init, CRUD operations
│   └── schemas.py                     # SQLite table definitions + Pydantic data models
│
├── tests/
│   ├── __init__.py
│   ├── test_memory.py                 # Memory store read/write tests
│   ├── test_retrieval.py              # Retrieval merge + dedup tests
│   ├── test_prompts.py                # Prompt template output validation
│   ├── test_graph.py                  # LangGraph node transitions + state propagation tests
│   ├── test_pdf_parser.py             # PDF chunking tests
│   └── test_integration.py            # End-to-end: input → question → memory write → retrieval
│
└── data/
    ├── db/                            # SQLite database files (gitignored, auto-created)
    ├── chroma/                        # ChromaDB persistence directory (gitignored)
    └── sample_pdfs/                   # Test PDFs for development
```

---

## 4. Data Models (`storage/schemas.py`)

These are the core data structures shared across all components. Define them once, import everywhere.

```python
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
```

---

## 5. SQLite Schemas (`storage/relational_store.py`)

```sql
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
    strengths TEXT,           -- JSON array
    weaknesses TEXT,          -- JSON array
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
    prior_conflict TEXT,      -- references claim_records.claim_id
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
```

---

## 6. LangGraph Session Graph

The Reasoning Layer is implemented as a LangGraph compiled graph. Each reasoning step is a node that reads from and writes to a shared `SessionState` TypedDict. Conditional edges route execution based on classification results and session phase.

### 6.1 Graph State (`reasoning/state.py`)

```python
from typing import TypedDict, Optional, Annotated
from operator import add
from storage.schemas import (
    MemoryBundle, Classification, ConflictResult,
    SessionRecord, ClaimRecord
)


class SessionState(TypedDict):
    """Shared state passed between all LangGraph nodes."""

    # Current turn
    user_input: str
    turn_number: int

    # Classification
    classification: Optional[Classification]

    # Memory
    memory_bundle: Optional[MemoryBundle]

    # Contradiction
    conflict_result: Optional[ConflictResult]

    # Generation
    agent_response: Optional[str]

    # Session phase
    phase: str                          # "practice" | "assessment" | "negotiation" | "update"

    # Accumulated session data
    turns: Annotated[list[dict], add]   # appended each turn
    claims: Annotated[list[ClaimRecord], add]

    # Session-end artifacts
    session_summary: Optional[SessionRecord]
    negotiation_items: Optional[list[dict]]
    negotiation_decisions: Optional[list[dict]]

    # Control
    session_active: bool
```

### 6.2 Graph Definition (`reasoning/graph.py`)

```python
from langgraph.graph import StateGraph, END
from reasoning.state import SessionState
from reasoning.nodes import (
    classify, retrieve, generate_question,
    detect_contradiction, summarise, score, negotiate
)
from reasoning.edges import route_after_classification, route_after_phase


def build_practice_graph() -> StateGraph:
    """Practice-phase subgraph: classify → retrieve → check → generate."""

    graph = StateGraph(SessionState)

    graph.add_node("classify", classify.run)
    graph.add_node("retrieve", retrieve.run)
    graph.add_node("detect_contradiction", detect_contradiction.run)
    graph.add_node("generate_question", generate_question.run)

    graph.set_entry_point("classify")
    graph.add_edge("classify", "retrieve")
    graph.add_edge("retrieve", "detect_contradiction")
    graph.add_conditional_edges(
        "detect_contradiction",
        route_after_classification,
        {
            "probe_weak": "generate_question",
            "escalate_contradiction": "generate_question",
            "request_evidence": "generate_question",
            "redirect": "generate_question",
        }
    )
    graph.add_edge("generate_question", END)

    return graph.compile()


def build_session_graph() -> StateGraph:
    """Full session graph including assessment and negotiation."""

    graph = StateGraph(SessionState)

    graph.add_node("practice_turn", build_practice_graph())
    graph.add_node("summarise", summarise.run)
    graph.add_node("score", score.run)
    graph.add_node("negotiate", negotiate.run)

    graph.set_entry_point("practice_turn")
    graph.add_conditional_edges(
        "practice_turn",
        route_after_phase,
        {
            "continue": "practice_turn",
            "end_session": "summarise",
        }
    )
    graph.add_edge("summarise", "score")
    graph.add_edge("score", "negotiate")
    graph.add_edge("negotiate", END)

    return graph.compile()
```

### 6.3 Conditional Edges (`reasoning/edges.py`)

```python
from reasoning.state import SessionState
from storage.schemas import ResponseClass, ConflictStatus


def route_after_classification(state: SessionState) -> str:
    """Select question generation strategy based on classification + conflict."""

    conflict = state.get("conflict_result")
    classification = state.get("classification")

    if conflict and conflict.status == ConflictStatus.TRUE_CONTRADICTION:
        return "escalate_contradiction"

    if classification is None:
        return "redirect"

    match classification.response_class:
        case ResponseClass.STRONG:
            return "probe_weak"
        case ResponseClass.WEAK:
            return "probe_weak"
        case ResponseClass.CONTRADICTION:
            return "escalate_contradiction"
        case ResponseClass.EVASION:
            return "redirect"

    return "request_evidence"


def route_after_phase(state: SessionState) -> str:
    """Decide whether to continue practice or move to assessment."""

    if state.get("session_active", True):
        return "continue"
    return "end_session"
```

### 6.4 Example Node (`reasoning/nodes/classify.py`)

```python
from datetime import datetime
from reasoning.state import SessionState
from reasoning.llm import call_llm_structured
from reasoning.prompts.classification import build_classification_prompt
from storage.schemas import Classification, ClaimRecord


def run(state: SessionState) -> dict:
    """Classify the user's response and return updated state."""

    prompt = build_classification_prompt(
        utterance=state["user_input"],
        memory_bundle=state.get("memory_bundle"),
    )
    raw = call_llm_structured(
        system_prompt=prompt["system"],
        user_prompt=prompt["user"],
    )
    classification = Classification(**raw)

    return {
        "classification": classification,
        "claims": [ClaimRecord(
            claim_id=f"{state.get('session_id')}-{state['turn_number']}",
            session_id=state.get("session_id", ""),
            turn_number=state["turn_number"],
            claim_text=state["user_input"],
            alignment=classification.alignment,
            mapped_to_slide=None,
            timestamp=datetime.now(),
        )],
    }
```

### 6.5 LLM Client (`reasoning/llm.py`)

```python
from langchain_community.chat_models import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
from config import settings
import json


_client = None


def get_llm_client() -> ChatOllama:
    global _client
    if _client is None:
        _client = ChatOllama(
            base_url=settings.ollama_base_url,
            model=settings.model_name,
            temperature=settings.temperature,
        )
    return _client


def call_llm_structured(system_prompt: str, user_prompt: str) -> dict:
    """Send a prompt and parse the JSON response."""
    llm = get_llm_client()
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ]
    response = llm.invoke(messages)
    return json.loads(response.content)
```

---

## 7. Key Interfaces

These are the function signatures that each team member exposes. Agree on these before writing any implementation.

### Memory Module (`memory/module.py`)

```python
class MemoryModule:
    def retrieve(
        self,
        query: str,
        stores: list[str],         # ["document", "episodic", "semantic", "common_ground"]
        top_k: int = 5
    ) -> MemoryBundle: ...

    def store_claim(self, claim: ClaimRecord) -> None: ...
    def store_session(self, session: SessionRecord, claims: list[ClaimRecord]) -> None: ...
    def store_common_ground(self, entry: CommonGroundEntry) -> None: ...
    def promote_patterns(self, session_id: str) -> list[SemanticPattern]: ...
```

### Session Runner (`reasoning/graph.py`)

```python
from reasoning.graph import build_session_graph
from reasoning.state import SessionState


class SessionRunner:
    """Thin wrapper that the UI calls into."""

    def __init__(self):
        self.graph = build_session_graph()
        self.state: SessionState = {
            "user_input": "",
            "turn_number": 0,
            "classification": None,
            "memory_bundle": None,
            "conflict_result": None,
            "agent_response": None,
            "phase": "practice",
            "turns": [],
            "claims": [],
            "session_summary": None,
            "negotiation_items": None,
            "negotiation_decisions": None,
            "session_active": True,
        }

    def handle_user_input(self, text: str) -> str:
        self.state["user_input"] = text
        self.state["turn_number"] += 1
        result = self.graph.invoke(self.state)
        self.state.update(result)
        return result["agent_response"]

    def end_session(self) -> SessionRecord:
        self.state["session_active"] = False
        result = self.graph.invoke(self.state)
        self.state.update(result)
        return result["session_summary"]

    def commit_negotiation(self, decisions: list[dict]) -> None:
        self.state["negotiation_decisions"] = decisions
        # Memory Module persists approved entries
```

### Input Pipeline (`interaction/`)

```python
# stt.py
def transcribe(audio_path: str) -> str: ...

# pdf_parser.py
def ingest_pdf(file_path: str) -> list[DocumentChunk]: ...
```

---

## 8. ChromaDB Collections

One collection per memory store. All use the same embedding function.

```python
import chromadb
from chromadb.utils import embedding_functions

ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

client = chromadb.PersistentClient(path="./data/chroma")

collections = {
    "document": client.get_or_create_collection("document_memory", embedding_function=ef),
    "episodic_claims": client.get_or_create_collection("episodic_claims", embedding_function=ef),
    "episodic_sessions": client.get_or_create_collection("episodic_sessions", embedding_function=ef),
    "semantic": client.get_or_create_collection("semantic_patterns", embedding_function=ef),
    "common_ground": client.get_or_create_collection("common_ground", embedding_function=ef),
}
```

---

## 9. Implementation Plan (3 Weeks)

### Week 1: Parallel Component Development

All four members work independently against the agreed interfaces.

| Member | Owns                                                                                                                   | Deliverable by Sunday                                                                                                                                                            |
|--------|------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **A**  | `memory/`, `storage/`                                                                                                  | All five stores operational. Can write and read claims, sessions, chunks, patterns, CG entries. Retrieval returns correct top-k per store with merge and dedup.                  |
| **B**  | `reasoning/llm.py`, `reasoning/prompts/`, `reasoning/nodes/detect_contradiction.py`                                    | All five prompt templates working against Ollama via ChatOllama. Each returns structured JSON matching the Pydantic models. Contradiction node works with hardcoded test claims. |
| **C**  | `interaction/stt.py`, `interaction/pdf_parser.py`, `memory/document.py`                                                | PDF ingestion pipeline end-to-end: upload → chunk → embed → store → query. Whisper transcription working.                                                                        |
| **D**  | `reasoning/graph.py`, `reasoning/state.py`, `reasoning/edges.py`, `reasoning/nodes/`, `interaction/ui.py`, `config.py` | LangGraph compiles and runs through all four phases with mocked node implementations. Gradio UI calls `SessionRunner.handle_user_input()` and displays responses.                |

**Coordination required in Week 1:**
- Day 1: All four members review and finalize the interfaces in Section 7 and the data models in Section 4. No code until this is agreed.
- Day 3: Brief sync to confirm everyone's component can produce/consume the shared Pydantic models and that LangGraph state keys align with node return dictionaries.

### Week 2: Integration + Negotiation

Members pair up to connect their components.

| Days    | Pair         | Task                                                                                                                                                                                                                                                                                                                                                   |
|---------|--------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Mon-Wed | **A + D**    | Wire Memory Module into LangGraph nodes. The `retrieve` node calls Member A's `MemoryModule.retrieve()`. The `classify` and `generate_question` nodes receive the memory bundle through graph state. Practice loop works end-to-end: user input → graph invocation → agent response displayed.                                                         |
| Mon-Wed | **B + C**    | Wire LLM prompts into PDF pipeline. The classification prompt receives document chunks from the memory bundle. Contradiction detection runs against episodic claims. Claim extraction from turns produces valid `ClaimRecord` objects.                                                                                                                 |
| Thu-Fri | **All four** | Build the negotiation flow together. The `summarise` → `score` → `negotiate` nodes run in sequence after `session_active` is set to false. The negotiate node presents items through Gradio. Approved entries are persisted to Common Ground and Episodic Memory via Member A's storage functions. This is the novel contribution and needs all hands. |

**End of Week 2 milestone:** A user can upload a PDF, practice over two sessions, receive memory-informed questions in session 2, and negotiate what to remember after each session.

### Week 3: Semantic Memory + Scoring + Polish

| Member | Task                                                                                                                                                                                                                                            |
|--------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **A**  | Semantic memory promotion logic. After each session, scan episodic records for recurring patterns (≥2 sessions), create/update semantic entries with confidence scores. Wire into retrieval so session 3+ questions draw on long-term patterns. |
| **B**  | Composite scoring system. Define the rubric (argument quality, evidence use, counterargument handling, PDF consistency). Write the scoring prompt. Store scores in session records. Tune all prompt templates based on real test runs.          |
| **C**  | Recency weighting on episodic retrieval. Edge case handling: first session (empty memory), PDF re-upload, resolved contradictions. Write integration tests (`tests/test_integration.py`).                                                       |
| **D**  | Evaluation logging: dump all session states, memory snapshots, scores, and negotiation decisions to JSON for analysis. Prepare demo script (3-session walkthrough). Clean up UI and graph edge cases.                                           |

**End of Week 3 milestone:** Complete system with semantic memory, scoring, evaluation logging. Ready for demo.

---

## 10. Configuration (`config.py`)

```python
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # LLM
    ollama_base_url: str = "http://localhost:11434"
    model_name: str = "mistral"
    temperature: float = 0.7
    max_tokens: int = 1024

    # Embeddings
    embedding_model: str = "all-MiniLM-L6-v2"

    # Memory
    working_memory_window: int = 20          # turns
    retrieval_top_k: int = 5                 # per store
    promotion_threshold: int = 2             # sessions before episodic → semantic
    recency_decay_factor: float = 0.85       # applied per session age

    # Storage
    sqlite_path: str = "./data/db/agent.db"
    chroma_path: str = "./data/chroma"

    # PDF
    max_chunk_tokens: int = 256
    chunk_overlap_tokens: int = 32

    class Config:
        env_file = ".env"


settings = Settings()
```

---

## 11. Git Workflow

```
main              ← stable, only merged after review
├── feat/memory      (Member A)
├── feat/reasoning   (Member B)
├── feat/input       (Member C)
├── feat/graph       (Member D)
└── integration      ← Week 2 merge target before main
```

Rules:
- No direct pushes to `main`.
- All merges via pull request with at least one reviewer.
- Integration branch created at start of Week 2 for connecting components.
- `main` updated from `integration` at end of Week 2 and Week 3.

---
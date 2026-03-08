# Testing

## Running Tests

```bash
# All tests (mocked only, no external services needed)
pytest tests/

# Include live LLM tests (requires Ollama running with qwen2.5:7b-instruct)
pytest tests/ --live

# By layer
pytest tests/llm/      # reasoning / LLM nodes
pytest tests/storage/   # SQLite + vector store
pytest tests/memory/    # memory sub-modules, orchestrator, integration
```

The `--live` flag is registered in `tests/llm/conftest.py`. Live tests also check Ollama reachability at `localhost:11434/api/tags` and skip if unreachable.

## Test Structure

```
tests/
├── llm/                          # Reasoning layer tests
│   ├── conftest.py               # --live flag, sample fixtures
│   ├── test_classify_node_mocked.py
│   ├── test_classify_node_live.py
│   ├── test_generate_question_mocked.py
│   ├── test_generate_question_node_live.py
│   ├── test_retrieve_node_mocked.py
│   ├── test_summarise_node_mocked.py
│   ├── test_summarise_node_live.py
│   ├── test_score_node_mocked.py
│   ├── test_score_node_live.py
│   ├── test_graph_runner_mocked.py
│   ├── test_graph_runner_live.py
│   ├── test_edges_classification.py   # pure unit (no mock needed)
│   ├── test_json_utils.py             # pure unit
│   └── test_prompts_classification.py # pure unit
├── storage/                      # Storage layer tests
│   ├── test_vector_store.py      # uses _FakeClient (in-memory)
│   └── test_relational_store.py  # uses tmp_path (real SQLite, temp files)
└── memory/                       # Memory layer tests
    ├── conftest.py               # shared vec_store / rel_store fixtures
    ├── test_working_memory.py    # pure in-memory
    ├── test_document_memory.py
    ├── test_episodic_memory.py
    ├── test_semantic_memory.py
    ├── test_common_ground_memory.py
    ├── test_retrieval.py         # pure unit (merge/dedup/ranking)
    ├── test_module.py            # MemoryModule orchestrator
    └── test_integration.py       # SessionRunner + MemoryModule end-to-end
```

**88 tests total** — 83 run without any external services, 5 require `--live` + Ollama.

## Test Patterns

### Mocked LLM tests (`tests/llm/`)
Each LLM-calling node has a `_mocked` and `_live` variant. Mocked tests patch at the node's import path:

```python
with patch("reasoning.nodes.classify.call_llm_structured", return_value=fake_json):
    result = run(state)
```

### Storage tests (`tests/storage/`)
- **VectorStore**: uses `_FakeClient` / `_FakeCollection` from `test_vector_store.py` — lightweight in-memory substitutes for ChromaDB (which is broken on Python 3.14 due to pydantic v1).
- **RelationalStore**: uses pytest's `tmp_path` fixture for real SQLite on temp files.

### Memory tests (`tests/memory/`)
Compose real `RelationalStore` (via `tmp_path`) and `VectorStore` (via `_FakeClient`) through shared fixtures in `tests/memory/conftest.py`. No mocking of internal classes — tests exercise the full store-and-retrieve path.

### Integration test
`test_integration.py` wires `SessionRunner` with a real `MemoryModule` (ephemeral stores) and mocked LLM calls. Verifies claims persist across turns and session records are stored at `end_session()`.

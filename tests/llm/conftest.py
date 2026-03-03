# tests/conftest.py
import os
import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--live",
        action="store_true",
        default=False,
        help="Run live tests that call the local Ollama server.",
    )


@pytest.fixture
def live_enabled(request) -> bool:
    return bool(request.config.getoption("--live"))


@pytest.fixture
def sample_classification_json():
    return {
        "response_class": "strong",
        "alignment": "supported",
        "confidence": 0.85,
        "reasoning": "Clear claim and rationale; consistent with provided context.",
    }
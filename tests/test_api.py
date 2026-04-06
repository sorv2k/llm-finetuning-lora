"""
Tests for the FastAPI endpoints using a mocked model state.
No real model is loaded — tests verify routing, validation, and response shapes.
"""

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient


def _make_mock_tokenizer():
    mock_tokenizer = MagicMock()
    mock_tokenizer.encode.return_value = [1, 2, 3, 4, 5]
    return mock_tokenizer


from src.api.main import app  # import once at module level


@pytest.fixture()
def client_with_model():
    """TestClient with a mock model pre-loaded. Patches load_fine_tuned_model so
    the lifespan startup does not attempt to download real weights."""
    mock_model = MagicMock()
    mock_tokenizer = _make_mock_tokenizer()

    with patch("src.api.main.load_fine_tuned_model", return_value=(mock_model, mock_tokenizer)):
        with patch("src.api.main.generate_answer", return_value="Paris"):
            with TestClient(app) as client:
                yield client


@pytest.fixture()
def client_degraded():
    """TestClient where model loading fails so the API starts in degraded state."""
    with patch("src.api.main.load_fine_tuned_model", side_effect=RuntimeError("no weights")):
        with TestClient(app) as client:
            yield client


class TestHealthEndpoint:
    def test_health_returns_200(self, client_with_model):
        response = client_with_model.get("/health")
        assert response.status_code == 200

    def test_health_reports_healthy_when_model_loaded(self, client_with_model):
        data = client_with_model.get("/health").json()
        assert data["status"] == "healthy"
        assert data["model_loaded"] is True

    def test_health_reports_degraded_when_model_not_loaded(self, client_degraded):
        data = client_degraded.get("/health").json()
        assert data["status"] == "degraded"
        assert data["model_loaded"] is False

    def test_health_response_schema(self, client_with_model):
        data = client_with_model.get("/health").json()
        required_keys = {"status", "model_loaded", "model_name", "device", "adapter_path"}
        assert required_keys.issubset(data.keys())


class TestGenerateEndpoint:
    VALID_PAYLOAD = {
        "context": "Paris is the capital and most populous city of France.",
        "question": "What is the capital of France?",
    }

    def test_generate_returns_200(self, client_with_model):
        response = client_with_model.post("/generate", json=self.VALID_PAYLOAD)
        assert response.status_code == 200

    def test_generate_response_contains_answer(self, client_with_model):
        data = client_with_model.post("/generate", json=self.VALID_PAYLOAD).json()
        assert "answer" in data
        assert data["answer"] == "Paris"

    def test_generate_response_schema(self, client_with_model):
        data = client_with_model.post("/generate", json=self.VALID_PAYLOAD).json()
        required_keys = {"answer", "context", "question", "model", "tokens_generated"}
        assert required_keys.issubset(data.keys())

    def test_generate_returns_503_when_model_not_loaded(self, client_degraded):
        response = client_degraded.post("/generate", json=self.VALID_PAYLOAD)
        assert response.status_code == 503

    def test_generate_validates_short_context(self, client_with_model):
        payload = {"context": "Hi", "question": "What is this?"}
        response = client_with_model.post("/generate", json=payload)
        assert response.status_code == 422  # Pydantic validation error

    def test_generate_validates_short_question(self, client_with_model):
        payload = {
            "context": "Paris is the capital of France.",
            "question": "Hi",
        }
        response = client_with_model.post("/generate", json=payload)
        assert response.status_code == 422

    def test_generate_respects_max_new_tokens(self, client_with_model):
        payload = {**self.VALID_PAYLOAD, "max_new_tokens": 50}
        response = client_with_model.post("/generate", json=payload)
        assert response.status_code == 200

    def test_generate_rejects_invalid_max_new_tokens(self, client_with_model):
        payload = {**self.VALID_PAYLOAD, "max_new_tokens": 1000}
        response = client_with_model.post("/generate", json=payload)
        assert response.status_code == 422

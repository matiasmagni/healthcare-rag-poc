"""Integration tests for the out-of-context refusal response."""
import pytest
from fastapi.testclient import TestClient

from app.config import get_settings
from app.main import create_app
from app.prompts import NO_ANSWER_RESPONSE


@pytest.fixture(scope="module")
def client() -> TestClient:
    """Create a test client with the real RAG pipeline initialized."""
    with TestClient(create_app(get_settings())) as test_client:
        yield test_client


OUT_OF_CONTEXT_QUESTIONS = [
    "What is the capital of France?",
    "Who won the 2024 FIFA World Cup?",
    "What is the boiling point of water at sea level?",
    "How do I bake sourdough bread?",
]


@pytest.mark.parametrize("question", OUT_OF_CONTEXT_QUESTIONS)
def test_out_of_context_question_returns_exact_refusal(
    client: TestClient,
    question: str,
) -> None:
    """An unrelated question must return the canonical refusal answer."""
    response = client.post(
        "/query",
        json={"question": question},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["answer"] == NO_ANSWER_RESPONSE
    assert isinstance(payload["context"], list)


def test_endpoint_alias_returns_exact_refusal(client: TestClient) -> None:
    """The hidden /endpoint alias must enforce the same refusal contract."""
    response = client.post(
        "/endpoint",
        json={"question": "What is the capital of France?"},
    )

    assert response.status_code == 200
    assert response.json()["answer"] == NO_ANSWER_RESPONSE

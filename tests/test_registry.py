"""Tests for PromptRegistry — store, pull, version, search, audit."""
import hashlib
import json
import pytest
from unittest.mock import MagicMock
import numpy as np

from ghostprompt.registry import PromptRegistry, PromptMeta
from ghostprompt.embedder import cosine_similarity


@pytest.fixture
def registry(tmp_path):
    return PromptRegistry(db_path=str(tmp_path / "test.db"))


# --- Store ---

def test_store_creates_prompt(registry):
    v = registry.store("greet", "Hello {{name}}", tags=["greeting"])
    assert v == 1


def test_store_returns_version(registry):
    registry.store("p", "v1 text")
    v = registry.store("p", "v2 text")
    assert v == 2


def test_store_same_template_no_version_bump(registry):
    registry.store("p", "same text")
    v = registry.store("p", "same text")
    assert v == 1


def test_store_with_tags(registry):
    registry.store("p", "text", tags=["a", "b"])
    info = registry.get("p")
    assert info["tags"] == ["a", "b"]


# --- Pull ---

def test_pull_returns_rendered_text(registry):
    registry.store("greet", "Hello {{name}}!")
    text, meta = registry.pull("greet", variables={"name": "Adam"})
    assert text == "Hello Adam!"


def test_pull_returns_meta(registry):
    registry.store("p", "template", tags=["t1"])
    _, meta = registry.pull("p")
    assert isinstance(meta, PromptMeta)
    assert meta.name == "p"
    assert meta.version == 1
    assert meta.tags == ["t1"]
    assert meta.hash  # SHA-256 present
    assert meta.pulled_at > 0


def test_pull_hash_is_sha256_of_rendered(registry):
    registry.store("p", "Hello {{x}}")
    text, meta = registry.pull("p", variables={"x": "world"})
    expected = hashlib.sha256("Hello world".encode()).hexdigest()
    assert meta.hash == expected


def test_pull_hash_deterministic(registry):
    registry.store("p", "fixed text")
    _, m1 = registry.pull("p")
    _, m2 = registry.pull("p")
    assert m1.hash == m2.hash


def test_pull_different_variables_different_hash(registry):
    registry.store("p", "{{x}}")
    _, m1 = registry.pull("p", variables={"x": "a"})
    _, m2 = registry.pull("p", variables={"x": "b"})
    assert m1.hash != m2.hash


def test_pull_unknown_raises(registry):
    with pytest.raises(KeyError, match="not found"):
        registry.pull("nonexistent")


def test_pull_no_variables(registry):
    registry.store("p", "plain text")
    text, _ = registry.pull("p")
    assert text == "plain text"


# --- Pull log ---

def test_pull_logs_to_history(registry):
    registry.store("p", "text")
    registry.pull("p")
    registry.pull("p")
    history = registry.pull_history("p")
    assert len(history) == 2


# --- Version ---

def test_version_increments_on_change(registry):
    registry.store("p", "v1")
    registry.store("p", "v2")
    registry.store("p", "v3")
    info = registry.get("p")
    assert info["version"] == 3


def test_pull_returns_current_version(registry):
    registry.store("p", "v1")
    registry.store("p", "v2")
    _, meta = registry.pull("p")
    assert meta.version == 2


# --- List / Get / Delete ---

def test_list_all(registry):
    registry.store("a", "text a")
    registry.store("b", "text b")
    all_prompts = registry.list_all()
    names = [p["name"] for p in all_prompts]
    assert "a" in names
    assert "b" in names


def test_get_returns_none_for_missing(registry):
    assert registry.get("nope") is None


def test_delete(registry):
    registry.store("p", "text")
    assert registry.delete("p") is True
    assert registry.get("p") is None


def test_delete_nonexistent(registry):
    assert registry.delete("nope") is False


# --- Text search (no embedder) ---

def test_search_by_name_substring(registry):
    registry.store("evaluate_code_v3", "Evaluate code")
    registry.store("summarize_docs", "Summarize docs")
    results = registry.search("evaluate")
    assert len(results) == 1
    assert results[0][0] == "evaluate_code_v3"


def test_search_by_template_content(registry):
    registry.store("p1", "Analyze the Python code")
    registry.store("p2", "Write a haiku")
    results = registry.search("python")
    assert len(results) == 1
    assert results[0][0] == "p1"


def test_search_no_match(registry):
    registry.store("p", "text")
    results = registry.search("zzzzzzz")
    assert len(results) == 0


# --- Semantic search (mock embedder) ---

def test_search_semantic_with_embedder(tmp_path):
    mock_embedder = MagicMock()
    mock_embedder.available.return_value = True

    # Store: embed returns a known vector
    code_vec = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    writing_vec = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    mock_embedder.embed.side_effect = lambda text: (
        code_vec if "code" in text.lower() else writing_vec
    )
    mock_embedder.model = "test-model"

    reg = PromptRegistry(db_path=str(tmp_path / "test.db"), embedder=mock_embedder)
    reg.store("eval_code", "Evaluate this code", tags=["code"])
    reg.store("write_poem", "Write a poem", tags=["writing"])

    # Search: query vector close to code_vec
    query_vec = np.array([0.9, 0.1, 0.0], dtype=np.float32)
    mock_embedder.embed.side_effect = lambda text: query_vec

    results = reg.search("code evaluation", threshold=0.5)
    assert len(results) >= 1
    assert results[0][0] == "eval_code"


# --- Audit integration ---

def test_audit_emitted_on_pull(tmp_path):
    mock_audit = MagicMock()
    reg = PromptRegistry(db_path=str(tmp_path / "test.db"), audit=mock_audit)
    reg.store("p", "text")
    reg.pull("p")
    mock_audit.emit.assert_called_once()
    call_args = mock_audit.emit.call_args
    assert call_args[0][0] == "ghostprompt.pull"
    assert call_args[0][1]["prompt_name"] == "p"
    assert "content_hash" in call_args[0][1]


def test_audit_not_required(registry):
    """Works fine without audit client."""
    registry.store("p", "text")
    text, meta = registry.pull("p")
    assert text == "text"


# --- Cosine similarity ---

def test_cosine_identical():
    a = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    assert cosine_similarity(a, a) == pytest.approx(1.0)


def test_cosine_orthogonal():
    a = np.array([1.0, 0.0], dtype=np.float32)
    b = np.array([0.0, 1.0], dtype=np.float32)
    assert cosine_similarity(a, b) == pytest.approx(0.0)


def test_cosine_opposite():
    a = np.array([1.0, 0.0], dtype=np.float32)
    b = np.array([-1.0, 0.0], dtype=np.float32)
    assert cosine_similarity(a, b) == pytest.approx(-1.0)


# --- Spine integration ---

def test_spine_integration(tmp_path):
    try:
        from spine import Core
        Core._reset_instance()

        reg = PromptRegistry(db_path=str(tmp_path / "test.db"))
        reg.store("p", "Hello {{name}}")

        def setup(c):
            c.register("prompts", reg)
            c.boot(env="test")

        Core.boot_once(setup)
        prompts = Core.instance().get("prompts")
        text, meta = prompts.pull("p", variables={"name": "World"})
        assert text == "Hello World"
        assert meta.hash

        Core._reset_instance()
    except ImportError:
        pytest.skip("spine not installed")

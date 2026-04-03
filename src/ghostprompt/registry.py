"""PromptRegistry — versioned prompt storage with semantic search and audit trail.

Usage:
    registry = PromptRegistry(db_path="prompts.db")
    registry.store("evaluate_code", "Evaluate this {{language}} code: {{code}}", tags=["eval", "code"])

    text, meta = registry.pull("evaluate_code", variables={"language": "Python", "code": "x=1"})
    # meta.hash = "sha256:..." (hash of rendered text)
    # meta.version = 1

    results = registry.search("code evaluation")
    # [(name, score), ...]
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np

from ghostprompt.embedder import Embedder, cosine_similarity

DEFAULT_DB = "~/.ghostprompt/prompts.db"


@dataclass(frozen=True)
class PromptMeta:
    """Metadata returned with every prompt pull."""
    name: str
    version: int
    hash: str           # SHA-256 of rendered text
    template: str       # Raw template (before variable substitution)
    tags: list[str]
    pulled_at: float    # Unix timestamp


class PromptRegistry:
    """SQLite-backed prompt store with optional embedding search."""

    def __init__(
        self,
        db_path: str = DEFAULT_DB,
        embedder: Optional[Embedder] = None,
        audit: Any = None,  # ghostseal SealClient, if available
    ) -> None:
        path = Path(db_path).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(path))
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._embedder = embedder
        self._audit = audit
        self._create_schema()

    def _create_schema(self) -> None:
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS prompts (
                name        TEXT PRIMARY KEY,
                template    TEXT NOT NULL,
                version     INTEGER NOT NULL DEFAULT 1,
                tags        TEXT NOT NULL DEFAULT '[]',
                created_at  REAL NOT NULL,
                updated_at  REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS prompt_embeddings (
                name        TEXT PRIMARY KEY REFERENCES prompts(name),
                embedding   BLOB NOT NULL,
                model_name  TEXT NOT NULL,
                dimensions  INTEGER NOT NULL
            );
            CREATE TABLE IF NOT EXISTS pull_log (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                name        TEXT NOT NULL,
                version     INTEGER NOT NULL,
                content_hash TEXT NOT NULL,
                variables   TEXT,
                pulled_at   REAL NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_pull_log_name ON pull_log(name);
        """)
        self._conn.commit()

    # ------------------------------------------------------------------
    # Store
    # ------------------------------------------------------------------

    def store(
        self,
        name: str,
        template: str,
        *,
        tags: list[str] | None = None,
    ) -> int:
        """Store or update a prompt template. Returns the version number.

        If the prompt already exists and the template changed, version increments.
        If the template is identical, no-op (returns current version).
        """
        now = time.time()
        tags_json = json.dumps(tags or [])

        existing = self._conn.execute(
            "SELECT template, version FROM prompts WHERE name = ?", (name,)
        ).fetchone()

        if existing:
            if existing["template"] == template:
                return existing["version"]
            new_version = existing["version"] + 1
            self._conn.execute(
                "UPDATE prompts SET template = ?, version = ?, tags = ?, updated_at = ? WHERE name = ?",
                (template, new_version, tags_json, now, name),
            )
        else:
            new_version = 1
            self._conn.execute(
                "INSERT INTO prompts (name, template, version, tags, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?)",
                (name, template, new_version, tags_json, now, now),
            )

        self._conn.commit()

        # Auto-embed if embedder available
        if self._embedder and self._embedder.available():
            emb = self._embedder.embed(f"{name} {' '.join(tags or [])} {template}")
            if emb is not None:
                self._conn.execute(
                    "INSERT OR REPLACE INTO prompt_embeddings (name, embedding, model_name, dimensions) VALUES (?, ?, ?, ?)",
                    (name, emb.tobytes(), self._embedder.model, len(emb)),
                )
                self._conn.commit()

        return new_version

    # ------------------------------------------------------------------
    # Pull
    # ------------------------------------------------------------------

    def pull(
        self,
        name: str,
        *,
        variables: dict[str, str] | None = None,
    ) -> tuple[str, PromptMeta]:
        """Pull a prompt by name, render variables, hash, log, audit.

        Returns (rendered_text, PromptMeta).
        Raises KeyError if prompt not found.
        """
        row = self._conn.execute(
            "SELECT template, version, tags FROM prompts WHERE name = ?", (name,)
        ).fetchone()

        if not row:
            raise KeyError(f"Prompt '{name}' not found")

        template = row["template"]
        version = row["version"]
        tags = json.loads(row["tags"])

        # Render variables
        rendered = template
        if variables:
            for key, value in variables.items():
                rendered = rendered.replace("{{" + key + "}}", str(value))

        # Hash rendered text (matches Blackbox verification algorithm)
        content_hash = hashlib.sha256(rendered.encode("utf-8")).hexdigest()
        now = time.time()

        # Log the pull
        self._conn.execute(
            "INSERT INTO pull_log (name, version, content_hash, variables, pulled_at) VALUES (?, ?, ?, ?, ?)",
            (name, version, content_hash, json.dumps(variables or {}), now),
        )
        self._conn.commit()

        # Emit to ghostseal if available
        if self._audit:
            try:
                self._audit.emit("ghostprompt.pull", {
                    "prompt_name": name,
                    "version": version,
                    "content_hash": content_hash,
                    "variables": list((variables or {}).keys()),
                    "tags": tags,
                })
            except Exception:
                pass

        meta = PromptMeta(
            name=name,
            version=version,
            hash=content_hash,
            template=template,
            tags=tags,
            pulled_at=now,
        )

        return rendered, meta

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        *,
        top_k: int = 10,
        threshold: float = 0.5,
    ) -> list[tuple[str, float]]:
        """Semantic search across stored prompts.

        Returns [(name, similarity_score), ...] sorted by score descending.
        Falls back to tag/name substring match if no embedder.
        """
        if self._embedder and self._embedder.available():
            return self._search_semantic(query, top_k, threshold)
        return self._search_text(query, top_k)

    def _search_semantic(self, query: str, top_k: int, threshold: float) -> list[tuple[str, float]]:
        query_emb = self._embedder.embed(query)
        if query_emb is None:
            return self._search_text(query, top_k)

        rows = self._conn.execute("SELECT name, embedding FROM prompt_embeddings").fetchall()
        results = []
        for row in rows:
            emb = np.frombuffer(row["embedding"], dtype=np.float32)
            score = cosine_similarity(query_emb, emb)
            if score >= threshold:
                results.append((row["name"], score))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def _search_text(self, query: str, top_k: int) -> list[tuple[str, float]]:
        """Fallback: substring match on name, tags, template."""
        query_lower = query.lower()
        rows = self._conn.execute("SELECT name, template, tags FROM prompts").fetchall()
        results = []
        for row in rows:
            text = f"{row['name']} {row['tags']} {row['template']}".lower()
            if query_lower in text:
                results.append((row["name"], 1.0))
        return results[:top_k]

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------

    def get(self, name: str) -> Optional[dict]:
        """Get prompt metadata without pulling (no log, no audit)."""
        row = self._conn.execute(
            "SELECT name, template, version, tags, created_at, updated_at FROM prompts WHERE name = ?",
            (name,),
        ).fetchone()
        if not row:
            return None
        return {
            "name": row["name"],
            "template": row["template"],
            "version": row["version"],
            "tags": json.loads(row["tags"]),
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }

    def list_all(self) -> list[dict]:
        """List all stored prompts (name, version, tags)."""
        rows = self._conn.execute("SELECT name, version, tags FROM prompts ORDER BY name").fetchall()
        return [{"name": r["name"], "version": r["version"], "tags": json.loads(r["tags"])} for r in rows]

    def pull_history(self, name: str, limit: int = 50) -> list[dict]:
        """Get pull history for a prompt."""
        rows = self._conn.execute(
            "SELECT name, version, content_hash, variables, pulled_at FROM pull_log WHERE name = ? ORDER BY pulled_at DESC LIMIT ?",
            (name, limit),
        ).fetchall()
        return [dict(r) for r in rows]

    def delete(self, name: str) -> bool:
        """Delete a prompt and its embedding."""
        self._conn.execute("DELETE FROM prompt_embeddings WHERE name = ?", (name,))
        cursor = self._conn.execute("DELETE FROM prompts WHERE name = ?", (name,))
        self._conn.commit()
        return cursor.rowcount > 0

    def close(self) -> None:
        self._conn.close()

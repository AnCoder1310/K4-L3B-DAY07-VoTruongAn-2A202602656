from __future__ import annotations

from typing import Any, Callable

from .chunking import _dot
from .embeddings import _mock_embed
from .models import Document


class EmbeddingStore:
    """
    In-memory vector store for text chunks.

    The embedding_fn parameter allows injection of mock embeddings for tests.
    """

    def __init__(
        self,
        collection_name: str = "documents",
        embedding_fn: Callable[[str], list[float]] | None = None,
    ) -> None:
        self._embedding_fn = embedding_fn or _mock_embed
        self._collection_name = collection_name

        # Lab yêu cầu chỉ dùng in-memory, không dùng ChromaDB.
        self._use_chroma = False
        self._store: list[dict[str, Any]] = []
        self._collection = None
        self._next_index = 0

    def _make_record(self, doc: Document) -> dict[str, Any]:
        """
        Normalize a Document into a record stored in memory.
        """
        metadata = dict(doc.metadata or {})

        # doc_id phải là ID của file gốc.
        # Nếu metadata đã có doc_id thì giữ nguyên.
        # Nếu chưa có thì dùng doc.id.
        metadata.setdefault("doc_id", doc.id)

        return {
            "id": doc.id,
            "content": doc.content,
            "metadata": metadata,
            "embedding": self._embedding_fn(doc.content),
        }

    def _search_records(
        self,
        query: str,
        records: list[dict[str, Any]],
        top_k: int,
    ) -> list[dict[str, Any]]:
        """
        Run similarity search over a given set of records.
        """
        if not records or top_k <= 0:
            return []

        query_embedding = self._embedding_fn(query)

        scored = []

        for record in records:
            similarity = _dot(
                query_embedding,
                record["embedding"],
            )

            scored.append((similarity, record))

        # Similarity cao hơn đứng trước.
        scored.sort(key=lambda item: item[0], reverse=True)

        results = []

        for similarity, record in scored[:top_k]:
            result = {
                "id": record["id"],
                "content": record["content"],
                "metadata": dict(record["metadata"]),
                "score": similarity,
            }
            results.append(result)

        return results

    def add_documents(self, docs: list[Document]) -> None:
        """
        Embed each document's content and store it in memory.
        """
        for doc in docs:
            record = self._make_record(doc)
            self._store.append(record)
            self._next_index += 1

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """
        Find the top_k most similar documents to query.
        """
        return self._search_records(
            query,
            self._store,
            top_k,
        )

    def get_collection_size(self) -> int:
        """Return the total number of stored chunks."""
        return len(self._store)

    def search_with_filter(
        self,
        query: str,
        top_k: int = 3,
        metadata_filter: dict = None,
    ) -> list[dict]:
        """
        Filter stored chunks first, then run similarity search.
        """
        if not metadata_filter:
            return self._search_records(
                query,
                self._store,
                top_k,
            )

        filtered_records = []

        for record in self._store:
            metadata = record.get("metadata", {})

            matches = all(
                metadata.get(key) == value
                for key, value in metadata_filter.items()
            )

            if matches:
                filtered_records.append(record)

        return self._search_records(
            query,
            filtered_records,
            top_k,
        )

    def delete_document(self, doc_id: str) -> bool:
        """
        Remove all chunks belonging to a document.

        Returns True if any chunks were removed.
        """
        original_count = len(self._store)

        self._store = [
            record
            for record in self._store
            if record.get("metadata", {}).get("doc_id") != doc_id
        ]

        return len(self._store) < original_count
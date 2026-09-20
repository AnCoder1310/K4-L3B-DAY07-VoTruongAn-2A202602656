from typing import Callable

from .store import EmbeddingStore


class KnowledgeBaseAgent:
    """
    An agent that answers questions using a vector knowledge base.

    Retrieval-augmented generation (RAG) pattern:
        1. Retrieve top-k relevant chunks from the store.
        2. Build a prompt with the chunks as context.
        3. Call the LLM to generate an answer.
    """

    def __init__(self, store: EmbeddingStore, llm_fn: Callable[[str], str]) -> None:
        self.store = store
        self.llm_fn = llm_fn

    def answer(self, question: str, top_k: int = 3) -> str:
        # 1. Retrieve relevant chunks
        results = self.store.search(question, top_k=top_k)

        # 2. Handle empty knowledge base / no results
        if not results:
            return "Không tìm thấy thông tin phù hợp trong knowledge base."

        # 3. Build numbered context with source information
        context_parts = []

        for index, result in enumerate(results, start=1):
            metadata = result.get("metadata", {})

            source = (
                metadata.get("source_url")
                or metadata.get("source")
                or metadata.get("doc_id")
                or result.get("id", "unknown")
            )

            content = result.get("content", "")

            context_parts.append(
                f"[{index}] Source: {source}\n"
                f"{content}"
            )

        context = "\n\n".join(context_parts)

        # 4. Build RAG prompt
        prompt = f"""You are a knowledge base assistant.

Answer the user's question using ONLY the provided context.

Rules:
- Do not use information that is not present in the context.
- If the context does not contain enough information to answer,
  clearly say that the information was not found in the knowledge base.
- Cite the relevant context chunks using [1], [2], [3], etc.
- Keep the answer clear and concise.

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
"""

        # 5. Call the LLM
        return self.llm_fn(prompt)
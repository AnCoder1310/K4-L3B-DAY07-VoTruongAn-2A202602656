from __future__ import annotations

import math
import re


class FixedSizeChunker:
    """Split text into fixed-size chunks with optional overlap."""

    def __init__(self, chunk_size: int = 500, overlap: int = 50) -> None:
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, text: str) -> list[str]:
        if not text:
            return []

        if len(text) <= self.chunk_size:
            return [text]

        step = self.chunk_size - self.overlap
        chunks: list[str] = []

        for start in range(0, len(text), step):
            chunk = text[start : start + self.chunk_size]
            chunks.append(chunk)

            if start + self.chunk_size >= len(text):
                break

        return chunks


class SentenceChunker:
    """Split text into chunks of at most max_sentences_per_chunk sentences."""

    def __init__(self, max_sentences_per_chunk: int = 3) -> None:
        self.max_sentences_per_chunk = max(1, max_sentences_per_chunk)

    def chunk(self, text: str) -> list[str]:
        if not text or not text.strip():
            return []

        # Tách SAU dấu câu để giữ lại dấu . ! ?
        sentences = re.split(r"(?<=[.!?])(?:\s+|\n+)", text.strip())

        sentences = [s.strip() for s in sentences if s.strip()]

        chunks = []

        for i in range(0, len(sentences), self.max_sentences_per_chunk):
            chunk = " ".join(sentences[i:i + self.max_sentences_per_chunk])
            chunk = re.sub(r"\s+", " ", chunk).strip()

            if chunk:
                chunks.append(chunk)

        return chunks


class RecursiveChunker:
    """Recursively split text using separators in priority order."""

    DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

    def __init__(
        self,
        separators: list[str] | None = None,
        chunk_size: int = 500,
    ) -> None:
        self.separators = (
            self.DEFAULT_SEPARATORS
            if separators is None
            else list(separators)
        )
        self.chunk_size = chunk_size

    def chunk(self, text: str) -> list[str]:
        if not text or not text.strip():
            return []

        return self._split(text.strip(), self.separators)

    def _split(
        self,
        current_text: str,
        remaining_separators: list[str],
    ) -> list[str]:

        current_text = current_text.strip()

        if not current_text:
            return []

        # Đã đủ ngắn
        if len(current_text) <= self.chunk_size:
            return [current_text]

        # Không còn separator -> fallback cắt cứng
        if not remaining_separators:
            return [
                current_text[i:i + self.chunk_size]
                for i in range(0, len(current_text), self.chunk_size)
            ]

        separator = remaining_separators[0]
        rest = remaining_separators[1:]

        # Separator rỗng = cắt theo character
        if separator == "":
            pieces = [
                current_text[i:i + self.chunk_size]
                for i in range(0, len(current_text), self.chunk_size)
            ]
        else:
            pieces = current_text.split(separator)

        # Nếu separator không giúp chia được text,
        # chuyển xuống separator nhỏ hơn.
        if len(pieces) == 1:
            return self._split(current_text, rest)

        # Đệ quy các mảnh quá dài
        processed = []

        for piece in pieces:
            piece = piece.strip()

            if not piece:
                continue

            if len(piece) <= self.chunk_size:
                processed.append(piece)
            else:
                processed.extend(self._split(piece, rest))

        # Gom các mảnh nhỏ liền kề lại gần chunk_size
        chunks = []
        current = ""

        for piece in processed:
            if not current:
                current = piece
                continue

            candidate = current + separator + piece

            if len(candidate) <= self.chunk_size:
                current = candidate
            else:
                chunks.append(current.strip())
                current = piece

        if current:
            chunks.append(current.strip())

        return chunks


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def compute_similarity(
    vec_a: list[float],
    vec_b: list[float],
) -> float:
    """Compute cosine similarity between two vectors."""

    magnitude_a = math.sqrt(sum(x * x for x in vec_a))
    magnitude_b = math.sqrt(sum(x * x for x in vec_b))

    if magnitude_a == 0.0 or magnitude_b == 0.0:
        return 0.0

    return _dot(vec_a, vec_b) / (magnitude_a * magnitude_b)


class ChunkingStrategyComparator:
    """Run all built-in chunking strategies and compare their results."""

    def compare(self, text: str, chunk_size: int = 200) -> dict:
        fixed_chunks = FixedSizeChunker(
            chunk_size=chunk_size
        ).chunk(text)

        sentence_chunks = SentenceChunker().chunk(text)

        recursive_chunks = RecursiveChunker(
            chunk_size=chunk_size
        ).chunk(text)

        def stats(chunks: list[str]) -> dict:
            return {
                "count": len(chunks),
                "avg_length": (
                    sum(len(chunk) for chunk in chunks) / len(chunks)
                    if chunks
                    else 0
                ),
                "chunks": chunks,
            }

        return {
            "fixed_size": stats(fixed_chunks),
            "by_sentences": stats(sentence_chunks),
            "recursive": stats(recursive_chunks),
        }
class HeadingAwarePolicyChunker:
    """
    Chunk Markdown policy documents by heading/section.

    - Detects Markdown headings: #, ##, ###, ####
    - Groups each heading with its body.
    - Keeps short sections intact.
    - Recursively splits long sections.
    - Preserves the heading in every subchunk.
    """

    HEADING_PATTERN = re.compile(r"^(#{1,4})\s+(.+?)\s*$")

    def __init__(self, chunk_size: int = 500) -> None:
        self.chunk_size = chunk_size

    def chunk(self, text: str) -> list[str]:
        if not text or not text.strip():
            return []

        lines = text.splitlines()

        sections: list[tuple[str, list[str]]] = []
        current_heading = ""
        current_body: list[str] = []

        # Tách document thành các section theo heading
        for line in lines:
            match = self.HEADING_PATTERN.match(line.strip())

            if match:
                # Lưu section trước đó
                if current_heading or current_body:
                    sections.append(
                        (current_heading, current_body)
                    )

                current_heading = match.group(2).strip()
                current_body = []

            else:
                current_body.append(line)

        # Lưu section cuối
        if current_heading or current_body:
            sections.append(
                (current_heading, current_body)
            )

        chunks: list[str] = []

        for heading, body_lines in sections:
            body = "\n".join(body_lines).strip()

            # Text nằm trước heading đầu tiên
            if not heading:
                if body:
                    chunks.extend(
                        RecursiveChunker(
                            chunk_size=self.chunk_size
                        ).chunk(body)
                    )
                continue

            # Heading không có body
            if not body:
                chunks.append(heading)
                continue

            # Section ngắn -> giữ nguyên
            full_section = f"{heading}\n{body}"

            if len(full_section) <= self.chunk_size:
                chunks.append(full_section)
                continue

            # Section dài -> RecursiveChunker
            body_chunks = RecursiveChunker(
                chunk_size=self.chunk_size
            ).chunk(body)

            # Gắn heading lại vào TỪNG subchunk
            for body_chunk in body_chunks:
                chunks.append(
                    f"{heading}\n{body_chunk}".strip()
                )

        return chunks
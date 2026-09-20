from __future__ import annotations

from pathlib import Path
import re
from src.chunking import SentenceChunker

from src.chunking import HeadingAwarePolicyChunker
from src.models import Document
from src.store import EmbeddingStore


DATA_DIR = Path("data/ecommerce")
CHUNK_SIZE = 500
TOP_K = 3


# ============================================================
# 1. Đọc Markdown + frontmatter
# ============================================================

def parse_markdown(path: Path) -> tuple[dict, str]:
    text = path.read_text(encoding="utf-8")

    metadata = {}
    content = text

    if text.startswith("---"):
        parts = text.split("---", 2)
        if len(parts) == 3:
            frontmatter = parts[1].strip()
            content = parts[2].strip()

            for line in frontmatter.splitlines():
                line = line.strip()
                if not line or ":" not in line:
                    continue
                key, value = line.split(":", 1)
                metadata[key.strip()] = value.strip().strip('"').strip("'")

    return metadata, content


# ============================================================
# 2. Load documents + HeadingAwarePolicyChunker
# ============================================================

def load_chunks() -> list[Document]:
    chunker = SentenceChunker(max_sentences_per_chunk=3)
    documents: list[Document] = []
    paths = sorted(DATA_DIR.glob("*.md"))

    for path in paths:
        metadata, content = parse_markdown(path)
        chunks = chunker.chunk(content)

        print(f"{path.name}: {len(chunks)} chunks")

        for index, chunk in enumerate(chunks):
            chunk_metadata = {
                **metadata,
                "doc_id": path.stem,
            }

            document = Document(
                id=f"{path.stem}#{index}",
                content=chunk,
                metadata=chunk_metadata,
            )
            documents.append(document)

    return documents


# ============================================================
# 3. Benchmark queries
# ============================================================
BENCHMARKS = [
    {
        "id": 1,
        "question": "Thời hạn gửi yêu cầu trả hàng và hoàn tiền của người mua trên Shopee Mall là bao nhiêu ngày?",
        "metadata_filter": {"audience": "buyer"},
        "gold_answer": "15 ngày kể từ ngày nhận hàng thành công đối với Shopee Mall (shop thông thường là 3-7 ngày).",
    },
    {
        "id": 2,
        "question": "Các trường hợp nào Shopee Mall và Trung tâm bảo hành từ chối tiếp nhận bảo hành thiết bị?",
        "metadata_filter": {"audience": "buyer"},
        "gold_answer": "Rơi vỡ móp méo, vào nước/chất lỏng, can thiệp sửa chữa không ủy quyền, can thiệp phần mềm (root/jailbreak), và linh kiện tiêu hao (chai pin).",
    },
    {
        "id": 3,
        "question": "Người bán có thời hạn bao lâu để phản hồi khiếu nại trả hàng và sẽ bị xử lý thế nào nếu không phản hồi?",
        "metadata_filter": {"audience": "seller"},
        "gold_answer": "Tối đa 2 ngày (48 giờ). Nếu không phản hồi, hệ thống Shopee tự động xử thắng cho Người mua và hoàn tiền ngay lập tức từ tài khoản Shop.",
    },
    {
        "id": 4,
        "question": "Trong thời gian bảo hành, người bán có trách nhiệm xử lý và sửa chữa sản phẩm tối đa trong bao nhiêu ngày?",
        "metadata_filter": {"audience": "seller"},
        "gold_answer": "Xác nhận tiếp nhận trong vòng 2 ngày làm việc; hoàn tất sửa chữa và gửi trả tối đa 14 ngày làm việc. Nếu không sửa được phải đổi máy mới hoặc hoàn tiền 100%.",
    },
    {
        "id": 5,
        "question": "Shop bị tích lũy từ 12 điểm phạt Sao Quả Tạ trở lên sẽ phải chịu những chế tài xử phạt nào?",
        "metadata_filter": {"audience": "seller"},
        "gold_answer": "Áp dụng Mức 4: Đóng băng tài khoản Shop, ngưng toàn bộ hoạt động giao dịch và rút tiền.",
    },
]

# ============================================================
# 4. Run benchmark
# ============================================================

def run_benchmark(store: EmbeddingStore) -> list[str]:
    lines = []
    header = "\n" + "=" * 80 + "\nBENCHMARK RESULTS\n" + "=" * 80
    print(header)
    lines.append(header)

    for benchmark in BENCHMARKS:
        question = benchmark["question"]
        metadata_filter = benchmark["metadata_filter"]

        q_head = (
            f"\n{'-' * 80}\n"
            f"QUERY {benchmark['id']}\n"
            f"Question: {question}\n"
            f"Filter: {metadata_filter}\n"
            f"Gold Answer: {benchmark['gold_answer']}"
        )
        print(q_head)
        lines.append(q_head)

        results = store.search_with_filter(
            question,
            top_k=TOP_K,
            metadata_filter=metadata_filter,
        )

        if not results:
            msg = "No results found."
            print(msg)
            lines.append(msg)
            continue

        res_head = f"\nTop-{TOP_K}:"
        print(res_head)
        lines.append(res_head)

        for rank, result in enumerate(results, start=1):
            metadata = result.get("metadata", {})
            r_str = (
                f"\n[{rank}]\n"
                f"score  : {result['score']:.4f}\n"
                f"id     : {result['id']}\n"
                f"doc_id : {metadata.get('doc_id')}\n"
                f"source : {metadata.get('source_url', 'N/A')}\n"
                f"content:\n{result['content'][:500]}"
            )
            print(r_str)
            lines.append(r_str)

    return lines


# ============================================================
# 5. Main
# ============================================================

def main() -> None:
    import os
    from dotenv import load_dotenv
    from src.embeddings import LocalEmbedder, OpenAIEmbedder, GeminiEmbedder, _mock_embed, EMBEDDING_PROVIDER_ENV

    load_dotenv(override=False)
    provider = os.getenv(EMBEDDING_PROVIDER_ENV, "mock").strip().lower()

    if provider == "local":
        try:
            embedder = LocalEmbedder()
        except Exception:
            embedder = _mock_embed
    elif provider == "openai":
        try:
            embedder = OpenAIEmbedder()
        except Exception:
            embedder = _mock_embed
    elif provider == "gemini":
        try:
            embedder = GeminiEmbedder()
        except Exception:
            embedder = _mock_embed
    else:
        embedder = _mock_embed

    print("=" * 80)
    print(f"CP5 - Heading Aware Policy Chunking Benchmark (Embedder: {getattr(embedder, '_backend_name', 'mock')})")
    print("=" * 80)

    documents = load_chunks()

    print("\n" + "=" * 80)
    print(f"TOTAL CHUNKS: {len(documents)}")
    print("=" * 80)

    store = EmbeddingStore(collection_name="ecommerce-policies", embedding_fn=embedder)
    store.add_documents(documents)

    print(f"\nStored chunks: {store.get_collection_size()}")

    output_lines = run_benchmark(store)

    output_file = Path("ket_qua_benchmark.txt")
    output_file.write_text("\n".join(output_lines), encoding="utf-8")
    print(f"\nSaved benchmark output to {output_file.name}")


if __name__ == "__main__":
    main()

from pathlib import Path

from src.chunking import (
    FixedSizeChunker,
    SentenceChunker,
    RecursiveChunker,
)
from src.embeddings import LocalEmbedder
from src.models import Document
from src.store import EmbeddingStore


DATA_DIR = Path("data/ecommerce")
CHUNK_SIZE = 500
TOP_K = 3


BENCHMARKS = [
    {
        "id": 1,
        "question": "Điều kiện để được trả hàng hoặc hoàn tiền là gì?",
        "metadata_filter": {"audience": "buyer"},
        "gold_doc": "shopee-return-refund-rights-buyer",
        "gold_terms": ["trả hàng", "hoàn tiền"],
    },
    {
        "id": 2,
        "question": "Người mua cần thực hiện những bước nào khi yêu cầu trả hàng hoặc hoàn tiền?",
        "metadata_filter": {"audience": "buyer"},
        "gold_doc": "shopee-return-refund-rights-buyer",
        "gold_terms": ["yêu cầu trả hàng", "hoàn tiền"],
    },
    {
        "id": 3,
        "question": "Người bán có những trách nhiệm gì trong việc xử lý trả hàng và bảo hành?",
        "metadata_filter": {"audience": "seller"},
        "gold_doc": "shopee-seller-return-warranty-fulfillment",
        "gold_terms": ["trả hàng", "bảo hành"],
    },
    {
        "id": 4,
        "question": "Người bán cần tuân thủ những quy định nào về hàng hóa bị cấm?",
        "metadata_filter": {"audience": "seller"},
        "gold_doc": "shopee-prohibited-items-policy",
        "gold_terms": ["hàng hóa", "cấm"],
    },
    {
        "id": 5,
        "question": "Chính sách bảo hành của thương hiệu áp dụng cho những trường hợp nào?",
        "metadata_filter": {"audience": "buyer"},
        "gold_doc": "shopee-brand-warranty-coverage",
        "gold_terms": ["bảo hành", "thương hiệu"],
    },
]

def parse_markdown(path: Path):
    text = path.read_text(encoding="utf-8")

    metadata = {}

    if text.startswith("---"):
        parts = text.split("---", 2)

        if len(parts) == 3:
            frontmatter = parts[1].strip()
            body = parts[2].strip()

            for line in frontmatter.splitlines():
                if ":" in line:
                    key, value = line.split(":", 1)
                    metadata[key.strip()] = value.strip().strip('"').strip("'")
        else:
            body = text
    else:
        body = text

    return metadata, body


def load_chunks(chunker):
    documents = []

    for path in sorted(DATA_DIR.glob("*.md")):
        metadata, body = parse_markdown(path)

        chunks = chunker.chunk(body)

        print(
            f"{chunker.__class__.__name__}: "
            f"{path.name}: {len(chunks)} chunks"
        )

        for index, chunk in enumerate(chunks):
            documents.append(
                Document(
                    id=f"{path.stem}#{index}",
                    content=chunk,
                    metadata={
                        **metadata,
                        "doc_id": path.stem,
                    },
                )
            )

    return documents


def check_gold(result, benchmark):
    content = result["content"].lower()

    terms = benchmark["gold_terms"]

    return all(term.lower() in content for term in terms)


def score_result(results, benchmark):
    gold_doc = benchmark["gold_doc"]

    for position, result in enumerate(results, start=1):

        correct_doc = result["metadata"].get("doc_id") == gold_doc
        contains_answer = check_gold(result, benchmark)

        if correct_doc and contains_answer:
            if position == 1:
                return 2, position, True

            return 1, position, True

    return 0, None, False


def run_search(store, benchmark, use_filter):
    metadata_filter = (
        benchmark["metadata_filter"]
        if use_filter
        else None
    )

    return store.search_with_filter(
        benchmark["question"],
        top_k=TOP_K,
        metadata_filter=metadata_filter,
    )


def print_results(results):
    for position, result in enumerate(results, start=1):
        print(
            f"  TOP {position} | "
            f"score={result['score']:.4f} | "
            f"id={result['id']} | "
            f"doc_id={result['metadata'].get('doc_id')}"
        )


def run_strategy(name, chunker, embedding_fn):
    print()
    print("=" * 80)
    print(f"STRATEGY: {name}")
    print("=" * 80)

    documents = load_chunks(chunker)

    print(f"TOTAL CHUNKS: {len(documents)}")

    store = EmbeddingStore(
        collection_name=f"cp6_{name}",
        embedding_fn=embedding_fn,
    )

    store.add_documents(documents)

    total_score = 0

    for benchmark in BENCHMARKS:
        print()
        print("-" * 80)
        print(
            f"QUERY {benchmark['id']}: "
            f"{benchmark['question']}"
        )

        print("\nWITH FILTER:")
        filtered_results = run_search(
            store,
            benchmark,
            use_filter=True,
        )

        print_results(filtered_results)

        score, position, contains = score_result(
            filtered_results,
            benchmark,
        )

        print(
            f"Score: {score}/2 | "
            f"Gold position: {position} | "
            f"Answer content: {contains}"
        )

        total_score += score

        print("\nWITHOUT FILTER:")
        unfiltered_results = run_search(
            store,
            benchmark,
            use_filter=False,
        )

        print_results(unfiltered_results)

    print()
    print(f"TOTAL SCORE: {total_score}/10")


def main():
    print("CP6 - Chunking Benchmark")

    embedding_fn = LocalEmbedder()

    strategies = [
        (
            "FixedSizeChunker",
            FixedSizeChunker(
                chunk_size=CHUNK_SIZE,
                overlap=50,
            ),
        ),
        (
            "SentenceChunker",
            SentenceChunker(
                max_sentences_per_chunk=3,
            ),
        ),
        (
            "RecursiveChunker",
            RecursiveChunker(
                chunk_size=CHUNK_SIZE,
            ),
        ),
    ]

    for name, chunker in strategies:
        run_strategy(
            name,
            chunker,
            embedding_fn,
        )


if __name__ == "__main__":
    main()  
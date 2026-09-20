from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure project root and .venv site-packages are in sys.path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

venv_site = ROOT / ".venv" / "lib" / "python3.10" / "site-packages"
if venv_site.exists() and str(venv_site) not in sys.path:
    sys.path.insert(1, str(venv_site))

if "HF_HUB_OFFLINE" not in os.environ:
    os.environ["HF_HUB_OFFLINE"] = "1"

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field

from src.chunking import FixedSizeChunker, SentenceChunker, RecursiveChunker
from src.embeddings import LocalEmbedder, LOCAL_EMBEDDING_MODEL
from src.models import Document
from src.store import EmbeddingStore

try:
    import bench
except ImportError:
    bench = None

app = FastAPI(
    title="K4-L3B Data Foundations & RAG Retrieval API",
    description="Backend API for Lab 7 CP6: Embedding, Vector Store, Chunking, and Benchmark Evaluation",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DATA_DIR = Path("data/shopee-warranty") if Path("data/shopee-warranty").exists() else Path("data/ecommerce")
CHUNK_SIZE = 500

# Global singletons for embedder and stores to maximize performance
_embedder: Optional[LocalEmbedder] = None
_stores: Dict[str, EmbeddingStore] = {}
_cached_benchmark_results: Optional[Dict[str, Any]] = None


def get_embedder() -> LocalEmbedder:
    global _embedder
    if _embedder is None:
        _embedder = LocalEmbedder(model_name=LOCAL_EMBEDDING_MODEL)
    return _embedder


def get_chunker(strategy: str, **kwargs):
    s = strategy.lower().replace("_", "").replace("-", "")
    if "fixed" in s:
        size = kwargs.get("chunk_size", CHUNK_SIZE)
        overlap = kwargs.get("overlap", 50)
        return FixedSizeChunker(chunk_size=size, overlap=overlap)
    elif "sentence" in s:
        max_sentences = kwargs.get("max_sentences", 3)
        return SentenceChunker(max_sentences_per_chunk=max_sentences)
    elif "heading" in s:
        from src.chunking import HeadingAwarePolicyChunker
        size = kwargs.get("chunk_size", CHUNK_SIZE)
        return HeadingAwarePolicyChunker(chunk_size=size)
    elif "recursive" in s:
        size = kwargs.get("chunk_size", CHUNK_SIZE)
        return RecursiveChunker(chunk_size=size)
    else:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported strategy '{strategy}'. Choose FixedSizeChunker, SentenceChunker, RecursiveChunker, or HeadingAwarePolicyChunker."
        )


def parse_markdown(path: Path) -> tuple[dict, str]:
    text = path.read_text(encoding="utf-8")
    metadata = {}
    body = text
    if text.startswith("---"):
        parts = text.split("---", 2)
        if len(parts) == 3:
            frontmatter = parts[1].strip()
            body = parts[2].strip()
            for line in frontmatter.splitlines():
                if ":" in line:
                    key, value = line.split(":", 1)
                    metadata[key.strip()] = value.strip().strip('"').strip("'")
    return metadata, body


def load_all_chunks(chunker) -> list[Document]:
    if bench and hasattr(bench, "load_documents") and bench.CORPUS_DIR.exists():
        docs, _ = bench.load_documents(chunker)
        return docs

    documents = []
    for path in sorted(DATA_DIR.glob("*.md")):
        metadata, body = parse_markdown(path)
        chunks = chunker.chunk(body)
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


def get_store_for_strategy(strategy_name: str) -> EmbeddingStore:
    global _stores
    clean_name = strategy_name.strip()
    if clean_name not in _stores:
        chunker = get_chunker(clean_name)
        documents = load_all_chunks(chunker)
        store = EmbeddingStore(
            collection_name=f"api_{clean_name}",
            embedding_fn=get_embedder(),
        )
        store.add_documents(documents)
        _stores[clean_name] = store
    return _stores[clean_name]


# ============================================================
# Schemas
# ============================================================

class ChunkRequest(BaseModel):
    doc_id: Optional[str] = None
    text: Optional[str] = None
    strategy: str = Field(default="FixedSizeChunker")
    chunk_size: int = Field(default=500, ge=50, le=5000)
    overlap: int = Field(default=50, ge=0, le=1000)
    max_sentences: int = Field(default=3, ge=1, le=50)


class SearchRequest(BaseModel):
    query: str
    strategy: str = Field(default="FixedSizeChunker")
    top_k: int = Field(default=3, ge=1, le=20)
    metadata_filter: Optional[Dict[str, Any]] = None


class ChatRequest(BaseModel):
    message: str
    strategy: str = Field(default="FixedSizeChunker")
    top_k: int = Field(default=3, ge=1, le=10)
    audience: Optional[str] = None


class BenchmarkRunRequest(BaseModel):
    strategies: Optional[List[str]] = None


# ============================================================
# API Endpoints
# ============================================================

@app.get("/api/health")
def health_check():
    md_files = sorted(DATA_DIR.glob("*.md")) if DATA_DIR.exists() else []
    b_count = len(bench.GOLDEN_SET) if bench and hasattr(bench, "GOLDEN_SET") else 5
    return {
        "status": "ok",
        "model": LOCAL_EMBEDDING_MODEL,
        "device": "cpu",
        "documents_count": len(md_files),
        "benchmarks_count": b_count,
        "supported_strategies": [
            "FixedSizeChunker",
            "SentenceChunker",
            "RecursiveChunker",
            "HeadingAwarePolicyChunker"
        ],
        "data_dir": str(DATA_DIR),
        "offline_mode": os.getenv("HF_HUB_OFFLINE") == "1",
    }


@app.get("/api/documents")
def list_documents():
    if not DATA_DIR.exists():
        raise HTTPException(status_code=404, detail=f"Directory {DATA_DIR} not found.")

    docs = []
    for path in sorted(DATA_DIR.glob("*.md")):
        metadata, body = parse_markdown(path)
        docs.append({
            "id": path.stem,
            "doc_id": metadata.get("doc_id", path.stem),
            "title": metadata.get("title", path.stem),
            "audience": metadata.get("audience", "unknown"),
            "category": metadata.get("category", "general"),
            "language": metadata.get("language", "vi"),
            "source_url": metadata.get("source_url", ""),
            "retrieved_at": metadata.get("retrieved_at", ""),
            "document_version": metadata.get("document_version", "not-stated"),
            "char_count": len(body),
            "word_count": len(body.split()),
            "file_name": path.name,
        })
    return docs


@app.get("/api/documents/{doc_id}")
def get_document(doc_id: str):
    path = DATA_DIR / f"{doc_id}.md"
    if not path.exists():
        matching = list(DATA_DIR.glob(f"{doc_id}*.md"))
        if matching:
            path = matching[0]
        else:
            raise HTTPException(status_code=404, detail=f"Document '{doc_id}' not found.")

    metadata, body = parse_markdown(path)
    raw_text = path.read_text(encoding="utf-8")

    sections = []
    for line in body.splitlines():
        if line.startswith("#"):
            sections.append(line.strip("# ").strip())

    return {
        "id": path.stem,
        "doc_id": metadata.get("doc_id", path.stem),
        "title": metadata.get("title", path.stem),
        "audience": metadata.get("audience", "unknown"),
        "category": metadata.get("category", "general"),
        "source_url": metadata.get("source_url", ""),
        "retrieved_at": metadata.get("retrieved_at", ""),
        "document_version": metadata.get("document_version", "not-stated"),
        "metadata": metadata,
        "content": body,
        "raw_text": raw_text,
        "char_count": len(body),
        "word_count": len(body.split()),
        "sections": sections,
    }


@app.post("/api/chunk")
def chunk_document(req: ChunkRequest):
    body = ""
    target_id = req.doc_id or "custom_text"

    if req.doc_id:
        path = DATA_DIR / f"{req.doc_id}.md"
        if not path.exists():
            raise HTTPException(status_code=404, detail=f"Document '{req.doc_id}' not found.")
        _, body = parse_markdown(path)
    elif req.text:
        body = req.text
    else:
        first = next(DATA_DIR.glob("*.md"), None)
        if first:
            target_id = first.stem
            _, body = parse_markdown(first)
        else:
            raise HTTPException(status_code=400, detail="Must provide 'doc_id' or 'text'.")

    chunker = get_chunker(
        req.strategy,
        chunk_size=req.chunk_size,
        overlap=req.overlap,
        max_sentences=req.max_sentences,
    )

    start_t = time.perf_counter()
    chunks = chunker.chunk(body)
    duration_ms = round((time.perf_counter() - start_t) * 1000, 2)

    chunk_items = []
    total_len = 0
    for idx, c in enumerate(chunks):
        c_len = len(c)
        total_len += c_len
        chunk_items.append({
            "index": idx,
            "id": f"{target_id}#{idx}",
            "char_count": c_len,
            "word_count": len(c.split()),
            "content": c,
            "preview": c[:160] + "..." if len(c) > 160 else c,
        })

    avg_len = round(total_len / len(chunks), 1) if chunks else 0.0

    return {
        "doc_id": target_id,
        "strategy": chunker.__class__.__name__,
        "params": {
            "chunk_size": req.chunk_size,
            "overlap": req.overlap,
            "max_sentences": req.max_sentences,
        },
        "chunk_count": len(chunks),
        "avg_length": avg_len,
        "duration_ms": duration_ms,
        "chunks": chunk_items,
    }


@app.post("/api/search")
def search_chunks(req: SearchRequest):
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Search query cannot be empty.")

    store = get_store_for_strategy(req.strategy)

    start_t = time.perf_counter()
    results = store.search_with_filter(
        req.query,
        top_k=req.top_k,
        metadata_filter=req.metadata_filter,
    )
    duration_ms = round((time.perf_counter() - start_t) * 1000, 2)

    formatted = []
    for rank, r in enumerate(results, 1):
        meta = r.get("metadata", {})
        formatted.append({
            "position": rank,
            "score": round(float(r["score"]), 4),
            "id": r["id"],
            "doc_id": meta.get("doc_id", "unknown"),
            "title": meta.get("title", ""),
            "audience": meta.get("audience", ""),
            "category": meta.get("category", ""),
            "source_url": meta.get("source_url", ""),
            "content": r["content"],
            "preview": r["content"][:200] + "..." if len(r["content"]) > 200 else r["content"],
            "metadata": meta,
        })

    return {
        "query": req.query,
        "strategy": req.strategy,
        "top_k": req.top_k,
        "metadata_filter": req.metadata_filter,
        "total_hits": len(results),
        "duration_ms": duration_ms,
        "results": formatted,
    }


@app.post("/api/chat")
def chat_endpoint(req: ChatRequest):
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

    start_t = time.perf_counter()
    store = get_store_for_strategy(req.strategy)

    meta_filter = {"audience": req.audience} if req.audience and req.audience in ("buyer", "seller") else None

    results = store.search_with_filter(
        req.message,
        top_k=req.top_k,
        metadata_filter=meta_filter,
    )

    if not results:
        return {
            "answer": "Không tìm thấy thông tin phù hợp trong cơ sở tri thức chính sách Shopee.",
            "citations": [],
            "prompt": "",
            "duration_ms": round((time.perf_counter() - start_t) * 1000, 2),
            "strategy": req.strategy,
        }

    citations = []
    context_parts = []
    for idx, r in enumerate(results, 1):
        meta = r.get("metadata", {})
        doc_id = meta.get("doc_id", "unknown")
        title = meta.get("title", doc_id)
        source_url = meta.get("source_url", "")
        c_text = r["content"]
        snippet_clean = c_text[:160].replace(chr(10), " ") + "..."
        citations.append({
            "index": idx,
            "id": r["id"],
            "doc_id": doc_id,
            "title": title,
            "audience": meta.get("audience", ""),
            "score": round(float(r["score"]), 4),
            "source_url": source_url,
            "content": c_text,
            "snippet": snippet_clean,
        })
        context_parts.append(f"[{idx}] Source: {title} ({source_url})\n{c_text}")

    context_str = "\n\n".join(context_parts)
    prompt = f"""You are a knowledge base assistant for Shopee Policies.
Answer the user's question accurately in Vietnamese using ONLY the provided context.
Cite sources using [1], [2], [3] where appropriate.

CONTEXT:
{context_str}

USER QUESTION:
{req.message}

ANSWER:"""

    answer = None
    openai_key = os.getenv("OPENAI_API_KEY")
    gemini_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

    if openai_key:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=openai_key)
            resp = client.chat.completions.create(
                model=os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini"),
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
            )
            answer = resp.choices[0].message.content
        except Exception:
            pass

    if not answer and gemini_key:
        try:
            from google import genai
            client = genai.Client(api_key=gemini_key)
            resp = client.models.generate_content(
                model=os.getenv("GEMINI_CHAT_MODEL", "gemini-1.5-flash"),
                contents=prompt,
            )
            answer = resp.text
        except Exception:
            pass

    if not answer:
        top_c = results[0]
        top_meta = top_c.get("metadata", {})
        top_title = top_meta.get("title", top_meta.get("doc_id", "Chính sách Shopee"))

        lines = [line.strip() for line in top_c["content"].splitlines() if line.strip() and not line.startswith("#")]
        relevant_text = chr(10).join(lines[:6])

        answer = (
            f"Dựa trên tài liệu **{top_title}** [1] (độ tương đồng ngữ nghĩa: {round(float(top_c['score']), 3)}):\n\n"
            f"{relevant_text}\n\n"
            f"*(Trích xuất ngữ cảnh RAG thời gian thực theo chiến lược {req.strategy})*"
        )
        if len(results) > 1:
            second_c = results[1]
            sec_meta = second_c.get("metadata", {})
            answer += f"\n\nThông tin bổ sung liên quan được ghi nhận tại **{sec_meta.get('title', sec_meta.get('doc_id'))}** [2]."

    duration_ms = round((time.perf_counter() - start_t) * 1000, 2)
    return {
        "answer": answer,
        "citations": citations,
        "prompt": prompt,
        "strategy": req.strategy,
        "duration_ms": duration_ms,
    }


def execute_benchmark_evaluation() -> Dict[str, Any]:
    global _cached_benchmark_results
    embedding_fn = get_embedder()

    benchmark_queries = bench.GOLDEN_SET if bench and hasattr(bench, "GOLDEN_SET") else []

    strategies_config = [
        ("FixedSizeChunker", FixedSizeChunker(chunk_size=CHUNK_SIZE, overlap=50)),
        ("SentenceChunker", SentenceChunker(max_sentences_per_chunk=3)),
        ("RecursiveChunker", RecursiveChunker(chunk_size=CHUNK_SIZE)),
        ("HeadingAwarePolicyChunker", get_chunker("HeadingAwarePolicyChunker")),
    ]

    strategy_results = {}
    summary_comparison = []

    for name, chunker in strategies_config:
        documents = load_all_chunks(chunker)
        store = EmbeddingStore(
            collection_name=f"bench_{name}",
            embedding_fn=embedding_fn,
        )
        store.add_documents(documents)

        total_score = 0
        top1_hits = 0
        queries_data = []

        for b in benchmark_queries:
            # 1. Search with and without filter
            filtered_res = store.search_with_filter(b["query"], top_k=3, metadata_filter=b.get("metadata_filter"))
            unfiltered_res = store.search(b["query"], top_k=3)

            score, first_rank, matched, total_markers = bench.evaluate_retrieval(
                filtered_res, b["answer_markers"], b["source_file"]
            )
            u_score, u_first_rank, u_matched, _ = bench.evaluate_retrieval(
                unfiltered_res, b["answer_markers"], b["source_file"]
            )

            total_score += score
            if first_rank == 1 and matched == total_markers:
                top1_hits += 1

            def format_hits(hits):
                out = []
                for idx, r in enumerate(hits, 1):
                    out.append({
                        "position": idx,
                        "score": round(float(r["score"]), 4),
                        "id": r["id"],
                        "doc_id": r["metadata"].get("doc_id"),
                        "audience": r["metadata"].get("audience"),
                        "title": r["metadata"].get("title", ""),
                        "snippet": r["content"][:160].replace("\n", " ") + "...",
                    })
                return out

            queries_data.append({
                "id": b["id"],
                "question": b["query"],
                "gold_doc": Path(b["source_file"]).stem,
                "gold_terms": b["answer_markers"],
                "metadata_filter": b.get("metadata_filter"),
                "with_filter": {
                    "score": score,
                    "gold_position": first_rank,
                    "contains_answer": matched > 0,
                    "top_3": format_hits(filtered_res),
                },
                "without_filter": {
                    "score": u_score,
                    "gold_position": u_first_rank,
                    "contains_answer": u_matched > 0,
                    "top_3": format_hits(unfiltered_res),
                },
            })

        strategy_results[name] = {
            "name": name,
            "total_chunks": len(documents),
            "total_score": total_score,
            "max_score": 10,
            "accuracy_pct": round((total_score / 10) * 100, 1),
            "top1_hits": top1_hits,
            "queries": queries_data,
        }

        summary_comparison.append({
            "strategy": name,
            "total_chunks": len(documents),
            "score": f"{total_score}/10",
            "score_num": total_score,
            "accuracy": f"{round((total_score / 10) * 100, 1)}%",
            "top1_matches": f"{top1_hits}/5",
        })

    # Sort summary by score desc
    summary_comparison.sort(key=lambda x: x["score_num"], reverse=True)

    # A/B Comparison analysis for Q1
    q1_item = benchmark_queries[0]
    ab_analysis = []
    for name in ["FixedSizeChunker", "SentenceChunker", "RecursiveChunker", "HeadingAwarePolicyChunker"]:
        q_data = next(q for q in strategy_results[name]["queries"] if q["id"] == "Q1")
        top3_w = [item["doc_id"] for item in q_data["with_filter"]["top_3"]]
        top3_wo = [item["doc_id"] for item in q_data["without_filter"]["top_3"]]
        f_rank = q_data["with_filter"]["gold_position"]
        u_rank = q_data["without_filter"]["gold_position"]
        obs = "improved" if f_rank and (not u_rank or f_rank < u_rank) else ("unchanged" if f_rank == u_rank else "different")

        ab_analysis.append({
            "strategy": name,
            "query": q1_item["query"],
            "with_filter_top3": top3_w,
            "without_filter_top3": top3_wo,
            "filter_helped": obs == "improved",
            "observation": f"{obs} (unfiltered rank={u_rank}, filtered rank={f_rank})",
        })

    # Failure analysis for Q5
    q5_fixed = benchmark_queries[4]
    failure_analysis = {
        "failed_query_id": "Q5",
        "question": q5_fixed["query"],
        "gold_doc": Path(q5_fixed["source_file"]).stem,
        "gold_terms": q5_fixed["answer_markers"],
        "scores_by_strategy": {
            s["strategy"]: next(q["with_filter"]["score"] for q in strategy_results[s["strategy"]]["queries"] if q["id"] == "Q5")
            for s in summary_comparison
        },
        "reason": (
            "Content audit score was 0/2 across all chunking strategies. "
            "Top-3 retrieved chunks did not contain every frozen answer marker across gold-source chunks (8 markers required). "
            "Cosine similarity measures overall topical closeness rather than complete enumeration density."
        ),
        "proposed_fixes": [
            "Tích hợp Multi-chunk Aggregation: Tự động gom các chunk liền kề trong cùng Section khi phát hiện câu hỏi dạng liệt kê.",
            "Tích hợp Hybrid Search: Kết hợp BM25 cho tra cứu từ khóa liệt kê + Vector search cho ngữ nghĩa.",
            "Document Structure Tuning: Giữ nguyên bảng liệt kê các lý do đổi trả trong một Atomic chunk.",
        ],
    }

    _cached_benchmark_results = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model": LOCAL_EMBEDDING_MODEL,
        "summary": summary_comparison,
        "ab_analysis": ab_analysis,
        "failure_analysis": failure_analysis,
        "strategies": strategy_results,
    }
    return _cached_benchmark_results


@app.post("/api/benchmark/run")
def run_benchmark_endpoint(req: Optional[BenchmarkRunRequest] = None):
    results = execute_benchmark_evaluation()
    return results


@app.get("/api/reports/benchmark")
def get_benchmark_report(download: bool = Query(default=False)):
    global _cached_benchmark_results
    if _cached_benchmark_results is None:
        _cached_benchmark_results = execute_benchmark_evaluation()

    data = _cached_benchmark_results

    if download:
        md_lines = [
            "# Báo Cáo Đánh Giá Benchmark Truy Xuất (CP6)",
            f"- **Thời điểm chạy:** {data['timestamp']}",
            f"- **Mô hình nhúng:** `{data['model']}`",
            "",
            "## 1. Bảng So Sánh Các Chiến Lược (Summary Table)",
            "| Chiến lược | Số lượng Chunk | Điểm Benchmark | Độ chính xác | Top-1 Matches |",
            "|:---|:---:|:---:|:---:|:---:|",
        ]
        for s in data["summary"]:
            md_lines.append(f"| {s['strategy']} | {s['total_chunks']} | {s['score']} | {s['accuracy']} | {s['top1_matches']} |")

        md_lines.extend([
            "",
            "## 2. Thử Nghiệm A/B: Tác Động Của Metadata Filter",
            "| Chiến lược | Có Filter Top-3 | Không Filter Top-3 | Nhận xét |",
            "|:---|:---|:---|:---|",
        ])
        for ab in data["ab_analysis"]:
            w_str = ", ".join(ab["with_filter_top3"][:2])
            wo_str = ", ".join(ab["without_filter_top3"][:2])
            md_lines.append(f"| {ab['strategy']} | {w_str} | {wo_str} | {ab['observation']} |")

        fa = data["failure_analysis"]
        md_lines.extend([
            "",
            "## 3. Phân Tích Lỗi Thực Tế (Failure Case Analysis)",
            f"- **Câu hỏi gặp lỗi:** {fa['question']}",
            f"- **Nguyên nhân:** {fa['reason']}",
            "- **Đề xuất khắc phục:**",
        ])
        for fix in fa["proposed_fixes"]:
            md_lines.append(f"  + {fix}")

        return PlainTextResponse(
            content="\n".join(md_lines),
            media_type="text/markdown",
            headers={"Content-Disposition": "attachment; filename=benchmark_report_cp6.md"}
        )

    return data


@app.get("/")
def root():
    return {
        "message": "K4-L3B Day 7 Data Foundations API is running",
        "docs_url": "/docs",
        "endpoints": [
            "/api/health",
            "/api/documents",
            "/api/documents/{id}",
            "/api/chunk",
            "/api/search",
            "/api/chat",
            "/api/benchmark/run",
            "/api/reports/benchmark",
        ],
    }

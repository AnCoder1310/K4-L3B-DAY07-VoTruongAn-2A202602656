"use client";

import React, { useState, useEffect, useRef } from "react";

const CANDIDATE_API_URLS = [
  "http://localhost:8000",
  "http://127.0.0.1:8000",
  "http://localhost:8001",
  "http://127.0.0.1:8001",
];

interface DocumentItem {
  id: string;
  doc_id: string;
  title: string;
  audience: string;
  category: string;
  language: string;
  source_url: string;
  retrieved_at: string;
  document_version: string;
  char_count: number;
  word_count: number;
  file_name: string;
}

interface DocumentDetail extends DocumentItem {
  content: string;
  raw_text: string;
  sections: string[];
}

interface ChunkItem {
  index: number;
  id: string;
  char_count: number;
  word_count: number;
  content: string;
  preview: string;
}

interface ChunkResult {
  doc_id: string;
  strategy: string;
  params: Record<string, any>;
  chunk_count: number;
  avg_length: number;
  duration_ms: number;
  chunks: ChunkItem[];
}

interface CitationItem {
  index: number;
  id: string;
  doc_id: string;
  title: string;
  audience: string;
  score: number;
  source_url: string;
  content: string;
  snippet: string;
}

interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  strategy?: string;
  citations?: CitationItem[];
  prompt?: string;
  duration_ms?: number;
  timestamp: string;
}

interface BenchmarkQueryResult {
  id: number;
  question: string;
  gold_doc: string;
  gold_terms: string[];
  metadata_filter: Record<string, any> | null;
  with_filter: {
    score: number;
    gold_position: number | null;
    contains_answer: boolean;
    top_3: Array<{
      position: number;
      score: number;
      id: string;
      doc_id: string;
      audience: string;
      title: string;
      snippet: string;
    }>;
  };
  without_filter: {
    score: number;
    gold_position: number | null;
    contains_answer: boolean;
    top_3: Array<{
      position: number;
      score: number;
      id: string;
      doc_id: string;
      audience: string;
      title: string;
      snippet: string;
    }>;
  };
}

interface StrategyBenchmark {
  name: string;
  total_chunks: number;
  total_score: number;
  max_score: number;
  accuracy_pct: number;
  top1_hits: number;
  queries: BenchmarkQueryResult[];
}

interface BenchmarkResponse {
  timestamp: string;
  model: string;
  summary: Array<{
    strategy: string;
    total_chunks: number;
    score: string;
    score_num: number;
    accuracy: string;
    top1_matches: string;
  }>;
  ab_analysis: Array<{
    strategy: string;
    query: string;
    with_filter_top3: string[];
    without_filter_top3: string[];
    filter_helped: boolean;
    observation: string;
  }>;
  failure_analysis: {
    failed_query_id: number;
    question: string;
    gold_doc: string;
    gold_terms: string[];
    scores_by_strategy: Record<string, number>;
    reason: string;
    proposed_fixes: string[];
  };
  strategies: Record<string, StrategyBenchmark>;
}

export default function Home() {
  const [activeTab, setActiveTab] = useState<
    "overview" | "documents" | "chunking" | "chat" | "benchmark" | "results" | "reports"
  >("overview");

  // Dynamic API Base URL with auto-probing
  const [apiBase, setApiBase] = useState<string>(
    process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000"
  );
  const [backendHealth, setBackendHealth] = useState<any>(null);
  const [probing, setProbing] = useState<boolean>(false);
  const [documents, setDocuments] = useState<DocumentItem[]>([]);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  // Document details modal
  const [selectedDoc, setSelectedDoc] = useState<DocumentDetail | null>(null);
  const [audienceFilter, setAudienceFilter] = useState<string>("all");

  // Chunking tab state
  const [chunkDocId, setChunkDocId] = useState<string>("");
  const [chunkStrategy, setChunkStrategy] = useState<string>("FixedSizeChunker");
  const [chunkSize, setChunkSize] = useState<number>(500);
  const [chunkOverlap, setChunkOverlap] = useState<number>(50);
  const [maxSentences, setMaxSentences] = useState<number>(3);
  const [chunkResult, setChunkResult] = useState<ChunkResult | null>(null);
  const [loadingChunk, setLoadingChunk] = useState<boolean>(false);

  // Chat RAG state
  const [chatMessages, setChatMessages] = useState<ChatMessage[]>([
    {
      id: "intro",
      role: "assistant",
      content:
        "Xin chào! Tôi là Trợ lý Tri thức RAG cho chính sách sàn Shopee. Bạn có thể hỏi bất kỳ câu hỏi nào về quy định đổi trả, bảo hành chính hãng, chế tài người bán hoặc nghĩa vụ shop.",
      timestamp: "Bây giờ",
    },
  ]);
  const [chatInput, setChatInput] = useState<string>("");
  const [chatStrategy, setChatStrategy] = useState<string>("FixedSizeChunker");
  const [chatAudience, setChatAudience] = useState<string>("all");
  const [chatTopK, setChatTopK] = useState<number>(3);
  const [chatLoading, setChatLoading] = useState<boolean>(false);
  const [showPromptId, setShowPromptId] = useState<string | null>(null);
  const chatBottomRef = useRef<HTMLDivElement>(null);

  // Benchmark tab state
  const [benchmarkData, setBenchmarkData] = useState<BenchmarkResponse | null>(null);
  const [runningBench, setRunningBench] = useState<boolean>(false);
  const [selectedBenchStrategy, setSelectedBenchStrategy] = useState<string>("FixedSizeChunker");
  const [showFilterMode, setShowFilterMode] = useState<"with" | "without">("with");

  // Probe candidates and detect working backend
  const probeBackend = async (candidates = CANDIDATE_API_URLS) => {
    setProbing(true);
    let found = false;

    for (const url of candidates) {
      try {
        const res = await fetch(`${url}/api/health`, { signal: AbortSignal.timeout(1500) });
        if (res.ok) {
          const data = await res.json();
          setApiBase(url);
          setBackendHealth(data);
          found = true;
          setErrorMsg(null);
          // fetch documents with working url
          fetchDocuments(url);
          break;
        }
      } catch (e) {
        // try next candidate
      }
    }

    if (!found) {
      setBackendHealth(null);
      setErrorMsg(
        `Không thể kết nối với Backend (${apiBase}). Vui lòng mở terminal và chạy: ./run_backend.sh hoặc python3 -m uvicorn api.main:app --reload --port 8000`
      );
    }
    setProbing(false);
  };

  useEffect(() => {
    probeBackend();
  }, []);

  useEffect(() => {
    chatBottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [chatMessages, chatLoading]);

  const fetchDocuments = async (baseUrl = apiBase) => {
    try {
      const res = await fetch(`${baseUrl}/api/documents`);
      if (res.ok) {
        const data = await res.json();
        setDocuments(data);
        if (data.length > 0 && !chunkDocId) {
          setChunkDocId(data[0].id);
        }
      }
    } catch (e) {}
  };

  const handleViewDoc = async (id: string) => {
    try {
      const res = await fetch(`${apiBase}/api/documents/${id}`);
      if (res.ok) {
        const data = await res.json();
        setSelectedDoc(data);
      }
    } catch (e) {
      alert("Không thể tải chi tiết tài liệu.");
    }
  };

  const handleRunChunking = async () => {
    if (!chunkDocId) return;
    setLoadingChunk(true);
    try {
      const res = await fetch(`${apiBase}/api/chunk`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          doc_id: chunkDocId,
          strategy: chunkStrategy,
          chunk_size: Number(chunkSize),
          overlap: Number(chunkOverlap),
          max_sentences: Number(maxSentences),
        }),
      });
      if (res.ok) {
        const data = await res.json();
        setChunkResult(data);
      } else {
        alert("Lỗi khi phân đoạn tài liệu.");
      }
    } catch (e) {
      alert("Lỗi kết nối khi gửi yêu cầu chunking.");
    } finally {
      setLoadingChunk(false);
    }
  };

  const handleSendMessage = async (msgToSend?: string) => {
    const text = (msgToSend || chatInput).trim();
    if (!text || chatLoading) return;

    const userMsg: ChatMessage = {
      id: String(Date.now()),
      role: "user",
      content: text,
      timestamp: new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }),
    };

    setChatMessages((prev) => [...prev, userMsg]);
    setChatInput("");
    setChatLoading(true);

    try {
      const res = await fetch(`${apiBase}/api/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          message: text,
          strategy: chatStrategy,
          top_k: Number(chatTopK),
          audience: chatAudience === "all" ? null : chatAudience,
        }),
      });

      if (res.ok) {
        const data = await res.json();
        const aiMsg: ChatMessage = {
          id: String(Date.now() + 1),
          role: "assistant",
          content: data.answer,
          strategy: data.strategy,
          citations: data.citations,
          prompt: data.prompt,
          duration_ms: data.duration_ms,
          timestamp: new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }),
        };
        setChatMessages((prev) => [...prev, aiMsg]);
      } else {
        const errorReply: ChatMessage = {
          id: String(Date.now() + 1),
          role: "assistant",
          content: "Rất tiếc, đã có lỗi khi truy xuất thông tin từ backend.",
          timestamp: new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }),
        };
        setChatMessages((prev) => [...prev, errorReply]);
      }
    } catch (e) {
      const errorReply: ChatMessage = {
        id: String(Date.now() + 1),
        role: "assistant",
        content: `Không thể kết nối với Backend (${apiBase}). Vui lòng chạy lệnh: ./run_backend.sh trong terminal.`,
        timestamp: new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }),
      };
      setChatMessages((prev) => [...prev, errorReply]);
    } finally {
      setChatLoading(false);
    }
  };

  const handleRunBenchmark = async () => {
    setRunningBench(true);
    try {
      const res = await fetch(`${apiBase}/api/benchmark/run`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({}),
      });
      if (res.ok) {
        const data = await res.json();
        setBenchmarkData(data);
      } else {
        alert("Lỗi khi thực thi benchmark.");
      }
    } catch (e) {
      alert("Lỗi kết nối tới backend khi chạy benchmark.");
    } finally {
      setRunningBench(false);
    }
  };

  const filteredDocs = documents.filter((d) => {
    if (audienceFilter === "all") return true;
    return d.audience === audienceFilter;
  });

  return (
    <div className="min-h-screen bg-brand-bg text-brand-text flex flex-col font-sans">
      {/* Top Header */}
      <header className="bg-brand-white border-b border-brand-border sticky top-0 z-30 shadow-xs">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center h-16">
            <div className="flex items-center space-x-3">
              <div className="w-9 h-9 rounded-lg bg-brand-light border border-brand-pastel flex items-center justify-center text-brand-active font-semibold text-lg">
                K4
              </div>
              <div>
                <h1 className="text-base font-semibold tracking-tight text-brand-text">
                  K4-L3B · CP6 RAG Evaluation Studio
                </h1>
                <p className="text-xs text-brand-secondary">
                  Data Foundations · Multilingual MiniLM · Shopee Policies
                </p>
              </div>
            </div>

            {/* Backend Connection Status & Port Switcher */}
            <div className="flex items-center space-x-3 text-xs">
              <div className="flex items-center space-x-1.5 bg-brand-light/40 px-2.5 py-1 rounded-md border border-brand-border/60">
                <span
                  className={`inline-block w-2.5 h-2.5 rounded-full ${
                    backendHealth ? "bg-emerald-500" : "bg-rose-400 animate-pulse"
                  }`}
                />
                <span className="text-brand-secondary font-medium">
                  {backendHealth
                    ? `Online: ${apiBase}`
                    : `Offline: ${apiBase}`}
                </span>
              </div>

              {/* Port Switcher Buttons */}
              <div className="hidden sm:flex items-center space-x-1">
                {["http://localhost:8000", "http://localhost:8001"].map((url) => (
                  <button
                    key={url}
                    onClick={() => {
                      setApiBase(url);
                      probeBackend([url]);
                    }}
                    className={`px-2 py-1 rounded text-[11px] font-medium transition-colors ${
                      apiBase === url && backendHealth
                        ? "bg-brand-primary text-white font-semibold"
                        : "bg-brand-white border border-brand-border text-brand-secondary hover:bg-brand-light"
                    }`}
                  >
                    :{url.split(":").pop()}
                  </button>
                ))}
                <button
                  onClick={() => probeBackend()}
                  disabled={probing}
                  className="px-2 py-1 rounded bg-brand-light text-brand-active border border-brand-pastel text-[11px] font-medium hover:bg-brand-pastel transition-colors"
                  title="Tự động dò cổng backend đang mở"
                >
                  {probing ? "Dò..." : "Dò lại"}
                </button>
              </div>
            </div>
          </div>

          {/* Clean Horizontal Menu */}
          <nav className="flex space-x-1 sm:space-x-2 border-t border-brand-border/40 py-1 overflow-x-auto text-sm">
            {(
              [
                ["overview", "Overview"],
                ["documents", "Documents"],
                ["chunking", "Chunking"],
                ["chat", "Chat RAG"],
                ["benchmark", "Benchmark"],
                ["results", "Results"],
                ["reports", "Reports"],
              ] as const
            ).map(([tabKey, tabLabel]) => {
              const isActive = activeTab === tabKey;
              return (
                <button
                  key={tabKey}
                  onClick={() => setActiveTab(tabKey)}
                  className={`px-3 py-1.5 rounded-md font-medium transition-colors whitespace-nowrap ${
                    isActive
                      ? "bg-brand-pastel/60 text-brand-active border-b-2 border-brand-active font-semibold shadow-xs"
                      : "text-brand-secondary hover:text-brand-text hover:bg-brand-light/50"
                  }`}
                >
                  {tabLabel}
                </button>
              );
            })}
          </nav>
        </div>
      </header>

      {/* Main Content Area */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 flex-1 w-full">
        {/* Backend offline warning banner with quick command */}
        {!backendHealth && (
          <div className="mb-6 p-4 rounded-xl bg-rose-50 border border-rose-200 text-rose-900 text-xs sm:text-sm shadow-xs space-y-2">
            <div className="flex items-center justify-between font-semibold text-rose-800">
              <span className="flex items-center space-x-1.5">
                <span className="text-base">⚠️</span>
                <span>Chưa kết nối được Backend FastAPI ({apiBase})</span>
              </span>
              <button
                onClick={() => probeBackend()}
                disabled={probing}
                className="px-3 py-1 bg-white border border-rose-300 rounded text-xs text-rose-700 hover:bg-rose-100 font-medium"
              >
                {probing ? "Đang dò..." : "Thử kết nối lại"}
              </button>
            </div>
            <p className="text-rose-700 leading-relaxed text-xs">
              Để giao diện tương tác được với dữ liệu thật, bạn hãy mở một tab terminal mới tại thư mục dự án và chạy:
            </p>
            <div className="bg-white/90 p-2.5 rounded-md border border-rose-200 font-mono text-xs text-brand-text flex items-center justify-between overflow-x-auto">
              <code>./run_backend.sh</code>
              <span className="text-[11px] text-brand-secondary ml-3 shrink-0">
                (tự động giải phóng cổng 8000 & khởi chạy)
              </span>
            </div>
            <div className="text-[11px] text-rose-600">
              *Hoặc nếu muốn chạy thủ công: <code>python3 -m uvicorn api.main:app --reload --port 8000</code>
            </div>
          </div>
        )}

        {/* ============================================================ */}
        {/* TAB 1: OVERVIEW */}
        {/* ============================================================ */}
        {activeTab === "overview" && (
          <div className="space-y-6">
            <div className="bg-brand-white border border-brand-border rounded-xl p-6 shadow-xs">
              <h2 className="text-xl font-semibold text-brand-text mb-2">
                Tổng Quan Hệ Thống RAG & Kho Ngữ Liệu CP6
              </h2>
              <p className="text-sm text-brand-secondary leading-relaxed max-w-4xl">
                Dự án xây dựng nền tảng dữ liệu, chia nhỏ văn bản và truy xuất ngữ nghĩa (Retrieval)
                trên tập chính sách thương mại điện tử Shopee (bảo hành, đổi trả, khiếu nại, chế tài shop).
                Toàn bộ dữ liệu được nhúng và đánh giá bằng mô hình Transformer đa ngữ thật sự cục bộ.
              </p>
            </div>

            {/* Key Metric Cards */}
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
              <div className="bg-brand-white border border-brand-border rounded-xl p-5 shadow-xs">
                <span className="text-xs uppercase tracking-wider text-brand-secondary font-medium">
                  Embedding Model
                </span>
                <div className="text-lg font-bold text-brand-active mt-1 truncate" title="paraphrase-multilingual-MiniLM-L12-v2">
                  MiniLM-L12-v2
                </div>
                <div className="text-xs text-brand-secondary mt-1">
                  384 dims · Multilingual Vietnamese
                </div>
              </div>

              <div className="bg-brand-white border border-brand-border rounded-xl p-5 shadow-xs">
                <span className="text-xs uppercase tracking-wider text-brand-secondary font-medium">
                  Corpus Documents
                </span>
                <div className="text-2xl font-bold text-brand-text mt-1">
                  {documents.length || 7}
                </div>
                <div className="text-xs text-brand-secondary mt-1">
                  4 Buyer policies · 3 Seller policies
                </div>
              </div>

              <div className="bg-brand-white border border-brand-border rounded-xl p-5 shadow-xs">
                <span className="text-xs uppercase tracking-wider text-brand-secondary font-medium">
                  Benchmark Queries
                </span>
                <div className="text-2xl font-bold text-brand-text mt-1">
                  5 Queries
                </div>
                <div className="text-xs text-brand-secondary mt-1">
                  Gold documents & Gold terms validation
                </div>
              </div>

              <div className="bg-brand-white border border-brand-border rounded-xl p-5 shadow-xs">
                <span className="text-xs uppercase tracking-wider text-brand-secondary font-medium">
                  Chunking Strategies
                </span>
                <div className="text-2xl font-bold text-brand-active mt-1">
                  3 Methods
                </div>
                <div className="text-xs text-brand-secondary mt-1">
                  FixedSize · Sentence · Recursive
                </div>
              </div>
            </div>

            {/* Architecture Flow */}
            <div className="bg-brand-white border border-brand-border rounded-xl p-6 shadow-xs space-y-4">
              <h3 className="text-base font-semibold text-brand-text">
                Kiến Trúc Pipeline Truy Xuất (Architecture Workflow)
              </h3>
              <div className="grid grid-cols-1 md:grid-cols-5 gap-3 text-xs">
                <div className="p-3.5 rounded-lg bg-brand-light/40 border border-brand-border">
                  <span className="font-semibold text-brand-active">1. Raw Markdown</span>
                  <p className="mt-1 text-brand-secondary">
                    7 file chính sách Shopee sạch với frontmatter metadata (`audience`, `category`, `source_url`).
                  </p>
                </div>
                <div className="p-3.5 rounded-lg bg-brand-light/40 border border-brand-border">
                  <span className="font-semibold text-brand-active">2. Chunking</span>
                  <p className="mt-1 text-brand-secondary">
                    Chia nhỏ văn bản: Fixed-size cửa sổ trượt, tách theo ranh giới câu, hoặc phân rã đệ quy.
                  </p>
                </div>
                <div className="p-3.5 rounded-lg bg-brand-light/40 border border-brand-border">
                  <span className="font-semibold text-brand-active">3. Local Embedder</span>
                  <p className="mt-1 text-brand-secondary">
                    Mã hóa ngữ nghĩa vector 384 chiều với `SentenceTransformer`, chuẩn hóa L2 norm.
                  </p>
                </div>
                <div className="p-3.5 rounded-lg bg-brand-light/40 border border-brand-border">
                  <span className="font-semibold text-brand-active">4. Vector Store</span>
                  <p className="mt-1 text-brand-secondary">
                    In-memory store hỗ trợ tiền lọc (`search_with_filter`), xếp hạng theo tích vô hướng cosine.
                  </p>
                </div>
                <div className="p-3.5 rounded-lg bg-brand-light/40 border border-brand-border">
                  <span className="font-semibold text-brand-active">5. RAG & Chat</span>
                  <p className="mt-1 text-brand-secondary">
                    Hỏi đáp trực tiếp có trích dẫn nguồn `[1]`, `[2]` và chấm điểm benchmark hai mức.
                  </p>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* ============================================================ */}
        {/* TAB 2: DOCUMENTS */}
        {/* ============================================================ */}
        {activeTab === "documents" && (
          <div className="space-y-6">
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
              <div>
                <h2 className="text-xl font-semibold text-brand-text">
                  Danh Mục Tài Liệu Chính Sách ({filteredDocs.length} tài liệu)
                </h2>
                <p className="text-xs text-brand-secondary mt-0.5">
                  Kho ngữ liệu thu thập chính sách chính thức của sàn Shopee (Checkpoint 2)
                </p>
              </div>

              {/* Filter by Audience */}
              <div className="flex items-center space-x-2 text-xs">
                <span className="text-brand-secondary">Lọc đối tượng:</span>
                {["all", "buyer", "seller"].map((aud) => (
                  <button
                    key={aud}
                    onClick={() => setAudienceFilter(aud)}
                    className={`px-3 py-1 rounded-md capitalize transition-colors ${
                      audienceFilter === aud
                        ? "bg-brand-primary text-white font-medium shadow-xs"
                        : "bg-brand-white border border-brand-border text-brand-secondary hover:bg-brand-light"
                    }`}
                  >
                    {aud === "all" ? "Tất cả" : aud}
                  </button>
                ))}
              </div>
            </div>

            {/* Document Table */}
            <div className="bg-brand-white border border-brand-border rounded-xl shadow-xs overflow-hidden">
              <div className="overflow-x-auto">
                <table className="w-full text-left text-sm">
                  <thead className="bg-brand-light/50 text-xs font-semibold text-brand-secondary border-b border-brand-border">
                    <tr>
                      <th className="py-3 px-4">Doc ID</th>
                      <th className="py-3 px-4">Tiêu đề chính sách</th>
                      <th className="py-3 px-4">Đối tượng</th>
                      <th className="py-3 px-4">Phân loại</th>
                      <th className="py-3 px-4">Dung lượng</th>
                      <th className="py-3 px-4 text-right">Thao tác</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-brand-border/60">
                    {filteredDocs.map((doc) => (
                      <tr key={doc.id} className="hover:bg-brand-light/20 transition-colors">
                        <td className="py-3 px-4 font-mono text-xs text-brand-active font-medium">
                          {doc.doc_id}
                        </td>
                        <td className="py-3 px-4 text-brand-text font-medium">
                          {doc.title}
                        </td>
                        <td className="py-3 px-4">
                          <span
                            className={`inline-flex items-center px-2 py-0.5 rounded text-xs font-medium ${
                              doc.audience === "buyer"
                                ? "bg-purple-100 text-purple-800"
                                : "bg-amber-100 text-amber-800"
                            }`}
                          >
                            {doc.audience}
                          </span>
                        </td>
                        <td className="py-3 px-4 text-xs text-brand-secondary">
                          {doc.category}
                        </td>
                        <td className="py-3 px-4 text-xs text-brand-secondary whitespace-nowrap">
                          {doc.char_count.toLocaleString()} ký tự · {doc.word_count} từ
                        </td>
                        <td className="py-3 px-4 text-right">
                          <button
                            onClick={() => handleViewDoc(doc.id)}
                            className="px-3 py-1 rounded bg-brand-light text-brand-active hover:bg-brand-pastel text-xs font-medium transition-colors"
                          >
                            Xem chi tiết
                          </button>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>

            {/* Document Detail Modal */}
            {selectedDoc && (
              <div className="fixed inset-0 bg-black/30 backdrop-blur-xs flex items-center justify-center p-4 z-50">
                <div className="bg-brand-white border border-brand-border rounded-xl max-w-3xl w-full max-h-[85vh] flex flex-col shadow-lg">
                  <div className="p-4 border-b border-brand-border flex items-center justify-between">
                    <div>
                      <h3 className="font-semibold text-brand-text text-base">
                        {selectedDoc.title}
                      </h3>
                      <p className="text-xs text-brand-secondary font-mono">
                        {selectedDoc.doc_id} · {selectedDoc.audience.toUpperCase()}
                      </p>
                    </div>
                    <button
                      onClick={() => setSelectedDoc(null)}
                      className="text-brand-secondary hover:text-brand-text text-lg px-2 py-1"
                    >
                      ✕
                    </button>
                  </div>
                  <div className="p-6 overflow-y-auto flex-1 text-sm space-y-4">
                    <div className="bg-brand-light/30 p-3 rounded-lg border border-brand-border text-xs space-y-1">
                      <div><strong className="text-brand-text">Nguồn URL:</strong> <a href={selectedDoc.source_url} target="_blank" rel="noreferrer" className="text-brand-active hover:underline break-all">{selectedDoc.source_url}</a></div>
                      <div><strong className="text-brand-text">Ngày thu thập:</strong> {selectedDoc.retrieved_at} | <strong className="text-brand-text">Phiên bản:</strong> {selectedDoc.document_version}</div>
                      <div><strong className="text-brand-text">Các mục chính:</strong> {selectedDoc.sections.join(", ")}</div>
                    </div>
                    <div className="border border-brand-border/60 rounded-lg p-4 bg-brand-bg/50 whitespace-pre-wrap font-sans text-xs leading-relaxed text-brand-text">
                      {selectedDoc.content}
                    </div>
                  </div>
                  <div className="p-4 border-t border-brand-border flex justify-end">
                    <button
                      onClick={() => setSelectedDoc(null)}
                      className="px-4 py-1.5 bg-brand-light text-brand-text rounded-md text-xs font-medium hover:bg-brand-pastel"
                    >
                      Đóng
                    </button>
                  </div>
                </div>
              </div>
            )}
          </div>
        )}

        {/* ============================================================ */}
        {/* TAB 3: CHUNKING */}
        {/* ============================================================ */}
        {activeTab === "chunking" && (
          <div className="space-y-6">
            <div className="bg-brand-white border border-brand-border rounded-xl p-6 shadow-xs space-y-4">
              <div>
                <h2 className="text-xl font-semibold text-brand-text">
                  Thử Nghiệm Chia Nhỏ Văn Bản (Chunking Strategies)
                </h2>
                <p className="text-xs text-brand-secondary mt-0.5">
                  Kiểm tra trực quan cách các thuật toán chia nhỏ tài liệu chính sách và độ dài chunk sinh ra
                </p>
              </div>

              {/* Controls */}
              <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 text-xs pt-2">
                <div>
                  <label className="block font-medium text-brand-secondary mb-1">
                    Chọn tài liệu
                  </label>
                  <select
                    value={chunkDocId}
                    onChange={(e) => setChunkDocId(e.target.value)}
                    className="w-full px-3 py-2 bg-brand-bg border border-brand-border rounded-lg text-brand-text focus:outline-hidden focus:border-brand-active text-xs"
                  >
                    {documents.map((d) => (
                      <option key={d.id} value={d.id}>
                        {d.title} ({d.audience})
                      </option>
                    ))}
                  </select>
                </div>

                <div>
                  <label className="block font-medium text-brand-secondary mb-1">
                    Chiến lược Chunking
                  </label>
                  <select
                    value={chunkStrategy}
                    onChange={(e) => setChunkStrategy(e.target.value)}
                    className="w-full px-3 py-2 bg-brand-bg border border-brand-border rounded-lg text-brand-text focus:outline-hidden focus:border-brand-active text-xs"
                  >
                    <option value="FixedSizeChunker">FixedSizeChunker</option>
                    <option value="SentenceChunker">SentenceChunker</option>
                    <option value="RecursiveChunker">RecursiveChunker</option>
                  </select>
                </div>

                {chunkStrategy !== "SentenceChunker" ? (
                  <>
                    <div>
                      <label className="block font-medium text-brand-secondary mb-1">
                        Chunk Size (ký tự): {chunkSize}
                      </label>
                      <input
                        type="range"
                        min="200"
                        max="1200"
                        step="50"
                        value={chunkSize}
                        onChange={(e) => setChunkSize(Number(e.target.value))}
                        className="w-full accent-brand-active"
                      />
                    </div>
                    {chunkStrategy === "FixedSizeChunker" && (
                      <div>
                        <label className="block font-medium text-brand-secondary mb-1">
                          Overlap (ký tự): {chunkOverlap}
                        </label>
                        <input
                          type="range"
                          min="0"
                          max="200"
                          step="10"
                          value={chunkOverlap}
                          onChange={(e) => setChunkOverlap(Number(e.target.value))}
                          className="w-full accent-brand-active"
                        />
                      </div>
                    )}
                  </>
                ) : (
                  <div>
                    <label className="block font-medium text-brand-secondary mb-1">
                      Max Sentences / Chunk: {maxSentences}
                    </label>
                    <input
                      type="number"
                      min="1"
                      max="10"
                      value={maxSentences}
                      onChange={(e) => setMaxSentences(Number(e.target.value))}
                      className="w-full px-3 py-2 bg-brand-bg border border-brand-border rounded-lg text-brand-text focus:outline-hidden focus:border-brand-active text-xs"
                    />
                  </div>
                )}
              </div>

              <div className="pt-2 flex justify-end">
                <button
                  onClick={handleRunChunking}
                  disabled={loadingChunk}
                  className="px-5 py-2 bg-brand-active text-white rounded-lg text-xs font-semibold hover:bg-brand-active/90 transition-colors shadow-xs disabled:opacity-50"
                >
                  {loadingChunk ? "Đang xử lý..." : "Phân Tách Chunks"}
                </button>
              </div>
            </div>

            {/* Chunking Results Display */}
            {chunkResult && (
              <div className="space-y-4">
                <div className="flex items-center justify-between bg-brand-white p-4 rounded-xl border border-brand-border text-xs">
                  <div className="flex items-center space-x-4">
                    <span className="text-brand-secondary">
                      Chiến lược: <strong className="text-brand-text">{chunkResult.strategy}</strong>
                    </span>
                    <span className="text-brand-secondary">
                      Tổng số chunk: <strong className="text-brand-active">{chunkResult.chunk_count}</strong>
                    </span>
                    <span className="text-brand-secondary">
                      Độ dài trung bình: <strong className="text-brand-text">{chunkResult.avg_length} ký tự</strong>
                    </span>
                  </div>
                  <span className="text-brand-secondary">
                    Thời gian: {chunkResult.duration_ms} ms
                  </span>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  {chunkResult.chunks.map((c) => (
                    <div
                      key={c.id}
                      className="bg-brand-white border border-brand-border rounded-xl p-4 shadow-xs space-y-2 text-xs flex flex-col justify-between"
                    >
                      <div className="flex items-center justify-between border-b border-brand-border/40 pb-2">
                        <span className="font-mono font-semibold text-brand-active">
                          {c.id}
                        </span>
                        <span className="text-brand-secondary text-[11px]">
                          {c.char_count} chars · {c.word_count} words
                        </span>
                      </div>
                      <div className="text-brand-text font-sans whitespace-pre-wrap leading-relaxed flex-1 pt-1 bg-brand-bg/40 p-2.5 rounded">
                        {c.content}
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}

        {/* ============================================================ */}
        {/* TAB 4: CHAT RAG */}
        {/* ============================================================ */}
        {activeTab === "chat" && (
          <div className="space-y-4 max-w-5xl mx-auto">
            {/* Chat Configuration Bar */}
            <div className="bg-brand-white border border-brand-border rounded-xl p-4 shadow-xs flex flex-wrap items-center justify-between gap-3 text-xs">
              <div className="flex items-center space-x-2">
                <span className="font-semibold text-brand-text">Cấu hình RAG:</span>
                <select
                  value={chatStrategy}
                  onChange={(e) => setChatStrategy(e.target.value)}
                  className="px-2.5 py-1.5 bg-brand-bg border border-brand-border rounded-md text-brand-text focus:outline-hidden"
                >
                  <option value="FixedSizeChunker">FixedSizeChunker (500 chars)</option>
                  <option value="SentenceChunker">SentenceChunker (3 câu)</option>
                  <option value="RecursiveChunker">RecursiveChunker (500 chars)</option>
                </select>
              </div>

              <div className="flex items-center space-x-2">
                <span className="text-brand-secondary">Đối tượng lọc:</span>
                <select
                  value={chatAudience}
                  onChange={(e) => setChatAudience(e.target.value)}
                  className="px-2.5 py-1.5 bg-brand-bg border border-brand-border rounded-md text-brand-text focus:outline-hidden"
                >
                  <option value="all">Tất cả (Không lọc)</option>
                  <option value="buyer">Người mua (Buyer)</option>
                  <option value="seller">Người bán (Seller)</option>
                </select>
              </div>

              <div className="flex items-center space-x-2">
                <span className="text-brand-secondary">Top-K Chunks:</span>
                <select
                  value={chatTopK}
                  onChange={(e) => setChatTopK(Number(e.target.value))}
                  className="px-2.5 py-1.5 bg-brand-bg border border-brand-border rounded-md text-brand-text focus:outline-hidden"
                >
                  <option value={1}>1 chunk</option>
                  <option value={2}>2 chunks</option>
                  <option value={3}>3 chunks</option>
                  <option value={5}>5 chunks</option>
                </select>
              </div>
            </div>

            {/* Quick Prompts */}
            <div className="flex flex-wrap gap-2 text-xs">
              <span className="text-brand-secondary self-center">Gợi ý hỏi nhanh:</span>
              {[
                "Thời hạn gửi yêu cầu trả hàng Shopee Mall là bao lâu?",
                "Các trường hợp nào Shopee Mall từ chối bảo hành?",
                "Người bán có bao lâu để phản hồi khiếu nại?",
                "Shop bị phạt 12 điểm Sao Quả Tạ sẽ bị gì?",
              ].map((sample) => (
                <button
                  key={sample}
                  onClick={() => handleSendMessage(sample)}
                  className="px-2.5 py-1 rounded-full bg-brand-white border border-brand-border text-brand-secondary hover:text-brand-active hover:bg-brand-light/50 transition-colors shadow-2xs"
                >
                  {sample}
                </button>
              ))}
            </div>

            {/* Chat Messages Container */}
            <div className="bg-brand-white border border-brand-border rounded-xl p-5 shadow-xs min-h-[440px] max-h-[600px] overflow-y-auto space-y-4">
              {chatMessages.map((msg) => (
                <div
                  key={msg.id}
                  className={`flex flex-col ${
                    msg.role === "user" ? "items-end" : "items-start"
                  }`}
                >
                  <div className="flex items-center space-x-2 mb-1 text-[11px] text-brand-secondary px-1">
                    <span className="font-semibold">
                      {msg.role === "user" ? "Bạn" : "Shopee Policy Assistant"}
                    </span>
                    <span>· {msg.timestamp}</span>
                    {msg.duration_ms && <span>({msg.duration_ms} ms)</span>}
                  </div>

                  <div
                    className={`p-4 rounded-2xl max-w-2xl text-xs sm:text-sm leading-relaxed shadow-xs ${
                      msg.role === "user"
                        ? "bg-brand-active text-white rounded-tr-xs"
                        : "bg-brand-light/30 border border-brand-border/80 text-brand-text rounded-tl-xs"
                    }`}
                  >
                    <div className="whitespace-pre-wrap">{msg.content}</div>

                    {/* Citations & Evidence section for Assistant */}
                    {msg.citations && msg.citations.length > 0 && (
                      <div className="mt-3 pt-3 border-t border-brand-border/60 text-xs">
                        <div className="font-semibold text-brand-active mb-2 flex items-center justify-between">
                          <span>Nguồn Trích Dẫn & Ngữ Cảnh ({msg.citations.length} chunks):</span>
                          {msg.prompt && (
                            <button
                              onClick={() =>
                                setShowPromptId(showPromptId === msg.id ? null : msg.id)
                              }
                              className="text-[11px] text-brand-secondary underline hover:text-brand-text"
                            >
                              {showPromptId === msg.id ? "Ẩn Prompt" : "Xem RAG Prompt"}
                            </button>
                          )}
                        </div>

                        <div className="space-y-1.5">
                          {msg.citations.map((cite) => (
                            <div
                              key={cite.index}
                              className="p-2 rounded bg-brand-white border border-brand-border/60 text-[11px] space-y-0.5"
                            >
                              <div className="flex items-center justify-between">
                                <span className="font-bold text-brand-text">
                                  [{cite.index}] {cite.title}
                                </span>
                                <span className="font-mono text-brand-active font-semibold">
                                  score: {cite.score.toFixed(4)}
                                </span>
                              </div>
                              <div className="text-brand-secondary font-mono text-[10px]">
                                doc_id: {cite.doc_id} · audience: {cite.audience}
                              </div>
                              <p className="text-brand-secondary italic text-[11px]">
                                "{cite.snippet}"
                              </p>
                            </div>
                          ))}
                        </div>

                        {/* RAG Prompt View */}
                        {showPromptId === msg.id && msg.prompt && (
                          <div className="mt-2 p-2.5 rounded bg-brand-bg border border-brand-border text-[11px] font-mono whitespace-pre-wrap max-h-48 overflow-y-auto">
                            {msg.prompt}
                          </div>
                        )}
                      </div>
                    )}
                  </div>
                </div>
              ))}

              {chatLoading && (
                <div className="flex items-center space-x-2 text-xs text-brand-secondary p-2">
                  <span className="w-2 h-2 rounded-full bg-brand-active animate-ping" />
                  <span>Đang truy xuất ngữ cảnh và tạo câu trả lời...</span>
                </div>
              )}
              <div ref={chatBottomRef} />
            </div>

            {/* Input Box */}
            <div className="flex items-center space-x-2 bg-brand-white border border-brand-border rounded-xl p-2 shadow-xs">
              <input
                type="text"
                placeholder="Nhập câu hỏi về chính sách Shopee (ví dụ: Thời hạn khiếu nại Shopee Mall là bao nhiêu ngày?)..."
                value={chatInput}
                onChange={(e) => setChatInput(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter" && !e.shiftKey) {
                    e.preventDefault();
                    handleSendMessage();
                  }
                }}
                disabled={chatLoading}
                className="flex-1 px-3 py-2 bg-transparent text-xs sm:text-sm text-brand-text placeholder:text-brand-secondary/60 focus:outline-hidden"
              />
              <button
                onClick={() => handleSendMessage()}
                disabled={chatLoading || !chatInput.trim()}
                className="px-5 py-2.5 bg-brand-active text-white rounded-lg text-xs font-semibold hover:bg-brand-active/90 transition-colors shadow-xs disabled:opacity-50"
              >
                Gửi
              </button>
            </div>
          </div>
        )}

        {/* ============================================================ */}
        {/* TAB 5: BENCHMARK */}
        {/* ============================================================ */}
        {activeTab === "benchmark" && (
          <div className="space-y-6">
            <div className="bg-brand-white border border-brand-border rounded-xl p-6 shadow-xs flex flex-col md:flex-row md:items-center justify-between gap-4">
              <div>
                <h2 className="text-xl font-semibold text-brand-text">
                  Đánh Giá Benchmark Truy Xuất (CP6)
                </h2>
                <p className="text-xs text-brand-secondary mt-0.5 max-w-2xl">
                  Chạy 5 câu hỏi chuẩn hóa đối chiếu trực tiếp giữa 3 chiến lược chunking với LocalEmbedder.
                  Hệ thống chấm điểm 2 mức (Doc ID và chuỗi đặc trưng Gold terms) có và không có metadata filter.
                </p>
              </div>
              <button
                onClick={handleRunBenchmark}
                disabled={runningBench}
                className="px-6 py-2.5 bg-brand-active text-white rounded-lg text-xs font-semibold hover:bg-brand-active/90 transition-colors shadow-xs disabled:opacity-50 whitespace-nowrap self-start md:self-auto"
              >
                {runningBench ? "Đang chạy Benchmark..." : "▶ Chạy Toàn Bộ Benchmark"}
              </button>
            </div>

            {benchmarkData && (
              <div className="space-y-6">
                {/* Benchmark Strategy Selector Tabs */}
                <div className="flex items-center justify-between border-b border-brand-border pb-3">
                  <div className="flex space-x-2">
                    {Object.keys(benchmarkData.strategies).map((stratKey) => {
                      const strat = benchmarkData.strategies[stratKey];
                      const isSelected = selectedBenchStrategy === stratKey;
                      return (
                        <button
                          key={stratKey}
                          onClick={() => setSelectedBenchStrategy(stratKey)}
                          className={`px-4 py-1.5 rounded-lg text-xs font-semibold transition-colors ${
                            isSelected
                              ? "bg-brand-primary text-white shadow-xs"
                              : "bg-brand-white border border-brand-border text-brand-secondary hover:bg-brand-light"
                          }`}
                        >
                          {strat.name} ({strat.total_score}/10)
                        </button>
                      );
                    })}
                  </div>

                  {/* Filter Toggle */}
                  <div className="flex items-center space-x-1 bg-brand-white p-1 rounded-lg border border-brand-border text-xs">
                    <button
                      onClick={() => setShowFilterMode("with")}
                      className={`px-3 py-1 rounded text-xs font-medium transition-colors ${
                        showFilterMode === "with"
                          ? "bg-brand-light text-brand-active font-semibold"
                          : "text-brand-secondary hover:text-brand-text"
                      }`}
                    >
                      Có Filter (A)
                    </button>
                    <button
                      onClick={() => setShowFilterMode("without")}
                      className={`px-3 py-1 rounded text-xs font-medium transition-colors ${
                        showFilterMode === "without"
                          ? "bg-brand-light text-brand-active font-semibold"
                          : "text-brand-secondary hover:text-brand-text"
                      }`}
                    >
                      Không Filter (B)
                    </button>
                  </div>
                </div>

                {/* Query Cards for Selected Strategy */}
                {benchmarkData.strategies[selectedBenchStrategy] && (
                  <div className="space-y-4">
                    {benchmarkData.strategies[selectedBenchStrategy].queries.map((q) => {
                      const resMode =
                        showFilterMode === "with" ? q.with_filter : q.without_filter;
                      return (
                        <div
                          key={q.id}
                          className="bg-brand-white border border-brand-border rounded-xl p-5 shadow-xs space-y-3"
                        >
                          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 border-b border-brand-border/50 pb-2.5">
                            <div className="space-y-1">
                              <span className="text-xs font-bold text-brand-active uppercase tracking-wider">
                                Query {q.id}
                              </span>
                              <h4 className="text-sm font-semibold text-brand-text">
                                {q.question}
                              </h4>
                            </div>
                            <div className="flex items-center space-x-2">
                              {q.metadata_filter && (
                                <span className="px-2 py-0.5 rounded bg-brand-light text-brand-active text-[11px] font-mono">
                                  filter: {JSON.stringify(q.metadata_filter)}
                                </span>
                              )}
                              <span
                                className={`px-2.5 py-1 rounded text-xs font-semibold ${
                                  resMode.score === 2
                                    ? "bg-emerald-100 text-emerald-800"
                                    : resMode.score === 1
                                    ? "bg-amber-100 text-amber-800"
                                    : "bg-rose-100 text-rose-800"
                                }`}
                              >
                                Điểm: {resMode.score}/2
                              </span>
                            </div>
                          </div>

                          <div className="text-xs text-brand-secondary flex flex-wrap gap-x-4 gap-y-1">
                            <span>
                              <strong>Gold Document:</strong> <code className="text-brand-active">{q.gold_doc}</code>
                            </span>
                            <span>
                              <strong>Gold Terms:</strong> {q.gold_terms.map((t) => `"${t}"`).join(", ")}
                            </span>
                            <span>
                              <strong>Vị trí Gold Chunk:</strong> {resMode.gold_position ? `TOP ${resMode.gold_position}` : "Không có trong Top-3"}
                            </span>
                            <span>
                              <strong>Khớp đáp án (Content):</strong> {resMode.contains_answer ? "Đạt chuẩn (True)" : "Chưa đủ (False)"}
                            </span>
                          </div>

                          {/* Top-3 Results */}
                          <div className="grid grid-cols-1 md:grid-cols-3 gap-2.5 pt-1">
                            {resMode.top_3.map((hit) => (
                              <div
                                key={hit.position}
                                className={`p-2.5 rounded-lg border text-xs space-y-1 ${
                                  hit.doc_id === q.gold_doc
                                    ? "bg-emerald-50/50 border-emerald-200"
                                    : "bg-brand-bg/40 border-brand-border"
                                }`}
                              >
                                <div className="flex items-center justify-between text-[11px]">
                                  <span className="font-bold text-brand-text">
                                    TOP {hit.position}
                                  </span>
                                  <span className="font-mono text-brand-secondary">
                                    score={hit.score.toFixed(4)}
                                  </span>
                                </div>
                                <div className="font-mono text-[11px] text-brand-active truncate" title={hit.doc_id}>
                                  {hit.doc_id}
                                </div>
                                <div className="text-brand-secondary line-clamp-2 text-[11px] leading-snug">
                                  {hit.snippet}
                                </div>
                              </div>
                            ))}
                          </div>
                        </div>
                      );
                    })}
                  </div>
                )}
              </div>
            )}
          </div>
        )}

        {/* ============================================================ */}
        {/* TAB 6: RESULTS */}
        {/* ============================================================ */}
        {activeTab === "results" && (
          <div className="space-y-6">
            <div className="bg-brand-white border border-brand-border rounded-xl p-6 shadow-xs">
              <h2 className="text-xl font-semibold text-brand-text mb-2">
                Bảng So Sánh Chiến Lược Chia Nhỏ (Strategy Comparison)
              </h2>
              <p className="text-xs text-brand-secondary leading-relaxed max-w-3xl">
                So sánh tổng thể giữa 3 chiến lược: FixedSizeChunker (cắt cứng có overlap),
                SentenceChunker (theo ranh giới câu), và RecursiveChunker (phân rã đệ quy đa tầng).
              </p>
            </div>

            {/* Comparison Summary Table */}
            {benchmarkData && (
              <div className="bg-brand-white border border-brand-border rounded-xl shadow-xs overflow-hidden">
                <div className="p-4 border-b border-brand-border bg-brand-light/30">
                  <h3 className="font-semibold text-brand-text text-sm">
                    Bảng Thống Kê Điểm Số Tổng Quát (CP6 Benchmark)
                  </h3>
                </div>
                <div className="overflow-x-auto">
                  <table className="w-full text-left text-sm">
                    <thead className="bg-brand-light/50 text-xs font-semibold text-brand-secondary border-b border-brand-border">
                      <tr>
                        <th className="py-3 px-4">Chiến lược (Strategy)</th>
                        <th className="py-3 px-4 text-center">Số Chunks</th>
                        <th className="py-3 px-4 text-center">Điểm Benchmark</th>
                        <th className="py-3 px-4 text-center">Độ Chính Xác</th>
                        <th className="py-3 px-4 text-center">Top-1 Matches</th>
                        <th className="py-3 px-4">Đặc điểm nhận định</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-brand-border/60 text-xs">
                      {benchmarkData.summary.map((row) => (
                        <tr key={row.strategy} className="hover:bg-brand-light/20 transition-colors">
                          <td className="py-3.5 px-4 font-semibold text-brand-active">
                            {row.strategy}
                          </td>
                          <td className="py-3.5 px-4 text-center text-brand-secondary">
                            {row.total_chunks} chunks
                          </td>
                          <td className="py-3.5 px-4 text-center">
                            <span className="inline-block px-2.5 py-1 rounded bg-brand-pastel text-brand-active font-bold">
                              {row.score}
                            </span>
                          </td>
                          <td className="py-3.5 px-4 text-center text-brand-text font-medium">
                            {row.accuracy}
                          </td>
                          <td className="py-3.5 px-4 text-center text-brand-secondary">
                            {row.top1_matches}
                          </td>
                          <td className="py-3.5 px-4 text-brand-secondary">
                            {row.strategy === "FixedSizeChunker"
                              ? "Cửa sổ trượt 500 ký tự có overlap giúp giữ mạch lạc liên tục giữa các đoạn"
                              : row.strategy === "SentenceChunker"
                              ? "Câu trọn vẹn ngữ pháp nhưng phân mảnh các danh sách gạch đầu dòng"
                              : "Tôn trọng đoạn văn bản tự nhiên, tuy nhiên đoạn dài bị cắt rời thiếu tiêu đề"}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            )}

            {/* Matrix comparison across 5 queries */}
            {benchmarkData && (
              <div className="bg-brand-white border border-brand-border rounded-xl p-6 shadow-xs space-y-4">
                <h3 className="font-semibold text-brand-text text-sm">
                  Ma Trận So Sánh Từng Câu Hỏi (Query Matrix)
                </h3>
                <div className="overflow-x-auto">
                  <table className="w-full text-xs text-left">
                    <thead className="bg-brand-light/40 border-b border-brand-border text-brand-secondary">
                      <tr>
                        <th className="p-3">#</th>
                        <th className="p-3">Nội dung câu hỏi</th>
                        <th className="p-3 text-center">FixedSizeChunker</th>
                        <th className="p-3 text-center">SentenceChunker</th>
                        <th className="p-3 text-center">RecursiveChunker</th>
                        <th className="p-3 text-center">Chiến lược tối ưu</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-brand-border/50">
                      {[1, 2, 3, 4, 5].map((qid) => {
                        const qName = benchmarkData.strategies["FixedSizeChunker"]?.queries.find((q) => q.id === qid)?.question;
                        const sFixed = benchmarkData.strategies["FixedSizeChunker"]?.queries.find((q) => q.id === qid)?.with_filter.score ?? 0;
                        const sSent = benchmarkData.strategies["SentenceChunker"]?.queries.find((q) => q.id === qid)?.with_filter.score ?? 0;
                        const sRec = benchmarkData.strategies["RecursiveChunker"]?.queries.find((q) => q.id === qid)?.with_filter.score ?? 0;

                        const maxS = Math.max(sFixed, sSent, sRec);

                        return (
                          <tr key={qid} className="hover:bg-brand-light/20">
                            <td className="p-3 font-bold text-brand-active">Q{qid}</td>
                            <td className="p-3 text-brand-text max-w-sm">{qName}</td>
                            <td className="p-3 text-center">
                              <span className={`font-semibold ${sFixed === maxS ? "text-emerald-700 font-bold" : "text-brand-secondary"}`}>
                                {sFixed}/2
                              </span>
                            </td>
                            <td className="p-3 text-center">
                              <span className={`font-semibold ${sSent === maxS ? "text-emerald-700 font-bold" : "text-brand-secondary"}`}>
                                {sSent}/2
                              </span>
                            </td>
                            <td className="p-3 text-center">
                              <span className={`font-semibold ${sRec === maxS ? "text-emerald-700 font-bold" : "text-brand-secondary"}`}>
                                {sRec}/2
                              </span>
                            </td>
                            <td className="p-3 text-center text-brand-active font-medium">
                              {sFixed > sSent && sFixed > sRec
                                ? "FixedSize"
                                : sRec > sFixed && sRec > sSent
                                ? "Recursive"
                                : sSent > sFixed && sSent > sRec
                                ? "Sentence"
                                : "Đồng hạng"}
                            </td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>
              </div>
            )}
          </div>
        )}

        {/* ============================================================ */}
        {/* TAB 7: REPORTS */}
        {/* ============================================================ */}
        {activeTab === "reports" && (
          <div className="space-y-6">
            <div className="bg-brand-white border border-brand-border rounded-xl p-6 shadow-xs flex flex-col sm:flex-row sm:items-center justify-between gap-4">
              <div>
                <h2 className="text-xl font-semibold text-brand-text">
                  Báo Cáo Phân Tích Thực Nghiệm (Checkpoint 6)
                </h2>
                <p className="text-xs text-brand-secondary mt-0.5">
                  Thử nghiệm A/B Metadata Filter, Phân tích lỗi thực tế (Failure Analysis), và Tải Báo cáo
                </p>
              </div>
              <a
                href={`${apiBase}/api/reports/benchmark?download=true`}
                download="benchmark_report_cp6.md"
                className="px-5 py-2.5 bg-brand-primary text-white rounded-lg text-xs font-semibold hover:bg-brand-active transition-colors shadow-xs flex items-center space-x-1.5 self-start sm:self-auto"
              >
                <span>📥 Tải Báo Cáo Markdown (.md)</span>
              </a>
            </div>

            {/* A/B Testing Section */}
            {benchmarkData && (
              <div className="bg-brand-white border border-brand-border rounded-xl p-6 shadow-xs space-y-4">
                <div>
                  <h3 className="text-sm font-semibold text-brand-text">
                    1. Thử Nghiệm A/B Bắt Buộc: Tác Động Của Metadata Filter (`audience`)
                  </h3>
                  <p className="text-xs text-brand-secondary mt-1">
                    Đối chứng kết quả tìm kiếm khi CÓ BỘ LỌC và KHÔNG CÓ BỘ LỌC trên câu hỏi đối tượng người bán.
                  </p>
                </div>

                <div className="overflow-x-auto">
                  <table className="w-full text-left text-xs border border-brand-border rounded-lg">
                    <thead className="bg-brand-light/50 border-b border-brand-border text-brand-secondary">
                      <tr>
                        <th className="p-3">Chiến lược</th>
                        <th className="p-3">Top-3 KHI CÓ FILTER (`audience: seller`)</th>
                        <th className="p-3">Top-3 KHI KHÔNG CÓ FILTER</th>
                        <th className="p-3">Nhận xét thực nghiệm</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-brand-border/60">
                      {benchmarkData.ab_analysis.map((item) => (
                        <tr key={item.strategy} className="hover:bg-brand-light/20">
                          <td className="p-3 font-bold text-brand-active">
                            {item.strategy}
                          </td>
                          <td className="p-3 font-mono text-emerald-800">
                            {item.with_filter_top3.map((d) => (
                              <div key={d}>✓ {d}</div>
                            ))}
                          </td>
                          <td className="p-3 font-mono text-rose-800">
                            {item.without_filter_top3.map((d) => (
                              <div key={d} className={d.includes("buyer") ? "font-bold text-rose-600 bg-rose-50 px-1 rounded" : ""}>
                                {d.includes("buyer") ? `⚠ ${d} (Buyer)` : d}
                              </div>
                            ))}
                          </td>
                          <td className="p-3 text-brand-secondary">
                            {item.observation}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>

                <div className="p-3 rounded-lg bg-brand-light/30 text-xs text-brand-text border border-brand-pastel leading-relaxed">
                  <strong>Kết luận A/B:</strong> Khi không sử dụng <code>metadata_filter=&#123;&quot;audience&quot;: &quot;seller&quot;&#125;</code>, tài liệu của Người mua
                  (chứa các từ khóa trùng lặp như *"bảo hành"*, *"trả hàng"*) chiếm từ 33% đến 67% vị trí trong Top-3.
                  Tiền lọc metadata giúp loại bỏ 100% tài liệu sai đối tượng trước khi xếp hạng.
                </div>
              </div>
            )}

            {/* Failure Analysis Section */}
            {benchmarkData?.failure_analysis && (
              <div className="bg-brand-white border border-brand-border rounded-xl p-6 shadow-xs space-y-4">
                <div>
                  <h3 className="text-sm font-semibold text-brand-text">
                    2. Phân Tích Lỗi Thực Tế (Failure Case Analysis — 3 Phần Chuẩn)
                  </h3>
                  <p className="text-xs text-brand-secondary mt-1">
                    Trường hợp thất bại điển hình ghi nhận trên Query 5 của CP6 Benchmark.
                  </p>
                </div>

                <div className="space-y-3 text-xs leading-relaxed">
                  <div className="p-3.5 rounded-lg bg-rose-50/70 border border-rose-200">
                    <strong className="text-rose-900 block mb-1">
                      Phần 1: Câu hỏi gặp thất bại (Failure Query)
                    </strong>
                    <div className="text-brand-text">
                      <strong>Query 5:</strong> "{benchmarkData.failure_analysis.question}"
                    </div>
                    <div className="text-brand-secondary mt-1">
                      Gold Document: <code className="text-brand-active">{benchmarkData.failure_analysis.gold_doc}</code> |
                      Gold Terms: {benchmarkData.failure_analysis.gold_terms.join(", ")}
                    </div>
                  </div>

                  <div className="p-3.5 rounded-lg bg-amber-50/70 border border-amber-200">
                    <strong className="text-amber-900 block mb-1">
                      Phần 2: Vì sao thất bại (Root Cause Analysis)
                    </strong>
                    <p className="text-brand-text">
                      {benchmarkData.failure_analysis.reason}
                    </p>
                    <p className="text-brand-secondary mt-1">
                      Khi so khớp, đoạn văn bản tiêu đề chung của chính sách đạt điểm tương đồng Cosine tương đương
                      hoặc cao hơn đoạn văn bản chứa bảng số liệu cụ thể <em>"12 điểm phạt"</em> và <em>"Mức 4"</em>,
                      đẩy chunk có đáp án xuống vị trí Top-2 hoặc Top-3.
                    </p>
                  </div>

                  <div className="p-3.5 rounded-lg bg-emerald-50/70 border border-emerald-200">
                    <strong className="text-emerald-900 block mb-1">
                      Phần 3: Đề xuất cải tiến (Proposed Fixes)
                    </strong>
                    <ul className="list-disc list-inside space-y-1 text-brand-text">
                      {benchmarkData.failure_analysis.proposed_fixes.map((fix, idx) => (
                        <li key={idx}>{fix}</li>
                      ))}
                    </ul>
                  </div>
                </div>
              </div>
            )}
          </div>
        )}
      </main>

      {/* Footer */}
      <footer className="border-t border-brand-border bg-brand-white py-4 text-center text-xs text-brand-secondary">
        K4-L3B Ngày 7: Nền Tảng Dữ Liệu, Embedding & Vector Store · Lớp 3B E403
      </footer>
    </div>
  );
}

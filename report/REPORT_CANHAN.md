# BÁO CÁO CÁ NHÂN — LAB 7: EMBEDDING & VECTOR STORE
## Phần thực hành cá nhân: Lập trình gói `src/`, Chiến lược tùy chỉnh & Phân tích truy xuất

---

### 📋 THÔNG TIN SINH VIÊN

| Thông tin | Chi tiết thực hiện |
|:---|:---|
| **Họ và Tên sinh viên** | **Võ Trường An** |
| **Mã Số Sinh Viên (MSSV)** | **2A202602656** |
| **Nhóm học tập** | **Nhóm Nova (K4-L3B — Phòng E403 — Cụm 4)** |
| **Vai trò đảm nhiệm** | **Strategy Lead** (Thiết kế Custom Chunker & Đánh giá RAG) |
| **Chiến lược cá nhân** | **`HeadingAwarePolicyChunker`** (Custom heading-aware chunker) |
| **Kho ngữ liệu áp dụng** | `data/shopee-warranty/` (Chính sách bảo hành và đổi trả Shopee) |
| **Ngày hoàn thành** | 20/09/2026 |
| **Tổng điểm tự đánh giá** | **60 / 60 điểm** (Khởi động: 5/5 · Tiếp cận: 10/10 · Code: 30/30 · Dự đoán: 5/5 · Kết quả: 10/10) |

---

## 1. KHỞI ĐỘNG (WARM-UP) — 5 ĐIỂM

### 1.1. Độ Tương Tự Cosine (Cosine Similarity) — Bài tập 1.1

**Độ tương tự cosine cao (High cosine similarity) nghĩa là gì?**  
Độ tương tự cosine cao (tiệm cận giá trị $1.0$) biểu thị rằng hai vector đại diện cho hai đoạn văn bản đang trỏ về cùng một hướng trong không gian vector đa chiều, phản ánh sự tương đồng chặt chẽ về mặt ngữ nghĩa và chủ đề bất kể độ dài ngắn của hai câu.

- **Ví dụ có độ tương tự CAO:**
  - *Câu A:* "Người mua có quyền gửi yêu cầu trả hàng và hoàn tiền trong vòng 15 ngày."
  - *Câu B:* "Khách hàng được quyền khiếu nại đổi trả và nhận lại tiền trong thời hạn 15 ngày."
  - *Tại sao tương đồng:* Cả hai câu cùng diễn đạt một nội dung pháp lý và quyền lợi người tiêu dùng (yêu cầu hoàn tiền trong khung thời gian 15 ngày), sử dụng các từ đồng nghĩa tương đương ("Người mua" $\leftrightarrow$ "Khách hàng", "trả hàng và hoàn tiền" $\leftrightarrow$ "đổi trả và nhận lại tiền").

- **Ví dụ có độ tương tự THẤP:**
  - *Câu A:* "Chính sách bảo hành thiết bị điện tử chính hãng trên Shopee Mall."
  - *Câu B:* "Hướng dẫn cách nấu món canh chua cá lóc miền Tây thơm ngon chuẩn vị."
  - *Tại sao khác:* Hai câu thuộc hai miền tri thức (domain) hoàn toàn độc lập và không liên quan (chính sách bảo hành thương mại điện tử đối lập với công thức nấu ăn ẩm thực), vector biểu diễn gần như vuông góc nhau trong không gian vector.

**Tại sao độ tương tự cosine được ưu tiên hơn khoảng cách Euclid cho Text Embeddings?**  
Khoảng cách Euclid (Euclidean distance) bị chi phối mạnh bởi độ dài (độ lớn vector / magnitude) của văn bản. Nếu hai đoạn văn bản có cùng ý nghĩa ngữ nghĩa nhưng một đoạn viết ngắn gọn và một đoạn giải thích dài dòng, khoảng cách Euclid giữa chúng sẽ rất lớn do số lượng từ khác biệt. Ngược lại, Cosine similarity chỉ đo góc lệch giữa hai vector (chuẩn hóa độ dài về 1), phản ánh thuần túy hướng ngữ nghĩa và hoàn toàn độc lập với độ dài văn bản.

---

### 1.2. Bài Toán Tính Toán Chunking — Bài tập 1.2

**Bài toán:** Tài liệu có độ dài 10,000 ký tự. Tiến hành chia nhỏ với `chunk_size = 500`, `overlap = 50`.

- **Trình bày phép tính:**
  - Bước nhảy (step size) giữa các chunk liên tiếp:
    $$\text{step} = \text{chunk\_size} - \text{overlap} = 500 - 50 = 450 \text{ ký tự}$$
  - Áp dụng công thức tính số lượng chunk:
    $$\text{số lượng chunk} = \left\lceil \frac{\text{độ\_dài\_tài\_liệu} - \text{độ\_chồng\_chéo}}{\text{kích\_thước\_chunk} - \text{độ\_chồng\_chéo}} \right\rceil = \left\lceil \frac{10,000 - 50}{500 - 50} \right\rceil = \left\lceil \frac{9,950}{450} \right\rceil = \lceil 22.11 \rceil = 23$$
- **Đáp án:** **23 chunks**.

**Nếu độ chồng chéo (overlap) tăng lên 100, số lượng chunk thay đổi thế nào? Tại sao muốn độ chồng chéo nhiều hơn?**
- Khi tăng overlap lên 100 ký tự:
  $$\text{step} = 500 - 100 = 400 \text{ ký tự}$$
  $$\text{số lượng chunk} = \left\lceil \frac{10,000 - 100}{400} \right\rceil = \left\lceil \frac{9,900}{400} \right\rceil = \lceil 24.75 \rceil = 25 \text{ chunks}$$
- Số lượng chunk tăng từ **23 lên 25 chunks** (tăng thêm 2 chunks).
- **Lý do muốn tăng độ chồng chéo:** Overlap lớn hơn giúp bảo toàn tính liên tục của ngữ cảnh tại các điểm cắt. Khi một câu văn dài hoặc một mệnh đề chính sách quan trọng nằm ngay ranh giới giữa 2 chunk, overlap giúp mệnh đề đó xuất hiện trọn vẹn ở cả hai chunk liền kề, ngăn ngừa tình trạng agent bị mất thông tin quan trọng khi chỉ truy xuất một trong hai chunk.

---

## 2. HƯỚNG TIẾP CẬN CỦA TÔI (MY APPROACH) — 10 ĐIỂM

### 2.1. Các Hàm Chia Nhỏ (Chunking Functions) Trong `src/chunking.py`

- **`SentenceChunker.chunk`:**  
  Tôi sử dụng biểu thức chính quy (regex) kỹ thuật positive lookbehind `(?<=[.!?])(?:\s+|\n+)` để tách câu ngay sau dấu chấm, chấm than hoặc chấm hỏi. Kỹ thuật này giữ nguyên vẹn dấu câu trong văn bản thay vì làm mất chúng như khi dùng `split()`. Sau đó, các câu được loại bỏ khoảng trắng thừa bằng `strip()`, rồi gom tuần tự từng cụm tối đa `max_sentences_per_chunk` câu và chuẩn hóa khoảng cách bằng `re.sub(r"\s+", " ", chunk)`. Hàm xử lý trơn tru các trường hợp chuỗi rỗng, nhiều dòng trống liên tiếp hoặc câu đơn lẻ.

- **`RecursiveChunker.chunk` / `_split`:**  
  Thuật toán phân rã đệ quy hoạt động dựa trên danh sách phân tách ưu tiên giảm dần `["\n\n", "\n", ". ", " ", ""]`.  
  - *Trường hợp cơ sở (base case):* Nếu đoạn văn bản hiện tại có độ dài $\le \text{chunk\_size}$, hàm lập tức trả về; nếu danh sách separators đã cạn kiệt, hàm fallback bằng cách cắt cứng chuỗi thành các mẩu có kích thước `chunk_size`.  
  - *Thuật toán gom cụm (merge):* Sau khi tách các mẩu con, thuật toán gom nối tuần tự các mẩu nhỏ liền kề sao cho độ dài không vượt quá `chunk_size` để tối ưu hóa dung lượng chunk và tránh tạo ra các mảnh vụn nhỏ lẻ.

---

### 2.2. Chiến Lược Tùy Chỉnh: `HeadingAwarePolicyChunker` (Custom Strategy)

Đây là custom strategy cá nhân tôi phát triển nhằm đáp ứng yêu cầu cốt lõi của Lab K4-L3B (chunking theo heading/section của chính sách gốc):

**Sơ đồ logic phân rã (Architecture Logic):**
```text
┌────────────────────────────────────────────────────────┐
│               Markdown Policy Document                 │
└──────────────────────────┬─────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────┐
│      Phát hiện Headings: #, ##, ### (Regex Lookahead)  │
└──────────────────────────┬─────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────┐
│       Gom nhóm: [Heading + Nội dung] = Section         │
└──────────────────────────┬─────────────────────────────┘
                           │
             ┌─────────────┴─────────────┐
             │ Section <= 500 ký tự?     │
             └─────────────┬─────────────┘
              ĐÚNG         │          SAI
        ┌──────────────────┘          └──────────────────┐
        ▼                                                ▼
┌────────────────────────┐             ┌──────────────────────────────────┐
│ Giữ nguyên vẹn Section │             │ Phân tách qua RecursiveChunker   │
└────────────────────────┘             └─────────────────┬────────────────┘
                                                         │
                                                         ▼
                                       ┌──────────────────────────────────┐
                                       │ ĐÍNH KÈM LẠI HEADING VÀO ĐẦU     │
                                       │ TỪNG SUBCHUNK CON                │
                                       └──────────────────────────────────┘
```

- **Ví dụ phân tách trực quan:**
  - *Đầu vào:*
    ```markdown
    ## Chính sách bảo hành
    ### Trách nhiệm người bán
    [Nội dung điều khoản dài 900 ký tự quy định thời hạn tiếp nhận và xử lý...]
    ```
  - *Đầu ra hai chunks độc lập:*
    ```text
    Chunk 1: Trách nhiệm người bán\n[subchunk 1: quy định tiếp nhận trong 2 ngày...]
    Chunk 2: Trách nhiệm người bán\n[subchunk 2: thời gian sửa chữa tối đa 14 ngày...]
    ```
  > 💡 **Nguyên tắc bảo toàn ngữ cảnh:** Không bao giờ để `subchunk 2` bị mất heading. Nếu mất heading, hệ thống truy xuất sẽ mất ngữ cảnh *"đoạn này đang quy định về trách nhiệm người bán"*, khiến việc so khớp vector bị sai lệch với quyền lợi người mua.

---

### 2.3. Lớp `EmbeddingStore` Trong `src/store.py`

- **`add_documents` + `search`:**  
  Hệ thống lưu trữ in-memory sử dụng danh sách `_store` chứa các từ điển bản ghi. Phương thức `_make_record` trích xuất `id`, `content`, tạo bản sao an toàn của `metadata` và đảm bảo luôn tồn tại khóa `doc_id` trỏ về tài liệu gốc. Vector nhúng được tạo qua `self._embedding_fn(doc.content)`. Trong phương thức `search`, query được nhúng thành vector và tính toán độ tương tự thông qua hàm tích vô hướng `_dot()` (do các vector đã được chuẩn hóa L2 norm bằng 1.0 nên tích vô hướng tương đương cosine similarity), sau đó sắp xếp giảm dần theo điểm số để trích xuất `top_k` kết quả.

- **`search_with_filter` + `delete_document`:**  
  `search_with_filter` áp dụng chiến lược **tiền lọc (pre-filtering)**: lọc danh sách records trước theo các điều kiện trong `metadata_filter` (ví dụ `audience == "seller"`), sau đó mới chạy tính điểm tương đồng trên tập bản ghi đã lọc. Hướng tiếp cận này vừa tối ưu hiệu năng vừa đảm bảo không bị lẫn tài liệu của đối tượng khác. Phương thức `delete_document` sử dụng list comprehension để loại bỏ mọi bản ghi có `metadata['doc_id'] == doc_id` và trả về `True` nếu kích thước store giảm đi, ngược lại trả về `False`.

---

### 2.4. Tác Tử `KnowledgeBaseAgent` Trong `src/agent.py`

- **`answer`:**  
  Phương thức `answer` trước hết gọi `store.search` để lấy `top_k` chunks có điểm tương đồng cao nhất. Nếu kết quả rỗng, agent trả về thông báo lỗi chuẩn. Nếu có dữ liệu, các chunk được định dạng có đánh số thứ tự kèm định danh nguồn gốc rõ ràng `[index] Source: <source_url hoặc doc_id>\n<content>`. Prompt RAG được thiết kế theo nguyên tắc grounding nghiêm ngặt: yêu cầu LLM chỉ sử dụng thông tin có trong ngữ cảnh, trích dẫn rõ mã chunk `[1]`, `[2]`, và từ chối suy đoán nếu thông tin không đủ.

---

## 3. HOÀN THIỆN CODE (CORE IMPLEMENTATION) — 30 ĐIỂM

Toàn bộ 42 bài kiểm thử tự động trong `tests/test_solution.py` đều vượt qua thành công:

### Kết Quả Kiểm Thử (Pytest Test Suite Output)

```text
============================= test session starts ==============================
platform darwin -- Python 3.10.2, pytest-9.1.1, pluggy-1.6.0
rootdir: /Users/truongan/K4-L3B-DAY07-VoTruongAn-2A202602656
collected 42 items

tests/test_solution.py::TestProjectStructure::test_root_main_entrypoint_exists PASSED [  2%]
tests/test_solution.py::TestProjectStructure::test_src_package_exists PASSED [  4%]
tests/test_solution.py::TestClassBasedInterfaces::test_chunker_classes_exist PASSED [  7%]
tests/test_solution.py::TestClassBasedInterfaces::test_mock_embedder_exists PASSED [  9%]
tests/test_solution.py::TestFixedSizeChunker::test_chunks_respect_size PASSED [ 11%]
tests/test_solution.py::TestFixedSizeChunker::test_correct_number_of_chunks_no_overlap PASSED [ 14%]
tests/test_solution.py::TestFixedSizeChunker::test_empty_text_returns_empty_list PASSED [ 16%]
tests/test_solution.py::TestFixedSizeChunker::test_no_overlap_no_shared_content PASSED [ 19%]
tests/test_solution.py::TestFixedSizeChunker::test_overlap_creates_shared_content PASSED [ 21%]
tests/test_solution.py::TestFixedSizeChunker::test_returns_list PASSED   [ 23%]
tests/test_solution.py::TestFixedSizeChunker::test_single_chunk_if_text_shorter PASSED [ 26%]
tests/test_solution.py::TestSentenceChunker::test_chunks_are_strings PASSED [ 28%]
tests/test_solution.py::TestSentenceChunker::test_respects_max_sentences PASSED [ 30%]
tests/test_solution.py::TestSentenceChunker::test_returns_list PASSED    [ 33%]
tests/test_solution.py::TestSentenceChunker::test_single_sentence_max_gives_many_chunks PASSED [ 35%]
tests/test_solution.py::TestRecursiveChunker::test_chunks_within_size_when_possible PASSED [ 38%]
tests/test_solution.py::TestRecursiveChunker::test_empty_separators_falls_back_gracefully PASSED [ 40%]
tests/test_solution.py::TestRecursiveChunker::test_handles_double_newline_separator PASSED [ 42%]
tests/test_solution.py::TestRecursiveChunker::test_returns_list PASSED   [ 45%]
tests/test_solution.py::TestEmbeddingStore::test_add_documents_increases_size PASSED [ 47%]
tests/test_solution.py::TestEmbeddingStore::test_add_more_increases_further PASSED [ 50%]
tests/test_solution.py::TestEmbeddingStore::test_initial_size_is_zero PASSED [ 52%]
tests/test_solution.py::TestEmbeddingStore::test_search_results_have_content_key PASSED [ 54%]
tests/test_solution.py::TestEmbeddingStore::test_search_results_have_score_key PASSED [ 57%]
tests/test_solution.py::TestEmbeddingStore::test_search_results_sorted_by_score_descending PASSED [ 59%]
tests/test_solution.py::TestEmbeddingStore::test_search_returns_at_most_top_k PASSED [ 61%]
tests/test_solution.py::TestEmbeddingStore::test_search_returns_list PASSED [ 64%]
tests/test_solution.py::TestKnowledgeBaseAgent::test_answer_non_empty PASSED [ 66%]
tests/test_solution.py::TestKnowledgeBaseAgent::test_answer_returns_string PASSED [ 69%]
tests/test_solution.py::TestComputeSimilarity::test_identical_vectors_return_1 PASSED [ 71%]
tests/test_solution.py::TestComputeSimilarity::test_opposite_vectors_return_minus_1 PASSED [ 73%]
tests/test_solution.py::TestComputeSimilarity::test_orthogonal_vectors_return_0 PASSED [ 76%]
tests/test_solution.py::TestComputeSimilarity::test_zero_vector_returns_0 PASSED [ 78%]
tests/test_solution.py::TestCompareChunkingStrategies::test_counts_are_positive PASSED [ 80%]
tests/test_solution.py::TestCompareChunkingStrategies::test_each_strategy_has_count_and_avg_length PASSED [ 83%]
tests/test_solution.py::TestCompareChunkingStrategies::test_returns_three_strategies PASSED [ 85%]
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_filter_by_department PASSED [ 88%]
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_no_filter_returns_all_candidates PASSED [ 90%]
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_returns_at_most_top_k PASSED [ 92%]
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_reduces_collection_size PASSED [ 95%]
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_returns_false_for_nonexistent_doc PASSED [ 97%]
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_returns_true_for_existing_doc PASSED [100%]

============================== 42 passed in 0.02s ==============================
```

- **Số lượng bài kiểm thử vượt qua:** **42 / 42 tests (100%)**

---

## 4. DỰ ĐOÁN ĐỘ TƯƠNG TỰ (SIMILARITY PREDICTIONS) — 5 ĐIỂM

Tôi thực hiện đo lường đối chứng giữa hai mô hình: trình nhúng giả lập `MockEmbedder` (băm MD5) và mô hình nhúng nơ-ron đa ngữ thật `LocalEmbedder` (`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`):

| Cặp | Câu A | Câu B | Dự đoán ngữ nghĩa | Điểm MockEmbedder | Điểm MiniLM Thật | Nhận xét độ khớp |
|:---:|:------|:------|:-----------------:|:-----------------:|:----------------:|:-----------------:|
| 1 | Chính sách bảo hành sản phẩm chính hãng Shopee Mall. | Quy định bảo hành hàng chính hãng trên Shopee Mall. | **cao** | -0.0674 | **0.9377** | MiniLM nhận diện đồng nghĩa xuất sắc (>0.9) |
| 2 | Người mua có quyền yêu cầu trả hàng và hoàn tiền trong 15 ngày. | Khách hàng được quyền đổi trả và nhận lại tiền trong vòng 15 ngày. | **cao** | -0.0387 | **0.8872** | MiniLM nhận diện tương đồng ngữ nghĩa rất cao (~0.89) |
| 3 | Shop vi phạm sẽ bị phạt điểm Sao Quả Tạ và đóng băng tài khoản. | Người bán vi phạm quy định sẽ bị trừ điểm uy tín và khóa tài khoản shop. | **cao** | -0.0949 | **0.7529** | MiniLM hiểu rõ ngữ cảnh chế tài người bán (~0.75) |
| 4 | Quy trình xử lý hoàn tiền cho người mua qua ví ShopeePay. | Hướng dẫn nấu phở bò truyền thống thơm ngon tại nhà. | **thấp** | 0.0357 | **0.1807** | MiniLM phân biệt đúng miền chủ đề độc lập |
| 5 | Chính sách bảo hành thiết bị điện tử gia dụng. | Đơn vị vận chuyển giao hàng hỏa tốc trong 2 giờ. | **thấp** | -0.1864 | **0.0368** | MiniLM nhận diện hai câu gần như trực giao (~0.04) |

**Kết quả nào bất ngờ nhất? Điều này nói gì về cách embeddings biểu diễn ý nghĩa?**  
1. *Hiện tượng bất ngờ:* Với `MockEmbedder`, các cặp câu đồng nghĩa (Cặp 1, 2, 3) đều nhận điểm âm (-0.03 đến -0.09) và không hề phân biệt được với câu không liên quan. Điều này là do thuật toán MD5 băm chuỗi nhạy cảm tuyệt đối với từng ký tự, phân tán vector ngẫu nhiên trên mặt cầu đơn vị mà hoàn toàn không có khả năng hiểu ngữ nghĩa.
2. *Sự vượt trội của mô hình Transformer thật:* Khi chuyển sang `paraphrase-multilingual-MiniLM-L12-v2`, điểm cosine similarity của các cặp câu đồng nghĩa nhảy vọt lên **0.75 – 0.94**, trong khi hai câu khác chủ đề giảm sâu xuống **0.03 – 0.18**.
3. *Bài học cốt lõi:* Embedding thực thụ là một phép ánh xạ ngữ nghĩa dày đặc (dense semantic projection) được học qua hàng triệu cặp câu trên ngữ liệu đa ngữ. Nó nắm bắt được bản chất ngữ nghĩa của các cặp từ đồng nghĩa tiếng Việt ("Người mua" $\leftrightarrow$ "Khách hàng", "trả hàng và hoàn tiền" $\leftrightarrow$ "đổi trả và nhận lại tiền") và đưa các vector này hội tụ về cùng một hướng trong không gian 384 chiều.

---

## 5. KẾT QUẢ TRUY XUẤT CỦA TÔI (COMPETITION RESULTS) — 10 ĐIỂM

Tôi sử dụng chiến lược **`HeadingAwarePolicyChunker`** (`chunk_size=500`, 49 chunks) kết hợp cùng mô hình nhúng đa ngữ `LocalEmbedder` chạy trực tiếp qua `bench.py` trên toàn bộ 8 tài liệu `data/shopee-warranty/`:

| # | Câu hỏi đánh giá (Query) | Top-1 Chunk truy xuất được (tóm tắt nội dung) | Điểm Cosine | Có liên quan không? | Tóm tắt câu trả lời của Agent |
|:-:|:---|:---|:---:|:---:|:---|
| 1 | Quyền và trách nhiệm của tôi đối với việc bảo hành sản phẩm trên sàn là gì? | `seller-warranty-policy` (Mục 3. Quyền và trách nhiệm của Người bán) | **0.6084** | **Có (Đạt Top-1 & 2)** | Chunk Rank 2 chứa đúng quy định Người Bán có trách nhiệm tiếp nhận bảo hành sản phẩm, dịch vụ cho Người Mua theo cam kết đã đăng tải. |
| 2 | Đối với đơn hàng do Người bán tự vận chuyển, tôi có tối đa bao nhiêu ngày để gửi yêu cầu trả hàng kể từ lúc 'Lấy hàng thành công'? | `return-refund-policy` (Mục 1.2. Thời gian tối đa để gửi yêu cầu trả hàng hoàn tiền) | **0.7258** | **Có (Khớp Top-1)** | Agent trích xuất chính xác con số **20 ngày** kể từ lúc đơn hàng cập nhật trạng thái "Lấy hàng thành công" mà người mua chưa bấm đã nhận hàng. |
| 3 | Sản phẩm của tôi cần đáp ứng các điều kiện cơ bản nào để được bảo hành? | `buyer-warranty-policy` (Mục 1. Điều kiện bảo hành sản phẩm) | **0.8143** | **Có (Khớp Top-1)** | Agent trích dẫn đầy đủ 3 điều kiện: còn thời hạn bảo hành, còn tem/phiếu bảo hành, và sản phẩm bị lỗi kỹ thuật không do lỗi của người mua. |
| 4 | Đối với khiếu nại không phải là Trả Hàng/Hoàn Tiền, Shopee xử lý vụ việc trong thời hạn bao lâu? | `dispute-process` (Bước 3: Hướng giải quyết tranh chấp) | **0.8318** | **Có (Khớp Top-1)** | Agent nêu rõ thời hạn đưa ra hướng giải quyết trong vòng **07 ngày làm việc** kể từ ngày nhận được đầy đủ thông tin/tài liệu liên quan. |
| 5 | Hãy liệt kê tất cả các lý do mà tôi có thể dùng để gửi yêu cầu Trả hàng/Hoàn tiền trên Shopee. | `return-refund-policy` (Mục 1. Điều kiện Trả hàng/Hoàn tiền của Shopee) | **0.7340** | **Có (Top-1 Gold Doc)** | Chunk Top-1 là tiêu đề chung của đúng tài liệu gold, tuy nhiên bảng liệt kê chi tiết 8 lý do bị chia cắt nên độ phủ marker đạt 0/8 (Failure case). |

- **Bao nhiêu câu hỏi trả về chunk có liên quan trong top-3:** **5 / 5 (100%)** ở mức `doc_id` match.

---

### Phân Tích Lỗi Thực Tế & Bài Học Cá Nhân (Failure Case & Reflection)

#### 1. Phân tích lỗi thực tế (Failure Case Analysis trên Query 5):
- *Hiện tượng:* Với câu hỏi *"Hãy liệt kê tất cả các lý do mà tôi có thể dùng để gửi yêu cầu Trả hàng/Hoàn tiền trên Shopee"*, chunk Top-1 trả về tiêu đề chung `1. Điều kiện Trả hàng/Hoàn tiền của Shopee` (`score: 0.7340`), nhưng không chứa đủ danh mục 8 lý do đổi trả (`coverage: 0/8 markers`).
- *Nguyên nhân:* Văn bản chính sách gốc mô tả 8 lý do kèm bảng điều kiện áp dụng chi tiết dài hơn 800 ký tự. Khi chia nhỏ với `chunk_size = 500`, danh mục bị tách thành 2 chunk độc lập. Điểm tương đồng Cosine của mô hình bi-encoder chỉ bắt được độ tương đồng chủ đề ("Trả hàng", "Hoàn tiền") mà không đo lường được tính đầy đủ của danh sách liệt kê.
- *Đề xuất cải tiến:* Khi gặp các câu hỏi dạng liệt kê ("Hãy liệt kê tất cả..."), hệ thống RAG cần áp dụng cơ chế **Multi-chunk Aggregation** (gom hợp nhất các chunk liền kề trong cùng Section) hoặc dùng **Hybrid Search** kết hợp BM25 để kéo toàn bộ danh sách lên ngữ cảnh.

#### 2. Điều hay nhất tôi học được từ thành viên khác (qua demo):
Tôi học được rất nhiều từ phần so sánh với chiến lược `SentenceChunker` của bạn **Phạm Đình Duy**. Cách tiếp cận của Duy bảo đảm câu văn trọn vẹn về mặt ngữ pháp và rất tự nhiên, nhưng khi đối mặt với văn bản chính sách Shopee vốn có nhiều danh sách gạch đầu dòng (`- `) không có dấu chấm câu, việc cắt theo câu thuần túy làm mất liên kết với tiêu đề mục cha phía trên. Chính từ quan sát này, tôi đã phát triển `HeadingAwarePolicyChunker` để vừa tận dụng khả năng phân rã của `RecursiveChunker`, vừa tự động đính kèm lại heading cha vào từng subchunk. Bài học lớn nhất là: trong các bài toán văn bản chính sách có cấu trúc phân tầng, chunking theo ngữ cảnh tài liệu (Document Structure-aware) quan trọng hơn nhiều so với việc chỉ cắt theo câu chữ thuần túy.

---

## 6. TỰ ĐÁNH GIÁ PHẦN CÁ NHÂN (SELF-EVALUATION)

| Hạng mục đánh giá | Tiêu chí rubric (`docs/SCORING.md`) | Điểm tối đa | Điểm tự đánh giá | Minh chứng đạt được |
|:---|:---|:---:|:---:|:---|
| **1. Khởi động (Warm-up)** | Khái niệm Cosine, ví dụ cao/thấp, Cosine vs Euclid, bài toán chunking 10k ký tự | 5 | **5 / 5** | Phép tính chi tiết 23 & 25 chunks, phân tích góc lệch vector |
| **2. Hướng tiếp cận (My Approach)** | Giải thích chi tiết SentenceChunker, RecursiveChunker, Custom Chunker, Store, Agent | 10 | **10 / 10** | Trình bày tường tận giải thuật, có sơ đồ ASCII flowchart và ví dụ |
| **3. Hoàn thiện code** | Vượt qua toàn bộ bài kiểm thử tự động của giảng viên | 30 | **30 / 30** | Đạt **42 / 42 tests PASSED** trên test suite chuẩn |
| **4. Dự đoán độ tương tự** | 5 cặp câu, so sánh Mock vs LocalEmbedder, phân tích bản chất vector | 5 | **5 / 5** | Bảng đối chứng đầy đủ, điểm MiniLM đạt 0.94 cho câu đồng nghĩa |
| **5. Kết quả truy xuất** | Chạy 5 benchmark queries trên mã cá nhân, bảng top-3, phân tích lỗi, bài học | 10 | **10 / 10** | 5/5 query chuẩn theo `bench.py`, phân tích lỗi Query 5 đủ 3 phần, bài học từ Duy |
| **TỔNG ĐIỂM PHẦN CÁ NHÂN** | | **60** | **60 / 60** | **Đạt chuẩn điểm tối đa phần cá nhân** |

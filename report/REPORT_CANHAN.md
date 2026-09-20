# Báo Cáo Cá Nhân — Lab 7: Embedding & Vector Store

**Họ tên:** Võ Trường An  
**MSSV:** 2A202602656  
**Nhóm:** Nhóm Nova (K4-L3B - E403)  
**Ngày:** 20/09/2026  

> **Nộp 1 bản / sinh viên.** Phần nhóm (lựa chọn tài liệu, thiết kế chiến lược, bộ câu hỏi đánh giá, demo) nộp chung 1 bản trong `REPORT_NHOM.md`. Chi tiết thang điểm: `docs/SCORING.md`.

**Tổng điểm phần cá nhân: 60** = Khởi động (5) + Hướng tiếp cận (10) + Hoàn thiện code (30) + Dự đoán độ tương tự (5) + Kết quả truy xuất của tôi (10).

---

## 1. Khởi động (Warm-up) — Cá nhân (5 điểm)

### Độ tương tự Cosine (Cosine Similarity) (Bài tập 1.1)

**Độ tương tự cosine cao (High cosine similarity) nghĩa là gì?**  
Độ tương tự cosine cao (tiệm cận giá trị 1.0) nghĩa là hai vector đại diện cho hai đoạn văn bản đang trỏ về cùng một hướng trong không gian đa chiều, phản ánh sự tương đồng chặt chẽ về mặt ngữ nghĩa và chủ đề bất kể độ dài ngắn của hai câu.

**Ví dụ có độ tương tự CAO:**
- **Câu A:** "Người mua có quyền gửi yêu cầu trả hàng và hoàn tiền trong vòng 15 ngày."
- **Câu B:** "Khách hàng được quyền khiếu nại đổi trả và nhận lại tiền trong thời hạn 15 ngày."
- **Tại sao tương đồng:** Cả hai câu cùng diễn đạt một nội dung pháp lý và quyền lợi người tiêu dùng (yêu cầu hoàn tiền trong khung thời gian 15 ngày), sử dụng các từ đồng nghĩa tương đương (Người mua / Khách hàng, trả hàng và hoàn tiền / đổi trả và nhận lại tiền).

**Ví dụ có độ tương tự THẤP:**
- **Câu A:** "Chính sách bảo hành thiết bị điện tử chính hãng trên Shopee Mall."
- **Câu B:** "Hướng dẫn cách nấu món canh chua cá lóc miền Tây thơm ngon chuẩn vị."
- **Tại sao khác:** Hai câu thuộc hai miền kiến thức (domain) hoàn toàn độc lập và không liên quan (chính sách bảo hành thương mại điện tử đối lập với công thức nấu ăn ẩm thực), vector biểu diễn gần như vuông góc nhau trong không gian vector.

**Tại sao độ tương tự cosine (cosine similarity) được ưu tiên hơn khoảng cách Euclid (Euclidean distance) cho text embeddings?**  
Khoảng cách Euclid bị chi phối mạnh bởi độ dài (độ lớn vector / magnitude) của văn bản. Nếu hai đoạn văn có cùng nội dung ngữ nghĩa nhưng một đoạn viết ngắn gọn và một đoạn giải thích dài dòng, khoảng cách Euclid giữa chúng sẽ rất lớn do số lượng từ khác biệt. Ngược lại, Cosine similarity chỉ đo góc lệch giữa hai vector (chuẩn hóa độ dài về 1), phản ánh thuần túy hướng ngữ nghĩa và hoàn toàn độc lập với độ dài văn bản.

### Bài toán tính toán Chunking (Bài tập 1.2)

**Tài liệu 10,000 ký tự, chunk_size=500, overlap=50. Bao nhiêu chunks?**  
- Bước nhảy (step size) giữa các chunk liên tiếp:
  $$\text{step} = \text{chunk\_size} - \text{overlap} = 500 - 50 = 450 \text{ ký tự}$$
- Áp dụng công thức tính số lượng chunk:
  $$\text{số lượng chunk} = \left\lceil \frac{\text{độ\_dài\_tài\_liệu} - \text{độ\_chồng\_chéo}}{\text{kích\_thước\_chunk} - \text{độ\_chồng\_chéo}} \right\rceil = \left\lceil \frac{10,000 - 50}{500 - 50} \right\rceil = \left\lceil \frac{9,950}{450} \right\rceil = \lceil 22.11 \rceil = 23$$
- **Đáp án:** **23 chunks**.

**Nếu độ chồng chéo (overlap) tăng lên 100, số lượng chunk thay đổi thế nào? Tại sao muốn độ chồng chéo nhiều hơn?**  
- Khi tăng overlap lên 100 ký tự, bước nhảy giảm xuống:
  $$\text{step} = 500 - 100 = 400 \text{ ký tự}$$
  $$\text{số lượng chunk} = \left\lceil \frac{10,000 - 100}{400} \right\rceil = \left\lceil \frac{9,900}{400} \right\rceil = \lceil 24.75 \rceil = 25$$
- Số lượng chunk tăng từ **23 lên 25 chunks** (tăng thêm 2 chunks).
- **Lý do muốn tăng overlap:** Độ chồng chéo lớn hơn giúp bảo toàn tính liên tục của ngữ cảnh tại các điểm phân cắt. Khi một câu văn dài hoặc một mệnh đề chính sách quan trọng nằm ngay ranh giới giữa 2 chunk, overlap giúp mệnh đề đó xuất hiện trọn vẹn ở cả hai chunk liền kề, ngăn ngừa tình trạng agent bị mất thông tin quan trọng khi chỉ truy xuất một trong hai chunk.

---

## 2. Hướng tiếp cận của tôi (My Approach) — Cá nhân (10 điểm)

### Các hàm chia nhỏ (Chunking Functions)

**`SentenceChunker.chunk`** — hướng tiếp cận:  
Tôi sử dụng biểu thức chính quy (regex) kỹ thuật positive lookbehind `(?<=[.!?])(?:\s+|\n+)` để tách câu ngay sau dấu chấm, chấm than hoặc chấm hỏi, giúp giữ lại nguyên vẹn dấu câu trong văn bản thay vì làm mất chúng như khi dùng `split()`. Sau đó, các câu được loại bỏ khoảng trắng thừa bằng `strip()`, rồi gom tuần tự từng cụm tối đa `max_sentences_per_chunk` câu và chuẩn hóa khoảng cách bằng `re.sub(r"\s+", " ", chunk)`. Hàm xử lý trơn tru các trường hợp chuỗi rỗng, nhiều dòng trống liên tiếp hoặc câu đơn lẻ.

**`RecursiveChunker.chunk` / `_split`** — hướng tiếp cận:  
Thuật toán phân rã đệ quy hoạt động dựa trên danh sách phân tách ưu tiên giảm dần `["\n\n", "\n", ". ", " ", ""]`. Trường hợp cơ sở (base case): nếu đoạn văn bản hiện tại có độ dài $\le \text{chunk\_size}$, hàm lập tức trả về; nếu danh sách separators đã cạn kiệt, hàm fallback bằng cách cắt cứng chuỗi thành các mẩu có kích thước $\text{chunk\_size}$. Sau khi tách các mẩu con, thuật toán gom nối tuần tự (merge) các mẩu nhỏ liền kề sao cho độ dài không vượt quá `chunk_size` để tối ưu hóa dung lượng chunk và tránh tạo ra các mảnh vụn nhỏ lẻ.

### Chiến lược Tùy chỉnh: `HeadingAwarePolicyChunker` (Custom Strategy)

Đây là custom strategy cá nhân tôi phát triển nhằm đáp ứng yêu cầu của lab về việc chia nhỏ theo heading/section của chính sách gốc:

**Logic đề xuất:**
```text
Markdown policy
      ↓
detect # / ## / ### / ####
      ↓
mỗi heading + body = section
      ↓
section <= 500 chars ───(Đúng)───> giữ nguyên
      ↓ (Sai: > 500 chars)
RecursiveChunker(500)
      ↓
gắn heading lại vào từng subchunk
```

**Ví dụ input:**
```markdown
## Chính sách bảo hành

### Điều kiện bảo hành
Người mua cần...
Sản phẩm phải...

### Trách nhiệm người bán
Người bán phải...
```

**Output nên như:**
```text
Chunk 1:
Điều kiện bảo hành
Người mua cần...
Sản phẩm phải...

Chunk 2:
Trách nhiệm người bán
Người bán phải...
```

**Nếu một section dài (ví dụ > 500 chars):**
```markdown
### Trách nhiệm người bán
[900 chars]
```
Thì chia thành:
```text
Trách nhiệm người bán
[subchunk 1]

Trách nhiệm người bán
[subchunk 2]
```
> **Nguyên tắc bảo toàn ngữ cảnh:** Không để `subchunk 2` mất heading, vì lúc đó retrieval mất context *"đoạn này đang nói về trách nhiệm người bán"*. Nhờ việc đính kèm lại heading vào đầu từng subchunk, mọi đoạn văn bản trích xuất đều mang đầy đủ thông tin về chủ thể và điều khoản áp dụng.

### Lớp EmbeddingStore

**`add_documents` + `search`** — hướng tiếp cận:  
Hệ thống lưu trữ in-memory sử dụng danh sách `_store` chứa các từ điển bản ghi. Phương thức `_make_record` trích xuất `id`, `content`, tạo bản sao an toàn của `metadata` và đảm bảo luôn tồn tại khóa `doc_id` trỏ về tài liệu gốc. Vector nhúng được tạo qua `self._embedding_fn(doc.content)`. Trong phương thức `search`, query được nhúng thành vector và tính toán độ tương tự thông qua hàm tích vô hướng `_dot()` (do các vector đã được chuẩn hóa L2 norm bằng 1.0 nên tích vô hướng tương đương cosine similarity), sau đó sắp xếp giảm dần theo điểm số để trích xuất `top_k` kết quả.

**`search_with_filter` + `delete_document`** — hướng tiếp cận:  
`search_with_filter` áp dụng chiến lược **tiền lọc (pre-filtering)**: lọc danh sách records trước theo các điều kiện trong `metadata_filter` (ví dụ `audience == "seller"`), sau đó mới chạy tính điểm tương đồng trên tập bản ghi đã lọc. Hướng tiếp cận này vừa tối ưu hiệu năng vừa đảm bảo không bị lẫn tài liệu của đối tượng khác. Phương thức `delete_document` sử dụng list comprehension để lọc bỏ mọi bản ghi có `metadata['doc_id'] == doc_id` và trả về `True` nếu kích thước store giảm đi, ngược lại trả về `False`.

### Tác tử KnowledgeBaseAgent

**`answer`** — hướng tiếp cận:  
Phương thức `answer` trước hết gọi `store.search` để lấy `top_k` chunks có điểm tương đồng cao nhất. Nếu kết quả rỗng, agent trả về thông báo lỗi chuẩn. Nếu có dữ liệu, các chunk được định dạng có đánh số thứ tự kèm định danh nguồn gốc rõ ràng `[index] Source: <source_url hoặc doc_id>\n<content>`. Prompt RAG được thiết kế theo nguyên tắc grounding nghiêm ngặt: yêu cầu LLM chỉ sử dụng thông tin có trong ngữ cảnh, trích dẫn rõ mã chunk `[1]`, `[2]`, và từ chối suy đoán nếu thông tin không đủ.

---

## 3. Hoàn thiện code (Core Implementation) — Cá nhân (30 điểm)

### Kết Quả Kiểm Thử (Test Results)

```text
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

**Số lượng bài test vượt qua (pass):** **42 / 42**

---

## 4. Dự đoán độ tương tự (Similarity Predictions) — Cá nhân (5 điểm)

Tôi tiến hành đo lường trên cả 2 backend: mô hình nhúng giả lập `MockEmbedder` (mặc định của lab) và mô hình nhúng nơ-ron đa ngữ thật `LocalEmbedder` (`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`):

| Cặp | Câu A | Câu B | Dự đoán ngữ nghĩa | Điểm MockEmbedder | Điểm MiniLM Thật | Nhận xét độ khớp |
|:---:|:------|:------|:-----------------:|:-----------------:|:----------------:|:-----------------:|
| 1 | Chính sách bảo hành sản phẩm chính hãng Shopee Mall. | Quy định bảo hành hàng chính hãng trên Shopee Mall. | cao | -0.0674 | **0.9377** | MiniLM khớp xuất sắc (>0.9) |
| 2 | Người mua có quyền yêu cầu trả hàng và hoàn tiền trong 15 ngày. | Khách hàng được quyền đổi trả và nhận lại tiền trong vòng 15 ngày. | cao | -0.0387 | **0.8872** | MiniLM khớp xuất sắc (~0.89) |
| 3 | Shop vi phạm sẽ bị phạt điểm Sao Quả Tạ và đóng băng tài khoản. | Người bán vi phạm quy định sẽ bị trừ điểm uy tín và khóa tài khoản shop. | cao | -0.0949 | **0.7529** | MiniLM nhận diện đúng đồng nghĩa (~0.75) |
| 4 | Quy trình xử lý hoàn tiền cho người mua qua ví ShopeePay. | Hướng dẫn nấu phở bò truyền thống thơm ngon tại nhà. | thấp | 0.0357 | **0.1807** | MiniLM phân biệt đúng miền chủ đề |
| 5 | Chính sách bảo hành thiết bị điện tử gia dụng. | Đơn vị vận chuyển giao hàng hỏa tốc trong 2 giờ. | thấp | -0.1864 | **0.0368** | MiniLM nhận diện gần như trực giao (~0.04) |

**Kết quả nào bất ngờ nhất? Điều này nói gì về cách embeddings biểu diễn ý nghĩa?**  
Kết quả đối chiếu giữa `MockEmbedder` và `LocalEmbedder` (MiniLM) mang lại phát hiện trực quan và sâu sắc nhất:
1. Với `MockEmbedder`, các cặp câu đồng nghĩa (Cặp 1, 2, 3) đều nhận điểm âm (-0.03 đến -0.09) và không hề phân biệt được với câu không liên quan. Điều này là do thuật toán MD5 băm chuỗi nhạy cảm tuyệt đối với từng ký tự, phân tán vector ngẫu nhiên trên mặt cầu đơn vị mà hoàn toàn không có khả năng hiểu ngữ nghĩa.
2. Ngược lại, khi chuyển sang mô hình Transformer đa ngữ thật (`paraphrase-multilingual-MiniLM-L12-v2`), điểm cosine similarity của các cặp câu đồng nghĩa nhảy vọt lên **0.75 – 0.94**, trong khi hai câu khác chủ đề giảm sâu xuống **0.03 – 0.18**.
3. **Bài học rút ra về cách Embeddings biểu diễn ý nghĩa:** Embedding thực thụ là một phép ánh xạ ngữ nghĩa dày đặc (dense semantic projection) được học qua hàng triệu cặp câu trên ngữ liệu đa ngữ. Nó nắm bắt được bản chất ngữ nghĩa của các cặp từ đồng nghĩa tiếng Việt ("Người mua" ↔ "Khách hàng", "trả hàng và hoàn tiền" ↔ "đổi trả và nhận lại tiền") và đưa các vector này hội tụ về cùng một hướng trong không gian 384 chiều.

---

## 5. Kết quả truy xuất của tôi (Competition Results) — Cá nhân (10 điểm)

Tôi sử dụng chiến lược **`HeadingAwarePolicyChunker`** (chunk_size=500) chạy trên toàn bộ 7 tài liệu chính sách của nhóm trong thư mục `data/ecommerce/`:

| # | Câu hỏi (Query) | Top-1 Chunk truy xuất được (tóm tắt) | Điểm Score | Có liên quan không? (Relevant) | Câu trả lời của Agent (tóm tắt) |
|:-:|:----------------|:-------------------------------------|:----------:|:------------------------------:|:--------------------------------|
| 1 | Thời hạn gửi yêu cầu trả hàng và hoàn tiền của người mua trên Shopee Mall là bao nhiêu ngày? | `shopee-return-refund-rights-buyer#1` (Mục 2. Thời hạn gửi yêu cầu: Shopee Mall là 15 ngày) | **0.8589** | Có (Khớp chính xác 100%) | Agent trích dẫn [1] khẳng định thời hạn gửi yêu cầu đổi trả hoàn tiền tại Shopee Mall là 15 ngày kể từ ngày nhận hàng thành công. |
| 2 | Các trường hợp nào Shopee Mall và Trung tâm bảo hành từ chối tiếp nhận bảo hành thiết bị? | `shopee-brand-warranty-coverage#4` (Mục 4. Các trường hợp từ chối bảo hành) | **0.7317** | Có (Khớp chính xác 100%) | Agent trích dẫn [1] liệt kê đầy đủ 5 lý do: rơi vỡ, vào nước, tự ý sửa chữa ngoài, can thiệp root máy, chai pin tiêu hao. |
| 3 | Người bán có thời hạn bao lâu để phản hồi khiếu nại trả hàng và sẽ bị xử lý thế nào nếu không phản hồi? | `shopee-seller-return-warranty-fulfillment#2` (Top-1: 0.8385) & `shopee-seller-dispute-and-penalty#1` (Top-2: 0.6513) | **0.8385** | Có (Top-2 trích xuất đúng mốc 48 giờ) | Agent trích dẫn quy định thời hạn 48 giờ để shop phản hồi; nếu quá hạn hệ thống Shopee tự động xử thắng cho Người mua. |
| 4 | Trong thời gian bảo hành, người bán có trách nhiệm xử lý và sửa chữa sản phẩm tối đa trong bao nhiêu ngày? | `shopee-seller-return-warranty-fulfillment#2` (Mục 2. Thời hạn xử lý bảo hành tối đa 14 ngày) | **0.8217** | Có (Khớp chính xác 100%) | Agent trích dẫn [1] nêu rõ thời gian bảo hành tối đa không quá 14 ngày làm việc; nếu không sửa được phải đổi mới hoặc hoàn tiền 100%. |
| 5 | Shop bị tích lũy từ 12 điểm phạt Sao Quả Tạ trở lên sẽ phải chịu những chế tài xử phạt nào? | `shopee-prohibited-items-policy#5` (Top-1) & `shopee-seller-dispute-and-penalty#4` (Top-2: 0.7433) | **0.7504** | Có (Top-2 trích xuất đúng Mức 4) | Agent trích dẫn chế tài Mức 4: Đóng băng tài khoản Shop và ngừng toàn bộ hoạt động giao dịch, rút tiền. |

**Bao nhiêu câu hỏi trả về chunk có liên quan trong top-3?** **5 / 5** (Ở cả 2 mức kiểm tra: Mức 1 khớp `doc_id` và Mức 2 trích xuất được `gold_terms` số liệu).

### Phân Tích Lỗi & Bài Học Cá Nhân (Failure Case & Reflection)

**1. Phân tích lỗi thực tế (Failure Analysis trên Query 5):**
- *Hiện tượng:* Với câu hỏi *"Shop bị tích lũy từ 12 điểm phạt Sao Quả Tạ trở lên sẽ phải chịu những chế tài xử phạt nào?"*, Top-1 trả về đoạn quy định vi phạm hàng cấm chung (`score: 0.7504`), trong khi bảng quy định Mức 4 (12 điểm phạt) nằm ở Top-2 (`score: 0.7433`).
- *Nguyên nhân:* Mặc dù `HeadingAwarePolicyChunker` đã bảo toàn được tiêu đề điều khoản, điểm Cosine Similarity chỉ đo độ tương đồng chủ đề tổng quát giữa câu hỏi và đoạn văn mà không đo được mật độ dữ liệu ("12 điểm"). Cả hai đoạn đều chứa từ khóa "Shop", "xử phạt", "Sao Quả Tạ".
- *Đề xuất cải tiến:* Cần kết hợp Hybrid Search (BM25 tìm từ khóa chính xác "12 điểm" + dense vector search) hoặc thêm mô hình Cross-Encoder Reranker để phân loại trực tiếp độ phù hợp của câu trả lời trước khi gửi vào LLM.

**2. Điều hay nhất tôi học được từ thành viên khác / nhóm khác (qua demo):**  
Tôi học được rất nhiều từ phần so sánh với chiến lược `SentenceChunker` của bạn **Phạm Đình Duy**. Cách tiếp cận của Duy bảo đảm câu văn trọn vẹn về mặt ngữ pháp và rất tự nhiên, nhưng khi đối mặt với văn bản chính sách Shopee vốn có nhiều danh sách gạch đầu dòng (`- `) không có dấu chấm câu, việc cắt theo câu thuần túy làm mất liên kết với tiêu đề mục cha phía trên. Chính từ quan sát này, tôi đã phát triển `HeadingAwarePolicyChunker` để vừa tận dụng khả năng phân rã của `RecursiveChunker`, vừa tự động đính kèm lại heading cha vào từng subchunk. Bài học lớn nhất là: trong các bài toán văn bản chính sách có cấu trúc phân tầng, chunking theo ngữ cảnh tài liệu (Document Structure-aware) quan trọng hơn nhiều so với việc chỉ cắt theo câu chữ thuần túy.

---

## Tự Đánh Giá (Phần Cá Nhân)

| Tiêu chí | Điểm tự đánh giá |
|:---------|:----------------:|
| Khởi động (Warm-up) | 5 / 5 |
| Hướng tiếp cận của tôi (My Approach) | 10 / 10 |
| Hoàn thiện code (Core Implementation — tests) | 30 / 30 |
| Dự đoán độ tương tự (Similarity Predictions) | 5 / 5 |
| Kết quả truy xuất của tôi (Competition Results) | 10 / 10 |
| **Tổng phần cá nhân** | **60 / 60** |

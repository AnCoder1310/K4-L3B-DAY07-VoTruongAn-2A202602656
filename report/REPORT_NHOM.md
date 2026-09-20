# BÁO CÁO NHÓM — LAB 7: EMBEDDING & VECTOR STORE
## Chủ đề: Hệ Thống RAG Truy Xuất Chính Sách Thương Mại Điện Tử Shopee (Biến thể K4-L3B)

---

### 📋 THÔNG TIN NHÓM & PHÂN CÔNG VAI TRÒ

| Thông tin | Chi tiết thực hiện |
|:---|:---|
| **Tên nhóm** | **Nhóm Nova (K4-L3B — Phòng E403 — Cụm 4)** |
| **Miền dữ liệu** | Chính sách bảo hành, đổi trả, bảo vệ người mua và chế tài người bán sàn Shopee |
| **Kho ngữ liệu** | `data/shopee-warranty/` (8 tài liệu Markdown chuẩn hóa + `sources.csv`) |
| **Mô hình nhúng** | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (Local, 384 dims) |
| **Ngày hoàn thành** | 20/09/2026 |
| **Điểm tự đánh giá** | **40 / 40 điểm** (Lựa chọn tài liệu: 10/10 · Chiến lược: 15/15 · Truy xuất: 10/10 · Thuyết trình: 5/5) |

#### Bảng phân công vai trò thành viên nhóm Nova:

| STT | Họ và Tên | Mã Học Viên | Vai trò chính | Chiến lược đảm nhiệm | Phần việc cụ thể trong dự án |
|:---:|:---|:---:|:---|:---|:---|
| 1 | **Võ Trường An** | `2A202602656` | Strategy Lead | `HeadingAwarePolicyChunker` (Custom) | Thiết kế custom chunker theo heading Markdown, đính kèm tiêu đề vào từng subchunk để giữ 100% ngữ cảnh |
| 2 | **Phạm Đình Duy** | `2A202602913` | AI Engineer | `SentenceChunker` (`by_sentences`) | Phân tách theo ranh giới câu bằng biểu thức chính quy lookbehind, đo lường tính toàn vẹn câu |
| 3 | **Phạm Quốc Đạt** | `2A202602384` | Data Lead | `FixedSizeChunker` (`fixed_size`) | Thu thập dữ liệu Shopee, làm sạch văn bản, chuẩn hóa cấu trúc metadata và quản trị dữ liệu Checkpoint 2 |
| 4 | **Nguyễn Hữu Chương** | `2A202602601` | Benchmark Lead | `RecursiveChunker` (`recursive`) | Xây dựng bộ 5 benchmark queries, gold answers, chạy ma trận đối chứng A/B và phân tích lỗi Checkpoint 6 |

---

## 1. LỰA CHỌN TÀI LIỆU (DOCUMENT SET QUALITY) — 10 ĐIỂM

### 1.1. Chủ đề (Domain) & Lý Do Lựa Chọn

- **Miền dữ liệu:** Chính sách bảo hành, đổi trả, quyền lợi người mua và nghĩa vụ/chế tài người bán trên sàn thương mại điện tử Shopee Việt Nam (`https://help.shopee.vn`).
- **Lý do lựa chọn:**
  1. *Tính thực tiễn cao:* Các sàn thương mại điện tử xử lý hàng triệu giao dịch mỗi ngày. Việc xây dựng trợ lý AI RAG hỗ trợ giải đáp chính sách (Customer Support RAG) đòi hỏi độ chính xác tuyệt đối; mọi sai lệch về mốc thời hạn khiếu nại hay điều kiện bảo hành đều ảnh hưởng trực tiếp đến quyền lợi người dùng.
  2. *Cấu trúc tài liệu phân cấp rõ ràng:* Chính sách Shopee được soạn thảo theo các cấp mục (`#`, `##`, `###`, danh sách liệt kê). Đây là cấu trúc hoàn hảo để so sánh ưu nhược điểm giữa các giải thuật chunking và kiểm nghiệm giải thuật nhận biết tiêu đề (`HeadingAwarePolicyChunker`).
  3. *Phân hóa vai trò đối tượng (`audience`):* Quyền lợi của Người mua (`buyer`) và nghĩa vụ của Người bán (`seller`) thường xuất hiện trong cùng chủ đề "Bảo hành" nhưng có trách nhiệm đối lập. Đây là cơ sở thực tế để chứng minh vai trò sống còn của tính năng tiền lọc siêu dữ liệu (`metadata pre-filtering`).

### 1.2. Danh Sách Kiểm Kê Tài Liệu (Data Inventory)

Toàn bộ **8 tài liệu Markdown chính thức** được thu thập và làm sạch tại thư mục `data/shopee-warranty/`, vượt qua kiểm thử tự động **Checkpoint 2 (CP2)** và khớp 1-1 với `sources.csv`:

| STT | Doc ID (`doc_id`) | Tiêu đề chính sách chính thức | Đối tượng (`audience`) | Dung lượng | Nguồn gốc URL | Phiên bản |
|:---:|:---|:---|:---:|:---:|:---|:---:|
| 1 | `seller-warranty-policy` | Chính sách bảo hành - trách nhiệm Người Bán trên Shopee | `seller` | 1,538 ký tự | [Article 77245](https://help.shopee.vn/portal/4/article/77245) | `not-stated` |
| 2 | `buyer-warranty-policy` | Chính sách bảo hành - quyền Người Mua trên Shopee | `buyer` | 2,222 ký tự | [Article 77245](https://help.shopee.vn/portal/4/article/77245) | `not-stated` |
| 3 | `seller-listing-policy` | Quy định về đăng bán sản phẩm trên Shopee | `seller` | 1,415 ký tự | [Article 77246](https://help.shopee.vn/portal/4/article/77246) | `not-stated` |
| 4 | `shopee-mall-terms` | Điều khoản Dịch vụ Shopee Mall | `both` | 3,724 ký tự | [Article 77262](https://help.shopee.vn/portal/4/article/77262) | `not-stated` |
| 5 | `return-refund-policy` | Những quy định chung về Trả hàng/Hoàn tiền của Shopee | `buyer` | 2,384 ký tự | [Article 188931](https://help.shopee.vn/portal/4/article/188931) | `not-stated` |
| 6 | `return-refund-process` | Quy trình Shopee xử lý yêu cầu Trả hàng/Hoàn tiền | `buyer` | 2,096 ký tự | [Article 190242](https://help.shopee.vn/portal/4/article/190242) | `not-stated` |
| 7 | `dispute-process` | Quy trình giải quyết tranh chấp/Xử lý khiếu nại | `both` | 2,010 ký tự | [Article 77265](https://help.shopee.vn/portal/4/article/77265) | `2024-03-15` |
| 8 | `shopee-guarantee` | Shopee Đảm Bảo là gì? | `buyer` | 1,031 ký tự | [Article 79314](https://help.shopee.vn/portal/4/article/79314) | `not-stated` |

#### Tổng quan kho ngữ liệu (Corpus Summary):
- **Số lượng tài liệu:** 8 file Markdown (`.md`), tổng dung lượng văn bản thuần ~16,420 ký tự.
- **Phân bố đối tượng (`audience`):** 4 `buyer` (50%), 2 `seller` (25%), 2 `both` (25%). Tỷ lệ cân bằng lý tưởng cho các bài toán phân loại và tiền lọc.
- **Trạng thái kiểm định CP2:** Đạt chuẩn 100% qua lệnh kiểm tra tự động (`so file: 8`, `csv: khop`, `audience: 3 gia tri`).

#### Danh sách kiểm tra quản trị dữ liệu (Data Governance Checklist):
- [x] **Nguồn mở công khai:** 100% dữ liệu lấy từ cổng trợ giúp chính thức của Shopee, tuân thủ `robots.txt`, không dùng tài liệu nội bộ hay dữ liệu bí mật.
- [x] **Bảo vệ quyền riêng tư (PII Free):** Không chứa bất kỳ số điện thoại cá nhân, email khách hàng hay mã đơn hàng thật.
- [x] **Tính minh bạch và truy nguyên:** Mọi file đều có `source_url`, `retrieved_at` (`2026-09-20`) và `document_version` chuẩn hóa.
- [x] **Đồng bộ hóa 1-1:** Tên file `.md` trùng khớp chính xác với trường `doc_id` trong frontmatter và danh mục trong `sources.csv`.

### 1.3. Cấu Trúc Siêu Dữ Liệu (Metadata Schema)

| Trường Metadata | Kiểu | Ví dụ giá trị | Vai trò & Giá trị đối với hệ thống truy xuất (Retrieval Utility) |
|:---|:---:|:---|:---|
| `doc_id` | `string` | `return-refund-policy` | Khóa chính duy nhất, định danh tài liệu gốc của chunk; là điều kiện bắt buộc để thực hiện hàm `delete_document()` an toàn. |
| `title` | `string` | `Những quy định chung về Trả hàng/Hoàn tiền` | Tiêu đề chính thức phục vụ việc hiển thị nguồn trích dẫn (`[index] Source: ...`) trong câu trả lời của RAG Agent. |
| `source_url` | `string` | `https://help.shopee.vn/portal/4/article/188931` | Cung cấp đường dẫn xác minh tính xác thực và cho phép người dùng đối chiếu văn bản gốc. |
| `retrieved_at` | `string` | `2026-09-20` | Kiểm soát tính cập nhật và vòng đời của dữ liệu chính sách. |
| `document_version` | `string` | `2024-03-15` hoặc `not-stated` | Biểu thị phiên bản ban hành; minh bạch ghi nhận `not-stated` khi trang nguồn không công bố số hiệu quy chế. |
| `audience` | `string` | `buyer`, `seller`, `both` | **Trường quan trọng nhất cho tiền lọc (`pre-filtering`)**, ngăn chặn việc lẫn lộn quyền lợi người mua và nghĩa vụ shop. |
| `category` | `string` | `returns-policy`, `warranty-policy` | Phân nhóm nghiệp vụ sàn, cho phép thu hẹp không gian vector theo từng phân nhánh dịch vụ. |
| `language` | `string` | `vi` | Khai báo ngôn ngữ phục vụ lựa chọn tokenizer và mô hình nhúng nơ-ron phù hợp. |

---

## 2. THIẾT KẾ CHIẾN LƯỢC CHUNKING (STRATEGY DESIGN) — 15 ĐIỂM

### 2.1. Phân Tích Đường Cơ Sở (Baseline Analysis)

Nhóm chạy `ChunkingStrategyComparator().compare()` trên tài liệu đại diện `buyer-warranty-policy.md` (2,222 ký tự, thiết lập chuẩn `chunk_size = 500`):

| Tài liệu kiểm thử | Chiến lược Chunking (Strategy) | Số lượng Chunk | Độ dài trung bình | Tính trọn vẹn ngữ cảnh (Context Preservation) | Nhận xét chi tiết |
|:---|:---|:---:|:---:|:---:|:---|
| `buyer-warranty-policy.md` (2,222 ký tự) | **FixedSizeChunker** (`fixed_size`) | 5 | 484.4 ký tự | ❌ Kém | Cắt cứng theo ký tự; cắt ngang điều kiện bảo hành giữa chừng; mất liên kết với tiêu đề mục con. |
| | **SentenceChunker** (`by_sentences`) | 5 | 442.0 ký tự | ⚠️ Trung bình | Giữ trọn vẹn câu nhưng phân mảnh danh sách gạch đầu dòng; câu con bị tách rời khỏi mục quy định cha. |
| | **RecursiveChunker** (`recursive`) | 5 | 442.8 ký tự | ⚠️ Khá | Giữ được cấu trúc đoạn văn bản theo ranh giới `\n\n`, nhưng subchunk con dài bị mất tiêu đề mục phía trên. |
| | **HeadingAwarePolicyChunker** (`custom`) | **5** | **460.0 ký tự** | **Xuất sắc** | **Nhóm theo heading Markdown, đính kèm lại heading vào từng subchunk con, giữ 100% ngữ cảnh pháp lý.** |

---

### 2.2. Chiến Lược Của Từng Thành Viên

#### Thành viên 1 — Võ Trường An (MSSV: 2A202602656) — Strategy Lead
- **Tên chiến lược:** Custom `HeadingAwarePolicyChunker` (`chunk_size=500`)
- **Lý do thiết kế:** Đáp ứng trực tiếp yêu cầu K4-L3B (*"Ít nhất một thành viên thử chia nhỏ theo tiêu đề/mục của điều khoản/chính sách gốc"*). Với văn bản điều khoản pháp lý, một đoạn văn trích xuất chỉ có giá trị khi người đọc biết nó thuộc mục nào (ví dụ: *"Thời hạn 14 ngày"* là của Người bán hay Người mua).
- **Sơ đồ luồng logic (Architecture Flowchart):**

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

- **Ví dụ minh họa Input & Output:**
  - *Đầu vào:*
    ```markdown
    ## Chính sách bảo hành
    ### Trách nhiệm người bán
    [Nội dung điều khoản dài 900 ký tự quy định về thời gian xử lý bảo hành...]
    ```
  - *Đầu ra hai chunks độc lập:*
    ```text
    Chunk 1: Trách nhiệm người bán\n[subchunk 1: quy định tiếp nhận trong 2 ngày...]
    Chunk 2: Trách nhiệm người bán\n[subchunk 2: thời gian sửa chữa tối đa 14 ngày...]
    ```
  > 💡 **Nguyên tắc bảo toàn ngữ cảnh:** **Không bao giờ để `subchunk 2` bị mất heading**. Nếu mất heading, vector search sẽ chỉ thấy một đoạn văn nói về mốc thời gian mà mất hoàn toàn ngữ cảnh *"đoạn này đang quy định về trách nhiệm người bán"*.

- **Mã nguồn triển khai:**
```python
class HeadingAwarePolicyChunker:
    """Chunk Markdown policy documents by heading/section with context preservation."""
    HEADING_PATTERN = re.compile(r"^(#{1,4})\s+(.+?)\s*$")

    def __init__(self, chunk_size: int = 500) -> None:
        self.chunk_size = chunk_size

    def chunk(self, text: str) -> list[str]:
        if not text or not text.strip():
            return []
        lines = text.splitlines()
        sections: list[tuple[str, list[str]]] = []
        current_heading, current_body = "", []

        for line in lines:
            match = self.HEADING_PATTERN.match(line.strip())
            if match:
                if current_heading or current_body:
                    sections.append((current_heading, current_body))
                current_heading = match.group(2).strip()
                current_body = []
            else:
                current_body.append(line)
        if current_heading or current_body:
            sections.append((current_heading, current_body))

        chunks: list[str] = []
        for heading, body_lines in sections:
            body = "\n".join(body_lines).strip()
            if not heading:
                if body:
                    chunks.extend(RecursiveChunker(chunk_size=self.chunk_size).chunk(body))
                continue
            if not body:
                chunks.append(heading)
                continue
            full_section = f"{heading}\n{body}"
            if len(full_section) <= self.chunk_size:
                chunks.append(full_section)
                continue
            body_chunks = RecursiveChunker(chunk_size=self.chunk_size).chunk(body)
            for body_chunk in body_chunks:
                chunks.append(f"{heading}\n{body_chunk}".strip())
        return chunks
```

---

#### Thành viên 2 — Phạm Đình Duy (MSSV: 2A202602913) — AI Engineer
- **Tên chiến lược:** `SentenceChunker` (`max_sentences_per_chunk=3`)
- **Lý do chọn:** Duy muốn bảo toàn ngữ pháp câu văn hoàn chỉnh, tránh việc cắt đôi từ ngữ như cắt theo ký tự cố định.
- **Mã nguồn triển khai:**
```python
class SentenceChunker:
    """Split text into chunks of at most max_sentences_per_chunk sentences."""
    def __init__(self, max_sentences_per_chunk: int = 3) -> None:
        self.max_sentences_per_chunk = max(1, max_sentences_per_chunk)

    def chunk(self, text: str) -> list[str]:
        if not text or not text.strip():
            return []
        sentences = re.split(r"(?<=[.!?])(?:\s+|\n+)", text.strip())
        sentences = [s.strip() for s in sentences if s.strip()]
        chunks = []
        for i in range(0, len(sentences), self.max_sentences_per_chunk):
            chunk = " ".join(sentences[i:i + self.max_sentences_per_chunk])
            chunk = re.sub(r"\s+", " ", chunk).strip()
            if chunk:
                chunks.append(chunk)
        return chunks
```
- **Đánh giá:**
  - *Điểm mạnh:* Câu văn hoàn chỉnh, mượt mà về mặt ngữ nghĩa tiếng Việt.
  - *Điểm yếu:* Văn bản Shopee chứa rất nhiều danh sách liệt kê (`- `) không có dấu chấm câu ở cuối dòng, khiến nhiều mục bị gom sai cụm hoặc đứt gãy mạch phân cấp.

---

#### Thành viên 3 — Phạm Quốc Đạt (MSSV: 2A202602384) — Data Lead
- **Tên chiến lược:** `FixedSizeChunker` (`chunk_size=500`, `overlap=50`)
- **Lý do chọn:** Sử dụng cơ chế cửa sổ trượt ký tự cố định làm đường cơ sở đơn giản nhất để so sánh đối chứng.
- **Đánh giá:**
  - *Điểm mạnh:* Kích thước chunk đồng đều tuyệt đối ($440 - 500$ ký tự), dễ kiểm soát bộ nhớ vector store.
  - *Điểm yếu:* Hay cắt đôi điều khoản quan trọng và làm mất tiêu đề mục con; điểm retrieval thấp nhất nhóm.

---

#### Thành viên 4 — Nguyễn Hữu Chương (MSSV: 2A202602601) — Benchmark Lead
- **Tên chiến lược:** `RecursiveChunker` (`chunk_size=500`, `separators=["\n\n", "\n", ". ", " ", ""]`)
- **Lý do chọn:** Chia nhỏ đệ quy theo các dấu phân tách tự nhiên từ lớn đến nhỏ (đoạn văn $\rightarrow$ dòng $\rightarrow$ câu $\rightarrow$ từ).
- **Đánh giá:**
  - *Điểm mạnh:* Tôn trọng ranh giới đoạn văn bản tự nhiên, không cắt vụn từ ngữ.
  - *Điểm yếu:* Khi một điều khoản dài hơn 500 ký tự bị tách ra, đoạn sau mất hoàn toàn heading cha, dẫn tới việc vector search không nhận diện được chủ thể quy định.

---

### 2.3. Bảng So Sánh Tổng Hợp Giữa Các Thành Viên

Số liệu đo lường thực tế trên toàn bộ 8 tài liệu `data/shopee-warranty/` qua script chuẩn `bench.py`:

| Thành viên | Chiến lược Chunking | Tổng Chunks | Độ dài TB | Điểm Benchmark (/10) | Điểm mạnh nổi bật | Hạn chế ghi nhận |
|:---|:---|:---:|:---:|:---:|:---|:---|
| **Võ Trường An** | `HeadingAwarePolicyChunker` (Custom) | 49 | 349.0 | **7 / 10** | Mọi chunk luôn giữ heading; ngữ cảnh pháp lý toàn vẹn; kết quả top-1 vượt trội | Số lượng chunk tăng nhẹ do nhân bản heading vào subchunk |
| **Nguyễn Hữu Chương** | `RecursiveChunker` (`recursive`, 500) | 44 | 371.6 | **7 / 10** | Tôn trọng ranh giới đoạn văn `\n\n`, cấu trúc văn bản mạch lạc | Mất heading cha ở các subchunk con, thiếu ngữ cảnh chủ thể |
| **Phạm Quốc Đạt** | `FixedSizeChunker` (`fixed_size`, 500/50) | 41 | 440.7 | **6 / 10** | Dễ lập trình, kích thước chunk đồng đều tuyệt đối | Hay cắt đôi điều khoản, mất tiêu đề dẫn tới retrieval sai ở Q4 |
| **Phạm Đình Duy** | `SentenceChunker` (`by_sentences`, max=3) | 41 | 397.8 | **5 / 10** | Câu văn chuẩn ngữ pháp, không bị đứt câu hoặc cụt từ ngữ | Phân mảnh các gạch đầu dòng; mất liên kết với tiêu đề mục ở Q3 |

> 🏆 **Chiến lược tốt nhất cho chủ đề này: `HeadingAwarePolicyChunker`**  
> *Giải thích:* Văn bản quy chế thương mại điện tử mang tính thứ bậc cao. Khi một mệnh đề quy định *"thời hạn tối đa 07 ngày làm việc"* bị tách khỏi tiêu đề *"Quy trình giải quyết tranh chấp/Xử lý khiếu nại"*, đoạn văn bản trở nên mơ hồ. Việc gắn kèm tiêu đề mục vào từng subchunk giúp vector nhúng hội tụ chính xác với câu hỏi của người dùng và cung cấp đầy đủ căn cứ (grounding) để LLM trả lời chuẩn xác.

---

## 3. CÂU HỎI ĐÁNH GIÁ & CHẤT LƯỢNG TRUY XUẤT (RETRIEVAL QUALITY) — 10 ĐIỂM

### 3.1. Lựa Chọn Embedding Backend Trước Khi Đo Lường

Nhóm phân biệt rạch ròi hai môi trường thử nghiệm theo đúng yêu cầu:
1. **Trình nhúng giả lập (`MockEmbedder`):** Sử dụng hàm băm MD5 tạo số giả ngẫu nhiên để kiểm thử luồng mã nguồn. Do MD5 không biểu diễn ngữ nghĩa, điểm số cosine của các câu đồng nghĩa dao động ngẫu nhiên quanh 0 (-0.03 đến -0.09).
2. **Trình nhúng nơ-ron đa ngữ cục bộ (`LocalEmbedder`):** Nhóm kích hoạt mô hình thật **`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`** (384 chiều) ở chế độ offline (`HF_HUB_OFFLINE=1`). Mô hình nạp trực tiếp từ bộ đệm máy tính, giúp điểm số cosine phản ánh đúng độ tương đồng ngữ nghĩa tiếng Việt (đạt **0.75 – 0.94** cho câu đồng nghĩa và giảm sâu xuống **0.03 – 0.18** cho câu khác miền).

### 3.2. Bộ 5 Câu Hỏi Đánh Giá & Câu Trả Lời Chuẩn (Gold Answers)

Bộ câu hỏi chuẩn hóa trong `bench.py` bám sát các nghiệp vụ trọng tâm của sàn Shopee:

| # | Mã Query | Dạng câu hỏi | Câu hỏi đánh giá (Query) | Bộ lọc (`filter`) | Câu trả lời chuẩn (Gold Answer) | Tài liệu nguồn |
|:-:|:---:|:---:|:---|:---:|:---|:---|
| 1 | **Q1** | Metadata Filter | Quyền và trách nhiệm của tôi đối với việc bảo hành sản phẩm trên sàn là gì? | `audience: seller` | Người Bán có trách nhiệm tiếp nhận bảo hành sản phẩm, dịch vụ cho Người Mua như cam kết trong Chính sách bảo hành sản phẩm của Người bán và/hoặc của nhà sản xuất và thông tin này phải được đăng tải trên Sàn Shopee trong phần mô tả sản phẩm. | `seller-warranty-policy.md` |
| 2 | **Q2** | Tra cứu số liệu | Đối với đơn hàng do Người bán tự vận chuyển, tôi có tối đa bao nhiêu ngày để gửi yêu cầu trả hàng kể từ lúc trạng thái cập nhật 'Lấy hàng thành công' mà tôi chưa bấm nhận hàng? | `None` | **20 ngày** kể từ lúc đơn hàng được cập nhật trạng thái "Lấy hàng thành công" và bạn không bấm "Đã nhận được hàng". | `return-refund-policy.md` |
| 3 | **Q3** | Điều kiện áp dụng | Sản phẩm của tôi cần đáp ứng các điều kiện cơ bản nào để được bảo hành? | `None` | Còn thời hạn bảo hành (dựa trên tem/phiếu bảo hành hoặc kích hoạt bảo hành điện tử); còn tem/phiếu bảo hành; sản phẩm bị lỗi kỹ thuật không phải do lỗi của Người Mua. | `buyer-warranty-policy.md` |
| 4 | **Q4** | Quy trình xử lý | Đối với khiếu nại không phải là Trả Hàng/Hoàn Tiền, Shopee xử lý vụ việc trong thời hạn bao lâu kể từ khi nhận đủ thông tin từ các bên? | `None` | Shopee yêu cầu các bên tranh chấp cung cấp đầy đủ thông tin/tài liệu, và đưa ra hướng giải quyết trong vòng **07 ngày làm việc** kể từ ngày nhận đủ thông tin; vụ việc phức tạp có thể kéo dài hơn. | `dispute-process.md` |
| 5 | **Q5** | Liệt kê danh mục | Hãy liệt kê tất cả các lý do mà tôi có thể dùng để gửi yêu cầu Trả hàng/Hoàn tiền trên Shopee. | `None` | Chưa nhận được hàng; thiếu hàng; Người bán gửi sai hàng; hàng lỗi, không hoạt động; khác với mô tả; hàng đã qua sử dụng; hàng giả/nhái; đổi ý (sản phẩm còn nguyên tem, nhãn mác, bao bì). | `return-refund-policy.md` |

---

### 3.3. Đánh Giá Hai Mức (Two-Level Evaluation — Tránh Thổi Phồng Điểm Số)

Theo chuẩn `docs/SCORING.md`, cách chấm ngây thơ chỉ kiểm tra `doc_id` của tài liệu gold sẽ **thổi phồng kết quả**, vì cả 3 chunk trong Top-3 có thể cùng thuộc tài liệu gold nhưng **không chunk nào chứa câu trả lời**. Nhóm áp dụng quy trình kiểm tra 2 mức nghiêm ngặt:

- **Mức 1 (Doc ID Match):** Kiểm tra `doc_id` của tài liệu gold có lọt vào Top-3 không.
- **Mức 2 (Content-level Verification):** Kiểm tra nội dung các chunk gold trong Top-3 có chứa đầy đủ chuỗi đặc trưng bắt buộc (`answer_markers`) hay không.
- **Thang điểm:** 2 điểm nếu bằng chứng xuất hiện ở Top-1; 1 điểm nếu xuất hiện ở Top-2 hoặc Top-3; 0 điểm nếu vắng mặt hoặc không đủ bằng chứng.

Kết quả kiểm tra chi tiết trên chiến lược `HeadingAwarePolicyChunker`:

| Query | Mức 1: Có đúng `doc_id`? | Thứ hạng bằng chứng đầu tiên | Mức 2: Độ phủ từ khóa (`answer_markers`) | Điểm số | Nhận xét chi tiết |
|:---:|:---:|:---:|:---:|:---:|:---|
| **Q1** | Có (Top 1 & 2) | Rank 2 | 3 / 3 markers đạt chuẩn | **1 / 2** | Chunk rank 1 là quyền và nghĩa vụ chung, chunk rank 2 chứa đúng điều khoản bảo hành của người bán. |
| **Q2** | Có (Top 1) | Rank 1 | 2 / 2 markers đạt chuẩn | **2 / 2** | Xuất sắc: Chunk Top-1 trích xuất chính xác con số "20 ngày" và mốc "Lấy hàng thành công". |
| **Q3** | Có (Top 1) | Rank 1 | 3 / 3 markers đạt chuẩn | **2 / 2** | Xuất sắc: Chunk Top-1 nêu đầy đủ 3 điều kiện (còn hạn, còn tem/phiếu, lỗi kỹ thuật). |
| **Q4** | Có (Top 1) | Rank 1 | 2 / 2 markers đạt chuẩn | **2 / 2** | Xuất sắc: Chunk Top-1 nêu đúng quy trình giải quyết trong "07 ngày làm việc". |
| **Q5** | Có (Top 1) | None | 0 / 8 markers (thiếu toàn bộ danh sách) | **0 / 2** | **Failure Case:** Chunk Top-1 chỉ là tiêu đề chung, không chứa trọn vẹn danh mục 8 lý do đổi trả. |
| **TỔNG** | **5 / 5 (100%)** | | | **7 / 10** | **Chênh lệch: Mức 1 đạt 10/10 nhưng Mức 2 phản ánh chính xác điểm thực tế 7/10** |

---

### 3.4. Thử Nghiệm Đối Chứng A/B Bắt Buộc (Metadata Filter)

Nhóm thực hiện A/B testing bắt buộc trên câu hỏi nhạy cảm về vai trò: *Q1 — "Quyền và trách nhiệm của tôi đối với việc bảo hành sản phẩm trên sàn là gì?"* trên toàn bộ 4 chiến lược chia nhỏ:

| Chiến lược Chunking | Thứ hạng khi KHÔNG CÓ FILTER | Thứ hạng khi CÓ FILTER (`audience: seller`) | Tác động thực nghiệm (A/B Observation) | Nhận xét chi tiết |
|:---|:---:|:---:|:---:|:---|
| **SentenceChunker** | **None** (Không có trong Top-3) | **Rank 3** (Lọt vào Top-3) | 🟢 **Improved (Cải thiện rõ rệt)** | Khi không lọc, 3 chunk của Buyer và Mall chiếm trọn Top-3. Tiền lọc `seller` cứu kết quả đưa gold chunk vào Rank 3. |
| **RecursiveChunker** | Rank 3 | **Rank 2** | 🟢 **Improved (Tăng hạng)** | Tiền lọc loại bỏ bớt tài liệu Buyer, đẩy chunk trách nhiệm người bán từ vị trí thứ 3 lên vị trí thứ 2. |
| **FixedSizeChunker** | Rank 1 | Rank 1 | ⚪ **Unchanged (Giữ nguyên)** | Chunk cắt cứng giữ được từ khóa câu hỏi ở ngay Rank 1 ở cả hai lượt chạy. |
| **HeadingAwarePolicyChunker** | Rank 2 | Rank 2 | ⚪ **Unchanged (Ổn định)** | Cả hai lượt chạy đều trả về Top-1 là quyền hạn chung và Top-2 là điều khoản bảo hành Người bán. |

> 📌 **Kết luận về tính hữu dụng của Metadata Filter:**  
> Dữ liệu đối chứng thực nghiệm chứng minh tiền lọc metadata mang tính **sống còn đối với các chiến lược phân tách tự nhiên** (đặc biệt là `SentenceChunker` — chuyển từ thất bại hoàn toàn `None` thành công `Rank 3`). Do Người mua và Người bán đều dùng chung các thuật ngữ "quyền và trách nhiệm", "bảo hành sản phẩm", tiền lọc metadata giúp vector store loại bỏ 100% tài liệu sai đối tượng trước khi xếp hạng cosine.

---

## 4. THUYẾT TRÌNH (DEMO), BÀI HỌC NHÓM & PHÂN TÍCH LỖI — 5 ĐIỂM

### 4.1. Những Phân Tích Hay Nhất Nhóm Sẽ Trình Bày (Demo Insights)
1. **Cấu trúc phân tầng là sống còn:** Trong văn bản quy chế chính sách, việc cắt nhỏ văn bản mà làm mất tiêu đề mục sẽ phá hủy hoàn toàn ngữ cảnh pháp lý. Chiến lược đính kèm heading vào từng subchunk (`HeadingAwarePolicyChunker`) giải quyết triệt để vấn đề này.
2. **Giá trị thực tế của Metadata Filter:** Tiền lọc siêu dữ liệu không phải là tính năng phụ trợ mà là rào chắn bắt buộc trong các bài toán đa đối tượng (Buyer vs Seller), giúp ngăn ngừa triệt để hiện tượng lẫn lộn trách nhiệm giữa các bên.
3. **Độ tương đồng Cosine không phản ánh mật độ câu trả lời:** Điểm số Cosine của mô hình vector dày đặc chỉ đo độ tương đồng về chủ đề tổng quát chứ không đo lường được liệu đoạn văn có chứa đầy đủ dữ liệu con số hay danh mục câu trả lời hay không.

### 4.2. Bài Học Rút Ra Khi So Sánh Trong Nhóm
Cùng một bộ 8 tài liệu chính sách, chiến lược chia nhỏ (chunking strategy) tạo ra sự khác biệt sống còn cho chất lượng câu trả lời của RAG Agent. `FixedSizeChunker` dù đơn giản nhưng làm đứt gãy câu văn và mất tiêu đề mục, khiến agent trích xuất sai ở câu hỏi quy trình (Q4: 0/2). `SentenceChunker` bảo đảm câu trọn vẹn nhưng phân mảnh danh sách gạch đầu dòng khiến câu hỏi điều kiện bảo hành bị trượt (Q3: 0/2). `HeadingAwarePolicyChunker` và `RecursiveChunker` đạt điểm cao nhất nhóm (7/10) nhờ giữ được tính mạch lạc của các khối điều khoản.

### 4.3. Nếu Làm Lại, Nhóm Sẽ Thay Đổi Gì Trong Chiến Lược Dữ Liệu?
- **Mở rộng lược đồ Metadata:** Bổ sung trường `policy_code` (mã điều khoản), `effective_date`, và `section_type` (`condition`, `process`, `penalty`).
- **Tích hợp Tìm kiếm lai (Hybrid Search):** Kết hợp thuật toán từ khóa BM25 (để bắt chính xác các con số như "20 ngày", "07 ngày làm việc") cùng mô hình vector MiniLM để đạt độ chính xác toàn diện 10/10.

---

### 4.4. Phân Tích Lỗi Thực Tế (Failure Case Analysis — 3 Phần Chuẩn)

Theo yêu cầu Checkpoint 6, nhóm phân tích trường hợp lỗi thực tế ghi nhận trên toàn bộ các chiến lược:

#### 1. Câu hỏi gặp sự cố (Failure Query):
- **Câu hỏi:** *Query 5 — "Hãy liệt kê tất cả các lý do mà tôi có thể dùng để gửi yêu cầu Trả hàng/Hoàn tiền trên Shopee."* (Không dùng filter).
- **Câu trả lời chuẩn mong đợi:** Đủ 8 lý do: *Chưa nhận được hàng; thiếu hàng; Người bán gửi sai hàng; hàng lỗi, không hoạt động; khác với mô tả; hàng đã qua sử dụng; hàng giả/nhái; đổi ý.*

#### 2. Vì sao thất bại (Root Cause Analysis):
- **Hiện tượng:** Cả 4 chiến lược đều chỉ đạt điểm **0 / 2** trên Query 5.
- **Nguyên nhân gốc rễ:**
  - *Vấn đề tiêu đề chung thắng đoạn liệt kê:* Chunk Top-1 trả về đoạn tiêu đề chung `1. Điều kiện Trả hàng/Hoàn tiền của Shopee` (`score: 0.7340`), trong khi đoạn bảng chi tiết liệt kê 8 lý do lại bị đẩy xuống sau hoặc bị chia nhỏ thành nhiều phần.
  - *Vấn đề giới hạn kích thước chunk (Chunk Size Limit):* Danh mục 8 lý do kèm điều kiện chi tiết trong văn bản gốc kéo dài hơn 800 ký tự. Khi áp dụng `chunk_size = 500`, danh mục bị phân mảnh sang 2-3 chunk khác nhau. Vì kiểm thử Top-K chỉ lấy 3 chunk và yêu cầu độ phủ 8/8 markers, không có chunk đơn lẻ nào chứa trọn vẹn toàn bộ 8 lý do.
  - *Hạn chế của Cosine Similarity:* Mô hình vector bi-encoder đo độ giống về chủ đề tổng quát ("Trả hàng", "Hoàn tiền") chứ **không đo lường được mật độ thông tin liệt kê (enumeration completeness)**.

#### 3. Đề xuất cải thiện (Proposed Fixes):
- **Giải pháp 1 — Multi-chunk Aggregation:** Khi phát hiện câu hỏi dạng liệt kê ("Hãy liệt kê tất cả..."), RAG Agent cần mở rộng context window, gom hợp nhất nội dung của các chunk lân cận thuộc cùng một Section thay vì chỉ gửi chunk rời rạc vào prompt.
- **Giải pháp 2 — Hybrid Search (BM25 + Dense Vector):** Bổ sung tìm kiếm từ khóa chính xác BM25 để kéo các chunk chứa danh sách từ khóa liệt kê đặc thù lên vị trí ưu tiên.
- **Giải pháp 3 — Document Structure Tuning:** Nhận diện các bảng biểu và danh sách liệt kê dạng gạch đầu dòng để giữ nguyên vẹn trong một khối chunk nguyên tử (Atomic Chunk) bất kể kích thước vượt nhẹ 500 ký tự.

---

## 5. TỰ ĐÁNH GIÁ PHẦN NHÓM (SELF-EVALUATION)

| Hạng mục đánh giá | Tiêu chí rubric (`docs/SCORING.md`) | Điểm tối đa | Điểm tự đánh giá | Minh chứng đạt được |
|:---|:---|:---:|:---:|:---|
| **1. Lựa chọn tài liệu** | Đủ 5-10 tài liệu, chủ đề rõ ràng, metadata minh bạch, nguồn kiểm chứng được | 10 | **10 / 10** | 8 tài liệu Markdown Shopee sạch, đủ 8 trường metadata, 100% pass CP2 |
| **2. Thiết kế chiến lược** | Baseline analysis, 4 thành viên thử nghiệm riêng, so sánh đối chứng chặt chẽ | 15 | **15 / 15** | Bảng baseline 4 dòng, custom chunker có flowchart/code, so sánh 4 người |
| **3. Chất lượng truy xuất** | 5 benchmark queries, gold answers, chấm 2 mức, thử nghiệm A/B metadata | 10 | **10 / 10** | 5 query chuẩn, bảng đánh giá 2 mức chi tiết, bảng A/B đối chứng đầy đủ |
| **4. Thuyết trình & Báo cáo** | Demo insights, bài học nhóm, phân tích lỗi thực tế đủ 3 phần | 5 | **5 / 5** | 3 insights sắc bén, phân tích lỗi Query 5 đủ 3 phần, đề xuất Hybrid/Aggregation |
| **TỔNG ĐIỂM PHẦN NHÓM** | | **40** | **40 / 40** | **Xuất sắc toàn diện theo chuẩn K4-L3B** |

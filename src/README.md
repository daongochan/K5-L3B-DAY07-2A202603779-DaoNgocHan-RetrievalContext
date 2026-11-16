# Hướng dẫn chạy code Day 7

Thư mục này chứa toàn bộ code thực nghiệm retrieval, chọn context và vòng lặp agent của bài lab. Các đường dẫn bên dưới tính từ thư mục gốc của repo.

## Các file trong `src/`

| File | Vai trò |
|---|---|
| `prepare_data.py` | Đọc chunk `B_para` và câu hỏi `dev` từ Day 6, chuyển sang định dạng Day 7 trong `data/` và sinh bộ câu hỏi `test` cùng các câu thiếu bằng chứng. |
| `ingest_pgvector.py` | Nạp lại 136 chunk từ `data/chunks.json` vào PostgreSQL + `pgvector` bằng `intfloat/multilingual-e5-base` khi chạy trên máy mới chưa có sẵn volume của Day 6. |
| `verify_benchmark.py` | Kiểm tra định dạng JSON, độ phủ chuỗi evidence trên từng trang, tính tách biệt giữa `dev` và `test`, và đối chiếu 136 chunk với bảng `chunks` trong database. |
| `retrievers.py` | Gồm 4 bộ truy xuất (`TfidfRetriever`, `Bm25Retriever`, `PgVectorDenseRetriever`, `HybridRrfRetriever`) và hàm `infer_query_filters` suy luận filter từ nội dung câu hỏi. |
| `context_builder.py` | Khử trùng lặp, mở rộng các chunk kế cận cùng Điều hoặc cùng trang Phụ lục, trộn vòng tròn kết quả qua nhiều lượt tìm kiếm và cắt theo ngân sách ký tự. |
| `eval_retrieval.py` | Chạy và lưu `summary.json`, `traces.json` cho các cấu hình retrieval trên `dev` hoặc `test` (cần truyền `--allow-test` khi chạy `test`). |
| `openrouter_client.py` | Gọi model qua OpenRouter API với tool calling, có retry khi lỗi mạng tạm thời và tính token, chi phí từng lượt. |
| `agent_runner.py` | Vòng lặp agent với các tool `search_legal_corpus`, `lookup_article_or_form`, `submit_final_response`. |
| `save_dev_reviews.py`, `save_test_reviews.py` | Ghi lại nhãn chấm thủ công (`reviews.json`, `manual_context_audit.json`) và gọi `starter/agent_metrics.py` để tổng hợp kết quả. |

## Chuẩn bị môi trường

1. Tạo môi trường ảo Python 3.11+ (mình chạy trên Python 3.13.5 với `uv`) và cài đặt thư viện:
   ```powershell
   uv venv
   uv pip install --python .\.venv\Scripts\python.exe -r requirements.txt
   ```

2. Tạo file `.env` từ `.env.example` và điền `OPENROUTER_API_KEY` (file `.env` đã nằm trong `.gitignore`):
   ```ini
   DATABASE_URL=postgresql://day6:day6_local_only@127.0.0.1:5433/day6_lab
   DAY6_STRATEGY=B_para
   EMBEDDING_MODEL=intfloat/multilingual-e5-base
   OPENROUTER_MODEL=openai/gpt-4o-mini
   OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
   OPENROUTER_API_KEY=<key_cua_ban>
   ```

3. Khởi chạy PostgreSQL + `pgvector`:
   - Nếu trên máy đã có sẵn container từ bài Day 6:
     ```powershell
     docker start day6_student_starter-postgres-1
     ```
   - Nếu chạy độc lập trên máy mới chỉ clone repo Day 7 này:
     ```powershell
     docker compose up -d
     .\.venv\Scripts\python.exe -m src.ingest_pgvector
     ```

## Lệnh chạy kiểm tra và tái lập kết quả

```powershell
# Chạy 26 unit test (chạy offline, không gọi API)
.\.venv\Scripts\python.exe -m unittest discover -s tests -v

# Kiểm tra dữ liệu, benchmark và kết nối pgvector
.\.venv\Scripts\python.exe -m src.verify_benchmark

# Chạy 7 cấu hình retrieval và đóng gói context trên tập dev
.\.venv\Scripts\python.exe -m src.eval_retrieval --split dev

# Chạy agent trên tập dev và tổng hợp nhãn chấm thủ công
.\.venv\Scripts\python.exe -m src.agent_runner --split dev --configs agent_baseline_tfidf_b800 agent_whyb_v2_b3200
.\.venv\Scripts\python.exe -m src.save_dev_reviews

# Chạy đánh giá trên tập test sau khi đã chốt cấu hình
.\.venv\Scripts\python.exe -m src.eval_retrieval --split test --allow-test --configs cfg1_tfidf_b800 imp3_whyb_qfilter_sibling_b2800
.\.venv\Scripts\python.exe -m src.agent_runner --split test --allow-test --configs agent_baseline_tfidf_b800 agent_whyb_v2_b3200
.\.venv\Scripts\python.exe -m src.save_test_reviews
```

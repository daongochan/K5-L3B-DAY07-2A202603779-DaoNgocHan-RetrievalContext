"""Persist human-audited held-out test context grades & agent review labels, and run starter/agent_metrics.py."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

MANUAL_CONTEXT_AUDIT_TEST = {
    "scale": "0 = không có evidence; 1 = có nhưng thiếu điều kiện/phần bằng chứng bắt buộc; 2 = đủ evidence và nguồn",
    "cfg1_tfidf_b800": {
        "counts": {"0": 1, "1": 1, "2": 5, "N": 7, "mean": round((0 * 1 + 1 * 1 + 2 * 5) / 7, 4)},
        "items": [
            {"query_id": "N_T01", "context_grade": 2, "context_ids": ["nghi-dinh-13-2023-nd-cp:p011:B_para:001"], "reason": "Có trọn khoản 1 Điều 13 (trang 11) nêu thực hiện một lần trước khi tiến hành xử lý dữ liệu."},
            {"query_id": "N_T02", "context_grade": 2, "context_ids": ["nghi-dinh-13-2023-nd-cp:p016:B_para:002"], "reason": "Có trọn Điều 18 (trang 16) nêu không cần đồng ý nhưng phải thông báo đang ghi âm, ghi hình và ngoại lệ."},
            {"query_id": "N_T03", "context_grade": 2, "context_ids": ["nghi-dinh-13-2023-nd-cp:p016:B_para:003"], "reason": "Có trọn khoản 1 và khoản 2 Điều 19 (trang 16)."},
            {"query_id": "N_T04", "context_grade": 2, "context_ids": ["nghi-dinh-13-2023-nd-cp:p013:B_para:002"], "reason": "Có trọn khoản 9 Điều 14 (trang 13) nêu chi phí in, sao, chụp, bưu chính, fax và phương thức, thời hạn thanh toán."},
            {"query_id": "N_T05", "context_grade": 2, "context_ids": ["nghi-dinh-13-2023-nd-cp:p026:B_para:003", "nghi-dinh-13-2023-nd-cp:p029:B_para:000"], "reason": "Có trọn khoản 1 và 2 Điều 35 (trang 26)."},
            {"query_id": "N_T06", "context_grade": 0, "context_ids": ["nghi-dinh-13-2023-nd-cp:p021:B_para:000"], "reason": "Không có filter Phụ lục nên TF-IDF lấy khoản 6 Điều 24 (trang 21) chỉ nhắc tên Mẫu số 05, thiếu hoàn toàn trang 36 của biểu mẫu."},
            {"query_id": "N_T07", "context_grade": 1, "context_ids": ["nghi-dinh-13-2023-nd-cp:p021:B_para:001"], "reason": "Có Điều 25 khoản 1 (trang 21) xác nhận Bên Xử lý thuộc Bên chuyển dữ liệu ra nước ngoài, nhưng thiếu định nghĩa phân biệt tại Điều 2 (trang 4)."},
        ],
    },
    "imp3_whyb_qfilter_sibling_b2800": {
        "counts": {"0": 0, "1": 2, "2": 5, "N": 7, "mean": round((0 * 0 + 1 * 2 + 2 * 5) / 7, 4)},
        "items": [
            {"query_id": "N_T01", "context_grade": 2, "context_ids": ["nghi-dinh-13-2023-nd-cp:p011:B_para:001", "nghi-dinh-13-2023-nd-cp:p011:B_para:002", "nghi-dinh-13-2023-nd-cp:p024:B_para:000", "nghi-dinh-13-2023-nd-cp:p011:B_para:003"], "reason": "Đầy đủ cả khoản 1 lẫn các trường hợp miễn thông báo tại khoản 4 Điều 13 nhờ mở rộng sibling."},
            {"query_id": "N_T02", "context_grade": 2, "context_ids": ["nghi-dinh-13-2023-nd-cp:p016:B_para:002", "nghi-dinh-13-2023-nd-cp:p016:B_para:000", "nghi-dinh-13-2023-nd-cp:p010:B_para:001", "nghi-dinh-13-2023-nd-cp:p016:B_para:001"], "reason": "Đầy đủ trọn vẹn Điều 18 (trang 16) và nguồn Công báo tr.18."},
            {"query_id": "N_T03", "context_grade": 2, "context_ids": ["nghi-dinh-13-2023-nd-cp:p016:B_para:003", "nghi-dinh-13-2023-nd-cp:p017:B_para:000", "nghi-dinh-13-2023-nd-cp:p015:B_para:003", "nghi-dinh-13-2023-nd-cp:p010:B_para:000"], "reason": "Đầy đủ trọn vẹn khoản 1 và 2 Điều 19 (trang 16)."},
            {"query_id": "N_T04", "context_grade": 2, "context_ids": ["nghi-dinh-13-2023-nd-cp:p013:B_para:002", "nghi-dinh-13-2023-nd-cp:p013:B_para:001", "nghi-dinh-13-2023-nd-cp:p012:B_para:000"], "reason": "Đầy đủ khoản 8-9 Điều 14 (trang 13) kèm khoản 3 (trang 12) nhờ mở rộng sibling."},
            {"query_id": "N_T05", "context_grade": 2, "context_ids": ["nghi-dinh-13-2023-nd-cp:p026:B_para:003", "nghi-dinh-13-2023-nd-cp:p026:B_para:001", "nghi-dinh-13-2023-nd-cp:p027:B_para:000", "nghi-dinh-13-2023-nd-cp:p024:B_para:001"], "reason": "Tự suy luận đúng filter article='Điều 35', xếp p026:B_para:003 lên hạng 1 với đủ 2 khoản."},
            {"query_id": "N_T06", "context_grade": 1, "context_ids": ["nghi-dinh-13-2023-nd-cp:p036:B_para:000", "nghi-dinh-13-2023-nd-cp:p037:B_para:000", "nghi-dinh-13-2023-nd-cp:p032:B_para:002"], "reason": "Tự suy luận đúng filter page_type='phu_luc_bieu_mau' nên lấy trúng tiêu đề Mẫu số 05 (p036:000), nhưng truy xuất đơn lượt budget 2800 chưa kịp nhét thêm chú thích cuối trang p036:003 (được khắc phục ở tầng Agent bằng lookup_article_or_form)."},
            {"query_id": "N_T07", "context_grade": 1, "context_ids": ["nghi-dinh-13-2023-nd-cp:p021:B_para:001", "nghi-dinh-13-2023-nd-cp:p021:B_para:002", "nghi-dinh-13-2023-nd-cp:p022:B_para:001"], "reason": "Truy vấn gộp đơn lượt lấy được Điều 25 khoản 1 (trang 21), nhưng thiếu định nghĩa phân biệt Bên Kiểm soát và Bên Xử lý ở Điều 2 (trang 4)."},
        ],
    },
}

TEST_REVIEWS = {
    "agent_baseline_tfidf_b800": [
        {
            "query_id": "N_T01",
            "answer_grade": 0,
            "citation_support": [False],
            "note": "Sau 3 lượt search với budget=800 không trộn batch, context cuối chỉ còn p024:B_para:000 (Điều 28 khoản 3), hoàn toàn không có mệnh đề 'một lần trước khi tiến hành' của Điều 13; model tự nhớ hoặc suy diễn rồi gắn sai citation p024:B_para:000.",
        },
        {
            "query_id": "N_T02",
            "answer_grade": 2,
            "citation_support": [True],
            "note": "Trả lời đúng và đủ cả điều kiện lẫn ngoại lệ theo Điều 18 (p016:B_para:002).",
        },
        {
            "query_id": "N_T03",
            "answer_grade": 2,
            "citation_support": [True],
            "note": "Trả lời đúng thứ tự hàng đồng ý và trường hợp không có thân nhân theo Điều 19 (p016:B_para:003).",
        },
        {
            "query_id": "N_T04",
            "answer_grade": 2,
            "citation_support": [True],
            "note": "Trả lời đúng và đủ chi phí in, sao, chụp, bưu chính, fax và phương thức/thời hạn thanh toán theo Điều 14 khoản 9 (p013:B_para:002).",
        },
        {
            "query_id": "N_T05",
            "answer_grade": 2,
            "citation_support": [True],
            "note": "Trả lời đúng đủ 2 khoản trách nhiệm của Bộ KH&CN tại Điều 35 (p026:B_para:003).",
        },
        {
            "query_id": "N_T06",
            "answer_grade": 0,
            "citation_support": [],
            "note": "Từ chối nhầm vì TF-IDF không có filter Phụ lục và không có lookup_article_or_form nên chỉ lấy được Điều 24 trang 21.",
        },
        {
            "query_id": "N_T07",
            "answer_grade": 0,
            "citation_support": [False],
            "note": "Context cuối chỉ có Điều 24 khoản 1 (p019:B_para:001); model tự suy diễn sai ngược pháp luật rằng Bên Xử lý không thuộc nhóm phải lập hồ sơ chuyển dữ liệu ra nước ngoài và gắn sai citation p019:B_para:001.",
        },
        {
            "query_id": "U_T01",
            "abstention_correct": True,
            "citation_support": [],
            "note": "Đọc đúng Điều 38 khoản 2 (p027:B_para:002) và từ chối chuẩn xác vì không có thời hạn tháng/năm.",
        },
        {
            "query_id": "U_T02",
            "abstention_correct": True,
            "citation_support": [],
            "note": "Từ chối đúng, không bịa danh sách quốc gia tương đương.",
        },
        {
            "query_id": "U_T03",
            "abstention_correct": True,
            "citation_support": [],
            "note": "Từ chối đúng, không bịa chuẩn thuật toán AES-256 hay RSA-2048.",
        },
    ],
    "agent_whyb_v2_b3200": [
        {
            "query_id": "N_T01",
            "answer_grade": 2,
            "citation_support": [True],
            "note": "Trả lời đúng một lần trước khi tiến hành và trích dẫn chuẩn khoản 1 Điều 13 (p011:B_para:001).",
        },
        {
            "query_id": "N_T02",
            "answer_grade": 2,
            "citation_support": [True],
            "note": "Trả lời đúng và đủ cả trách nhiệm thông báo lẫn ngoại lệ theo Điều 18 (p016:B_para:002).",
        },
        {
            "query_id": "N_T03",
            "answer_grade": 2,
            "citation_support": [True],
            "note": "Tra cứu kiểm tra Điều 17, 18, 19 cẩn trọng và trả lời đúng đủ cả 2 khoản Điều 19 (p016:B_para:003).",
        },
        {
            "query_id": "N_T04",
            "answer_grade": 2,
            "citation_support": [True],
            "note": "Trả lời đúng và đủ theo khoản 9 Điều 14 (p013:B_para:002).",
        },
        {
            "query_id": "N_T05",
            "answer_grade": 2,
            "citation_support": [True],
            "note": "Suy luận đúng filter Điều 35 và trả lời đầy đủ 2 trách nhiệm của Bộ KH&CN (p026:B_para:003).",
        },
        {
            "query_id": "N_T06",
            "answer_grade": 2,
            "citation_support": [True, True, True, True],
            "note": "Dùng lookup_article_or_form(form='Mẫu số 05') lấy trọn cả 4 đoạn trang 36, trả lời đúng cả mục đích thông báo thay đổi hồ sơ lẫn 2 loại hồ sơ áp dụng ở chú thích chân trang.",
        },
        {
            "query_id": "N_T07",
            "answer_grade": 1,
            "citation_support": [True, False, False],
            "note": "Đúng một phần: trả lời chính xác vế 1 phân biệt Bên Kiểm soát và Bên Xử lý theo Điều 2 (p004:000 hỗ trợ cả 2 định nghĩa, nhưng đoạn định nghĩa Bên Xử lý bị gắn nhầm sang p004:001 cạnh đó). Ở vế 2, lượt tìm kiếm thứ 4 gọi nhầm lookup Điều 14 thay vì Điều 25 nên context chỉ có khoản 5-6 Điều 25 (p022:001) mà thiếu khoản 1 Điều 25 (p021:001), dẫn tới suy luận sai rằng Bên Xử lý không phải lập hồ sơ.",
        },
        {
            "query_id": "U_T01",
            "abstention_correct": True,
            "citation_support": [],
            "note": "Gọi lookup_article_or_form('Điều 38') và tìm kiểm chứng 4 lượt trên toàn văn rồi từ chối chính xác, không ảo giác thời hạn lưu nhật ký.",
        },
        {
            "query_id": "U_T02",
            "abstention_correct": True,
            "citation_support": [],
            "note": "Từ chối đúng, xác nhận Nghị định không quy định danh sách quốc gia tương đương để miễn hồ sơ.",
        },
        {
            "query_id": "U_T03",
            "abstention_correct": True,
            "citation_support": [],
            "note": "Từ chối đúng, xác nhận Nghị định không quy định chuẩn mật mã AES-256 hay RSA-2048.",
        },
    ],
}


def main() -> None:
    ctx_audit_path = ROOT / "results" / "retrieval" / "test" / "manual_context_audit.json"
    ctx_audit_path.parent.mkdir(parents=True, exist_ok=True)
    ctx_audit_path.write_text(
        json.dumps(MANUAL_CONTEXT_AUDIT_TEST, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    cases_path = ROOT / "data" / "agent_cases.json"
    for cfg, reviews in TEST_REVIEWS.items():
        cfg_dir = ROOT / "results" / "agent" / "test" / cfg
        rev_path = cfg_dir / "reviews.json"
        out_path = cfg_dir / "metrics_summary.json"
        rev_path.write_text(json.dumps(reviews, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "starter" / "agent_metrics.py"),
                "--cases",
                str(cases_path),
                "--traces",
                str(cfg_dir / "traces.json"),
                "--reviews",
                str(rev_path),
                "--split",
                "test",
                "--output",
                str(out_path),
            ],
            check=True,
        )


if __name__ == "__main__":
    main()

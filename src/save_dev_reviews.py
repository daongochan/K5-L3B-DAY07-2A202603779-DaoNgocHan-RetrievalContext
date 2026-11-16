"""Persist human-audited dev review labels and run starter/agent_metrics.py on all dev agent runs."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


DEV_REVIEWS = {
    "agent_baseline_tfidf_b800": [
        {"query_id": "D01", "answer_grade": 0, "citation_support": [], "note": "Từ chối nhầm do TF-IDF + budget 800 ký tự làm mất p029:B_para:001."},
        {"query_id": "D02", "answer_grade": 0, "citation_support": [], "note": "Từ chối nhầm do TF-IDF trượt định nghĩa ở Điều 2."},
        {"query_id": "D03", "answer_grade": 1, "citation_support": [True, True], "note": "Nêu đúng Điều 22 (cấm mua bán khi không đồng ý) nhưng thiếu nguyên tắc gốc Điều 3 khoản 4 ('không được mua, bán dưới mọi hình thức, trừ trường hợp luật có quy định khác')."},
        {"query_id": "D04", "answer_grade": 2, "citation_support": [True], "note": "Đúng 72 giờ và ngoại lệ theo Điều 9 khoản 6 (p008:B_para:000)."},
        {"query_id": "D05", "answer_grade": 0, "citation_support": [], "note": "Từ chối nhầm do TF-IDF không khớp câu hỏi diễn đạt lại."},
        {"query_id": "D06", "answer_grade": 0, "citation_support": [], "note": "Từ chối nhầm vì context rỗng 0 ký tự (chunk đúng dài >800 ký tự kể cả header)."},
        {"query_id": "D07", "answer_grade": 0, "citation_support": [], "note": "Từ chối nhầm vì chunk hạng 1 (p017:B_para:000) vượt budget 800 ký tự bị bỏ qua."},
        {"query_id": "D08", "answer_grade": 1, "citation_support": [False], "note": "Nêu đúng 72 giờ nhưng thiếu tên Bộ Công an (Cục A05), và chunk trích dẫn p018:B_para:002 (khoản 3) hoàn toàn không chứa mệnh đề 72 giờ."},
        {"query_id": "D09", "answer_grade": 2, "citation_support": [True], "note": "Đúng 60 ngày kể từ ngày tiến hành xử lý dữ liệu theo Điều 24 khoản 4."},
        {"query_id": "D10", "answer_grade": 2, "citation_support": [True], "note": "Đúng 02 năm đầu và nêu rõ ngoại lệ theo Điều 43."},
        {"query_id": "D11", "answer_grade": 2, "citation_support": [True], "note": "Đúng theo Điều 43 khoản 3."},
        {"query_id": "D12", "answer_grade": 0, "citation_support": [], "note": "Từ chối nhầm vì p024:B_para:001 đứng hạng 3 bị hết budget 800 ký tự."},
        {"query_id": "D13", "answer_grade": 2, "citation_support": [True], "note": "Đúng và đầy đủ theo Điều 12 khoản 1."},
        {"query_id": "D14", "answer_grade": 0, "citation_support": [False], "note": "Trả lời sai ngược quy định: kết luận Nghị định không quy định thời gian tối đa, trong khi khoản 3 Điều 14 quy định 72 giờ; p013:B_para:002 không chứng minh cho kết luận đó."},
        {"query_id": "D15", "answer_grade": 0, "citation_support": [], "note": "Từ chối nhầm sau 3 lượt tìm kiếm."},
        {"query_id": "D16", "answer_grade": 0, "citation_support": [], "note": "Từ chối nhầm sau 3 lượt tìm kiếm."},
        {"query_id": "D17", "answer_grade": 1, "citation_support": [True], "note": "Chỉ trả lời được vế miễn trừ 02 năm ở Điều 43 (p029:B_para:001), thiếu quy định cụ thể của Điều 28 do budget 800 ký tự làm rơi p023:B_para:002."},
        {"query_id": "U_D01", "abstention_correct": True, "citation_support": [], "note": "Từ chối đúng, không bịa mức phạt tiền."},
        {"query_id": "U_D02", "abstention_correct": True, "citation_support": [], "note": "Từ chối đúng, không bịa lệ phí thẩm định."},
        {"query_id": "U_D03", "abstention_correct": True, "citation_support": [], "note": "Từ chối đúng, không bịa chứng chỉ/năm kinh nghiệm."},
    ],
    "agent_whyb_interleave_b3200": [
        {"query_id": "D01", "answer_grade": 2, "citation_support": [True], "note": "Đúng ngày 01/7/2023 theo Điều 43."},
        {"query_id": "D02", "answer_grade": 2, "citation_support": [True, False], "note": "Định nghĩa đúng theo p003:B_para:000, nhưng gắn thừa citation p002:B_para:001 (dữ liệu cơ bản)."},
        {"query_id": "D03", "answer_grade": 2, "citation_support": [True], "note": "Đúng nguyên tắc và ngoại lệ tại Điều 3 khoản 4."},
        {"query_id": "D04", "answer_grade": 0, "citation_support": [], "note": "Từ chối nhầm sau khi tự đoán sai filter Điều 18 và gõ nhầm đuôi mã p008:B_para:006."},
        {"query_id": "D05", "answer_grade": 2, "citation_support": [True], "note": "Đúng 72 giờ và ngoại lệ theo Điều 16 khoản 5."},
        {"query_id": "D06", "answer_grade": 2, "citation_support": [True], "note": "Đúng theo Điều 11 khoản 6."},
        {"query_id": "D07", "answer_grade": 2, "citation_support": [True], "note": "Đúng từ đủ 7 tuổi trở lên kèm đồng ý của cha/mẹ/giám hộ theo Điều 20 khoản 2."},
        {"query_id": "D08", "answer_grade": 2, "citation_support": [True], "note": "Đúng Bộ Công an (Cục A05) và chậm nhất 72 giờ theo Điều 23 khoản 1."},
        {"query_id": "D09", "answer_grade": 2, "citation_support": [True], "note": "Đúng 60 ngày theo Điều 24 khoản 4."},
        {"query_id": "D10", "answer_grade": 2, "citation_support": [True], "note": "Đúng 02 năm đầu và ngoại lệ theo Điều 43."},
        {"query_id": "D11", "answer_grade": 2, "citation_support": [True], "note": "Đúng theo Điều 43 khoản 3."},
        {"query_id": "D12", "answer_grade": 2, "citation_support": [True], "note": "Đúng Cục A05 - Bộ Công an theo Điều 29 khoản 1."},
        {"query_id": "D13", "answer_grade": 2, "citation_support": [True, True], "note": "Đúng và đầy đủ cả khoản 1 (p010:003) và khoản 4 (p011:000) Điều 12."},
        {"query_id": "D14", "answer_grade": 2, "citation_support": [True], "note": "Đúng 72 giờ theo Điều 14 khoản 3."},
        {"query_id": "D15", "answer_grade": 2, "citation_support": [True], "note": "Đúng đối tượng cá nhân của Mẫu số 01 (p031:000)."},
        {"query_id": "D16", "answer_grade": 2, "citation_support": [True], "note": "Đúng tiêu đề Mẫu số 02 dành cho tổ chức, doanh nghiệp (p032:000)."},
        {"query_id": "D17", "answer_grade": 2, "citation_support": [True, True, True], "note": "Kết hợp đầy đủ cả Điều 28 (p023:002, p024:000) và Điều 43 (p029:001)."},
        {"query_id": "U_D01", "abstention_correct": True, "citation_support": [], "note": "Từ chối đúng."},
        {"query_id": "U_D02", "abstention_correct": True, "citation_support": [], "note": "Từ chối đúng."},
    ],
    "agent_whyb_v2_b3200": [
        {"query_id": "D01", "answer_grade": 2, "citation_support": [True], "note": "Đúng ngày 01/7/2023 theo Điều 43 khoản 1."},
        {"query_id": "D02", "answer_grade": 2, "citation_support": [True], "note": "Đúng định nghĩa dữ liệu cá nhân nhạy cảm và chỉ trích dẫn đúng p003:B_para:000."},
        {"query_id": "D03", "answer_grade": 2, "citation_support": [True], "note": "Đúng nguyên tắc và ngoại lệ tại Điều 3 khoản 4."},
        {"query_id": "D04", "answer_grade": 1, "citation_support": [False], "note": "Nêu đúng thời hạn 72 giờ nhưng trích dẫn nhầm sang p008:B_para:001 (khoản 8 Quyền phản đối xử lý) thay vì p008:B_para:000 (khoản 6 Quyền hạn chế xử lý)."},
        {"query_id": "D05", "answer_grade": 2, "citation_support": [True], "note": "Đúng 72 giờ và ngoại lệ theo Điều 16 khoản 5."},
        {"query_id": "D06", "answer_grade": 2, "citation_support": [True], "note": "Đúng theo Điều 11 khoản 6."},
        {"query_id": "D07", "answer_grade": 2, "citation_support": [True], "note": "Đúng theo Điều 20 khoản 2."},
        {"query_id": "D08", "answer_grade": 2, "citation_support": [True], "note": "Đúng Bộ Công an (Cục A05) và chậm nhất 72 giờ theo Điều 23 khoản 1."},
        {"query_id": "D09", "answer_grade": 2, "citation_support": [True], "note": "Đúng 60 ngày theo Điều 24 khoản 4."},
        {"query_id": "D10", "answer_grade": 2, "citation_support": [True], "note": "Đúng 02 năm đầu và ngoại lệ theo Điều 43."},
        {"query_id": "D11", "answer_grade": 2, "citation_support": [True], "note": "Đúng theo Điều 43 khoản 3."},
        {"query_id": "D12", "answer_grade": 2, "citation_support": [True], "note": "Đúng Cục A05 - Bộ Công an theo Điều 29 khoản 1."},
        {"query_id": "D13", "answer_grade": 2, "citation_support": [True], "note": "Đúng theo Điều 12 khoản 1."},
        {"query_id": "D14", "answer_grade": 2, "citation_support": [True], "note": "Gọi lookup_article_or_form('Điều 14') ở lượt 2 và trả lời đúng 72 giờ kèm ngoại lệ (p012:B_para:000)."},
        {"query_id": "D15", "answer_grade": 2, "citation_support": [True, True], "note": "Đúng đối tượng cá nhân (p031:000) và nêu đúng trường nơi cư trú ở mục 4 (p031:001)."},
        {"query_id": "D16", "answer_grade": 2, "citation_support": [True], "note": "Đúng tiêu đề Mẫu số 02 dành cho tổ chức, doanh nghiệp (p032:000)."},
        {"query_id": "D17", "answer_grade": 2, "citation_support": [True, True, True], "note": "Kết hợp đầy đủ cả Điều 28 (p023:002, p024:000) và Điều 43 (p029:001)."},
        {"query_id": "U_D01", "abstention_correct": True, "citation_support": [], "note": "Từ chối đúng, không suy đoán mức phạt."},
        {"query_id": "U_D02", "abstention_correct": True, "citation_support": [], "note": "Từ chối đúng, không suy đoán lệ phí."},
        {"query_id": "U_D03", "abstention_correct": True, "citation_support": [], "note": "Từ chối đúng, không suy đoán chứng chỉ/năm kinh nghiệm."},
    ],
}


def main() -> None:
    cases_path = ROOT / "data" / "agent_cases.json"
    for cfg, reviews in DEV_REVIEWS.items():
        cfg_dir = ROOT / "results" / "agent" / "dev" / cfg
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
                "dev",
                "--output",
                str(out_path),
            ],
            check=True,
        )


if __name__ == "__main__":
    main()

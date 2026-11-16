"""Step 1: Convert Day 6 corpus/chunks and build the frozen Day 7 benchmark.

Produces in `data/`:
- `manifest.json`: Provenance, Day 6 vector store & embedding spec, pre-committed selection criteria.
- `chunks.json`: 136 `B_para` chunks in Day 7 JSON array contract (`id`, `text`, `source_id`, `source`, `page`, `metadata`).
- `questions.json`: 24 answerable questions (17 dev + 7 new test) compatible with `starter/baseline.py` + `required_gold_groups` for strict multi-hop evaluation.
- `agent_cases.json`: All 30 agent benchmark cases (20 dev = 17 answerable + 3 unanswerable; 10 test = 7 answerable + 3 unanswerable) compatible with `starter/agent_metrics.py`.
- `benchmark_full.json`: Full benchmark specification with split rationale, multi-hop grading policy, and absence verification notes.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

DEFAULT_DAY6_ROOT = Path(__file__).resolve().parents[2] / "K5-L3B-DAY06-DaoNgocHan-2A202603779-DataIngestion"
SOURCE_ID = "nghi-dinh-13-2023-nd-cp"


def _resolve_exact_evidence(evidence: str, chunks: list[dict[str, Any]], page: int) -> str:
    """Resolve whitespace-normalized Day 6 evidence to exact verbatim substring in Day 7 chunk text."""
    page_chunks = [c for c in chunks if c["page"] == page and c["source_id"] == SOURCE_ID]
    for c in page_chunks:
        if evidence in c["text"]:
            return evidence
    pattern = r"\s+".join(re.escape(part) for part in evidence.split())
    for c in page_chunks:
        m = re.search(pattern, c["text"])
        if m:
            return m.group(0)
    raise ValueError(f"Không tìm thấy evidence trên trang {page}: {evidence!r}")


def convert_day6_chunk(raw: dict[str, Any]) -> dict[str, Any]:
    meta = dict(raw.get("metadata") or {})
    return {
        "id": raw["chunk_id"],
        "text": raw["text"],
        "source_id": raw["document_id"],
        "source": raw["source"],
        "page": int(raw["page"]),
        "metadata": meta,
    }


def convert_day6_dev_question(q: dict[str, Any], chunks: list[dict[str, Any]]) -> dict[str, Any]:
    gold = [
        {
            "source_id": SOURCE_ID,
            "page": int(g["physical_page"]),
            "printed_page": int(g["printed_page"]),
            "article": g["article"],
            "evidence": _resolve_exact_evidence(g["evidence"], chunks, int(g["physical_page"])),
        }
        for g in q["gold"]
    ]
    return {
        "id": q["id"],
        "split": "dev",
        "kind": q["kind"],
        "question": q["question"],
        "expected_answer": q["answer"],
        "gold": gold,
        "required_gold_groups": [gold],
    }


def build_new_questions() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return (additional_answerable_questions, unanswerable_cases)."""
    # 1 new multi-hop dev question + 7 brand-new test questions on untouched pages/articles
    new_answerable: list[dict[str, Any]] = [
        {
            "id": "D17",
            "split": "dev",
            "kind": "multi_hop",
            "question": (
                "Một doanh nghiệp khởi nghiệp mới thành lập (không trực tiếp kinh doanh hoạt động "
                "xử lý dữ liệu cá nhân) có xử lý dữ liệu cá nhân nhạy cảm thì theo quy định chung về "
                "bảo vệ dữ liệu nhạy cảm phải làm gì đối với bộ phận và nhân sự bảo vệ dữ liệu cá nhân, "
                "nhưng doanh nghiệp có được hưởng quyền miễn trừ trong bao lâu không?"
            ),
            "expected_answer": (
                "Theo Điều 28, bảo vệ dữ liệu cá nhân nhạy cảm yêu cầu chỉ định bộ phận có chức năng bảo vệ "
                "dữ liệu cá nhân, chỉ định nhân sự phụ trách và trao đổi thông tin với Cơ quan chuyên trách. "
                "Tuy nhiên, theo Điều 43, doanh nghiệp khởi nghiệp không trực tiếp kinh doanh hoạt động xử lý "
                "dữ liệu cá nhân được quyền lựa chọn miễn trừ quy định về chỉ định cá nhân và bộ phận bảo vệ "
                "dữ liệu cá nhân trong 02 năm đầu kể từ khi thành lập doanh nghiệp."
            ),
            "gold": [
                {
                    "source_id": SOURCE_ID,
                    "page": 23,
                    "printed_page": 25,
                    "article": "Điều 28",
                    "evidence": "Chỉ định bộ phận có chức năng bảo vệ dữ liệu cá nhân, chỉ định nhân sự phụ trách bảo vệ dữ liệu cá nhân",
                },
                {
                    "source_id": SOURCE_ID,
                    "page": 29,
                    "printed_page": 31,
                    "article": "Điều 43",
                    "evidence": "được quyền lựa chọn miễn trừ quy định về chỉ định cá nhân và bộ phận bảo vệ dữ liệu cá nhân trong thời gian 02 năm đầu kể từ khi thành lập doanh nghiệp",
                },
            ],
            "required_gold_groups": [
                [
                    {
                        "source_id": SOURCE_ID,
                        "page": 23,
                        "printed_page": 25,
                        "article": "Điều 28",
                        "evidence": "Chỉ định bộ phận có chức năng bảo vệ dữ liệu cá nhân, chỉ định nhân sự phụ trách bảo vệ dữ liệu cá nhân",
                    }
                ],
                [
                    {
                        "source_id": SOURCE_ID,
                        "page": 29,
                        "printed_page": 31,
                        "article": "Điều 43",
                        "evidence": "được quyền lựa chọn miễn trừ quy định về chỉ định cá nhân và bộ phận bảo vệ dữ liệu cá nhân trong thời gian 02 năm đầu kể từ khi thành lập doanh nghiệp",
                    }
                ],
            ],
            "multi_hop_note": "Cần kết hợp cả Điều 28 khoản 2 (trang 23) và Điều 43 khoản 2-3 (trang 29).",
        },
        # --- Brand-new held-out TEST answerable questions (N_T01 .. N_T07) ---
        {
            "id": "N_T01",
            "split": "test",
            "kind": "direct",
            "question": "Việc thông báo xử lý dữ liệu cá nhân cho chủ thể dữ liệu được thực hiện mấy lần và vào thời điểm nào?",
            "expected_answer": "Được thực hiện một lần trước khi tiến hành đối với hoạt động xử lý dữ liệu cá nhân (trừ các trường hợp được miễn theo quy định).",
            "gold": [
                {
                    "source_id": SOURCE_ID,
                    "page": 11,
                    "printed_page": 13,
                    "article": "Điều 13",
                    "evidence": "Việc thông báo được thực hiện một lần trước khi tiến hành đối với hoạt động xử lý dữ liệu cá nhân.",
                }
            ],
            "required_gold_groups": [
                [
                    {
                        "source_id": SOURCE_ID,
                        "page": 11,
                        "printed_page": 13,
                        "article": "Điều 13",
                        "evidence": "Việc thông báo được thực hiện một lần trước khi tiến hành đối với hoạt động xử lý dữ liệu cá nhân.",
                    }
                ]
            ],
        },
        {
            "id": "N_T02",
            "split": "test",
            "kind": "conditional",
            "question": (
                "Khi cơ quan, tổ chức có thẩm quyền ghi âm, ghi hình tại nơi công cộng nhằm bảo vệ an ninh "
                "quốc gia và trật tự an toàn xã hội thì có cần sự đồng ý của chủ thể dữ liệu không, và cơ quan "
                "đó có trách nhiệm thông báo gì?"
            ),
            "expected_answer": (
                "Không cần có sự đồng ý của chủ thể dữ liệu, nhưng cơ quan, tổ chức có thẩm quyền có trách nhiệm "
                "thông báo để chủ thể hiểu được mình đang bị ghi âm, ghi hình, trừ trường hợp pháp luật có quy định khác."
            ),
            "gold": [
                {
                    "source_id": SOURCE_ID,
                    "page": 16,
                    "printed_page": 18,
                    "article": "Điều 18",
                    "evidence": "Khi thực hiện việc ghi âm, ghi hình, cơ quan, tổ chức có thẩm quyền có trách nhiệm thông báo để chủ thể hiểu được mình đang bị ghi âm, ghi hình, trừ trường hợp pháp luật có quy định khác.",
                }
            ],
            "required_gold_groups": [
                [
                    {
                        "source_id": SOURCE_ID,
                        "page": 16,
                        "printed_page": 18,
                        "article": "Điều 18",
                        "evidence": "Khi thực hiện việc ghi âm, ghi hình, cơ quan, tổ chức có thẩm quyền có trách nhiệm thông báo để chủ thể hiểu được mình đang bị ghi âm, ghi hình, trừ trường hợp pháp luật có quy định khác.",
                    }
                ]
            ],
        },
        {
            "id": "N_T03",
            "split": "test",
            "kind": "conditional",
            "question": (
                "Ngoài các trường hợp tại Điều 17 và Điều 18, việc xử lý dữ liệu cá nhân của người đã chết "
                "phải được sự đồng ý của những ai; nếu không có bất kỳ người thân nào thuộc diện nêu trên thì xử lý như thế nào?"
            ),
            "expected_answer": (
                "Phải được sự đồng ý của vợ, chồng hoặc con thành niên của người đó; nếu không có những người này "
                "thì phải được sự đồng ý của cha, mẹ của người đã chết. Nếu không có tất cả những người nêu trên "
                "thì được coi là không có sự đồng ý."
            ),
            "gold": [
                {
                    "source_id": SOURCE_ID,
                    "page": 16,
                    "printed_page": 18,
                    "article": "Điều 19",
                    "evidence": "Trường hợp không có tất cả những người được nêu tại khoản 1 Điều này thì được coi là không có sự đồng ý.",
                }
            ],
            "required_gold_groups": [
                [
                    {
                        "source_id": SOURCE_ID,
                        "page": 16,
                        "printed_page": 18,
                        "article": "Điều 19",
                        "evidence": "Trường hợp không có tất cả những người được nêu tại khoản 1 Điều này thì được coi là không có sự đồng ý.",
                    }
                ]
            ],
        },
        {
            "id": "N_T04",
            "split": "test",
            "kind": "rephrased",
            "question": (
                "Khi giải quyết yêu cầu cung cấp dữ liệu cá nhân hợp lệ, Bên Kiểm soát dữ liệu cá nhân cần "
                "thông báo những nội dung gì về chi phí và thanh toán (nếu có) cho người yêu cầu?"
            ),
            "expected_answer": (
                "Thông báo chi phí thực tế để in, sao, chụp, gửi thông tin qua dịch vụ bưu chính, fax (nếu có) "
                "và phương thức, thời hạn thanh toán (cùng thời hạn, địa điểm, hình thức cung cấp dữ liệu)."
            ),
            "gold": [
                {
                    "source_id": SOURCE_ID,
                    "page": 13,
                    "printed_page": 15,
                    "article": "Điều 14",
                    "evidence": "chi phí thực tế để in, sao, chụp, gửi thông tin qua dịch vụ bưu chính, fax (nếu có) và phương thức, thời hạn thanh toán",
                }
            ],
            "required_gold_groups": [
                [
                    {
                        "source_id": SOURCE_ID,
                        "page": 13,
                        "printed_page": 15,
                        "article": "Điều 14",
                        "evidence": "chi phí thực tế để in, sao, chụp, gửi thông tin qua dịch vụ bưu chính, fax (nếu có) và phương thức, thời hạn thanh toán",
                    }
                ]
            ],
        },
        {
            "id": "N_T05",
            "split": "test",
            "kind": "direct",
            "question": "Theo Điều 35, Bộ Khoa học và Công nghệ có những trách nhiệm gì trong bảo vệ dữ liệu cá nhân?",
            "expected_answer": (
                "1) Phối hợp với Bộ Công an trong xây dựng Tiêu chuẩn bảo vệ dữ liệu cá nhân và các khuyến nghị áp dụng "
                "Tiêu chuẩn bảo vệ dữ liệu cá nhân; 2) Nghiên cứu, trao đổi với Bộ Công an về các biện pháp bảo vệ dữ liệu "
                "cá nhân theo kịp sự phát triển của khoa học, công nghệ."
            ),
            "gold": [
                {
                    "source_id": SOURCE_ID,
                    "page": 26,
                    "printed_page": 28,
                    "article": "Điều 35",
                    "evidence": "Phối hợp với Bộ Công an trong xây dựng Tiêu chuẩn bảo vệ dữ liệu cá nhân và các khuyến nghị áp dụng Tiêu chuẩn bảo vệ dữ liệu cá nhân.",
                }
            ],
            "required_gold_groups": [
                [
                    {
                        "source_id": SOURCE_ID,
                        "page": 26,
                        "printed_page": 28,
                        "article": "Điều 35",
                        "evidence": "Phối hợp với Bộ Công an trong xây dựng Tiêu chuẩn bảo vệ dữ liệu cá nhân và các khuyến nghị áp dụng Tiêu chuẩn bảo vệ dữ liệu cá nhân.",
                    }
                ]
            ],
            "query_filter_hint_note": "Query nêu tường minh 'Theo Điều 35'; hệ thống có thể tự suy luận filter article='Điều 35' từ query.",
        },
        {
            "id": "N_T06",
            "split": "test",
            "kind": "direct",
            "question": "Mẫu số 05 tại Phụ lục của Nghị định dùng để thông báo về việc gì và áp dụng cho những loại hồ sơ nào?",
            "expected_answer": (
                "Mẫu số 05 là Thông báo thay đổi nội dung hồ sơ (kính gửi Bộ Công an qua Cục An ninh mạng và phòng, "
                "chống tội phạm sử dụng công nghệ cao), áp dụng cho Hồ sơ đánh giá tác động xử lý dữ liệu cá nhân "
                "hoặc Hồ sơ đánh giá tác động chuyển dữ liệu cá nhân ra nước ngoài."
            ),
            "gold": [
                {
                    "source_id": SOURCE_ID,
                    "page": 36,
                    "printed_page": 38,
                    "article": "Phụ lục - Mẫu số 05",
                    "evidence": "THAY ĐỔI NỘI DUNG HỒ SƠ",
                },
                {
                    "source_id": SOURCE_ID,
                    "page": 36,
                    "printed_page": 38,
                    "article": "Phụ lục - Mẫu số 05",
                    "evidence": "1 Tên hồ sơ: Hồ sơ đánh giá tác động xử lý dữ liệu cá nhân hoặc Hồ sơ đánh giá tác động chuyển dữ liệu cá nhân ra nước ngoài.",
                },
            ],
            "required_gold_groups": [
                [
                    {
                        "source_id": SOURCE_ID,
                        "page": 36,
                        "printed_page": 38,
                        "article": "Phụ lục - Mẫu số 05",
                        "evidence": "THAY ĐỔI NỘI DUNG HỒ SƠ",
                    }
                ],
                [
                    {
                        "source_id": SOURCE_ID,
                        "page": 36,
                        "printed_page": 38,
                        "article": "Phụ lục - Mẫu số 05",
                        "evidence": "1 Tên hồ sơ: Hồ sơ đánh giá tác động xử lý dữ liệu cá nhân hoặc Hồ sơ đánh giá tác động chuyển dữ liệu cá nhân ra nước ngoài.",
                    }
                ],
            ],
            "query_filter_hint_note": (
                "Query nêu tường minh 'Mẫu số 05 tại Phụ lục'; tiêu đề nằm ở chunk p036:000 và chú thích loại hồ sơ nằm ở chunk p036:003."
            ),
        },
        {
            "id": "N_T07",
            "split": "test",
            "kind": "multi_hop",
            "question": (
                "Nghị định định nghĩa 'Bên Xử lý dữ liệu cá nhân' khác 'Bên Kiểm soát dữ liệu cá nhân' ở điểm nào, "
                "và khi chuyển dữ liệu cá nhân của công dân Việt Nam ra nước ngoài thì Bên Xử lý dữ liệu cá nhân có "
                "thuộc nhóm Bên chuyển dữ liệu phải lập Hồ sơ đánh giá tác động chuyển dữ liệu ra nước ngoài không?"
            ),
            "expected_answer": (
                "Theo Điều 2, Bên Kiểm soát dữ liệu cá nhân là tổ chức, cá nhân quyết định mục đích và phương tiện xử lý "
                "dữ liệu cá nhân, còn Bên Xử lý dữ liệu cá nhân là tổ chức, cá nhân thực hiện việc xử lý dữ liệu thay mặt "
                "cho Bên Kiểm soát dữ liệu thông qua hợp đồng hoặc thỏa thuận. Theo Điều 25 khoản 1, Bên Xử lý dữ liệu cá nhân "
                "có thuộc nhóm Bên chuyển dữ liệu ra nước ngoài và phải lập Hồ sơ đánh giá tác động chuyển dữ liệu cá nhân "
                "ra nước ngoài khi thực hiện chuyển dữ liệu cá nhân của công dân Việt Nam ra nước ngoài."
            ),
            "gold": [
                {
                    "source_id": SOURCE_ID,
                    "page": 4,
                    "printed_page": 6,
                    "article": "Điều 2",
                    "evidence": "Bên Xử lý dữ liệu cá nhân là tổ chức, cá nhân thực hiện việc xử lý dữ liệu thay mặt cho Bên Kiểm soát dữ liệu, thông qua một hợp đồng hoặc thỏa thuận với Bên Kiểm soát dữ liệu.",
                },
                {
                    "source_id": SOURCE_ID,
                    "page": 21,
                    "printed_page": 23,
                    "article": "Điều 25",
                    "evidence": "Bên chuyển dữ liệu ra nước ngoài bao gồm Bên Kiểm soát dữ liệu cá nhân, Bên Kiểm soát và xử lý dữ liệu cá nhân, Bên Xử lý dữ liệu cá nhân, Bên thứ ba.",
                },
            ],
            "required_gold_groups": [
                [
                    {
                        "source_id": SOURCE_ID,
                        "page": 4,
                        "printed_page": 6,
                        "article": "Điều 2",
                        "evidence": "Bên Xử lý dữ liệu cá nhân là tổ chức, cá nhân thực hiện việc xử lý dữ liệu thay mặt cho Bên Kiểm soát dữ liệu, thông qua một hợp đồng hoặc thỏa thuận với Bên Kiểm soát dữ liệu.",
                    }
                ],
                [
                    {
                        "source_id": SOURCE_ID,
                        "page": 21,
                        "printed_page": 23,
                        "article": "Điều 25",
                        "evidence": "Bên chuyển dữ liệu ra nước ngoài bao gồm Bên Kiểm soát dữ liệu cá nhân, Bên Kiểm soát và xử lý dữ liệu cá nhân, Bên Xử lý dữ liệu cá nhân, Bên thứ ba.",
                    }
                ],
            ],
            "multi_hop_note": "Cần kết hợp cả Điều 2 khoản 9-10 (trang 4) và Điều 25 khoản 1 (trang 21).",
        },
    ]

    unanswerable_cases: list[dict[str, Any]] = [
        # --- DEV unanswerable (3 cases) ---
        {
            "id": "U_D01",
            "split": "dev",
            "question": (
                "Mức phạt tiền hành chính tối đa (tính bằng đồng hoặc phần trăm doanh thu) đối với hành vi "
                "mua, bán trái phép dữ liệu cá nhân theo Nghị định 13/2023/NĐ-CP là bao nhiêu?"
            ),
            "answerable": False,
            "expected_behavior": "search_and_abstain",
            "absence_reason": (
                "Kiểm tra toàn bộ 37 trang PDF (136 chunks): Điều 3 khoản 4 và Điều 22 cấm mua, bán dữ liệu cá nhân; "
                "Điều 4 (trang 5) chỉ nêu nguyên tắc bị xử lý kỷ luật, xử phạt vi phạm hành chính, xử lý hình sự theo quy định, "
                "không quy định bất kỳ mức phạt tiền cụ thể hay tỷ lệ phần trăm doanh thu nào (0 kết quả cho 'phạt tiền', 'triệu đồng', 'tỷ đồng', 'doanh thu')."
            ),
        },
        {
            "id": "U_D02",
            "split": "dev",
            "question": (
                "Khi nộp Hồ sơ đánh giá tác động chuyển dữ liệu cá nhân ra nước ngoài cho Bộ Công an, doanh nghiệp "
                "phải nộp mức lệ phí thẩm định hồ sơ là bao nhiêu tiền theo Nghị định 13/2023/NĐ-CP?"
            ),
            "answerable": False,
            "expected_behavior": "search_and_abstain",
            "absence_reason": (
                "Kiểm tra toàn bộ 37 trang PDF (136 chunks), đặc biệt Điều 24, Điều 25 và Phụ lục Mẫu số 04, 06: "
                "văn bản quy định thủ tục gửi hồ sơ về Bộ Công an (Cục A05) nhưng hoàn toàn không quy định lệ phí thẩm định hồ sơ "
                "(0 kết quả cho 'lệ phí', 'thẩm định')."
            ),
        },
        {
            "id": "U_D03",
            "split": "dev",
            "question": (
                "Theo Nghị định 13/2023/NĐ-CP, nhân sự được chỉ định phụ trách bảo vệ dữ liệu cá nhân bắt buộc phải "
                "có chứng chỉ chuyên môn gì và tối thiểu bao nhiêu năm kinh nghiệm?"
            ),
            "answerable": False,
            "expected_behavior": "search_and_abstain",
            "absence_reason": (
                "Kiểm tra toàn bộ 37 trang PDF (136 chunks), gồm Điều 28 khoản 2, Điều 30 và các Mẫu số 03-06: "
                "chỉ yêu cầu chỉ định bộ phận/nhân sự và cung cấp họ tên, chức danh, SĐT, email; không quy định chứng chỉ bắt buộc "
                "hay số năm kinh nghiệm tối thiểu (0 kết quả cho 'chứng chỉ', 'năm kinh nghiệm')."
            ),
        },
        # --- TEST unanswerable (3 cases) ---
        {
            "id": "U_T01",
            "split": "test",
            "question": (
                "Theo Điều 38 Nghị định 13/2023/NĐ-CP, Bên Kiểm soát dữ liệu cá nhân phải lưu trữ nhật ký hệ thống "
                "quá trình xử lý dữ liệu cá nhân trong thời hạn tối thiểu bao nhiêu tháng hoặc bao nhiêu năm?"
            ),
            "answerable": False,
            "expected_behavior": "search_and_abstain",
            "absence_reason": (
                "Kiểm tra toàn bộ 37 trang PDF (136 chunks): cụm 'nhật ký hệ thống' chỉ xuất hiện duy nhất 1 lần tại "
                "Điều 38 khoản 2 (trang 27, chunk p027:B_para:002): 'Ghi lại và lưu trữ nhật ký hệ thống quá trình xử lý dữ liệu cá nhân.', "
                "không nêu thời hạn lưu trữ tối thiểu bao nhiêu tháng hay năm."
            ),
        },
        {
            "id": "U_T02",
            "split": "test",
            "question": (
                "Nghị định 13/2023/NĐ-CP công bố danh sách những quốc gia hoặc vùng lãnh thổ nào có mức độ bảo vệ "
                "dữ liệu cá nhân tương đương để doanh nghiệp được miễn lập Hồ sơ đánh giá tác động chuyển dữ liệu ra nước ngoài?"
            ),
            "answerable": False,
            "expected_behavior": "search_and_abstain",
            "absence_reason": (
                "Kiểm tra toàn bộ Điều 2 khoản 14, Điều 25 và Phụ lục Mẫu số 06 trên toàn văn 37 trang PDF: "
                "Nghị định yêu cầu lập Hồ sơ đánh giá tác động chuyển dữ liệu cá nhân ra nước ngoài theo Điều 25, "
                "hoàn toàn không có danh sách quốc gia/vùng lãnh thổ an toàn hoặc cơ chế miễn trừ theo quốc gia tiếp nhận "
                "(0 kết quả cho 'danh sách quốc gia', 'vùng lãnh thổ')."
            ),
        },
        {
            "id": "U_T03",
            "split": "test",
            "question": (
                "Theo Nghị định 13/2023/NĐ-CP, khi áp dụng biện pháp kỹ thuật mã hóa dữ liệu cá nhân, tổ chức bắt buộc "
                "phải dùng chuẩn thuật toán mật mã cụ thể nào (ví dụ AES-256 hay RSA-2048)?"
            ),
            "answerable": False,
            "expected_behavior": "search_and_abstain",
            "absence_reason": (
                "Kiểm tra toàn bộ 37 trang PDF (136 chunks): từ 'mã hóa' chỉ xuất hiện duy nhất 1 lần tại Điều 2 khoản 7 (trang 3) "
                "khi liệt kê các hoạt động xử lý dữ liệu cá nhân; Điều 26-28 nêu áp dụng biện pháp kỹ thuật nhưng không quy định "
                "tên chuẩn thuật toán mật mã cụ thể nào (0 kết quả cho 'AES', 'RSA', 'thuật toán')."
            ),
        },
    ]

    return new_answerable, unanswerable_cases


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare Day 7 data and frozen benchmark from Day 6.")
    parser.add_argument("--day6-root", type=Path, default=DEFAULT_DAY6_ROOT)
    parser.add_argument("--output-dir", type=Path, default=Path("data"))
    args = parser.parse_args()

    chunks_jsonl = args.day6_root / "starter" / "results" / "chunks_B_para.jsonl"
    day6_questions_path = args.day6_root / "benchmark" / "questions.json"
    day6_manifest_path = args.day6_root / "data" / "manifest.json"
    day6_ingest_path = args.day6_root / "starter" / "results" / "ingest_vectors.json"

    raw_chunks = [
        json.loads(line)
        for line in chunks_jsonl.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    chunks = [convert_day6_chunk(c) for c in raw_chunks]

    day6_q_doc = json.loads(day6_questions_path.read_text(encoding="utf-8"))
    dev_from_day6 = [
        convert_day6_dev_question(q, chunks)
        for q in day6_q_doc["questions"]
        if q.get("split") == "dev"
    ]
    new_answerable, unanswerable_cases = build_new_questions()
    questions = dev_from_day6 + new_answerable

    # Build agent_cases.json combining answerable + unanswerable in split order (dev then test)
    agent_cases: list[dict[str, Any]] = []
    for split in ("dev", "test"):
        for q in questions:
            if q["split"] == split:
                agent_cases.append(
                    {
                        "id": q["id"],
                        "split": split,
                        "question": q["question"],
                        "answerable": True,
                        "expected_behavior": "search_and_answer",
                        "kind": q["kind"],
                    }
                )
        for u in unanswerable_cases:
            if u["split"] == split:
                agent_cases.append(dict(u))

    day6_manifest = json.loads(day6_manifest_path.read_text(encoding="utf-8"))
    day6_ingest = json.loads(day6_ingest_path.read_text(encoding="utf-8"))

    manifest = {
        "day7_version": 1,
        "inherited_from_day6": {
            "source_id": SOURCE_ID,
            "chunk_strategy": "B_para",
            "chunk_count": len(chunks),
            "embedding": day6_ingest["embedding"],
            "vector_store": {
                **day6_ingest["vector_store"],
                "default_port_windows": 5433,
                "container_name": "day6_student_starter-postgres-1",
            },
            "corpus_document": day6_manifest["documents"][0],
        },
        "benchmark_summary": {
            "total_cases": len(agent_cases),
            "answerable_total": len(questions),
            "unanswerable_total": len(unanswerable_cases),
            "dev": {
                "answerable": sum(1 for q in questions if q["split"] == "dev"),
                "unanswerable": sum(1 for u in unanswerable_cases if u["split"] == "dev"),
                "total": sum(1 for c in agent_cases if c["split"] == "dev"),
            },
            "test": {
                "answerable": sum(1 for q in questions if q["split"] == "test"),
                "unanswerable": sum(1 for u in unanswerable_cases if u["split"] == "test"),
                "total": sum(1 for c in agent_cases if c["split"] == "test"),
            },
        },
    }

    benchmark_full = {
        "benchmark_version": 1,
        "frozen_before_experiments": True,
        "split_policy": {
            "dev": "17 câu có đáp án (D01-D16 kế thừa từ dev Day 6 + D17 câu multi-hop mới kết hợp Điều 28 và Điều 43) + 3 câu thiếu bằng chứng (U_D01-U_D03) = 20 câu.",
            "test": (
                "7 câu có đáp án hoàn toàn mới (N_T01-N_T07) trên các trang/điều chưa từng dùng ở cả dev lẫn test cũ của Day 6 "
                "(gồm câu hỏi thẳng, điều kiện/ngoại lệ, diễn đạt lại, câu nêu Điều 35, câu nêu Mẫu số 05 tại Phụ lục, và câu multi-hop Điều 2 + Điều 25) "
                "+ 3 câu thiếu bằng chứng hoàn toàn mới (U_T01-U_T03) = 10 câu."
            ),
            "zero_gold_leakage_rule": (
                "Agent và Retriever chỉ nhận chuỗi `question` của người dùng. Không truyền `gold`, `expected_answer`, `answerable`, "
                "`absence_reason`, hay filter gán nhãn sẵn vào agent hoặc công cụ tìm kiếm. Mọi metadata filter phải do hệ thống/agent "
                "tự suy luận từ nội dung câu hỏi của người dùng."
            ),
            "multi_hop_grading_rule": (
                "Mỗi câu có đáp án đều có `required_gold_groups` (danh sách các nhóm bằng chứng bắt buộc). "
                "Với câu 1-hop, có 1 nhóm. Với câu multi-hop hoặc cần 2 phần bằng chứng (D17, N_T06, N_T07), có 2 nhóm bắt buộc: "
                "một tập chunk (top-k hoặc context cuối) chỉ được tính là đủ bằng chứng (strict hit) khi thỏa ĐỒNG THỜI tất cả các nhóm trong `required_gold_groups`."
            ),
        },
        "selection_criteria_precommitted": [
            "Ưu tiên 1 (Chất lượng Context & Retrieval trên dev): Strict Context Evidence Rate và Strict Evidence Hit@3 cao nhất trên 17 câu dev có đáp án.",
            "Ưu tiên 2 (Chất lượng Agent trên dev): Grounded answer trung bình (/2) cao, Citation hợp lệ đạt 100%, và Abstention đúng trên 3 câu dev thiếu bằng chứng mà không từ chối nhầm câu có đáp án.",
            "Ưu tiên 3 (Hiệu quả chi phí & độ trễ): Giữa các cấu hình có chất lượng tương đương trên dev, ưu tiên cấu hình có số lượt gọi LLM/tool hợp lý, độ trễ thấp và chi phí API thấp.",
        ],
        "answerable_questions": questions,
        "unanswerable_cases": unanswerable_cases,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "chunks.json": chunks,
        "questions.json": questions,
        "agent_cases.json": agent_cases,
        "manifest.json": manifest,
        "benchmark_full.json": benchmark_full,
    }
    for filename, payload in outputs.items():
        out_path = args.output_dir / filename
        out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(manifest["benchmark_summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

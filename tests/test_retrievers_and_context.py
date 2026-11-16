import unittest

from src.context_builder import (
    expand_adjacent_siblings,
    interleave_search_batches,
    pack_context_smart,
)
from src.eval_retrieval import strict_groups_hit
from src.retrievers import (
    Bm25Retriever,
    TfidfRetriever,
    fuse_rrf,
    infer_query_filters,
    tokenize_vi,
)


class RetrieverAndContextTests(unittest.TestCase):
    def setUp(self):
        self.sample_chunks = [
            {
                "id": "doc:p1:0",
                "source_id": "doc",
                "source": "doc.pdf",
                "page": 1,
                "text": "Điều 10. Nghĩa vụ của chủ thể dữ liệu cá nhân là tự bảo vệ dữ liệu của mình.",
                "metadata": {"article": "Điều 10", "page_type": "noi_dung", "printed_page": 3},
            },
            {
                "id": "doc:p1:1",
                "source_id": "doc",
                "source": "doc.pdf",
                "page": 1,
                "text": "Khoản 2 Điều 10 quy định tôn trọng, bảo vệ dữ liệu cá nhân của người khác.",
                "metadata": {"article": "Điều 10", "page_type": "noi_dung", "printed_page": 3},
            },
            {
                "id": "doc:p2:0",
                "source_id": "doc",
                "source": "doc.pdf",
                "page": 2,
                "text": "Mẫu số 01 PHIẾU YÊU CẦU CUNG CẤP DỮ LIỆU CÁ NHÂN (Dành cho cá nhân)",
                "metadata": {"article": None, "form": "Mẫu số 01", "page_type": "phu_luc_bieu_mau", "printed_page": 4},
            },
            {
                "id": "doc:p2:1",
                "source_id": "doc",
                "source": "doc.pdf",
                "page": 2,
                "text": "Chú thích cuối trang Mẫu số 01: Ghi rõ họ tên người đại diện.",
                "metadata": {"article": None, "form": "Mẫu số 01", "page_type": "phu_luc_bieu_mau", "printed_page": 4},
            },
        ]

    def test_tokenize_vi_bigrams(self):
        toks = tokenize_vi("Dữ liệu cá nhân", use_bigrams=True)
        self.assertIn("dữ", toks)
        self.assertIn("dữ_liệu", toks)
        self.assertIn("cá_nhân", toks)

    def test_infer_query_filters_high_precision_and_no_false_exclusions(self):
        f1, _ = infer_query_filters("Mẫu số 01 tại Phụ lục của Nghị định là phiếu dành cho ai?")
        self.assertEqual(f1, {"page_type": "phu_luc_bieu_mau"})

        f2, _ = infer_query_filters("Theo Điều 35, Bộ Khoa học và Công nghệ có trách nhiệm gì?")
        self.assertEqual(f2, {"article": "Điều 35"})

        # Must NOT filter when articles are mentioned as exclusions or multiple articles appear
        f3, _ = infer_query_filters("Ngoài các trường hợp tại Điều 17 và Điều 18, xử lý dữ liệu thế nào?")
        self.assertIsNone(f3)

        f4, _ = infer_query_filters("Trừ trường hợp tại Điều 17, chủ thể có quyền đồng ý không?")
        self.assertIsNone(f4)

        f5, _ = infer_query_filters("Khi nào doanh nghiệp phải xóa dữ liệu cá nhân?")
        self.assertIsNone(f5)

    def test_bm25_and_tfidf_with_filter(self):
        bm25 = Bm25Retriever(self.sample_chunks, use_bigrams=True)
        hits = bm25.search("phiếu yêu cầu cung cấp dữ liệu", filters={"page_type": "phu_luc_bieu_mau"}, top_k=2)
        self.assertTrue(hits)
        self.assertEqual(hits[0]["id"], "doc:p2:0")
        self.assertTrue(all(h["metadata"]["page_type"] == "phu_luc_bieu_mau" for h in hits))

        tfidf = TfidfRetriever(self.sample_chunks)
        t_hits = tfidf.search("nghĩa vụ tự bảo vệ", top_k=2)
        self.assertEqual(t_hits[0]["id"], "doc:p1:0")

    def test_rrf_fusion_combines_ranks(self):
        l1 = [{"id": "a", "score": 0.9, "text": "x", "source_id": "s", "source": "s", "page": 1, "metadata": {}},
              {"id": "b", "score": 0.8, "text": "y", "source_id": "s", "source": "s", "page": 1, "metadata": {}}]
        l2 = [{"id": "b", "score": 12.0, "text": "y", "source_id": "s", "source": "s", "page": 1, "metadata": {}},
              {"id": "c", "score": 5.0, "text": "z", "source_id": "s", "source": "s", "page": 1, "metadata": {}}]
        fused = fuse_rrf([("dense", l1, 1.0), ("bm25", l2, 1.0)], rrf_k=60, top_k=3)
        # 'b' appears in both lists (rank 2 + rank 1) so it must outrank 'a' (only rank 1 in l1)
        self.assertEqual(fused[0]["id"], "b")

    def test_sibling_expansion_keeps_primary_first(self):
        primary = [self.sample_chunks[0], self.sample_chunks[2]]
        expanded = expand_adjacent_siblings(primary, self.sample_chunks)
        # Primary hits must remain in positions 0 and 1
        self.assertEqual([c["id"] for c in expanded[:2]], ["doc:p1:0", "doc:p2:0"])
        # Siblings of Điều 10 and page 2 appendix must follow
        self.assertIn("doc:p1:1", [c["id"] for c in expanded[2:]])
        self.assertIn("doc:p2:1", [c["id"] for c in expanded[2:]])

        ctx, ids = pack_context_smart(
            primary,
            budget=2000,
            corpus_chunks=self.sample_chunks,
            expand_siblings=True,
            header_style="rich",
        )
        self.assertEqual(len(ids), 4)
        self.assertIn("Công báo tr.3", ctx)

    def test_interleave_search_batches_and_strict_multihop(self):
        b1 = [{"id": "c1"}, {"id": "c2"}]
        b2 = [{"id": "c3"}, {"id": "c1"}]
        merged = interleave_search_batches([b1, b2])
        self.assertEqual([c["id"] for c in merged], ["c1", "c3", "c2"])

        q_multihop = {
            "gold": [
                {"source_id": "doc", "page": 1, "evidence": "tự bảo vệ"},
                {"source_id": "doc", "page": 2, "evidence": "PHIẾU YÊU CẦU"},
            ],
            "required_gold_groups": [
                [{"source_id": "doc", "page": 1, "evidence": "tự bảo vệ"}],
                [{"source_id": "doc", "page": 2, "evidence": "PHIẾU YÊU CẦU"}],
            ],
        }
        self.assertFalse(strict_groups_hit([self.sample_chunks[0]], q_multihop))
        self.assertTrue(strict_groups_hit([self.sample_chunks[0], self.sample_chunks[2]], q_multihop))


if __name__ == "__main__":
    unittest.main()

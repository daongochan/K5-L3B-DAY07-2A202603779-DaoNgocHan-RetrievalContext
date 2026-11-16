import json
import unittest

from src.agent_runner import run_agent_benchmark, run_agent_case
from src.retrievers import TfidfRetriever


class ScriptedLLMClient:
    model = "mock/scripted-test-only"

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def chat_with_tools(self, messages, tools, *, tool_choice="auto", temperature=0.0, max_tokens=900):
        self.calls.append({"messages": messages, "tool_choice": tool_choice})
        if not self.responses:
            raise RuntimeError("Out of scripted responses")
        nxt = self.responses.pop(0)
        if isinstance(nxt, Exception):
            raise nxt
        return {
            "message": nxt,
            "model": self.model,
            "prompt_tokens": 100,
            "completion_tokens": 30,
            "total_tokens": 130,
            "cost_usd": 0.000033,
            "elapsed_ms": 10.0,
        }


class AgentRunnerTests(unittest.TestCase):
    def setUp(self):
        self.chunks = [
            {
                "id": "nd13:p23:0",
                "source_id": "nd13",
                "source": "nd13.pdf",
                "page": 23,
                "text": "Điều 28 quy định bảo vệ dữ liệu cá nhân nhạy cảm phải chỉ định bộ phận bảo vệ dữ liệu.",
                "metadata": {"article": "Điều 28", "page_type": "noi_dung", "printed_page": 25},
            },
            {
                "id": "nd13:p29:0",
                "source_id": "nd13",
                "source": "nd13.pdf",
                "page": 29,
                "text": "Điều 43 quy định doanh nghiệp khởi nghiệp được miễn trừ chỉ định bộ phận trong 02 năm đầu.",
                "metadata": {"article": "Điều 43", "page_type": "noi_dung", "printed_page": 31},
            },
        ]
        self.retriever = TfidfRetriever(self.chunks)

    def test_multihop_search_and_lookup_then_grounded_answer(self):
        llm = ScriptedLLMClient(
            [
                {
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "tc1",
                            "type": "function",
                            "function": {
                                "name": "search_legal_corpus",
                                "arguments": json.dumps({"query": "dữ liệu nhạy cảm chỉ định bộ phận"}),
                            },
                        },
                        {
                            "id": "tc2",
                            "type": "function",
                            "function": {
                                "name": "lookup_article_or_form",
                                "arguments": json.dumps({"article": "Điều 43"}),
                            },
                        },
                    ],
                },
                {
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "tc3",
                            "type": "function",
                            "function": {
                                "name": "submit_final_response",
                                "arguments": json.dumps(
                                    {
                                        "status": "answered",
                                        "answer": "Phải chỉ định bộ phận [nd13:p23:0], nhưng khởi nghiệp được miễn 02 năm [nd13:p29:0].",
                                        "citations": ["nd13:p23:0", "nd13:p29:0"],
                                        "reasoning": "Kết hợp Điều 28 và Điều 43.",
                                    }
                                ),
                            },
                        }
                    ],
                },
            ]
        )
        res = run_agent_case(
            "Doanh nghiệp khởi nghiệp xử lý dữ liệu nhạy cảm có được miễn chỉ định bộ phận không?",
            llm_client=llm,
            retriever=self.retriever,
            corpus_chunks=self.chunks,
            budget=2000,
        )
        self.assertEqual(res["status"], "answered")
        self.assertEqual(res["search_calls"], 2)
        self.assertEqual(res["citations"], ["nd13:p23:0", "nd13:p29:0"])
        self.assertEqual(sum(1 for e in res["events"] if e["event"] == "search"), 2)

    def test_citation_retry_fixes_typo_and_inlines_valid_citation(self):
        llm = ScriptedLLMClient(
            [
                {
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "tc1",
                            "type": "function",
                            "function": {
                                "name": "search_legal_corpus",
                                "arguments": json.dumps({"query": "miễn trừ 02 năm"}),
                            },
                        }
                    ],
                },
                # First attempt cites non-existent mistyped ID nd13:p29:999 -> triggers retry
                {
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "tc2",
                            "type": "function",
                            "function": {
                                "name": "submit_final_response",
                                "arguments": json.dumps(
                                    {
                                        "status": "answered",
                                        "answer": "Được miễn 02 năm đầu [nd13:p29:999].",
                                        "citations": ["nd13:p29:999"],
                                    }
                                ),
                            },
                        }
                    ],
                },
                # Second attempt fixes ID in structured citations array; inline tag is normalized
                {
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "tc3",
                            "type": "function",
                            "function": {
                                "name": "submit_final_response",
                                "arguments": json.dumps(
                                    {
                                        "status": "answered",
                                        "answer": "Được miễn 02 năm đầu.",
                                        "citations": ["nd13:p29:0"],
                                    }
                                ),
                            },
                        }
                    ],
                },
            ]
        )
        res = run_agent_case(
            "Miễn trừ bao lâu?",
            llm_client=llm,
            retriever=self.retriever,
            corpus_chunks=self.chunks,
        )
        self.assertEqual(res["status"], "answered")
        self.assertIn("[nd13:p29:0]", res["answer"])

    def test_provider_error_is_recorded_cleanly_in_benchmark(self):
        llm = ScriptedLLMClient([RuntimeError("Simulated API outage")])
        cases = [{"id": "c1", "split": "dev", "question": "Câu hỏi lỗi?"}]
        summary, traces = run_agent_benchmark(
            config_name="err_test",
            cases=cases,
            llm_client=llm,
            retriever=self.retriever,
            corpus_chunks=self.chunks,
            agent_kwargs={"top_k": 2, "budget": 800},
        )
        self.assertEqual(summary["error_cases"], 1)
        self.assertEqual(traces[0]["status"], "error")


if __name__ == "__main__":
    unittest.main()

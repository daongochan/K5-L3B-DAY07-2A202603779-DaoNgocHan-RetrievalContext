import json
import unittest
from pathlib import Path

from src.verify_benchmark import verify_files


class Day7BenchmarkTests(unittest.TestCase):
    def test_benchmark_files_are_valid_and_isolated(self):
        root = Path(__file__).resolve().parent.parent
        summary = verify_files(root / "data", check_db=False)
        self.assertEqual(summary["status"], "OK")
        self.assertEqual(summary["chunks"], 136)
        self.assertGreaterEqual(summary["dev"]["answerable"], 15)
        self.assertGreaterEqual(summary["dev"]["unanswerable"], 2)
        self.assertGreaterEqual(summary["test"]["answerable"], 5)
        self.assertGreaterEqual(summary["test"]["unanswerable"], 2)
        self.assertEqual(summary["shared_gold_pages_between_dev_and_test"], [])

    def test_agent_cases_hide_gold_labels(self):
        root = Path(__file__).resolve().parent.parent
        cases = json.loads((root / "data" / "agent_cases.json").read_text(encoding="utf-8"))
        self.assertEqual(len(cases), 30)
        for c in cases:
            self.assertNotIn("gold", c)
            self.assertNotIn("expected_answer", c)
            self.assertNotIn("filter", c)


if __name__ == "__main__":
    unittest.main()

import unittest
import json
import os
from typing import Any

TEST_RESULT_JSONL = "test_result.jsonl"

DATAPIPELINE = {}

class MockSOp:
    def __init__(self, tokens):
        self.tokens = tokens

    def __eq__(self, other):
        return [t == other for t in self.tokens]

def make_frac_prevs(values):
    """Mock implementation of make_frac_prevs."""
    count_open = 0
    results = []
    for i, val in enumerate(values):
        if val:
            count_open += 1
        results.append(count_open / (i + 1))
    return results

class RASP:
    @staticmethod
    def numerical(sequence):
        return sequence

    @staticmethod
    def LinearSequenceMap(seq1, seq2, w1, w2):
        return [(w1 * a) - (w2 * b) for a, b in zip(seq1, seq2)]

rasp = RASP()

def make_pair_balance(sop: MockSOp, open_token: str, close_token: str) -> MockSOp:
    bools_open = rasp.numerical(sop == open_token)
    opens = rasp.numerical(make_frac_prevs(bools_open))

    bools_close = rasp.numerical(sop == close_token)
    closes = rasp.numerical(make_frac_prevs(bools_close))

    pair_balance = rasp.numerical(rasp.LinearSequenceMap(opens, closes, 1, -1))
    return pair_balance

class TestMakePairBalance(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippets = data[173]  # Get the 174th JSON element
        if len(cls.code_snippets) < 1:
            raise ValueError("Expected at least one code snippet in the 174th JSON array")

    def test_make_pair_balance(self):
        """Dynamically test make_pair_balance function."""
        passed_count = 0
        failed_count = 0
        results = []

        for i, code in enumerate(self.code_snippets):
            with self.subTest(code_index=i):
                print(f"Running test for code snippet {i}...")

                # Test inputs
                sop = MockSOp(["a", "(", ")", "b", "(", "c", ")", ")"])
                expected_result = [0, 0.5, 0, 0, 0.8, 0.6666666666666666, 0, -0.125]

                try:
                    # Execute the function
                    result = make_pair_balance(sop, "(", ")")

                    # Validate the results
                    self.assertEqual(result, expected_result, f"Code snippet {i} produced incorrect results.")
                    
                    print(f"Code snippet {i}: PASSED all assertions.\n")
                    passed_count += 1
                    results.append({
                        "function_name": "make_pair_balance",
                        "code": code,
                        "result": "passed"
                    })
                except Exception as e:
                    print(f"Code snippet {i}: FAILED with error: {e}\n")
                    failed_count += 1
                    results.append({
                        "function_name": "make_pair_balance",
                        "code": code,
                        "result": "failed"
                    })

        print(f"\nTest Summary: {passed_count} passed, {failed_count} failed, total {len(self.code_snippets)}\n")
        self.assertEqual(passed_count + failed_count, len(self.code_snippets), "Test count mismatch!")

        # Write results to test_result.jsonl
        existing_records = []
        if os.path.exists(TEST_RESULT_JSONL):
            with open(TEST_RESULT_JSONL, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    existing_records.append(json.loads(line))

        existing_records = [
            rec for rec in existing_records
            if rec.get("function_name") != "make_pair_balance"
        ]

        existing_records.extend(results)

        with open(TEST_RESULT_JSONL, "w", encoding="utf-8") as f:
            for record in existing_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        print("Results have been written to test_result.jsonl")

if __name__ == "__main__":
    unittest.main()
import unittest
import json
import os
import numpy as np
from typing import Sequence

TEST_RESULT_JSONL = "test_result.jsonl"

class TestEvalSelectorAndResult(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippets = data[147]  # Get the 148th element
        if len(cls.code_snippets) < 1:
            raise ValueError("Expected at least one code snippet in the 148th JSON array")

    def test_eval_selector_and(self):
        """Dynamically test all code snippets for 'eval_selector_and'."""
        passed_count = 0  # Counter for passed tests
        failed_count = 0  # Counter for failed tests
        results = []      # To collect test results

        for i, code in enumerate(self.code_snippets):
            with self.subTest(code_index=i):
                print(f"Running test for code snippet {i}...")

                # ------------------- Static checks -------------------
                if "def eval_selector_and" not in code:
                    print(f"Code snippet {i}: FAILED, function 'eval_selector_and' not found.\n")
                    failed_count += 1
                    # Record failure
                    results.append({
                        "function_name": "eval_selector_and",
                        "code": code,
                        "result": "failed"
                    })
                    continue

                # ------------------- Dynamic testing -------------------
                exec_globals = {'np': np, 'Sequence': Sequence}
                exec_locals = {}

                try:
                    # Dynamically execute code snippet
                    exec(code, exec_globals, exec_locals)

                    # Check if 'eval_selector_and' is actually implemented
                    if 'eval_selector_and' not in exec_locals:
                        raise RuntimeError("Function 'eval_selector_and' not found after execution.")

                    # Prepare a mock object with the needed interface for testing
                    class SelectorMock:
                        def __init__(self, fst, snd):
                            self.fst = fst
                            self.snd = snd

                    class EvalClassMock:
                        def evaluate(self, sel, xs):
                            if sel == 'fst':
                                return [True, False, True]
                            if sel == 'snd':
                                return [True, True, False]

                    # Prepare the function and mock instances
                    eval_selector_and = exec_locals['eval_selector_and']
                    mock = EvalClassMock()

                    # Test cases
                    selector = SelectorMock('fst', 'snd')
                    xs = []

                    # Evaluate and verify
                    result = eval_selector_and(mock, selector, xs)
                    expected_result = [True, False, False]
                    self.assertEqual(result, expected_result,
                                     f"Code snippet {i} evaluation gave {result}, "
                                     f"expected {expected_result}.")

                    print(f"Code snippet {i}: PASSED all assertions.\n")
                    passed_count += 1
                    results.append({
                        "function_name": "eval_selector_and",
                        "code": code,
                        "result": "passed"
                    })
                except Exception as e:
                    print(f"Code snippet {i}: FAILED with error: {e}\n")
                    failed_count += 1
                    results.append({
                        "function_name": "eval_selector_and",
                        "code": code,
                        "result": "failed"
                    })

        # Final statistics
        print(f"\nTest Summary: {passed_count} passed, {failed_count} failed, total {len(self.code_snippets)}\n")
        self.assertEqual(passed_count + failed_count, len(self.code_snippets), "Test count mismatch!")

        # ============= Write results to test_result.jsonl =============
        # Read existing test_result.jsonl (ignore if not present)
        existing_records = []
        if os.path.exists(TEST_RESULT_JSONL):
            with open(TEST_RESULT_JSONL, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    existing_records.append(json.loads(line))

        # Remove old records with function_name == "eval_selector_and"
        existing_records = [
            rec for rec in existing_records
            if rec.get("function_name") != "eval_selector_and"
        ]

        # Append new results
        existing_records.extend(results)

        # Rewrite test_result.jsonl
        with open(TEST_RESULT_JSONL, "w", encoding="utf-8") as f:
            for record in existing_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        print("Results have been written to test_result.jsonl")

if __name__ == "__main__":
    unittest.main()
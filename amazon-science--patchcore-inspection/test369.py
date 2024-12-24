import unittest
import json
import sys
import re
import os
import numpy as np
import torch

TEST_RESULT_JSONL = "test_result.jsonl"

class TestCarperAIScoreFunction(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippet = data[368]  # Get the 369th JSON element

    def test_score_function(self):
        """Dynamically test the score function."""
        passed_count = 0  # Counter for passed tests
        failed_count = 0  # Counter for failed tests
        results = []      # Collect test results to write into JSONL

        code = self.code_snippet
        # ------------------- Additional Logical Checks -------------------
        if "def score" not in code:
            print(f"Code snippet: FAILED, function 'score' not found.\n")
            failed_count += 1
            # Write failure record
            results.append({
                "function_name": "score",
                "code": code,
                "result": "failed"
            })
            return
        
        func_pattern = r"def\s+score\s*\("
        if not re.search(func_pattern, code):
            print(f"Code snippet: FAILED, incorrect signature for 'score'.\n")
            failed_count += 1
            results.append({
                "function_name": "score",
                "code": code,
                "result": "failed"
            })
            return

        # ------------------- Dynamic Execution and Testing Logic -------------------
        exec_globals = {
            'np': np,
            'torch': torch,
        }
        exec_locals = {}

        try:
            # Dynamically execute the code snippet
            exec(code, exec_globals, exec_locals)

            # Check if score function was found
            if 'score' not in exec_locals:
                print("Code snippet: FAILED, 'score' not found in exec_locals.\n")
                failed_count += 1
                results.append({
                    "function_name": "score",
                    "code": code,
                    "result": "failed"
                })
                return

            score_func = exec_locals['score']

            # Test cases
            test_cases = [
                (np.array([[1, 2], [3, 0]]), np.array([2, 3])),
                (torch.tensor([[4, 5], [6, 7]]), torch.tensor([5, 7])),
                (torch.tensor([9]), torch.tensor(9)),
                (np.array([10, 20, 30]), np.array(30))
            ]

            for i, (input_val, expected_output) in enumerate(test_cases):
                with self.subTest(test_case=i):
                    output = score_func(self=None, x=input_val)
                    if isinstance(expected_output, np.ndarray):
                        np.testing.assert_array_equal(output, expected_output)
                    else:
                        self.assertTrue(torch.equal(output, expected_output))
            
            print("Code snippet: PASSED all assertions.\n")
            passed_count += 1
            results.append({
                "function_name": "score",
                "code": code,
                "result": "passed"
            })
        
        except Exception as e:
            print(f"Code snippet: FAILED with error: {e}\n")
            failed_count += 1
            results.append({
                "function_name": "score",
                "code": code,
                "result": "failed"
            })

        # Final statistics
        print(f"\nTest Summary: {passed_count} passed, {failed_count} failed\n")

        # ============= Write test results to test_result.jsonl =============
        existing_records = []

        if os.path.exists(TEST_RESULT_JSONL):
            with open(TEST_RESULT_JSONL, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    existing_records.append(json.loads(line))

        # Remove old records for the 'score' function
        existing_records = [
            rec for rec in existing_records
            if rec.get("function_name") != "score"
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
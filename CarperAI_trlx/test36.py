import unittest
import json
import torch
import numpy as np
import os

TEST_RESULT_JSONL = "test_result.jsonl"

class TestTopKMaskResult(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippets = data[35]  # Get the 36th JSON element (index 35)
        if len(cls.code_snippets) < 1:
            raise ValueError("Expected at least one code snippet in the selected JSON array")

    def test_code_snippets(self):
        """Dynamically test all code snippets for the function 'topk_mask'."""
        passed_count = 0  # Counter for passed tests
        failed_count = 0  # Counter for failed tests
        results = []      # Collect test results to write into JSONL

        for i, code in enumerate(self.code_snippets):
            with self.subTest(code_index=i):
                print(f"Running test for code snippet {i}...")

                # Check if 'topk_mask' function is present
                if "def topk_mask" not in code:
                    print(f"Code snippet {i}: FAILED, function 'topk_mask' not found.\n")
                    failed_count += 1
                    results.append({
                        "function_name": "topk_mask",
                        "code": code,
                        "result": "failed"
                    })
                    continue

                # Execute the code snippet
                exec_globals = {
                    'torch': torch,
                    'np': np
                }
                exec_locals = {}

                try:
                    exec(code, exec_globals, exec_locals)

                    # Ensure 'topk_mask' is correctly defined
                    if 'topk_mask' not in exec_locals:
                        print(f"Code snippet {i}: FAILED, 'topk_mask' not found in exec_locals.\n")
                        failed_count += 1
                        results.append({
                            "function_name": "topk_mask",
                            "code": code,
                            "result": "failed"
                        })
                        continue

                    # Testing the function 'topk_mask'
                    topk_mask = exec_locals['topk_mask']

                    # Define test cases
                    test_cases = [
                        (torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]), 2, np.array([[-np.inf, 2.0, 3.0], [5.0, 6.0, -np.inf]])),
                        (torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]), 3, np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])),
                        (torch.tensor([[1.0, 2.0], [3.0, 4.0]]), 1, np.array([[-np.inf, 2.0], [3.0, -np.inf]]))
                    ]

                    all_passed = True
                    for xs, k, expected in test_cases:
                        result = topk_mask(xs, k)
                        np.testing.assert_array_almost_equal(
                            result.numpy(), expected,
                            err_msg=f"Failed on input {xs} with k={k}"
                        )

                    if all_passed:
                        print(f"Code snippet {i}: PASSED all test cases.\n")
                        passed_count += 1
                        results.append({
                            "function_name": "topk_mask",
                            "code": code,
                            "result": "passed"
                        })
                except Exception as e:
                    print(f"Code snippet {i}: FAILED with error: {e}\n")
                    failed_count += 1
                    results.append({
                        "function_name": "topk_mask",
                        "code": code,
                        "result": "failed"
                    })

        # Test summary
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
            if rec.get("function_name") != "topk_mask"
        ]

        existing_records.extend(results)

        with open(TEST_RESULT_JSONL, "w", encoding="utf-8") as f:
            for record in existing_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        print("Results have been written to test_result.jsonl")

if __name__ == "__main__":
    unittest.main()
import unittest
import json
import os
import torch
from typing import Any 

TEST_RESULT_JSONL = "test_result.jsonl"

class TestReplaceNoneWithZero(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippets = data[276]  # Get the 277th JSON element (index 276)
        if len(cls.code_snippets) < 1:
            raise ValueError("Expected at least one code snippet in the 277th JSON array")

    def test_code_snippets(self):
        """Dynamically test all code snippets for the 'replace_none_with_zero' function."""
        passed_count = 0  # Counter for passed tests
        failed_count = 0  # Counter for failed tests
        results = []      # Collecting results to write to JSONL

        for i, code in enumerate(self.code_snippets):
            with self.subTest(code_index=i):
                print(f"Running test for code snippet {i}...")
                
                # Static check: Check if 'replace_none_with_zero' is defined
                if "def replace_none_with_zero" not in code:
                    print(f"Code snippet {i}: FAILED, 'replace_none_with_zero' not found in code.\n")
                    failed_count += 1
                    results.append({
                        "function_name": "replace_none_with_zero",
                        "code": code,
                        "result": "failed"
                    })
                    continue

                # Dynamic execution and test logic
                exec_globals = {
                    'torch': torch,
                    'Any': Any,
                }
                exec_locals = {}

                try:
                    # Dynamically execute the code snippet
                    exec(code, exec_globals, exec_locals)

                    if 'replace_none_with_zero' not in exec_locals:
                        print(f"Code snippet {i}: FAILED, 'replace_none_with_zero' not found in exec_locals.\n")
                        failed_count += 1
                        results.append({
                            "function_name": "replace_none_with_zero",
                            "code": code,
                            "result": "failed"
                        })
                        continue

                    # Test the function
                    replace_none_with_zero = exec_locals['replace_none_with_zero']
                    tensor_list = [torch.tensor([1, 2]), None, torch.tensor([3, 4])]
                    reference = [torch.tensor([0, 0]), torch.tensor([5, 6]), torch.tensor([7, 8])]
                    expected_output = (
                        torch.tensor([1, 2]),
                        torch.tensor([0, 0]),
                        torch.tensor([3, 4])
                    )

                    # Verify the output
                    output = replace_none_with_zero(tensor_list, reference)
                    for o, e in zip(output, expected_output):
                        self.assertTrue(torch.equal(o, e), f"Output mismatch for code snippet {i}.")

                    print(f"Code snippet {i}: PASSED all assertions.\n")
                    passed_count += 1
                    results.append({
                        "function_name": "replace_none_with_zero",
                        "code": code,
                        "result": "passed"
                    })

                except Exception as e:
                    print(f"Code snippet {i}: FAILED with error: {e}\n")
                    failed_count += 1
                    results.append({
                        "function_name": "replace_none_with_zero",
                        "code": code,
                        "result": "failed"
                    })

        # Final summary
        print(f"\nTest Summary: {passed_count} passed, {failed_count} failed, total {len(self.code_snippets)}\n")
        self.assertEqual(passed_count + failed_count, len(self.code_snippets), "Test count mismatch!")

        # Write the results to test_result.jsonl
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
            if rec.get("function_name") != "replace_none_with_zero"
        ]

        existing_records.extend(results)

        with open(TEST_RESULT_JSONL, "w", encoding="utf-8") as f:
            for record in existing_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        print("Results have been written to test_result.jsonl")

if __name__ == "__main__":
    unittest.main()
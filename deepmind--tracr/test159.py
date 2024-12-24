import unittest
import json
import os
from typing import Sequence, Any

TEST_RESULT_JSONL = "test_result.jsonl"

class TestNumSopsFunction(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippets = data[158]  # Get the 159th JSON element (index 158)
        if len(cls.code_snippets) < 1:
            raise ValueError("Expected at least one code snippet in the JSON array")

    def test_code_snippets(self):
        """Dynamically test all code snippets."""
        passed_count = 0  # Counter for passed tests
        failed_count = 0  # Counter for failed tests
        results = []      # Collect test results to write to the JSONL file

        for i, code in enumerate(self.code_snippets):
            with self.subTest(code_index=i):
                print(f"Running test for code snippet {i}...")

                # Prepare dynamic execution environment
                exec_globals = {
                    'graph': {
                        'nodes': {
                            1: {'expr': 'some_expr'},
                            2: {'expr': 'SOp_expr'},
                            3: {'expr': 'another_expr'}
                        }
                    },
                    'NodeID': int,
                    'rasp': type('rasp', (object,), {'SOp': 'SOp_expr'}),
                    'nodes': type('nodes', (object,), {'EXPR': 'expr'}),
                    'Sequence': Sequence,
                    'Any': Any
                }
                exec_locals = {}

                try:
                    # Dynamically execute the code snippet
                    exec(code, exec_globals, exec_locals)

                    # Check if 'num_sops' function is defined
                    if 'num_sops' not in exec_locals:
                        print(f"Code snippet {i}: FAILED, 'num_sops' not defined.\n")
                        failed_count += 1
                        results.append({
                            "function_name": "num_sops",
                            "code": code,
                            "result": "failed"
                        })
                        continue

                    # Test the num_sops function with a mock path
                    num_sops = exec_locals['num_sops']
                    test_path = [1, 2, 3]
                    expected_result = 1  # Only the node with ID 2 matches

                    self.assertEqual(
                        num_sops(test_path),
                        expected_result,
                        f"Code snippet {i}: 'num_sops' did not return the expected result."
                    )

                    print(f"Code snippet {i}: PASSED all assertions.\n")
                    passed_count += 1
                    results.append({
                        "function_name": "num_sops",
                        "code": code,
                        "result": "passed"
                    })
                except Exception as e:
                    print(f"Code snippet {i}: FAILED with error: {e}\n")
                    failed_count += 1
                    results.append({
                        "function_name": "num_sops",
                        "code": code,
                        "result": "failed"
                    })

        # Final summary
        print(f"\nTest Summary: {passed_count} passed, {failed_count} failed, total {len(self.code_snippets)}\n")
        self.assertEqual(passed_count + failed_count, len(self.code_snippets), "Test count mismatch!")

        # Write the test results to test_result.jsonl
        existing_records = []
        if os.path.exists(TEST_RESULT_JSONL):
            with open(TEST_RESULT_JSONL, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    existing_records.append(json.loads(line))

        # Remove old records for "num_sops"
        existing_records = [
            rec for rec in existing_records
            if rec.get("function_name") != "num_sops"
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
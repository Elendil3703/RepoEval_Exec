import unittest
import json
import sys
import os
from typing import Any

TEST_RESULT_JSONL = "test_result.jsonl"

class TestBasesFunResult(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippets = data[164]  # Get the 165th JSON element (index 164)
        if len(cls.code_snippets) < 1:
            raise ValueError("Expected at least one code snippet in the JSON array")

    def test_code_snippets(self):
        """Dynamically test all code snippets in the JSON with additional checks."""
        passed_count = 0  # Counter for passed tests
        failed_count = 0  # Counter for failed tests
        results = []      # Collected test results to write to JSONL

        for i, code in enumerate(self.code_snippets):
            with self.subTest(code_index=i):
                print(f"Running test for code snippet {i}...")
                # ------------------- Static Checks -------------------
                # Check that the code snippet contains `def bases_fun`.
                if "def bases_fun" not in code:
                    print(f"Code snippet {i}: FAILED, function 'bases_fun' not found.\n")
                    failed_count += 1
                    results.append({
                        "function_name": "bases_fun",
                        "code": code,
                        "result": "failed"
                    })
                    continue

                # ------------------- Dynamic Execution and Tests -------------------
                exec_globals = {
                    'sys': sys,
                    'Any': Any,
                    'bases': type('bases', (), {'BasisDirection': lambda name, result: (name, result)}),  # Mock
                    'fun': lambda *vals: sum(vals),  # Mocked fun: sum of values
                }
                exec_locals = {}

                try:
                    # Dynamically execute the code snippet
                    exec(code, exec_globals, exec_locals)

                    # Check if bases_fun was defined
                    if 'bases_fun' not in exec_locals:
                        print(f"Code snippet {i}: FAILED, 'bases_fun' not found in exec_locals.\n")
                        failed_count += 1
                        results.append({
                            "function_name": "bases_fun",
                            "code": code,
                            "result": "failed"
                        })
                        continue

                    # Test cases for bases_fun
                    output_direction_name = 'direction'
                    args = [type('D', (object,), {'value': v}) for v in [1, 2, 3]]

                    result_with_direction = exec_locals['bases_fun'](*args, output_direction_name=output_direction_name)
                    expected_result_with_direction = ('direction', 6)

                    # Perform checks
                    self.assertEqual(
                        result_with_direction,
                        expected_result_with_direction,
                        f"Code snippet {i} did not return expected result with direction.",
                    )

                    # Test without output direction
                    result_without_direction = exec_locals['bases_fun'](*args, output_direction_name=None)
                    expected_result_without_direction = 6

                    self.assertEqual(
                        result_without_direction,
                        expected_result_without_direction,
                        f"Code snippet {i} did not return expected result without direction.",
                    )

                    print(f"Code snippet {i}: PASSED all assertions.\n")
                    passed_count += 1
                    results.append({
                        "function_name": "bases_fun",
                        "code": code,
                        "result": "passed"
                    })
                except Exception as e:
                    print(f"Code snippet {i}: FAILED with error: {e}\n")
                    failed_count += 1
                    results.append({
                        "function_name": "bases_fun",
                        "code": code,
                        "result": "failed"
                    })

        # Final summary
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

        # Remove old records with function_name == "bases_fun"
        existing_records = [
            rec for rec in existing_records
            if rec.get("function_name") != "bases_fun"
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
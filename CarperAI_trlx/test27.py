import unittest
import json
import os
import re

TEST_RESULT_JSONL = "test_result.jsonl"

class TestRangeToPattern(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippets = data[26]  # Get the 28th JSON element
        if len(cls.code_snippets) < 1:
            raise ValueError("Expected at least one code snippet in the JSON array")

    def test_range_to_pattern(self):
        """Dynamically test all range_to_pattern implementations in the JSON."""
        passed_count = 0  # Counter for passed tests
        failed_count = 0  # Counter for failed tests
        results = []      # Collect test results for JSONL output

        for i, code in enumerate(self.code_snippets):
            with self.subTest(code_index=i):
                print(f"Running test for code snippet {i}...")
                
                # Validate presence of 'range_to_pattern' in the code
                if "def range_to_pattern" not in code:
                    print(f"Code snippet {i}: FAILED, function 'range_to_pattern' not defined.\n")
                    failed_count += 1
                    results.append({
                        "function_name": "range_to_pattern",
                        "code": code,
                        "result": "failed"
                    })
                    continue

                exec_globals = {}
                exec_locals = {}

                try:
                    # Execute the code
                    exec(code, exec_globals, exec_locals)

                    # Check if 'range_to_pattern' is in the executed local scope
                    if 'range_to_pattern' not in exec_locals:
                        print(f"Code snippet {i}: FAILED, 'range_to_pattern' not found in exec_locals.\n")
                        failed_count += 1
                        results.append({
                            "function_name": "range_to_pattern",
                            "code": code,
                            "result": "failed"
                        })
                        continue

                    # Test cases for range_to_pattern
                    test_cases = [
                        ((1, 9), r"[1-9]"),
                        ((5, 5), "5"),
                        ((15, 19), r"1[5-9]"),
                        ((10, 19), r"1\d"),
                        ((100, 109), r"10\d"),
                        ((15, 25), r"[1-2][5]"),
                        ((110, 119), r"11\d"),
                        ((200, 209), r"20\d"),
                    ]

                    for (start, stop), expected in test_cases:
                        with self.subTest(start=start, stop=stop):
                            result = exec_locals['range_to_pattern'](start, stop)
                            self.assertEqual(
                                result, expected,
                                f"Code snippet {i} failed: range_to_pattern({start}, {stop}) = '{result}' != '{expected}'"
                            )

                    print(f"Code snippet {i}: PASSED all assertions.\n")
                    passed_count += 1
                    results.append({
                        "function_name": "range_to_pattern",
                        "code": code,
                        "result": "passed"
                    })
                except Exception as e:
                    print(f"Code snippet {i}: FAILED with error: {e}\n")
                    failed_count += 1
                    results.append({
                        "function_name": "range_to_pattern",
                        "code": code,
                        "result": "failed"
                    })

        # Summary
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

        # Remove old records for 'range_to_pattern'
        existing_records = [
            rec for rec in existing_records
            if rec.get("function_name") != "range_to_pattern"
        ]

        # Add new results
        existing_records.extend(results)

        # Rewrite test_result.jsonl
        with open(TEST_RESULT_JSONL, "w", encoding="utf-8") as f:
            for record in existing_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        print("Results have been written to test_result.jsonl")

if __name__ == "__main__":
    unittest.main()
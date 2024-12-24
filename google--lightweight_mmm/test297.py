import unittest
import json
import os
from typing import Any

TEST_RESULT_JSONL = "test_result.jsonl"

class TestCalculateNumberRowsPlot(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippet = data[296]  # Get the 297th JSON element (index 296)

    def test_calculate_number_rows_plot(self):
        """Test the _calculate_number_rows_plot function."""
        passed_count = 0  # Counter for passed tests
        failed_count = 0  # Counter for failed tests
        results = []      # Collect the test results to write to JSONL

        # Extract the code snippet
        code = self.code_snippet

        # Execute the code snippet dynamically
        exec_globals = {
            'Any': Any,  # Ensure Any is available
        }
        exec_locals = {}

        try:
            exec(code, exec_globals, exec_locals)

            if '_calculate_number_rows_plot' not in exec_locals:
                raise ValueError("_calculate_number_rows_plot function not found in the executed locals.")

            _calculate_number_rows_plot = exec_locals['_calculate_number_rows_plot']

            # Define test cases
            test_cases = [
                (3, 2, 3),  # 3 media channels, 2 columns => (3+1) / 2 = 2 rows + 1 = 3 total rows
                (4, 2, 3),  # 4 media channels, 2 columns => (4+1) / 2 = 2 rows + 1 = 3 total rows
                (5, 3, 3),  # 5 media channels, 3 columns => (5+1) / 3 = 2 total rows
                (6, 3, 3),  # 6 media channels, 3 columns => (6+1) / 3 = 2 total rows
                (1, 2, 2)   # 1 media channel, 2 columns => (1+1) / 2 = 1 row + 1 = 2 total rows
            ]

            # Run and check test cases
            for i, (n_media_channels, n_columns, expected) in enumerate(test_cases):
                with self.subTest(test_index=i):
                    print(f"Running test case {i}: n_media_channels={n_media_channels}, n_columns={n_columns}...")
                    result = _calculate_number_rows_plot(n_media_channels, n_columns)
                    self.assertEqual(result, expected, f"Failed test case {i}: expected {expected}, got {result}")
                    print(f"Test case {i} PASSED.\n")
                    passed_count += 1
                    results.append({
                        "function_name": "_calculate_number_rows_plot",
                        "input": (n_media_channels, n_columns),
                        "expected_output": expected,
                        "result": "passed"
                    })

        except Exception as e:
            print(f"Execution failed with error: {e}")
            failed_count += 1
            results.append({
                "function_name": "_calculate_number_rows_plot",
                "code": code,
                "result": "failed"
            })

        # Final statistics
        print(f"Test Summary: {passed_count} tests passed, {failed_count} tests failed.")

        # Write test results to test_result.jsonl
        existing_records = []
        if os.path.exists(TEST_RESULT_JSONL):
            with open(TEST_RESULT_JSONL, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    existing_records.append(json.loads(line))

        # Filter out old records
        existing_records = [
            rec for rec in existing_records
            if rec.get("function_name") != "_calculate_number_rows_plot"
        ]

        # Append new results
        existing_records.extend(results)

        # Write to test_result.jsonl
        with open(TEST_RESULT_JSONL, "w", encoding="utf-8") as f:
            for record in existing_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    unittest.main()
import unittest
import json
import sys
import os
import numpy as np

TEST_RESULT_JSONL = "test_result.jsonl"

class TestCustomForwardFunction(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippets = data[74]  # Get the 75th JSON element (index 74)
        if len(cls.code_snippets) < 1:
            raise ValueError("Expected at least one code snippet in the JSON array")

    def test_code_snippets(self):
        """Dynamically test all code snippets to ensure forward function works correctly."""
        passed_count = 0  # Counter for passed tests
        failed_count = 0  # Counter for failed tests
        results = []      # Collect test results for writing to JSONL

        for i, code in enumerate(self.code_snippets):
            with self.subTest(code_index=i):
                print(f"Running test for code snippet {i}...")
                
                # Dynamic execution context setup
                exec_globals = {
                    'np': np,
                    'self': None,  # Mocking a class instance
                    'exists': lambda value: value is not None
                }
                exec_locals = {}

                try:
                    # Define a mock class with necessary methods
                    class MockClass:
                        def groupnorm(self, x):
                            return (x - np.mean(x)) / (np.std(x) + 1e-5)

                        def activation(self, x):
                            return np.maximum(x, 0)  # ReLU

                        def project(self, x):
                            return x * 2  # Example projection

                    exec_globals['self'] = MockClass()

                    # Execute the code snippet
                    exec(code, exec_globals, exec_locals)

                    # Validating the existence of forward function
                    if 'forward' not in exec_locals:
                        print(f"Code snippet {i}: FAILED, 'forward' function not found.\n")
                        failed_count += 1
                        results.append({
                            "function_name": "forward",
                            "code": code,
                            "result": "failed"
                        })
                        continue

                    forward = exec_locals['forward']

                    # Test 1: With no scale and shift, verify basic transformation
                    input_data = np.array([1.0, 2.0, 3.0])
                    result = forward(self=exec_globals['self'], x=input_data)
                    expected_result = exec_globals['self'].project(
                        exec_globals['self'].activation(
                            exec_globals['self'].groupnorm(input_data)
                        )
                    )
                    np.testing.assert_almost_equal(result, expected_result, decimal=5)

                    # Test 2: With scale and shift
                    scale_shift = (np.array([0.1, 0.2, 0.3]), np.array([0.1, 0.2, 0.3]))
                    result_with_scale_shift = forward(self=exec_globals['self'], x=input_data, scale_shift=scale_shift)
                    expected_result_with_scale_shift = exec_globals['self'].project(
                        exec_globals['self'].activation(
                            (exec_globals['self'].groupnorm(input_data) * (scale_shift[0] + 1)) + scale_shift[1]
                        )
                    )
                    np.testing.assert_almost_equal(result_with_scale_shift, expected_result_with_scale_shift, decimal=5)

                    print(f"Code snippet {i}: PASSED all assertions.\n")
                    passed_count += 1
                    results.append({
                        "function_name": "forward",
                        "code": code,
                        "result": "passed"
                    })
                except Exception as e:
                    print(f"Code snippet {i}: FAILED with error: {e}\n")
                    failed_count += 1
                    results.append({
                        "function_name": "forward",
                        "code": code,
                        "result": "failed"
                    })

        # Final Test Summary
        print(f"\nTest Summary: {passed_count} passed, {failed_count} failed, total {len(self.code_snippets)}\n")
        self.assertEqual(passed_count + failed_count, len(self.code_snippets), "Test count mismatch!")

        # Write results to JSONL
        existing_records = []
        if os.path.exists(TEST_RESULT_JSONL):
            with open(TEST_RESULT_JSONL, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    existing_records.append(json.loads(line))

        # Remove old records for forward
        existing_records = [
            rec for rec in existing_records
            if rec.get("function_name") != "forward"
        ]

        # Append new results
        existing_records.extend(results)

        # Rewrite the test_result.jsonl
        with open(TEST_RESULT_JSONL, "w", encoding="utf-8") as f:
            for record in existing_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        print("Results have been written to test_result.jsonl")

if __name__ == "__main__":
    unittest.main()
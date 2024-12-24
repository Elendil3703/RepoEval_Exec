import unittest
import json
import os
from typing import Any  # Ensure Any is available in the injected environment

TEST_RESULT_JSONL = "test_result.jsonl"

class TestMockModelFunction(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippets = data[313]  # Get the 314th JSON element
        if len(cls.code_snippets) < 1:
            raise ValueError("Expected at least one code snippet in the 314th JSON array")

    def test_code_snippets(self):
        """Dynamically test all code snippets in the JSON with additional checks."""
        passed_count = 0  # Counter for passed tests
        failed_count = 0  # Counter for failed tests
        results = []      # Collect test results to write to JSONL

        for i, code in enumerate(self.code_snippets):
            with self.subTest(code_index=i):
                print(f"Running test for code snippet {i}...")
                
                # ------------------- Static Check -------------------
                if "numpyro.deterministic" not in code:
                    print(f"Code snippet {i}: FAILED, 'numpyro.deterministic' not found in code.\n")
                    failed_count += 1
                    # Write failure record
                    results.append({
                        "function_name": "mock_model_function",
                        "code": code,
                        "result": "failed"
                    })
                    continue
                
                if '"intercept_values"' not in code:
                    print(f"Code snippet {i}: FAILED, key 'intercept_values' not found in code.\n")
                    failed_count += 1
                    # Write failure record
                    results.append({
                        "function_name": "mock_model_function",
                        "code": code,
                        "result": "failed"
                    })
                    continue

                if "intercept.simple_intercept" not in code:
                    print(f"Code snippet {i}: FAILED, 'intercept.simple_intercept' not found.\n")
                    failed_count += 1
                    # Write failure record
                    results.append({
                        "function_name": "mock_model_function",
                        "code": code,
                        "result": "failed"
                    })
                    continue
                
                # ------------------- Dynamic Execution and Logic Test -------------------
                exec_globals = {
                    'numpyro': __import__('numpyro'),
                    'intercept': __import__('unittest.mock', fromlist=['simple_intercept']),  # Mocking intercept
                    'Any': Any,  # Inject Any
                }
                exec_locals = {}

                try:
                    intercept_mock = exec_globals['intercept']
                    intercept_mock.simple_intercept = unittest.mock.Mock(return_value="mocked_value")
                    exec_globals['intercept'] = intercept_mock

                    # Dynamically execute code snippet
                    exec(code, exec_globals, exec_locals)

                    # Check mock_model_function exists
                    if 'mock_model_function' not in exec_locals:
                        print(f"Code snippet {i}: FAILED, 'mock_model_function' not found in exec_locals.\n")
                        failed_count += 1
                        results.append({
                            "function_name": "mock_model_function",
                            "code": code,
                            "result": "failed"
                        })
                        continue

                    # Use the function to ensure correct behavior
                    exec_locals['mock_model_function']("test_data")

                    intercept_mock.simple_intercept.assert_called_once_with(
                        data="test_data", custom_priors={}
                    )

                    print(f"Code snippet {i}: PASSED all assertions.\n")
                    passed_count += 1
                    results.append({
                        "function_name": "mock_model_function",
                        "code": code,
                        "result": "passed"
                    })
                except Exception as e:
                    print(f"Code snippet {i}: FAILED with error: {e}\n")
                    failed_count += 1
                    results.append({
                        "function_name": "mock_model_function",
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

        # Remove old records with function_name == "mock_model_function"
        existing_records = [
            rec for rec in existing_records
            if rec.get("function_name") != "mock_model_function"
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
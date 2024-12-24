import unittest
import json
import sys
import os
import numpyro
from unittest.mock import MagicMock

TEST_RESULT_JSONL = "test_result.jsonl"

class TestMockModelFunction(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippet = data[321]  # Get the 322nd JSON element (0-indexed)

    def test_mock_model_function(self):
        """Test the mock_model_function for correct behavior."""
        passed_count = 0
        failed_count = 0
        results = []  # Collect results to write to JSONL

        code = self.code_snippet

        # Setup global and local environments for executing the snippet
        exec_globals = {
            'numpyro': MagicMock(),
        }
        exec_locals = {}

        try:
            # Dynamically execute the code snippet
            exec(code, exec_globals, exec_locals)

            # Check if mock_model_function is defined
            if 'mock_model_function' not in exec_locals:
                print("mock_model_function not found in executed code.")
                failed_count += 1
                results.append({
                    "function_name": "mock_model_function",
                    "code": code,
                    "result": "failed"
                })
                return

            # Get the mock_model_function
            mock_model_fn = exec_locals['mock_model_function']

            # Prepare mock input data
            mock_data = MagicMock()
            number_lags = MagicMock()

            # Call the function
            mock_model_fn(mock_data, number_lags)

            # Verify that numpyro.deterministic was called correctly
            numpyro_mock = exec_globals['numpyro']
            numpyro_mock.deterministic.assert_called_with(
                "carryover",
                numpyro_mock.lagging.carryover(
                    data=mock_data, custom_priors={}, number_lags=number_lags
                )
            )

            print("mock_model_function: PASSED all assertions.")
            passed_count += 1
            results.append({
                "function_name": "mock_model_function",
                "code": code,
                "result": "passed"
            })
        except Exception as e:
            print(f"mock_model_function: FAILED with error: {e}")
            failed_count += 1
            results.append({
                "function_name": "mock_model_function",
                "code": code,
                "result": "failed"
            })

        print(f"\nTest Summary: {passed_count} passed, {failed_count} failed\n")
        self.assertEqual(passed_count + failed_count, 1, "Test count mismatch!")

        # Write test results to test_result.jsonl
        existing_records = []
        if os.path.exists(TEST_RESULT_JSONL):
            with open(TEST_RESULT_JSONL, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    existing_records.append(json.loads(line))

        # Remove old records for "mock_model_function"
        existing_records = [
            rec for rec in existing_records
            if rec.get("function_name") != "mock_model_function"
        ]

        # Append the new results
        existing_records.extend(results)

        # Rewrite test_result.jsonl
        with open(TEST_RESULT_JSONL, "w", encoding="utf-8") as f:
            for record in existing_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        print("Results have been written to test_result.jsonl")

if __name__ == "__main__":
    unittest.main()
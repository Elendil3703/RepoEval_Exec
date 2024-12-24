import unittest
import json
import os
import numpy as np
from typing import Optional
from unittest.mock import MagicMock
import jax.numpy as jnp

TEST_RESULT_JSONL = "test_result.jsonl"

class TestMakeSinglePrediction(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippets = data[295]  # Get the 296th JSON element
        if len(cls.code_snippets) < 1:
            raise ValueError("Expected at least one code snippet in the 296th JSON array")

    def test_make_single_prediction(self):
        """Dynamically test all code snippets in the JSON with additional checks."""
        passed_count = 0  # Counter for passed tests
        failed_count = 0  # Counter for failed tests
        results = []      # To collect test results to write into JSONL

        for i, code in enumerate(self.code_snippets):
            with self.subTest(code_index=i):
                print(f"Running test for code snippet {i}...")
                # Inject the _make_single_prediction function into the test environment
                exec_globals = {
                    'jnp': jnp,
                    'Optional': Optional,
                    'MagicMock': MagicMock,
                }
                exec_locals = {}

                try:
                    # Execute the code snippet dynamically
                    exec(code, exec_globals, exec_locals)

                    # Check if _make_single_prediction is defined
                    if '_make_single_prediction' not in exec_locals:
                        print(f"Code snippet {i}: FAILED, '_make_single_prediction' not found in exec_locals.\n")
                        failed_count += 1
                        results.append({
                            "function_name": "_make_single_prediction",
                            "code": code,
                            "result": "failed"
                        })
                        continue

                    # Use mock objects to test _make_single_prediction logic
                    mock_media = jnp.array([1.0, 2.0, 3.0])
                    extra_features = jnp.array([[0.1, 0.2, 0.3]])
                    seed = 42
                    
                    mock_media_mix_model = MagicMock()
                    mock_predicted_values = jnp.array([[10.0, 20.0, 30.0]])
                    mock_media_mix_model.predict.return_value = mock_predicted_values

                    result = exec_locals['_make_single_prediction'](
                        media_mix_model=mock_media_mix_model,
                        mock_media=mock_media,
                        extra_features=extra_features,
                        seed=seed
                    )

                    # Assert the correctness of the result
                    expected_result = mock_predicted_values.mean(axis=0)
                    np.testing.assert_array_almost_equal(
                        result, 
                        expected_result, 
                        err_msg=f"Code snippet {i} did not produce the expected prediction results."
                    )

                    print(f"Code snippet {i}: PASSED all assertions.\n")
                    passed_count += 1
                    results.append({
                        "function_name": "_make_single_prediction",
                        "code": code,
                        "result": "passed"
                    })
                except Exception as e:
                    print(f"Code snippet {i}: FAILED with error: {e}\n")
                    failed_count += 1
                    results.append({
                        "function_name": "_make_single_prediction",
                        "code": code,
                        "result": "failed"
                    })

        # Final statistics
        print(f"\nTest Summary: {passed_count} passed, {failed_count} failed, total {len(self.code_snippets)}\n")
        self.assertEqual(passed_count + failed_count, len(self.code_snippets), "Test count mismatch!")

        # Write test results to test_result.jsonl
        existing_records = []
        if os.path.exists(TEST_RESULT_JSONL):
            with open(TEST_RESULT_JSONL, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    existing_records.append(json.loads(line))

        # Remove old records for _make_single_prediction
        existing_records = [
            rec for rec in existing_records
            if rec.get("function_name") != "_make_single_prediction"
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
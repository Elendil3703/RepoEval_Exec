import unittest
import json
import os
import sys
from unittest.mock import MagicMock
from typing import Any

TEST_RESULT_JSONL = "test_result.jsonl"

class NotFittedModelError(Exception):
    pass

# Placeholder for the actual module
class lightweight_mmm:
    class LightweightMMM:
        pass
    NotFittedModelError = NotFittedModelError

# Placeholder for actual jnp module
import numpy as np
jnp = np

class TestCalculateMediaContribution(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippet = data[297]  # Get the 298th element (zero-indexed)

    def test_calculate_media_contribution(self):
        """Dynamically test code snippets from the JSON with additional checks."""
        passed_count = 0
        failed_count = 0
        results = []

        code = self.code_snippet
        func_name = "_calculate_media_contribution"

        # ------------------- Dynamic Execution and Testing -------------------
        exec_globals = {
            'lightweight_mmm': lightweight_mmm,
            'jnp': jnp,
            'Any': Any,
        }
        exec_locals = {}

        try:
            # Execute the code segment
            exec(code, exec_globals, exec_locals)

            if func_name not in exec_locals:
                raise ValueError(f"Function {func_name} was not found after execution.")

            # Define a test case
            mock_model = MagicMock()
            mock_model.trace = {
                "media_transformed": jnp.random.rand(10, 5, 3),
                "coef_media": jnp.random.rand(10, 3),
            }

            calc_media_contribution = exec_locals[func_name]

            # Calculate contribution
            media_contribution = calc_media_contribution(mock_model)

            # Verify the output is a numpy array
            self.assertIsInstance(media_contribution, jnp.ndarray, "Output is not a jnp.ndarray.")
            print("Output is a jnp.ndarray")

            # Verify shape consistency based on einsum pattern logic
            expected_shape = (10, 5, 3)  # (s, t, c)
            self.assertEqual(media_contribution.shape, expected_shape, "Unexpected output shape.")
            print("Output shape is correct")

            # Expected behavior when model.trace["media_transformed"].ndim > 3
            mock_model.trace["media_transformed"] = jnp.random.rand(10, 5, 3, 4)
            media_contribution = calc_media_contribution(mock_model)
            self.assertEqual(media_contribution.ndim, 3, "Expected output ndim after geo aggregation.")

            # Simulate unfitted model to raise error
            delattr(mock_model, "trace")
            with self.assertRaises(lightweight_mmm.NotFittedModelError):
                calc_media_contribution(mock_model)
            print("Correctly raised NotFittedModelError for unfitted model")
            
            passed_count += 1
            results.append({
                "function_name": func_name,
                "code": code,
                "result": "passed"
            })
        except Exception as e:
            print(f"Failed with error: {e}")
            failed_count += 1
            results.append({
                "function_name": func_name,
                "code": code,
                "result": "failed"
            })

        # Assert test results are as expected
        self.assertEqual(passed_count + failed_count, 1, "Test count mismatch!")
        
        # Write results to test_result.jsonl
        existing_records = []
        if os.path.exists(TEST_RESULT_JSONL):
            with open(TEST_RESULT_JSONL, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    existing_records.append(json.loads(line))

        # Remove old records for the function
        existing_records = [rec for rec in existing_records if rec.get("function_name") != func_name]
        
        # Append new results
        existing_records.extend(results)
        
        # Rewrite test_result.jsonl
        with open(TEST_RESULT_JSONL, "w", encoding="utf-8") as f:
            for record in existing_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        print("Results have been written to test_result.jsonl")

if __name__ == "__main__":
    unittest.main()
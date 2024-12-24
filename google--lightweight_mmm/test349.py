import unittest
import json
import os
import numpy as np
import jax.numpy as jnp

TEST_RESULT_JSONL = "test_result.jsonl"

class TestCalculateSeasonality(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_to_test = data[348]  # Get the 349th JSON element (index 348)

        # Inject the necessary imports and symbols
        exec_globals = {
            'jnp': jnp,
        }
        exec_locals = {}

        # Execute the provided code snippet to define `calculate_seasonality`
        exec(cls.code_to_test, exec_globals, exec_locals)
        
        # Store the reference to calculate_seasonality for testing
        cls.calculate_seasonality = exec_locals.get('calculate_seasonality')

        if cls.calculate_seasonality is None:
            raise ValueError("Function 'calculate_seasonality' could not be found in the provided code.")

    def test_seasonality_calculation(self):
        """Test specific scenarios for calculate_seasonality function."""
        results = []  # Collect results for JSONL output

        # Define test cases
        test_cases = [
            {
                "args": {
                    "number_periods": 5,
                    "degrees": 2,
                    "frequency": 4,
                    "gamma_seasonality": jnp.array([1.0, 0.5]),
                },
                "expected_shape": (5,),  # Expected output shape
            },
            {
                "args": {
                    "number_periods": 10,
                    "degrees": 3,
                    "frequency": 12,
                    "gamma_seasonality": jnp.array([0.5, 0.2, 0.1]),
                },
                "expected_shape": (10,),
            },
        ]

        for i, case in enumerate(test_cases):
            with self.subTest(test_index=i):
                try:
                    output = self.calculate_seasonality(**case["args"])
                    self.assertEqual(output.shape, case["expected_shape"],
                        f"Test case {i} failed: expected shape {case['expected_shape']}, got {output.shape}")
                    results.append({
                        "function_name": "calculate_seasonality",
                        "args": case["args"],
                        "result": "passed"
                    })
                    print(f"Test case {i}: PASSED")
                except Exception as e:
                    results.append({
                        "function_name": "calculate_seasonality",
                        "args": case["args"],
                        "result": "failed",
                        "error": str(e)
                    })
                    print(f"Test case {i}: FAILED with error {e}")

        # Write results to test_result.jsonl
        self.write_results(results)

    @staticmethod
    def write_results(results):
        """Write the test results to test_result.jsonl"""
        # Read existing records if any
        existing_records = []
        if os.path.exists(TEST_RESULT_JSONL):
            with open(TEST_RESULT_JSONL, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    existing_records.append(json.loads(line))

        # Remove old records for `calculate_seasonality`
        existing_records = [
            rec for rec in existing_records
            if rec.get("function_name") != "calculate_seasonality"
        ]

        # Append new results
        existing_records.extend(results)

        # Write back all results
        with open(TEST_RESULT_JSONL, "w", encoding="utf-8") as f:
            for record in existing_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    unittest.main()
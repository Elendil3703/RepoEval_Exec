import unittest
import json
import os
import numpy as np
from jax import numpy as jnp

TEST_RESULT_JSONL = "test_result.jsonl"

class TestSinusoidalSeasonality(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file and select the specific code snippet
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippet = data[308]  # Get the JSON element at index 308

    def test_sinusoidal_seasonality(self):
        """Test the sinusoidal_seasonality function logic."""
        results = []
        exec_globals = {
            'np': np,
            'jnp': jnp
        }
        exec_locals = {}

        code = self.code_snippet

        try:
            # Dynamically execute the code snippet
            exec(code, exec_globals, exec_locals)

            # Ensure the required function is defined
            self.assertIn(
                '_sinusoidal_seasonality',
                exec_locals,
                "Function '_sinusoidal_seasonality' not found in the code."
            )

            # Extract function to test
            sinusoidal_seasonality = exec_locals['_sinusoidal_seasonality']
            
            # ----- Add custom test cases here -----
            # Customizable test inputs
            data = np.array([[1], [2], [3]])
            frequency = 52
            degrees_seasonality = 3
            gamma_seasonality = jnp.array([0.1, 0.2, 0.3])

            # Expected shape based on logic
            expected_shape = (data.shape[0], degrees_seasonality)

            # Call the function with inputs
            result = sinusoidal_seasonality(
                seasonality_arange=jnp.expand_dims(a=jnp.arange(data.shape[0]), axis=-1),
                degrees_arange=jnp.arange(degrees_seasonality),
                frequency=frequency,
                gamma_seasonality=gamma_seasonality
            )

            # Check the resulting shape
            self.assertEqual(
                result.shape, expected_shape,
                f"Expected shape {expected_shape}, but got {result.shape}."
            )

            results.append({
                "function_name": "sinusoidal_seasonality",
                "code": code,
                "result": "passed"
            })

        except Exception as e:
            results.append({
                "function_name": "sinusoidal_seasonality",
                "code": code,
                "result": f"failed with error {e}"
            })

        # Write results to test_result.jsonl
        self._write_test_results(results)

    def _write_test_results(self, results):
        # Read existing records if the file exists
        existing_records = []
        if os.path.exists(TEST_RESULT_JSONL):
            with open(TEST_RESULT_JSONL, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    existing_records.append(json.loads(line))

        # Remove old records for the tested function
        existing_records = [
            rec for rec in existing_records
            if rec.get("function_name") != "sinusoidal_seasonality"
        ]

        # Append new results
        existing_records.extend(results)

        # Write back to the file
        with open(TEST_RESULT_JSONL, "w", encoding="utf-8") as f:
            for record in existing_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    unittest.main()
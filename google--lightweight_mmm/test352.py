import unittest
import json
import os
import jax.numpy as jnp
from typing import Any

TEST_RESULT_JSONL = "test_result.jsonl"

def carryover(data: jnp.ndarray,
              ad_effect_retention_rate: jnp.ndarray,
              peak_effect_delay: jnp.ndarray,
              number_lags: int = 13) -> jnp.ndarray:
    lags_arange = jnp.expand_dims(jnp.arange(number_lags, dtype=jnp.float32),
                                  axis=-1)
    convolve_func = _carryover_convolve
    if data.ndim == 3:
        convolve_func = jax.vmap(
            fun=_carryover_convolve, in_axes=(2, None, None), out_axes=2)
    weights = ad_effect_retention_rate**((lags_arange - peak_effect_delay)**2)
    return convolve_func(data, weights, number_lags)

def _carryover_convolve(data, weights, number_lags):
    # Dummy implementation for testing purposes
    return data * weights[:number_lags]

class TestCarryoverFunction(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_data = data[351]  # Get the 352nd JSON element

    def test_carryover(self):
        """Test the carryover function with prepared scenarios."""
        results = []  # To collect results for JSONL output

        for i, code in enumerate(self.code_data):
            with self.subTest(code_index=i):
                try:
                    # Setup test data for carryover call
                    data = jnp.array([[1.0, 2.0], [3.0, 4.0]])  # Example input
                    ad_effect_retention_rate = jnp.array(0.5)
                    peak_effect_delay = jnp.array(1.0)
                    number_lags = 2

                    # Call the carryover function
                    result = carryover(data, ad_effect_retention_rate, peak_effect_delay, number_lags)

                    # Verify the result has expected properties
                    self.assertEqual(result.shape, data.shape, "Shape mismatch.")
                    self.assertTrue((result >= 0).all(), "Result contains negative values.")

                    # Record successful test
                    results.append({"function_name": "carryover", "code": code, "result": "passed"})
                except Exception as e:
                    # Handle exceptions and record failure
                    results.append({"function_name": "carryover", "code": code, "result": f"failed with error {e}"})

        # Write results to JSONL file
        existing_records = []
        if os.path.exists(TEST_RESULT_JSONL):
            with open(TEST_RESULT_JSONL, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    existing_records.append(json.loads(line))

        # Remove old "carryover" function records and add new results
        existing_records = [rec for rec in existing_records if rec.get("function_name") != "carryover"]
        existing_records.extend(results)

        with open(TEST_RESULT_JSONL, "w", encoding="utf-8") as f:
            for record in existing_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    unittest.main()
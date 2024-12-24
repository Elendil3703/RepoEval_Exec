import unittest
import json
import os
import jax.numpy as jnp
from typing import Any

TEST_RESULT_JSONL = "test_result.jsonl"

def _hill(
    data: jnp.ndarray,
    half_max_effective_concentration: jnp.ndarray,
    slope: jnp.ndarray,
) -> jnp.ndarray:
    """Calculates the hill function for a given array of values."""
    save_transform = jnp.power(data / half_max_effective_concentration, -slope)
    return jnp.where(save_transform == 0, x=0, y=1. / (1 + save_transform))

class TestHillFunction(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippet = data[314]  # Get the 315th JSON element

    def test_hill_function(self):
        """Tests the _hill function with various inputs."""
        results = []

        # Define test cases
        test_cases = [
            {
                "data": jnp.array([2.0, 3.0, 10.0]),
                "half_max_effective_concentration": jnp.array([5.0]),
                "slope": jnp.array([1.0]),
                "expected": jnp.array([0.2857143, 0.375, 0.6666667]),
            },
            {
                "data": jnp.array([0.0, 1.0, 5.0]),
                "half_max_effective_concentration": jnp.array([2.0]),
                "slope": jnp.array([2.0]),
                "expected": jnp.array([0.0, 0.2, 0.8]),
            }
        ]

        for i, case in enumerate(test_cases):
            with self.subTest(test_case_index=i):
                try:
                    result = _hill(case['data'], case['half_max_effective_concentration'], case['slope'])
                    jnp.testing.assert_allclose(result, case['expected'], rtol=1e-6)

                    results.append({
                        "function_name": "_hill",
                        "code": self.code_snippet,
                        "result": "passed"
                    })
                except Exception as e:
                    results.append({
                        "function_name": "_hill",
                        "code": self.code_snippet,
                        "result": "failed",
                        "error": str(e)
                    })

        # Write results to test_result.jsonl
        existing_records = []
        if os.path.exists(TEST_RESULT_JSONL):
            with open(TEST_RESULT_JSONL, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    existing_records.append(json.loads(line))

        # Remove old records for _hill
        existing_records = [
            rec for rec in existing_records
            if rec.get("function_name") != "_hill"
        ]

        # Append new results
        existing_records.extend(results)

        # Rewrite test_result.jsonl
        with open(TEST_RESULT_JSONL, "w", encoding="utf-8") as f:
            for record in existing_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    unittest.main()
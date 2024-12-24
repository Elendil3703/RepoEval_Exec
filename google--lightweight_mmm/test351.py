import unittest
import json
import os
import jax.numpy as jnp
from typing import Any

TEST_RESULT_JSONL = "test_result.jsonl"

class TestHillFunction(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file and get the 351st code snippet
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippet = data[350]  # Get the 351st JSON element
        if not cls.code_snippet:
            raise ValueError("Expected a code snippet for index 350")

    def test_hill_function(self):
        """Test the hill function calculation logic."""
        results = []  # Collect results to write to JSONL

        # Prepare the execution environment
        exec_globals = {
            'jnp': jnp,
        }
        exec_locals = {}

        code = self.code_snippet

        # Execute the code snippet
        try:
            exec(code, exec_globals, exec_locals)

            # Check if 'hill' function is defined
            if 'hill' not in exec_locals:
                self.fail("Function 'hill' is not defined in the executed code.")

            # Accessing the hill function
            hill_func = exec_locals['hill']
            
            # Sample test cases
            test_cases = [
                # Format: (data, half_max_effective_concentration, slope, expected)
                (jnp.array([1.0, 10.0, 50.0]), jnp.array([10.0, 10.0, 10.0]), jnp.array([1.0, 2.0, 1.0]), 
                 jnp.array([0.5, 0.83333, 0.95238])),
                (jnp.array([0.1, 0.5, 2.0]), jnp.array([1.0, 1.0, 1.0]), jnp.array([1.0, 1.0, 1.0]), 
                 jnp.array([0.09091, 0.33333, 0.66667])),
            ]
            
            for i, (data, ec50, slope, expected) in enumerate(test_cases):
                with self.subTest(test_case=i):
                    result = hill_func(data, ec50, slope)
                    # Compare rounded results for numerical precision
                    self.assertTrue(jnp.allclose(result, expected, atol=1e-5), f"Failed test case {i}")
                    
            # Log the passing result
            results.append({
                "function_name": "hill",
                "code": code,
                "result": "passed"
            })
        except Exception as e:
            # Log the failing result
            results.append({
                "function_name": "hill",
                "code": code,
                "result": "failed",
                "error": str(e)
            })
            self.fail(f"Caught exception while executing hill function: {e}")

        # ============= Write test results to test_result.jsonl =============
        existing_records = []
        if os.path.exists(TEST_RESULT_JSONL):
            with open(TEST_RESULT_JSONL, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    existing_records.append(json.loads(line))

        # Remove old records related to the hill function
        existing_records = [
            rec for rec in existing_records
            if rec.get("function_name") != "hill"
        ]

        # Append new results
        existing_records.extend(results)

        # Rewrite test_result.jsonl
        with open(TEST_RESULT_JSONL, "w", encoding="utf-8") as f:
            for record in existing_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    unittest.main()
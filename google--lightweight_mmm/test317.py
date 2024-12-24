import unittest
import json
import os
import numpyro
import jax.numpy as jnp
from typing import Any

TEST_RESULT_JSONL = "test_result.jsonl"

class TestExponentFunction(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippets = data[316]  # Get the 317th JSON element (index 316)
        if len(cls.code_snippets) < 1:
            raise ValueError("Expected at least one code snippet in the JSON element")

    def test_exponent_function(self):
        passed_count = 0
        failed_count = 0
        results = []

        for i, code in enumerate(self.code_snippets):
            with self.subTest(code_index=i):
                print(f"Running test for code snippet {i}...")
                
                # Ensure that `numpyro` and `jnp` are used within the snippet
                if "numpyro" not in code or "jnp" not in code:
                    print(f"Code snippet {i}: FAILED, 'numpyro' or 'jnp' not found in code.\n")
                    failed_count += 1
                    results.append({
                        "function_name": "exponent",
                        "code": code,
                        "result": "failed"
                    })
                    continue

                # Prepare the mock environment for execution
                exec_globals = {
                    'numpyro': numpyro,
                    'jnp': jnp,
                    'Any': Any,
                }
                exec_locals = {}

                try:
                    # Execute the code snippet to check for errors
                    exec(code, exec_globals, exec_locals)

                    # Validate the expected logic for `_exponent` transformation
                    if '_exponent' not in exec_locals:
                        print(f"Code snippet {i}: FAILED, '_exponent' function not defined in exec_locals.\n")
                        failed_count += 1
                        results.append({
                            "function_name": "exponent",
                            "code": code,
                            "result": "failed"
                        })
                        continue

                    # Create mock data and priors to simulate the function execution
                    mock_data = jnp.ones((2, 3, 4))
                    mock_priors = {'get_default_priors': lambda: {'EXPONENT': lambda: jnp.array([1.0, 1.5, 2.0])}}
                    mock_custom_priors = {'EXPONENT': lambda: jnp.array([1.2, 1.7, 2.2])}
                    mock_prefix = "test_"

                    # Call the function and verify it behaves as expected
                    result = exec_locals['_exponent'](data=mock_data, priors=mock_priors, custom_priors=mock_custom_priors, prefix=mock_prefix)

                    # Validate the dimensions or values based on your needs.
                    self.assertEqual(result.shape[-1], mock_data.shape[-1], f"Code snippet {i} did not transform data correctly.")
                    
                    print(f"Code snippet {i}: PASSED all assertions.\n")
                    passed_count += 1
                    results.append({
                        "function_name": "exponent",
                        "code": code,
                        "result": "passed"
                    })
                except Exception as e:
                    print(f"Code snippet {i}: FAILED with error: {e}\n")
                    failed_count += 1
                    results.append({
                        "function_name": "exponent",
                        "code": code,
                        "result": "failed"
                    })

        print(f"\nTest Summary: {passed_count} passed, {failed_count} failed, total {len(self.code_snippets)}\n")

        # Ensure all snippets were tested
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

        # Remove old records for "exponent"
        existing_records = [
            rec for rec in existing_records
            if rec.get("function_name") != "exponent"
        ]

        # Append new results
        existing_records.extend(results)

        # Write updated records to the file
        with open(TEST_RESULT_JSONL, "w", encoding="utf-8") as f:
            for record in existing_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        print("Results have been written to test_result.jsonl")

if __name__ == "__main__":
    unittest.main()
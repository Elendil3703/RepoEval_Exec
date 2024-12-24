import unittest
import json
import os
import jax.numpy as jnp
from typing import Any  # Make sure the execution environment includes Any

TEST_RESULT_JSONL = "test_result.jsonl"

class TestZeroMLPsFunction(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippet = data[249]  # Get the 250th JSON element (index 249)
        if len(cls.code_snippet) < 1:
            raise ValueError("Expected at least one code snippet for index 249")

    def test_zero_mlps(self):
        """Test the _zero_mlps function."""
        passed_count = 0  # Counter for passed tests
        failed_count = 0  # Counter for failed tests
        results = []      # Collect results for writing to JSONL

        code = self.code_snippet

        # ------------------- Static check -------------------
        if "_zero_mlps" not in code:
            print("Code snippet 249: FAILED, '_zero_mlps' not found in code.\n")
            failed_count += 1
            # Write failure record
            results.append({
                "function_name": "_zero_mlps",
                "code": code,
                "result": "failed"
            })
            return

        func_pattern = r"def\s+_zero_mlps\s*\("
        if not re.search(func_pattern, code):
            print("Code snippet 249: FAILED, incorrect signature for '_zero_mlps'.\n")
            failed_count += 1
            # Write failure record
            results.append({
                "function_name": "_zero_mlps",
                "code": code,
                "result": "failed"
            })
            return

        # ------------------- Dynamic execution and testing -------------------
        exec_globals = {
            'jnp': jnp,
            'Any': Any,  # Inject Any
        }
        exec_locals = {}

        try:
            # Dynamically execute the code snippet
            exec(code, exec_globals, exec_locals)

            # Check if _zero_mlps is present
            if '_zero_mlps' not in exec_locals:
                print("Code snippet 249: FAILED, '_zero_mlps' not found in exec_locals.\n")
                failed_count += 1
                results.append({
                    "function_name": "_zero_mlps",
                    "code": code,
                    "result": "failed"
                })
                return

            zero_mlps_func = exec_locals['_zero_mlps']

            # Create test cases
            test_params = {
                "layer1": {
                    "weights": jnp.array([1.0, 2.0]),
                    "bias": jnp.array([0.5])
                },
                "mlp_layer": {
                    "weights": jnp.array([0.8, 1.5]),
                    "bias": jnp.array([0.3])
                }
            }

            expected_params = {
                "layer1": {
                    "weights": jnp.array([1.0, 2.0]),
                    "bias": jnp.array([0.5])
                },
                "mlp_layer": {
                    "weights": jnp.zeros_like(jnp.array([0.8, 1.5])),
                    "bias": jnp.zeros_like(jnp.array([0.3]))
                }
            }

            # Run the function and test the result
            result = zero_mlps_func(None, test_params)

            for layer in expected_params:
                for param in expected_params[layer]:
                    self.assertTrue(
                        jnp.array_equal(result[layer][param], expected_params[layer][param]),
                        f"Parameter {param} in {layer} did not match expected zeroed values."
                    )

            print("Code snippet 249: PASSED all assertions.\n")
            passed_count += 1
            results.append({
                "function_name": "_zero_mlps",
                "code": code,
                "result": "passed"
            })
        except Exception as e:
            print(f"Code snippet 249: FAILED with error: {e}\n")
            failed_count += 1
            results.append({
                "function_name": "_zero_mlps",
                "code": code,
                "result": "failed"
            })

        # Final statistics
        print(f"\nTest Summary: {passed_count} passed, {failed_count} failed, total 1\n")
        self.assertEqual(passed_count + failed_count, 1, "Test count mismatch!")

        # ============= Write test results to test_result.jsonl =============
        # Read existing test_result.jsonl (ignore if it doesn't exist)
        existing_records = []
        if os.path.exists(TEST_RESULT_JSONL):
            with open(TEST_RESULT_JSONL, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    existing_records.append(json.loads(line))

        # Remove old records with function_name == "_zero_mlps"
        existing_records = [
            rec for rec in existing_records
            if rec.get("function_name") != "_zero_mlps"
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
import unittest
import json
import os
import jax.numpy as jnp
from typing import Any

TEST_RESULT_JSONL = "test_result.jsonl"

class TestUnembedFunction(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file and get the test code
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippet = data[151]  # Get the 152nd JSON element
        
        if len(cls.code_snippet) < 1:
            raise ValueError("Expected at least one code snippet in the JSON array")

    def test_unembed_function(self):
        """Test the unembed function with various scenarios and inputs."""
        code = self.code_snippet

        passed_count = 0  # Counter for passed tests
        failed_count = 0  # Counter for failed tests
        results = []      # Collect results to write to JSONL

        # ------------------- Dynamically execute and test the logic -------------------
        exec_globals = {
            'jnp': jnp,
            'res_to_out': None,
        }
        exec_locals = {}

        try:
            # Dynamically execute the code snippet to bring unembed into scope
            exec(code, exec_globals, exec_locals)

            if 'unembed' not in exec_locals:
                print(f"Test failed: 'unembed' function not found in exec_locals.\n")
                failed_count += 1
                results.append({
                    "function_name": "unembed",
                    "code": code,
                    "result": "failed",
                    "reason": "'unembed' function not found"
                })
            else:
                unembed = exec_locals['unembed']

                # Test assuming res_to_out.matrix is well-defined and we pass correct arguments
                class MockResToOut:
                    def __init__(self, matrix):
                        self.matrix = matrix

                res_to_out = MockResToOut(jnp.array([[1, 0], [0, 1]]))
                exec_globals['res_to_out'] = res_to_out

                # Prepare a sample x and test unembed
                x = jnp.array([[0.5, 0.2], [0.8, 0.9]])
                
                # Case 1: using unembed with use_unembed_argmax=True
                result = unembed(x, use_unembed_argmax=True)
                expected = jnp.array([0, 1])
                try:
                    self.assertTrue(jnp.array_equal(result, expected))
                    print(f"Test case 1: PASSED.\n")
                    results.append({
                        "function_name": "unembed",
                        "code": code,
                        "result": "passed"
                    })
                    passed_count += 1
                except AssertionError:
                    print(f"Test case 1: FAILED.\n")
                    results.append({
                        "function_name": "unembed",
                        "code": code,
                        "result": "failed",
                        "reason": "AssertionError in use_unembed_argmax=True"
                    })
                    failed_count += 1

                # Additional tests can be added here
                # ...

        except Exception as e:
            print(f"Dynamic execution failed with error: {e}\n")
            failed_count += 1
            results.append({
                "function_name": "unembed",
                "code": code,
                "result": "failed",
                "reason": str(e)
            })

        # Final statistics
        print(f"\nTest Summary: {passed_count} passed, {failed_count} failed, total tests {passed_count + failed_count}\n")

        # ============= Write test results to test_result.jsonl =============
        existing_records = []
        if os.path.exists(TEST_RESULT_JSONL):
            with open(TEST_RESULT_JSONL, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    existing_records.append(json.loads(line))

        # Remove old records with function_name == "unembed"
        existing_records = [
            rec for rec in existing_records
            if rec.get("function_name") != "unembed"
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
import unittest
import json
import os
import jax.numpy as jnp
import lightweight_mmm
from typing import Any

TEST_RESULT_JSONL = "test_result.jsonl"

class TestSetUpClassResult(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippets = data[352]  # Get the 353rd JSON element
        if len(cls.code_snippets) < 1:
            raise ValueError("Expected at least one code snippet in the JSON array")

    def test_set_up_class(self):
        """Dynamically test all code snippets in the JSON with additional checks."""
        passed_count = 0  # Counter for passed tests
        failed_count = 0  # Counter for failed tests
        results = []      # Collect test results to write to JSONL

        for i, code in enumerate(self.code_snippets):
            with self.subTest(code_index=i):
                print(f"Running test for code snippet {i}...")
                # ------------------- Logic checks -------------------
                # Static Check: Verify `lightweight_mmm.LightweightMMM()` is present in the code
                if "lightweight_mmm.LightweightMMM()" not in code:
                    print(f"Code snippet {i}: FAILED, 'lightweight_mmm.LightweightMMM()' not found in code.\n")
                    failed_count += 1
                    # Write failure record
                    results.append({
                        "function_name": "setUpClass",
                        "code": code,
                        "result": "failed"
                    })
                    continue
                
                # ------------------- Dynamic Execution and Testing Logic -------------------
                exec_globals = {
                    'jnp': jnp,
                    'lightweight_mmm': lightweight_mmm,
                }
                exec_locals = {}

                try:
                    # Execute the code snippet
                    exec(code, exec_globals, exec_locals)

                    # Check if setUpClass executed correctly and objects are created
                    test_instance = exec_locals.get('LightweightMmmTest')()

                    # Access the class-level attributes to ensure set up works
                    self.assertIsNotNone(test_instance.national_mmm, 
                                         f"Code snippet {i} did not set 'national_mmm'.")
                    self.assertIsNotNone(test_instance.geo_mmm, 
                                         f"Code snippet {i} did not set 'geo_mmm'.")

                    print(f"Code snippet {i}: PASSED all checks.\n")
                    passed_count += 1
                    results.append({
                        "function_name": "setUpClass",
                        "code": code,
                        "result": "passed"
                    })
                except Exception as e:
                    print(f"Code snippet {i}: FAILED with error: {e}\n")
                    failed_count += 1
                    results.append({
                        "function_name": "setUpClass",
                        "code": code,
                        "result": "failed"
                    })

        # Test Summary
        print(f"\nTest Summary: {passed_count} passed, {failed_count} failed, total {len(self.code_snippets)}\n")
        self.assertEqual(passed_count + failed_count, len(self.code_snippets), "Test count mismatch!")

        # ============= Write the results to test_result.jsonl =============
        existing_records = []
        if os.path.exists(TEST_RESULT_JSONL):
            with open(TEST_RESULT_JSONL, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    existing_records.append(json.loads(line))

        # Remove old records with function_name == "setUpClass"
        existing_records = [
            rec for rec in existing_records
            if rec.get("function_name") != "setUpClass"
        ]

        # Extend with new results
        existing_records.extend(results)

        # Rewrite test_result.jsonl
        with open(TEST_RESULT_JSONL, "w", encoding="utf-8") as f:
            for record in existing_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        print("Results have been written to test_result.jsonl")

if __name__ == "__main__":
    unittest.main()
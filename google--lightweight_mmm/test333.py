import unittest
import json
import jax.numpy as jnp
import os

TEST_RESULT_JSONL = "test_result.jsonl"

class TestFitFunction(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file and fetch the 333rd group of code snippets
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippets = data[332]  # Get the 333rd group

        if len(cls.code_snippets) < 1:
            raise ValueError("Expected at least one code snippet in the 333rd JSON array")

    def test_fit_function(self):
        """Test the fit function behavior."""
        passed_count = 0
        failed_count = 0
        results = []

        for i, code in enumerate(self.code_snippets):
            with self.subTest(code_index=i):
                print(f"Running test for code snippet {i}...")

                # Prepare the execution environment
                exec_globals = {
                    'jnp': jnp,
                }
                exec_locals = {}

                try:
                    # Dynamically execute the code snippet
                    exec(code, exec_globals, exec_locals)

                    # Check if 'fit' function exists in locals
                    if 'fit' not in exec_locals:
                        print(f"Code snippet {i}: FAILED, 'fit' not found in exec_locals.\n")
                        failed_count += 1
                        results.append({
                            "function_name": "fit",
                            "code": code,
                            "result": "failed"
                        })
                        continue

                    # Instantiate a mock class with the fit method
                    class MockTransformer:
                        divide_by = 1.0
                        multiply_by = 1.0

                    transformer_instance = MockTransformer()
                    transformer_instance.fit = exec_locals['fit'].__get__(transformer_instance, MockTransformer)

                    # Providing sample data for testing
                    data = jnp.array([[1.0, 2.0], [3.0, 4.0]])

                    # Trigger the fit function
                    transformer_instance.fit(data)

                    # Verify if the transformations seem correctly initialized
                    self.assertIsNotNone(transformer_instance.divide_by, f"Code snippet {i} did not set divide_by.")
                    self.assertIsNotNone(transformer_instance.multiply_by, f"Code snippet {i} did not set multiply_by.")

                    print(f"Code snippet {i}: PASSED all assertions.\n")
                    passed_count += 1
                    results.append({
                        "function_name": "fit",
                        "code": code,
                        "result": "passed"
                    })
                except Exception as e:
                    print(f"Code snippet {i}: FAILED with error: {e}\n")
                    failed_count += 1
                    results.append({
                        "function_name": "fit",
                        "code": code,
                        "result": "failed"
                    })

        # Test summary
        print(f"\nTest Summary: {passed_count} passed, {failed_count} failed, total {len(self.code_snippets)}\n")
        self.assertEqual(passed_count + failed_count, len(self.code_snippets), "Test count mismatch!")

        # Append results to test_result.jsonl
        existing_records = []
        if os.path.exists(TEST_RESULT_JSONL):
            with open(TEST_RESULT_JSONL, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    existing_records.append(json.loads(line))

        # Remove old records for 'fit'
        existing_records = [
            rec for rec in existing_records
            if rec.get("function_name") != "fit"
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
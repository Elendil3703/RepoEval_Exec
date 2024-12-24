import unittest
import json
import os

TEST_RESULT_JSONL = "test_result.jsonl"

class TestDoValidation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippets = data[283]  # Get the 284th JSON element
        if len(cls.code_snippets) < 1:
            raise ValueError("Expected at least one code snippet in the 284th JSON array")

    def test_do_validation(self):
        """Dynamically test all code snippets in the JSON with additional checks."""
        passed_count = 0  # Counter for passed tests
        failed_count = 0  # Counter for failed tests
        results = []      # Collect results to write to JSONL

        for i, code in enumerate(self.code_snippets):
            with self.subTest(code_index=i):
                print(f"Running test for code snippet {i}...")

                # Static checks
                if "def do_validation" not in code:
                    print(f"Code snippet {i}: FAILED, function 'do_validation' not found.\n")
                    failed_count += 1
                    results.append({
                        "function_name": "do_validation",
                        "code": code,
                        "result": "failed"
                    })
                    continue

                # Dynamic execution and logic tests
                exec_globals = {}
                exec_locals = {}

                try:
                    # Execute the code snippet
                    exec(code, exec_globals, exec_locals)

                    # Define a test class with the required methods
                    class TestClass:
                        def is_implemented(self, feature):
                            return feature == "validation"

                        def is_rank_zero(self):
                            return True

                    # Test do_validation function
                    test_instance = exec_locals.get("TestClass", TestClass)()
                    do_validation_method = getattr(test_instance, "do_validation", None)

                    if not callable(do_validation_method):
                        print(f"Code snippet {i}: FAILED, 'do_validation' not callable.\n")
                        failed_count += 1
                        results.append({
                            "function_name": "do_validation",
                            "code": code,
                            "result": "failed"
                        })
                        continue

                    self.assertTrue(
                        do_validation_method(),
                        f"Code snippet {i} did not return True as expected."
                    )

                    print(f"Code snippet {i}: PASSED all assertions.\n")
                    passed_count += 1
                    results.append({
                        "function_name": "do_validation",
                        "code": code,
                        "result": "passed"
                    })

                except Exception as e:
                    print(f"Code snippet {i}: FAILED with error: {e}\n")
                    failed_count += 1
                    results.append({
                        "function_name": "do_validation",
                        "code": code,
                        "result": "failed"
                    })

        # Summary Information
        print(f"\nTest Summary: {passed_count} passed, {failed_count} failed, total {len(self.code_snippets)}\n")
        self.assertEqual(passed_count + failed_count, len(self.code_snippets), "Test count mismatch!")

        # Write results to test_result.jsonl
        existing_records = []
        if os.path.exists(TEST_RESULT_JSONL):
            with open(TEST_RESULT_JSONL, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    existing_records.append(json.loads(line))

        # Remove old records with the same function name
        existing_records = [
            rec for rec in existing_records
            if rec.get("function_name") != "do_validation"
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
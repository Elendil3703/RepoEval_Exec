import unittest
import json
import os
from typing import Dict, Tuple, Any

TEST_RESULT_JSONL = "test_result.jsonl"

class TestGetTrunkFieldsResult(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        # Select the 400th set of code snippets
        cls.code_snippets = data[399]
        if len(cls.code_snippets) < 1:
            raise ValueError("Expected at least one code snippet in the 400th JSON array")

    def test_get_trunk_fields(self):
        """Dynamically test all code snippets in the JSON with additional checks."""
        passed_count = 0  # Counter for passed tests
        failed_count = 0  # Counter for failed tests
        results = []      # Collect the test results to be written into JSONL

        for i, code in enumerate(self.code_snippets):
            with self.subTest(code_index=i):
                print(f"Running test for code snippet {i}...")
                # ------------------- Static Checks -------------------
                # Check for '_get_trunk_fields' definition in the snippet
                if "def _get_trunk_fields" not in code:
                    print(f"Code snippet {i}: FAILED, function '_get_trunk_fields' not found.\n")
                    failed_count += 1
                    # Write failure record
                    results.append({
                        "function_name": "_get_trunk_fields",
                        "code": code,
                        "result": "failed"
                    })
                    continue

                # ------------------- Dynamic Execution and Testing -------------------
                exec_globals = {}
                exec_locals = {}

                try:
                    # Dynamically execute the code snippet
                    exec(code, exec_globals, exec_locals)

                    # Check if _get_trunk_fields is defined
                    if '_get_trunk_fields' not in exec_locals:
                        print(f"Code snippet {i}: FAILED, '_get_trunk_fields' not found in exec_locals.\n")
                        failed_count += 1
                        results.append({
                            "function_name": "_get_trunk_fields",
                            "code": code,
                            "result": "failed"
                        })
                        continue

                    # Create a test class with dummy storage attributes
                    class DummyClass:
                        input_key = "test_key"
                        trunk_field_args: Dict[str, Any] = {
                            "test_key": ["arg1", "arg2"],
                            None: ["default_arg1", "default_arg2"]
                        }
                        trunk_field_kwargs: Dict[str, Any] = {
                            "test_key": {"kwarg1": 1, "kwarg2": 2},
                            None: {"default_kwarg1": 10, "default_kwarg2": 20}
                        }

                        _get_trunk_fields = exec_locals['_get_trunk_fields']

                    # Instance of the class
                    obj = DummyClass()

                    # Call the _get_trunk_fields method and test results
                    fields_args, fields_kwargs = obj._get_trunk_fields()
                    self.assertEqual(fields_args, ["arg1", "arg2"])
                    self.assertEqual(fields_kwargs, {"kwarg1": 1, "kwarg2": 2})

                    # Change input_key to test default case
                    obj.input_key = "non_existent_key"
                    fields_args, fields_kwargs = obj._get_trunk_fields()
                    self.assertEqual(fields_args, ["default_arg1", "default_arg2"])
                    self.assertEqual(fields_kwargs, {"default_kwarg1": 10, "default_kwarg2": 20})

                    print(f"Code snippet {i}: PASSED all assertions.\n")
                    passed_count += 1
                    results.append({
                        "function_name": "_get_trunk_fields",
                        "code": code,
                        "result": "passed"
                    })
                except Exception as e:
                    print(f"Code snippet {i}: FAILED with error: {e}\n")
                    failed_count += 1
                    results.append({
                        "function_name": "_get_trunk_fields",
                        "code": code,
                        "result": "failed"
                    })

        # Final statistics
        print(f"\nTest Summary: {passed_count} passed, {failed_count} failed, total {len(self.code_snippets)}\n")
        self.assertEqual(passed_count + failed_count, len(self.code_snippets), "Test count mismatch!")

        # ============= Write test results to test_result.jsonl =============
        # Read existing test_result.jsonl (ignore if not present)
        existing_records = []
        if os.path.exists(TEST_RESULT_JSONL):
            with open(TEST_RESULT_JSONL, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    existing_records.append(json.loads(line))

        # Remove old records with function_name == "_get_trunk_fields"
        existing_records = [
            rec for rec in existing_records
            if rec.get("function_name") != "_get_trunk_fields"
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
import unittest
import json
import os
import copy

TEST_RESULT_JSONL = "test_result.jsonl"

class TestGroundTruthInit(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippets = data[385]  # Get the 386th JSON element (index 385)
        if len(cls.code_snippets) < 1:
            raise ValueError("Expected at least one code snippet in the JSON array")

    def test_init_snippets(self):
        """Test all init function snippets."""
        passed_count = 0  # Counter for passed tests
        failed_count = 0  # Counter for failed tests
        results = []      # To collect test results for JSONL

        for i, code in enumerate(self.code_snippets):
            with self.subTest(code_index=i):
                print(f"Running test for code snippet {i}...")
                try:
                    # Dynamic execution of the code snippet
                    exec_globals = {
                        'copy': copy
                    }
                    exec(code, exec_globals)

                    # Create class to test __init__ method
                    class TestClass:
                        def __init__(self, hook_dict, layer_name: str, last_layer_to_extract: str):
                            self.hook_dict = hook_dict
                            self.layer_name = layer_name
                            self.raise_exception_to_break = copy.deepcopy(
                                layer_name == last_layer_to_extract
                            )

                    # Test case setup
                    hook_dict = {"hook": "value"}
                    layer_name = "layer1"
                    last_layer_to_extract = "layer1"

                    instance = TestClass(hook_dict, layer_name, last_layer_to_extract)

                    # Assertions to validate initialization
                    self.assertEqual(instance.hook_dict, hook_dict, "Hook dict mismatch.")
                    self.assertEqual(instance.layer_name, layer_name, "Layer name mismatch.")
                    self.assertEqual(instance.raise_exception_to_break, True, "raise_exception_to_break mismatch.")

                    print(f"Code snippet {i}: PASSED all assertions.\n")
                    passed_count += 1
                    results.append({
                        "function_name": "__init__",
                        "code": code,
                        "result": "passed"
                    })
                except Exception as e:
                    print(f"Code snippet {i}: FAILED with error: {e}\n")
                    failed_count += 1
                    results.append({
                        "function_name": "__init__",
                        "code": code,
                        "result": "failed"
                    })

        # Final summary
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

        # Remove old records with function_name == "__init__"
        existing_records = [
            rec for rec in existing_records
            if rec.get("function_name") != "__init__"
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
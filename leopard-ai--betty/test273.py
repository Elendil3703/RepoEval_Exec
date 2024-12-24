import unittest
import json
import os
from typing import Any

TEST_RESULT_JSONL = "test_result.jsonl"

class TestInitMethod(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippets = data[272]  # Get the 273rd JSON element
        if len(cls.code_snippets) < 1:
            raise ValueError("Expected at least one code snippet in this JSON array")

    def test_code_snippets(self):
        """Dynamically test all code snippets for the '__init__' method."""
        passed_count = 0  # Counter for passed tests
        failed_count = 0  # Counter for failed tests
        results = []      # To store test results and write to JSONL

        for i, code in enumerate(self.code_snippets):
            with self.subTest(code_index=i):
                print(f"Running test for code snippet {i}...")
                
                # ------------------- Static Checks -------------------
                # Check if '__init__' is in the snippet
                if "def __init__(" not in code:
                    print(f"Code snippet {i}: FAILED, '__init__' method not found in code.\n")
                    failed_count += 1
                    results.append({
                        "function_name": "__init__",
                        "code": code,
                        "result": "failed"
                    })
                    continue

                # ------------------- Dynamic Execution and Testing Logic -------------------
                exec_globals = {
                    'Any': Any,  # Ensure Any is defined
                }
                exec_locals = {}

                try:
                    # Dynamically execute the code snippet
                    exec(code, exec_globals, exec_locals)

                    # Define a dummy configuration class and data loader for testing
                    class DummyConfig:
                        pass

                    class DummyDataLoader:
                        pass

                    # Instantiate the class to test the '__init__' constructor
                    if '__init__' in exec_globals:
                        test_instance = exec_locals['__init__'](
                            "test_name",
                            DummyConfig(),
                            module=None,
                            optimizer=None,
                            scheduler=None,
                            train_data_loader=DummyDataLoader(),
                            extra_config=None
                        )

                        # Verify the instance has expected attributes (state dict caches)
                        self.assertTrue(
                            hasattr(test_instance, 'module_state_dict_cache'),
                            f"Code snippet {i}: 'module_state_dict_cache' attribute not found."
                        )
                        self.assertTrue(
                            hasattr(test_instance, 'opitmizer_state_dict_cache'),
                            f"Code snippet {i}: 'opitmizer_state_dict_cache' attribute not found."
                        )

                        print(f"Code snippet {i}: PASSED all assertions.\n")
                        passed_count += 1
                        results.append({
                            "function_name": "__init__",
                            "code": code,
                            "result": "passed"
                        })
                    else:
                        print(f"Code snippet {i}: FAILED, '__init__' not defined properly.\n")
                        failed_count += 1
                        results.append({
                            "function_name": "__init__",
                            "code": code,
                            "result": "failed"
                        })
                except Exception as e:
                    print(f"Code snippet {i}: FAILED with error: {e}\n")
                    failed_count += 1
                    results.append({
                        "function_name": "__init__",
                        "code": code,
                        "result": "failed"
                    })

        # Summary
        print(f"\nTest Summary: {passed_count} passed, {failed_count} failed, total {len(self.code_snippets)}\n")
        self.assertEqual(passed_count + failed_count, len(self.code_snippets), "Test count mismatch!")

        # ============= Write results to test_result.jsonl =============
        # Read existing test_result.jsonl if it exists
        existing_records = []
        if os.path.exists(TEST_RESULT_JSONL):
            with open(TEST_RESULT_JSONL, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    existing_records.append(json.loads(line))

        # Remove old records for function_name == "__init__"
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
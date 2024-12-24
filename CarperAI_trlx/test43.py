import unittest
import json
import sys
import os
import inspect
from typing import Optional, Any
import transformers

TEST_RESULT_JSONL = "test_result.jsonl"

class TestInitFunction(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippet = data[42]  # Get the 43rd JSON element (index 42)
        if not cls.code_snippet:
            raise ValueError("Code snippet for __init__ function is missing in JSON array")

    def test_init_function(self):
        """Test __init__ functionality with enhanced checks."""
        passed_count = 0
        failed_count = 0
        results = []

        code = self.code_snippet

        with self.subTest():
            try:
                print("Running test for __init__ function snippet...")

                # Inject necessary globals
                exec_globals = {
                    'sys': sys,
                    'transformers': transformers,
                    'Optional': Optional,
                    'Any': Any,
                    'inspect': inspect,
                }
                exec_locals = {}

                # Dynamically execute code snippet
                exec(code, exec_globals, exec_locals)
                
                # Check if __init__ correctly initializes with a base model
                class TestModel(transformers.PreTrainedModel):
                    def __init__(self, config):
                        super().__init__(config)

                    def forward(self):
                        pass

                model_instance = TestModel(transformers.PretrainedConfig())
                test_init_instance = exec_locals["__init__"](base_model=model_instance)

                # Check if base_model is set
                self.assertIsNotNone(
                    test_init_instance.base_model,
                    "Base model should not be None."
                )

                # Check if forward_kwargs are initialized as expected
                expected_args = inspect.getfullargspec(model_instance.forward).args
                self.assertEqual(
                    test_init_instance.forward_kwargs,
                    expected_args,
                    "Forward_kwargs should match the forward args of the base_model."
                )

                print("__init__ function: PASSED all assertions.\n")
                passed_count += 1
                results.append({
                    "function_name": "__init__",
                    "code": code,
                    "result": "passed"
                })
            except Exception as e:
                print(f"__init__ function: FAILED with error: {e}\n")
                failed_count += 1
                results.append({
                    "function_name": "__init__",
                    "code": code,
                    "result": "failed"
                })

        # Test summary
        print(f"\nTest Summary: {passed_count} passed, {failed_count} failed, total 1\n")
        
        # Ensure test count consistency
        self.assertEqual(passed_count + failed_count, 1, "Test count mismatch!")

        # Write the test results to test_result.jsonl
        existing_records = []
        if os.path.exists(TEST_RESULT_JSONL):
            with open(TEST_RESULT_JSONL, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    existing_records.append(json.loads(line))

        # Remove old records for "__init__"
        existing_records = [
            rec for rec in existing_records
            if rec.get("function_name") != "__init__"
        ]

        # Append new results
        existing_records.extend(results)

        # Rewrite the test_result.jsonl
        with open(TEST_RESULT_JSONL, "w", encoding="utf-8") as f:
            for record in existing_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        print("Results have been written to test_result.jsonl")

if __name__ == "__main__":
    unittest.main()
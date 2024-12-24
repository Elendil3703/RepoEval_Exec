import unittest
import json
import sys
import re
import os
from unittest.mock import MagicMock

TEST_RESULT_JSONL = "test_result.jsonl"

class TestParseConfigFunction(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippets = data[277]  # Get the 278th JSON element
        if len(cls.code_snippets) < 1:
            raise ValueError("Expected at least one code snippet in the JSON element")

    def test_parse_config_snippets(self):
        """Dynamically test all code snippets in the JSON with additional checks."""
        passed_count = 0  # Counter for passed tests
        failed_count = 0  # Counter for failed tests
        results = []      # Collect the test results to write into JSONL

        for i, code in enumerate(self.code_snippets):
            with self.subTest(code_index=i):
                print(f"Running test for code snippet {i}...")
                
                if "def parse_config" not in code:
                    print(f"Code snippet {i}: FAILED, function 'parse_config' not found.\n")
                    failed_count += 1
                    results.append({
                        "function_name": "parse_config",
                        "code": code,
                        "result": "failed"
                    })
                    continue

                exec_globals = {
                    'EarlyStopping': MagicMock(),
                }
                exec_locals = {}

                try:
                    # Dynamically execute code snippet
                    exec(code, exec_globals, exec_locals)

                    # Check if parse_config exists
                    if 'parse_config' not in exec_locals:
                        print(f"Code snippet {i}: FAILED, 'parse_config' not found in exec_locals.\n")
                        failed_count += 1
                        results.append({
                            "function_name": "parse_config",
                            "code": code,
                            "result": "failed"
                        })
                        continue

                    # Create a mock config and class to enhance parse_config
                    class ConfigMock:
                        def __init__(self):
                            self.train_iters = 10
                            self.valid_step = 5
                            self.logger_type = "default"
                            self.roll_back = True
                            self.strategy = "default_strategy"
                            self.backend = "default_backend"
                            self.early_stopping = True
                            self.early_stopping_metric = "accuracy"
                            self.early_stopping_mode = "max"
                            self.early_stopping_tolerance = 0.01

                    class HasConfig:
                        def __init__(self):
                            self.config = ConfigMock()
                    
                        parse_config = exec_locals['parse_config']

                    # Instantiate and run the parse_config
                    instance = HasConfig()
                    instance.parse_config(instance)

                    # Assertions
                    self.assertEqual(instance.train_iters, 10)
                    self.assertEqual(instance.valid_step, 5)
                    self.assertEqual(instance.logger_type, "default")
                    self.assertEqual(instance._roll_back, True)
                    self.assertEqual(instance._strategy, "default_strategy")
                    self.assertEqual(instance._backend, "default_backend")
                    exec_globals['EarlyStopping'].assert_called_once_with(
                        metric="accuracy", mode="max", tolerance=0.01
                    )

                    print(f"Code snippet {i}: PASSED all assertions.\n")
                    passed_count += 1
                    results.append({
                        "function_name": "parse_config",
                        "code": code,
                        "result": "passed"
                    })
                except Exception as e:
                    print(f"Code snippet {i}: FAILED with error: {e}\n")
                    failed_count += 1
                    results.append({
                        "function_name": "parse_config",
                        "code": code,
                        "result": "failed"
                    })

        print(f"\nTest Summary: {passed_count} passed, {failed_count} failed, total {len(self.code_snippets)}\n")
        self.assertEqual(passed_count + failed_count, len(self.code_snippets), "Test count mismatch!")

        existing_records = []
        if os.path.exists(TEST_RESULT_JSONL):
            with open(TEST_RESULT_JSONL, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    existing_records.append(json.loads(line))

        existing_records = [
            rec for rec in existing_records
            if rec.get("function_name") != "parse_config"
        ]

        existing_records.extend(results)

        with open(TEST_RESULT_JSONL, "w", encoding="utf-8") as f:
            for record in existing_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    unittest.main()
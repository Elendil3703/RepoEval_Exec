import unittest
import json
import os
import re
from typing import Any

TEST_RESULT_JSONL = "test_result.jsonl"

class TestLayerNaming(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file and get the 249th element (index 248)
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippets = data[248]
        if len(cls.code_snippets) < 1:
            raise ValueError("Expected at least one code snippet in the selected JSON array")

    def test_check_layer_naming(self):
        """Dynamically test all code snippets in the JSON related to `_check_layer_naming`."""
        passed_count = 0  # Counter for passed tests
        failed_count = 0  # Counter for failed tests
        results = []      # Store the test results to write into JSONL

        for i, code in enumerate(self.code_snippets):
            with self.subTest(code_index=i):
                print(f"Running test for code snippet {i}...")
                
                if "_check_layer_naming" not in code:
                    print(f"Code snippet {i}: FAILED, '_check_layer_naming' not found in code.\n")
                    failed_count += 1
                    results.append({
                        "function_name": "_check_layer_naming",
                        "code": code,
                        "result": "failed"
                    })
                    continue

                func_pattern = r"def\s+_check_layer_naming\s*\("
                if not re.search(func_pattern, code):
                    print(f"Code snippet {i}: FAILED, incorrect signature for '_check_layer_naming'.\n")
                    failed_count += 1
                    results.append({
                        "function_name": "_check_layer_naming",
                        "code": code,
                        "result": "failed"
                    })
                    continue
                
                exec_globals = {
                    'Any': Any
                }
                exec_locals = {}

                try:
                    # Simulate a testing environment for _check_layer_naming
                    class DummyTest(unittest.TestCase):
                        def setUp(self):
                            self.passed_keys = []
                        
                        def assertEqual(self, a, b, msg=None):
                            if a != b:
                                raise AssertionError(msg or f"{a} != {b}")

                        def assertStartsWith(self, string, prefix, msg=None):
                            if not string.startswith(prefix):
                                raise AssertionError(msg or f"{string} does not start with {prefix}")

                        def assertIn(self, member, container, msg=None):
                            if member not in container:
                                raise AssertionError(msg or f"{member} not found in {container}")

                    exec_globals['unittest'] = unittest
                    
                    # Dynamic execution
                    exec(code, exec_globals, exec_locals)

                    # Check if _check_layer_naming is defined
                    if '_check_layer_naming' not in exec_locals:
                        print(f"Code snippet {i}: FAILED, '_check_layer_naming' not found in exec_locals.\n")
                        failed_count += 1
                        results.append({
                            "function_name": "_check_layer_naming",
                            "code": code,
                            "result": "failed"
                        })
                        continue

                    # Test the extracted _check_layer_naming function with realistic `params`
                    sample_params = {
                        "transformer/layer_0/mlp/linear_1": Any,
                        "transformer/layer_0/attn/key": Any,
                        "transformer/layer_1/layer_norm": Any,
                        "transformer/layer_1/mlp/linear_2": Any
                    }
                    dummy_test_instance = DummyTest()
                    dummy_test_instance.setUp()
                    exec_locals['_check_layer_naming'](dummy_test_instance, sample_params)

                    print(f"Code snippet {i}: PASSED all assertions.\n")
                    passed_count += 1
                    results.append({
                        "function_name": "_check_layer_naming",
                        "code": code,
                        "result": "passed"
                    })
                except Exception as e:
                    print(f"Code snippet {i}: FAILED with error: {e}\n")
                    failed_count += 1
                    results.append({
                        "function_name": "_check_layer_naming",
                        "code": code,
                        "result": "failed"
                    })

        # Summary of results
        print(f"\nTest Summary: {passed_count} passed, {failed_count} failed, total {len(self.code_snippets)}\n")
        self.assertEqual(passed_count + failed_count, len(self.code_snippets), "Test count mismatch!")

        # ===== Write results to test_result.jsonl ====
        existing_records = []
        if os.path.exists(TEST_RESULT_JSONL):
            with open(TEST_RESULT_JSONL, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    existing_records.append(json.loads(line))
        
        # Remove existing records for _check_layer_naming
        existing_records = [
            rec for rec in existing_records
            if rec.get("function_name") != "_check_layer_naming"
        ]
        
        # Add new results
        existing_records.extend(results)

        with open(TEST_RESULT_JSONL, "w", encoding="utf-8") as f:
            for record in existing_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        print("Results have been written to test_result.jsonl")

if __name__ == "__main__":
    unittest.main()
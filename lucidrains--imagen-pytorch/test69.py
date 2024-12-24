import unittest
import json
import os
from typing import Any
import torch
import torch.nn as nn

TEST_RESULT_JSONL = "test_result.jsonl"

class TestGroundTruthInit(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file and get the specific code snippet
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippet = data[68]
    
    def test_init_function(self):
        """Test the __init__ function behavior."""
        passed_count = 0
        failed_count = 0
        results = []

        code = self.code_snippet

        try:
            exec_globals = {
                'torch': torch,
                'nn': nn,
                'default': lambda val, default_val: val if val is not None else default_val
            }
            exec_locals = {}
            
            # Dynamically execute the code snippet
            exec(code, exec_globals, exec_locals)

            # Check if the '__init__' function exists
            assert '__init__' in exec_locals, "Function '__init__' not found in code."

            # Create a dummy class to test the constructor
            class TestClass(exec_locals['__init__']):
                def init_conv_(self, conv_layer):
                    pass

            # Instantiate the class
            instance = TestClass(4, 8)
            self.assertIsInstance(instance, TestClass)
            self.assertIsInstance(instance.net, nn.Sequential)

            conv_layer = instance.net[0]
            self.assertIsInstance(conv_layer, nn.Conv2d)
            self.assertEqual(conv_layer.out_channels, 32)

            print(f"Code snippet: PASSED all assertions.\n")
            passed_count += 1
            results.append({
                "function_name": "__init__",
                "code": code,
                "result": "passed"
            })
        except Exception as e:
            print(f"Code snippet: FAILED with error: {e}\n")
            failed_count += 1
            results.append({
                "function_name": "__init__",
                "code": code,
                "result": "failed"
            })

        print(f"\nTest Summary: {passed_count} passed, {failed_count} failed\n")
        self.assertEqual(passed_count + failed_count, 1, "Test count mismatch!")

        # ============= Write Test Results to test_result.jsonl =============
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
            if rec.get("function_name") != "__init__"
        ]

        existing_records.extend(results)

        with open(TEST_RESULT_JSONL, "w", encoding="utf-8") as f:
            for record in existing_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    unittest.main()
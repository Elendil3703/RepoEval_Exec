import unittest
import json
import sys
import os
from typing import Any

TEST_RESULT_JSONL = "test_result.jsonl"

class TestFromHiddenFunction(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippets = data[198]  # Get the 199th JSON element (index 198)
        if len(cls.code_snippets) < 1:
            raise ValueError("Expected at least one code snippet in the 199th JSON array")

    def test_code_snippets(self):
        """Dynamically test all code snippets in the JSON for the 'from_hidden' function."""
        passed_count = 0  # Counter for passed tests
        failed_count = 0  # Counter for failed tests
        results = []      # 收集要写入 JSONL 的测试结果

        for i, code in enumerate(self.code_snippets):
            with self.subTest(code_index=i):
                print(f"Running test for code snippet {i}...")

                if "def from_hidden" not in code:
                    print(f"Code snippet {i}: FAILED, function 'from_hidden' not found.\n")
                    failed_count += 1
                    results.append({
                        "function_name": "from_hidden",
                        "code": code,
                        "result": "failed"
                    })
                    continue

                exec_globals = {
                    'sys': sys,
                    'Any': Any,
                    'bases': bases
                }
                exec_locals = {}

                try:
                    # 添加 bases 的模拟实现
                    class MockBases:
                        class BasisDirection:
                            def __init__(self, name, value):
                                self.name = name
                                self.value = value
                            
                            def __eq__(self, other):
                                return self.name == other.name and self.value == other.value
                    
                    exec_globals['bases'] = MockBases

                    # 动态执行代码片段
                    exec(code, exec_globals, exec_locals)

                    if 'from_hidden' not in exec_locals:
                        print(f"Code snippet {i}: FAILED, 'from_hidden' not found in exec_locals.\n")
                        failed_count += 1
                        results.append({
                            "function_name": "from_hidden",
                            "code": code,
                            "result": "failed"
                        })
                        continue
                    
                    # Simulate a hidden object
                    class MockHidden:
                        value = ('x', 1, 'y', 2)

                    # Execute the function
                    hidden_object = MockHidden()
                    x_dir, y_dir = exec_locals['from_hidden'](hidden_object)

                    # Validate the results
                    self.assertEqual(x_dir, MockBases.BasisDirection('x', 1), f"Code snippet {i} failed for x_dir.")
                    self.assertEqual(y_dir, MockBases.BasisDirection('y', 2), f"Code snippet {i} failed for y_dir.")

                    print(f"Code snippet {i}: PASSED all assertions.\n")
                    passed_count += 1
                    results.append({
                        "function_name": "from_hidden",
                        "code": code,
                        "result": "passed"
                    })

                except Exception as e:
                    print(f"Code snippet {i}: FAILED with error: {e}\n")
                    failed_count += 1
                    results.append({
                        "function_name": "from_hidden",
                        "code": code,
                        "result": "failed"
                    })

        # Test summary
        print(f"\nTest Summary: {passed_count} passed, {failed_count} failed, total {len(self.code_snippets)}\n")
        self.assertEqual(passed_count + failed_count, len(self.code_snippets), "Test count mismatch!")

        # Writing results to test_result.jsonl
        existing_records = []
        if os.path.exists(TEST_RESULT_JSONL):
            with open(TEST_RESULT_JSONL, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    existing_records.append(json.loads(line))

        # Remove old records for from_hidden
        existing_records = [
            rec for rec in existing_records
            if rec.get("function_name") != "from_hidden"
        ]

        # Append new results
        existing_records.extend(results)

        with open(TEST_RESULT_JSONL, "w", encoding="utf-8") as f:
            for record in existing_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        print("Results have been written to test_result.jsonl")

if __name__ == "__main__":
    unittest.main()
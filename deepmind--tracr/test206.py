import unittest
import json
import sys
import re
import os
from typing import Any

TEST_RESULT_JSONL = "test_result.jsonl"

class TestActionFunction(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippets = data[205]  # Get the 206th JSON element
        if len(cls.code_snippets) < 1:
            raise ValueError("Expected at least one code snippet in the 206th JSON array")

    def test_code_snippets(self):
        """Dynamically test all code snippets for the 'action' function."""
        passed_count = 0  # Counter for passed tests
        failed_count = 0  # Counter for failed tests
        results = []      # 收集要写入 JSONL 的测试结果

        for i, code in enumerate(self.code_snippets):
            with self.subTest(code_index=i):
                print(f"Running test for code snippet {i}...")

                # ------------------- Static checks -------------------
                if "def action" not in code:
                    print(f"Code snippet {i}: FAILED, 'action' function not found in code.\n")
                    failed_count += 1
                    results.append({
                        "function_name": "action",
                        "code": code,
                        "result": "failed"
                    })
                    continue

                # Pattern for function signature
                func_pattern = r"def\s+action\s*\(.*bases\.BasisDirection.*\)\s*->\s*bases\.VectorInBasis\s*:.*"
                if not re.search(func_pattern, code, re.DOTALL):
                    print(f"Code snippet {i}: FAILED, incorrect signature for 'action'.\n")
                    failed_count += 1
                    results.append({
                        "function_name": "action",
                        "code": code,
                        "result": "failed"
                    })
                    continue

                # ------------------- Dynamic execution and logic testing -------------------
                exec_globals = {
                    'sys': sys,
                    'bases': None,  # Assume the presence of 'bases' module
                }
                exec_locals = {}

                try:
                    exec(code, exec_globals, exec_locals)

                    # Check if 'action' is defined
                    if 'action' not in exec_locals:
                        print(f"Code snippet {i}: FAILED, 'action' not found in exec_locals.\n")
                        failed_count += 1
                        results.append({
                            "function_name": "action",
                            "code": code,
                            "result": "failed"
                        })
                        continue

                    # Define mocks and test 'action' logic
                    class MockBasisDirection: pass
                    class MockVectorInBasis:
                        def __add__(self, other): return self
                        
                    class MockSpace:
                        @staticmethod
                        def null_vector():
                            return MockVectorInBasis()
                        
                        def vector_from_basis_direction(self, x):
                            return MockVectorInBasis()
                    
                    # Configure exec_globals with mock components
                    joint_output_space = MockSpace()
                    fns = [MockSpace()]
                    exec_globals['joint_output_space'] = joint_output_space
                    exec_globals['fns'] = fns
                    exec_globals['bases'] = {
                        'BasisDirection': MockBasisDirection,
                        'VectorInBasis': MockVectorInBasis
                    }

                    # Execute the action function
                    x = MockBasisDirection()
                    func_result = exec_locals['action'](x)

                    # Simple assertion: function should return an instance of MockVectorInBasis
                    self.assertIsInstance(func_result, MockVectorInBasis, f"Code snippet {i} failed to return correct type.")

                    print(f"Code snippet {i}: PASSED all assertions.\n")
                    passed_count += 1
                    results.append({
                        "function_name": "action",
                        "code": code,
                        "result": "passed"
                    })
                except Exception as e:
                    print(f"Code snippet {i}: FAILED with error: {e}\n")
                    failed_count += 1
                    results.append({
                        "function_name": "action",
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

        # Remove old records for 'action'
        existing_records = [
            rec for rec in existing_records
            if rec.get("function_name") != "action"
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
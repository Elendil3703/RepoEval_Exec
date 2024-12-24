import unittest
import json
import os
import sys
from typing import Any  # 确保注入的环境中有 Any

TEST_RESULT_JSONL = "test_result.jsonl"

class TestCarperAITrlxResult(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippet = data[207]  # Get the 208th JSON element (index 207)
        if not cls.code_snippet:
            raise ValueError("Expected a code snippet in the 208th JSON element")

    def test_action_function(self):
        """Test the action function by simulating typical use cases and edge cases."""
        results = []
        passed_count = 0
        failed_count = 0

        code = self.code_snippet

        exec_globals = {
            'to_space': MockToSpace(),
            'VectorInBasis': MockVectorInBasis,
            'sys': sys,
            'Any': Any
        }
        exec_locals = {}

        try:
            # Execute the code snippet dynamically
            exec(code, exec_globals, exec_locals)

            # Check if 'action' function is present in the executed local scope
            if 'action' not in exec_locals:
                print("Code snippet FAILED, 'action' function not found after execution.\n")
                failed_count += 1
                results.append({
                    "function_name": "action",
                    "code": code,
                    "result": "failed"
                })
                self.fail("Function 'action' not found in code snippet execution.")
                return

            # Retrieve the 'action' function
            action = exec_locals['action']

            # Define test cases
            test_cases = [
                (MockBasisDirection(True), MockVectorInBasis("from_basis")),
                (MockBasisDirection(False), MockVectorInBasis("null_vector")),
            ]

            for direction, expected_result in test_cases:
                result = action(direction)
                self.assertEqual(
                    result.value,
                    expected_result.value,
                    f"Failed on direction {direction.in_space}: expected {expected_result}, got {result}"
                )
            
            print("Code snippet PASSED all assertions.\n")
            passed_count += len(test_cases)
            for _ in test_cases:
                results.append({
                    "function_name": "action",
                    "code": code,
                    "result": "passed"
                })
        except Exception as e:
            print(f"Code snippet FAILED with error: {e}\n")
            failed_count += 1
            results.append({
                "function_name": "action",
                "code": code,
                "result": "failed"
            })

        # Print summary and assert
        print(f"\nTest Summary: {passed_count} passed, {failed_count} failed, total {len(test_cases)}\n")
        self.assertEqual(passed_count + failed_count, len(test_cases), "Test count mismatch!")

        # Write results to test_result.jsonl
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
            if rec.get("function_name") != "action"
        ]

        existing_records.extend(results)

        with open(TEST_RESULT_JSONL, "w", encoding="utf-8") as f:
            for record in existing_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        print("Results have been written to test_result.jsonl")


class MockBasisDirection:
    def __init__(self, in_space: bool):
        self.in_space = in_space


class MockToSpace:
    def vector_from_basis_direction(self, direction):
        return MockVectorInBasis("from_basis")

    @staticmethod
    def null_vector():
        return MockVectorInBasis("null_vector")


class MockVectorInBasis:
    def __init__(self, value: str):
        self.value = value

    def __eq__(self, other):
        if isinstance(other, MockVectorInBasis):
            return self.value == other.value
        return False

    def __repr__(self):
        return f"MockVectorInBasis({self.value})"


if __name__ == "__main__":
    unittest.main()
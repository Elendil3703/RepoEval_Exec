import unittest
import json
import os
from typing import Any

TEST_RESULT_JSONL = "test_result.jsonl"

class TestFirstLayerAction(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippet = data[191]  # Get the specified JSON element
        if not cls.code_snippet:
            raise ValueError("Expected code snippet not found in JSON data")

    def test_first_layer_action(self):
        """Dynamically test the 'first_layer_action' in the JSON with specific checks."""
        passed_count = 0  # Counter for passed tests
        failed_count = 0  # Counter for failed tests
        results = []      # Collect test results to write into JSONL

        # Static check: Ensure the function definition for 'first_layer_action' exists
        if "def first_layer_action" not in self.code_snippet:
            print(f"Code snippet: FAILED, function 'first_layer_action' not found in code.\n")
            failed_count += 1
            results.append({
                "function_name": "first_layer_action",
                "code": self.code_snippet,
                "result": "failed"
            })
        else:
            func_pattern = r"def\s+first_layer_action\s*\("
            if not re.search(func_pattern, self.code_snippet):
                print(f"Code snippet: FAILED, incorrect signature for 'first_layer_action'.\n")
                failed_count += 1
                results.append({
                    "function_name": "first_layer_action",
                    "code": self.code_snippet,
                    "result": "failed"
                })
            else:
                # Dynamic execution and test logic
                exec_globals = {
                    'bases': mock_bases_module(),
                    'hidden_space': mock_hidden_space(),
                }
                exec_locals = {}
                try:
                    # Dynamically execute the code snippet
                    exec(self.code_snippet, exec_globals, exec_locals)

                    # Check if 'first_layer_action' really exists
                    if 'first_layer_action' not in exec_locals:
                        print(f"Code snippet: FAILED, 'first_layer_action' not found in exec_locals.\n")
                        failed_count += 1
                        results.append({
                            "function_name": "first_layer_action",
                            "code": self.code_snippet,
                            "result": "failed"
                        })
                    else:
                        first_layer_action = exec_locals['first_layer_action']
                        # Perform tests on 'first_layer_action'
                        direction_input1 = exec_globals['bases'].BasisDirection('input1')
                        direction_input2 = exec_globals['bases'].BasisDirection('input2')
                        expected_output_input1 = exec_globals['bases'].VectorInBasis([1, 0])
                        expected_output_input2 = exec_globals['bases'].VectorInBasis([0, 1])

                        # Test for input1 basis direction
                        result = first_layer_action(direction_input1)
                        self.assertEqual(result, expected_output_input1, "Mismatch output for input1 direction")
                    
                        # Test for input2 basis direction
                        result = first_layer_action(direction_input2)
                        self.assertEqual(result, expected_output_input2, "Mismatch output for input2 direction")

                        print(f"Code snippet: PASSED all assertions.\n")
                        passed_count += 1
                        results.append({
                            "function_name": "first_layer_action",
                            "code": self.code_snippet,
                            "result": "passed"
                        })

                except Exception as e:
                    print(f"Code snippet: FAILED with error: {e}\n")
                    failed_count += 1
                    results.append({
                        "function_name": "first_layer_action",
                        "code": self.code_snippet,
                        "result": "failed"
                    })

        # Test summary
        print(f"\nTest Summary: {passed_count} passed, {failed_count} failed.\n")

        # Writing the test results into test_result.jsonl
        # Read existing records (if any)
        existing_records = []
        if os.path.exists(TEST_RESULT_JSONL):
            with open(TEST_RESULT_JSONL, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    existing_records.append(json.loads(line))

        # Remove old records for the function name
        existing_records = [
            rec for rec in existing_records
            if rec.get("function_name") != "first_layer_action"
        ]

        # Append new results
        existing_records.extend(results)

        # Overwrite the test_result.jsonl file
        with open(TEST_RESULT_JSONL, "w", encoding="utf-8") as f:
            for record in existing_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

def mock_bases_module():
    """Mocks the 'bases' module with necessary classes and logic."""
    class BasisDirection:
        def __init__(self, name):
            self.name = name

    class VectorInBasis:
        def __init__(self, vector):
            self.vector = vector

        def __add__(self, other):
            return VectorInBasis([x + y for x, y in zip(self.vector, other.vector)])

        def __sub__(self, other):
            return VectorInBasis([x - y for x, y in zip(self.vector, other.vector)])

        def __eq__(self, other):
            return self.vector == other.vector

    return type('bases', (), {
        'BasisDirection': BasisDirection,
        'VectorInBasis': VectorInBasis
    })

def mock_hidden_space():
    """Mocks hidden_space with necessary logic."""
    class MockHiddenSpace:
        @staticmethod
        def null_vector():
            return mock_bases_module().VectorInBasis([0, 0])

    return MockHiddenSpace()

if __name__ == "__main__":
    unittest.main()
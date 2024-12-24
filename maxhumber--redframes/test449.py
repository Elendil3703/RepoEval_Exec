import unittest
import json
import os
from typing import Any

TEST_RESULT_JSONL = "test_result.jsonl"

class DataFrameMock:
    """Mock class for DataFrame to test memory function."""
    def __init__(self, data):
        self._data = data

    def memory_usage(self, deep=False):
        """Mock memory usage calculation."""
        # Assume simplistic memory usage calculation for testing purposes
        if deep:
            return {
                "foo": 3 * 8,  # Assuming each integer takes 8 bytes
                "bar": sum(len(item) for item in self._data["bar"])  # Each character is 1 byte
            }

class TestMemoryFunction(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippets = data[448]  # Get the 449th JSON element
        if len(cls.code_snippets) < 1:
            raise ValueError("Expected at least one code snippet in the 449th JSON array")
    
    def test_memory_function(self):
        """Test the memory function from the code snippet."""
        passed_count = 0  # Counter for passed tests
        failed_count = 0  # Counter for failed tests
        results = []      # To collect test results for JSONL

        for i, code in enumerate(self.code_snippets):
            with self.subTest(code_index=i):
                print(f"Running test for code snippet {i}...")
                
                # Inject necessary components for code execution
                exec_globals = {
                    'DataFrameMock': DataFrameMock,
                    'Any': Any,
                }
                
                exec_locals = {}

                try:
                    # Dynamically execute the code snippet
                    exec(code, exec_globals, exec_locals)

                    # Check if "memory" function exists within the executed snippet
                    if "memory" not in exec_locals:
                        print(f"Code snippet {i}: FAILED, 'memory' function not found in exec_locals.\n")
                        failed_count += 1
                        results.append({
                            "function_name": "memory",
                            "code": code,
                            "result": "failed"
                        })
                        continue

                    # Access the memory function
                    memory_func = exec_locals["memory"]

                    # Create a mock DataFrame
                    df = DataFrameMock(data={
                        "foo": [1, 2, 3],
                        "bar": ["A", "B", "C"]
                    })

                    # Inject the mock object and test the memory function
                    df.memory = memory_func.__get__(df)
                    result = df.memory()

                    # Check the expected output
                    expected_memory_usage = str((3*8 + 1 + 1 + 1)) + " B"  # Total mock memory
                    self.assertEqual(result, expected_memory_usage,
                                     f"Code snippet {i}: FAILED memory usage is incorrect, got {result}, expected {expected_memory_usage}")
                    
                    print(f"Code snippet {i}: PASSED all assertions.\n")
                    passed_count += 1
                    results.append({
                        "function_name": "memory",
                        "code": code,
                        "result": "passed"
                    })
                except Exception as e:
                    print(f"Code snippet {i}: FAILED with error: {e}\n")
                    failed_count += 1
                    results.append({
                        "function_name": "memory",
                        "code": code,
                        "result": "failed"
                    })

        # Print test summary
        print(f"\nTest Summary: {passed_count} passed, {failed_count} failed, total {len(self.code_snippets)}\n")
        self.assertEqual(passed_count + failed_count, len(self.code_snippets), "Test count mismatch!")

        # ============= Write the test results into test_result.jsonl =============
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
            if rec.get("function_name") != "memory"
        ]

        existing_records.extend(results)

        with open(TEST_RESULT_JSONL, "w", encoding="utf-8") as f:
            for record in existing_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        print("Results have been written to test_result.jsonl")

if __name__ == "__main__":
    unittest.main()
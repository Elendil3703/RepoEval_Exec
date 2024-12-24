import unittest
import json
import os
from typing import Any

TEST_RESULT_JSONL = "test_result.jsonl"

class TestMatrixInitialization(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippets = data[208]  # Get the 209th JSON element (index 208)
        if len(cls.code_snippets) < 1:
            raise ValueError("Expected at least one code snippet in the 209th JSON element")

    def test_post_init(self):
        """Test the __post_init__ method for size validation."""
        passed_count = 0
        failed_count = 0
        results = []

        for i, code in enumerate(self.code_snippets):
            with self.subTest(code_index=i):
                print(f"Running test for code snippet {i}...")
                if "__post_init__" not in code:
                    print(f"Code snippet {i}: FAILED, '__post_init__' not found.\n")
                    failed_count += 1
                    results.append({
                        "function_name": "__post_init__",
                        "code": code,
                        "result": "failed"
                    })
                    continue

                exec_globals = {
                    'Any': Any
                }
                exec_locals = {}

                try:
                    exec(code, exec_globals, exec_locals)

                    # Assuming these mock classes are required by the test
                    class MockSpace:
                        def __init__(self, num_dims):
                            self.num_dims = num_dims

                    class MockMatrix:
                        def __init__(self, shape):
                            self.shape = shape

                    # Injecting mocks and testing __post_init__
                    mock_left_space = MockSpace(3)
                    mock_right_space = MockSpace(4)
                    mock_matrix_correct = MockMatrix((3, 4))
                    mock_matrix_incorrect = MockMatrix((2, 5))

                    # Define a mock dataclass to test __post_init__
                    class MockDataClass:
                        def __init__(self, matrix, left_space, right_space):
                            self.matrix = matrix
                            self.left_space = left_space
                            self.right_space = right_space
                            exec_locals['__post_init__'](self)

                    # Correct matrix should not raise an error
                    try:
                        MockDataClass(mock_matrix_correct, mock_left_space, mock_right_space)
                    except AssertionError:
                        raise AssertionError("Code snippet incorrectly raised an AssertionError with correct sizes.")

                    # Incorrect matrix should raise an error
                    with self.assertRaises(AssertionError):
                        MockDataClass(mock_matrix_incorrect, mock_left_space, mock_right_space)

                    print(f"Code snippet {i}: PASSED all assertions.\n")
                    passed_count += 1
                    results.append({
                        "function_name": "__post_init__",
                        "code": code,
                        "result": "passed"
                    })
                except Exception as e:
                    print(f"Code snippet {i}: FAILED with error: {e}\n")
                    failed_count += 1
                    results.append({
                        "function_name": "__post_init__",
                        "code": code,
                        "result": "failed"
                    })

        # Test summary
        print(f"\nTest Summary: {passed_count} passed, {failed_count} failed, total {len(self.code_snippets)}\n")
        self.assertEqual(passed_count + failed_count, len(self.code_snippets), "Test count mismatch!")

        # Write test results to test_result.jsonl
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
            if rec.get("function_name") != "__post_init__"
        ]

        existing_records.extend(results)

        with open(TEST_RESULT_JSONL, "w", encoding="utf-8") as f:
            for record in existing_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        print("Results have been written to test_result.jsonl")

if __name__ == "__main__":
    unittest.main()
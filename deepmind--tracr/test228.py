import unittest
import json
import sys
import os
from typing import Any

TEST_RESULT_JSONL = "test_result.jsonl"

class TestPostInitFunction(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippet = data[227]  # Get the 228th JSON element (at index 227)
        if not cls.code_snippet:
            raise ValueError("Expected code snippet is not found in the JSON data")

    def test_post_init_function(self):
        """Test code snippet to ensure __post_init__ function behaves correctly."""
        passed_count = 0
        failed_count = 0
        results = []

        code = self.code_snippet
        i = 227  # index for reference
        
        print(f"Running test for code snippet {i}...")

        exec_globals = {
            'Any': Any,
            '__name__': '__main__',  # Necessary for some dynamic checks
            'bases': type('bases', (), {
                'join_vector_spaces': lambda x, y: x + " & " + y
            })
        }
        exec_locals = {}

        try:
            # Dynamically execute the code snippet
            exec(code, exec_globals, exec_locals)

            # Ensure there's a __post_init__ function
            if '__post_init__' not in exec_locals:
                print(f"Code snippet {i}: FAILED, '__post_init__' not found.\n")
                failed_count += 1
                results.append({
                    "function_name": "__post_init__",
                    "code": code,
                    "result": "failed"
                })
                raise ValueError("__post_init__ method is missing in the executed code.")

            # Mock necessary parts for testing
            class MockSpace:
                def __init__(self, space_name, parent_space=None):
                    self.space_name = space_name
                    self.parent_space = parent_space

                def issubspace(self, other_space):
                    return self.space_name in other_space.space_name

            class MockObject:
                def __init__(self):
                    self.input_space = MockSpace("A")
                    self.output_space = MockSpace("B")

            # Mock object with necessary attributes
            test_instance = type('MockTest', (), {
                'fst': MockObject(),
                'snd': MockObject(),
                'residual_space': None
            })()

            exec_locals['__post_init__'](test_instance)

            # Validate function behavior
            self.assertIsNotNone(test_instance.residual_space,
                                 f"Code snippet {i}: residual_space should not be None.")

            self.assertTrue(test_instance.fst.input_space.issubspace(test_instance.residual_space),
                            f"Code snippet {i}: fst.input_space should be a subspace of residual_space.")

            self.assertTrue(test_instance.snd.output_space.issubspace(test_instance.residual_space),
                            f"Code snippet {i}: snd.output_space should be a subspace of residual_space.")

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

        print(f"\nTest Summary: {passed_count} passed, {failed_count} failed\n")

        # ============= Write the test results to test_result.jsonl =============
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
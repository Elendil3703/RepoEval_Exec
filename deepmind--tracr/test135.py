import unittest
import json
import sys
import os
from typing import Any

TEST_RESULT_JSONL = "test_result.jsonl"

# Placeholder class to simulate Selector
class Selector:
    pass

class TestInitMethod(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippet = data[134]  # Get the specified JSON element
        if not cls.code_snippet.strip():
            raise ValueError("Expected a non-empty code snippet")

    def test_init_method(self):
        """Test the behavior of the __init__ method within the code snippet."""
        passed_count = 0  # Counter for passed tests
        failed_count = 0  # Counter for failed tests
        results = []      # Collect results to write to JSONL

        # Execute the snippet and inject necessary components
        exec_globals = {
            'Selector': Selector,
            'Any': Any,
        }
        exec_locals = {}

        try:
            # Dynamically execute the code snippet
            exec(self.code_snippet, exec_globals, exec_locals)

            # Check if the implemented class or method is present
            if '__init__' not in exec_locals:
                failed_count += 1
                results.append({
                    "function_name": "__init__",
                    "code": self.code_snippet,
                    "result": "failed"
                })
            else:
                # Test instantiation with proper arguments
                class TestClass:
                    def __init__(self, fst: Selector, snd: Selector):
                        exec_locals['__init__'](self, fst, snd)

                instance = TestClass(Selector(), Selector())
                
                # Assertions after instantiation
                self.assertIsInstance(instance.fst, Selector)
                self.assertIsInstance(instance.snd, Selector)

                passed_count += 1
                results.append({
                    "function_name": "__init__",
                    "code": self.code_snippet,
                    "result": "passed"
                })

        except Exception as e:
            print(f"__init__ method test FAILED with error: {e}\n")
            failed_count += 1
            results.append({
                "function_name": "__init__",
                "code": self.code_snippet,
                "result": "failed"
            })

        # Write results to test_result.jsonl
        existing_records = []
        if os.path.exists(TEST_RESULT_JSONL):
            with open(TEST_RESULT_JSONL, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        existing_records.append(json.loads(line))

        # Remove old records for __init__
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
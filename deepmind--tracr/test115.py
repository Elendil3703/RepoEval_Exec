import unittest
import json
import os
from typing import Any

TEST_RESULT_JSONL = "test_result.jsonl"

class TestCustomObjectGetItem(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippet = data[114]  # Get the 115th JSON element
        if not cls.code_snippet:
            raise ValueError("Expected a code snippet for the 115th element")

    def test_get_item(self):
        """Test the __getitem__ method of the custom class."""
        passed_count = 0
        failed_count = 0
        results = []

        code = self.code_snippet

        # Define expected variables for execution
        exec_globals = {
            'Any': Any,  # Inject Any
            'DEFAULT_ANNOTATORS': {'test_key': lambda x: f"annotated_{x}"}
        }
        exec_locals = {}

        try:
            # Execute the code snippet
            exec(code, exec_globals, exec_locals)

            if '__getitem__' not in exec_locals:
                raise ValueError("Method '__getitem__' not found in executed context.")

            # Define a mock class to test the method
            class MockClass:
                def __init__(self):
                    self._inner_dict = {}
                    self._expr = "expression"

                __getitem__ = exec_locals['__getitem__']

            # Create an instance of the mock class
            obj = MockClass()

            # Test the functionality
            try:
                # Test accessing an existing key
                obj._inner_dict['existing_key'] = 'value'
                result = obj['existing_key']
                self.assertEqual(result, 'value', "Failed to retrieve existing key value.")
                passed_count += 1
                results.append({
                    "function_name": "__getitem__",
                    "code": code,
                    "result": "passed"
                })
            except Exception as e:
                failed_count += 1
                results.append({
                    "function_name": "__getitem__",
                    "code": code,
                    "result": "failed",
                    "error": str(e)
                })

            try:
                # Test accessing a default annotator key
                result = obj['test_key']
                self.assertEqual(result, 'annotated_expression', "Failed to retrieve and use default annotator.")
                passed_count += 1
                results.append({
                    "function_name": "__getitem__",
                    "code": code,
                    "result": "passed"
                })
            except Exception as e:
                failed_count += 1
                results.append({
                    "function_name": "__getitem__",
                    "code": code,
                    "result": "failed",
                    "error": str(e)
                })

            try:
                # Test accessing a non-existing key
                with self.assertRaises(KeyError):
                    obj['non_existing_key']
                passed_count += 1
                results.append({
                    "function_name": "__getitem__",
                    "code": code,
                    "result": "passed"
                })
            except Exception as e:
                failed_count += 1
                results.append({
                    "function_name": "__getitem__",
                    "code": code,
                    "result": "failed",
                    "error": str(e)
                })

        except Exception as top_exception:
            failed_count += 3  # Reflection of passed, non-tested scenarios
            results.append({
                "function_name": "__getitem__",
                "code": code,
                "result": "failed",
                "error": str(top_exception)
            })

        print(f"\nTest Summary: {passed_count} passed, {failed_count} failed, total {3}\n")
        self.assertEqual(passed_count + failed_count, 3, "Test count mismatch!")

        # Write results to JSONL file
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
            if rec.get("function_name") != "__getitem__"
        ]

        existing_records.extend(results)

        with open(TEST_RESULT_JSONL, "w", encoding="utf-8") as f:
            for record in existing_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    unittest.main()
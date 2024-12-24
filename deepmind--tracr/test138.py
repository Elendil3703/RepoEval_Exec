import unittest
import json
import os
from typing import Any

TEST_RESULT_JSONL = "test_result.jsonl"

class TestCustomInitFunction(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippet = data[137]  # Get the 138th JSON element

    def test_init_function(self):
        """Test the __init__ function from the 138th JSON code snippet."""
        code = self.code_snippet
        passed_count = 0
        failed_count = 0
        results = []

        # Prepare an execution environment
        exec_globals = {}
        exec_locals = {}

        try:
            # Execute the code snippet to bring class definitions into scope
            exec(code, exec_globals, exec_locals)

            # Define mock classes for Selector and SOp for testing
            class Selector: pass
            class SOp: pass

            # Inject mock classes into the execution environment
            exec_locals['Selector'] = Selector
            exec_locals['SOp'] = SOp

            # Test the __init__ function by creating a class instance
            if '__init__' not in code:
                raise AssertionError("__init__ method not found in provided code snippet.")

            instance = exec_locals['MyClass'](selector=Selector(), sop=SOp(), default="default")

            # Perform assertions
            assert isinstance(instance.selector, Selector)
            assert isinstance(instance.sop, SOp)
            assert instance.default == "default"
            passed_count += 1
            results.append({
                "function_name": "__init__",
                "code": code,
                "result": "passed"
            })
            print(f"Code snippet: PASSED all assertions.\n")
        except Exception as e:
            print(f"Code snippet: FAILED with error: {e}\n")
            failed_count += 1
            results.append({
                "function_name": "__init__",
                "code": code,
                "result": "failed"
            })

        # Final summary
        print(f"\nTest Summary: {passed_count} passed, {failed_count} failed, total 1\n")
        self.assertEqual(passed_count + failed_count, 1, "Test count mismatch!")

        # Write the results to test_result.jsonl
        existing_records = []
        if os.path.exists(TEST_RESULT_JSONL):
            with open(TEST_RESULT_JSONL, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    existing_records.append(json.loads(line))

        # Delete old records for function name "__init__"
        existing_records = [
            rec for rec in existing_records
            if rec.get("function_name") != "__init__"
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
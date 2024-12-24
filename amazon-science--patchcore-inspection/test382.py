import unittest
import json
import os
from typing import Any

TEST_RESULT_JSONL = "test_result.jsonl"

class TestCarperAITrlxResult(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippet = data[381]  # Get the 382th JSON element (index 381)
        if len(cls.code_snippet) < 1:
            raise ValueError("Expected at least one code snippet in JSON.")

    def test_constructor(self):
        """Test the constructor (__init__) of the provided class."""
        code = self.code_snippet

        passed_count = 0  # Counter for passed tests
        failed_count = 0  # Counter for failed tests
        results = []      # Collect test results for writing into JSONL

        exec_globals = {
            'Any': Any,  # Inject Any if needed
        }
        exec_locals = {}

        try:
            # Dynamically execute the code snippet
            exec(code, exec_globals, exec_locals)

            # Assume a class is defined in the code
            class_names = [name for name, obj in exec_locals.items() if isinstance(obj, type)]

            if not class_names:
                print("Code snippet: FAILED, no class definition found.\n")
                failed_count += 1
                results.append({
                    "function_name": "__init__",
                    "code": code,
                    "result": "failed"
                })
            else:
                # Instantiate the class and check the constructor logic
                class_name = class_names[0]
                test_class = exec_locals[class_name]

                # Creating an instance with default arguments
                instance = test_class('cuda')

                # Check the attribute values
                self.assertEqual(instance.device, 'cuda', "Mismatch in 'device' attribute.")
                self.assertEqual(instance.target_size, 224, "Mismatch in 'target_size' attribute.")
                self.assertEqual(instance.smoothing, 4, "Mismatch in 'smoothing' attribute.")

                print("Code snippet: PASSED all assertions.\n")
                passed_count += 1
                results.append({
                    "function_name": "__init__",
                    "code": code,
                    "result": "passed"
                })

        except Exception as e:
            print(f"Code snippet: FAILED with error: {e}\n")
            failed_count += 1
            results.append({
                "function_name": "__init__",
                "code": code,
                "result": "failed"
            })

        # Final summary
        print(f"\nTest Summary: {passed_count} passed, {failed_count} failed, total tests 1\n")
        self.assertEqual(passed_count + failed_count, 1, "Test count mismatch!")

        # ============= Write test results to test_result.jsonl =============
        existing_records = []
        if os.path.exists(TEST_RESULT_JSONL):
            with open(TEST_RESULT_JSONL, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    existing_records.append(json.loads(line))

        # Remove old records with function_name == "__init__"
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
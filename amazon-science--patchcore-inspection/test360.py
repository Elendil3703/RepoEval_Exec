import unittest
import json
import sys
import os

TEST_RESULT_JSONL = "test_result.jsonl"

class TestFunctionInit(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippet = data[359]  # Get the 360th JSON element
        
    def test_init_function(self):
        """Dynamically test the __init__ function from the JSON code snippet."""
        passed_count = 0  # Counter for passed tests
        failed_count = 0  # Counter for failed tests
        results = []      # Collect test results to write to JSONL

        code = self.code_snippet
        try:
            # ------------------- Dynamic Execution and Assertion Checks -------------------
            exec_globals = {}
            exec_locals = {}

            # Execute the code snippet
            exec(code, exec_globals, exec_locals)

            # Check if the class is defined properly
            defined_class_name = None
            for key in exec_locals:
                if callable(exec_locals[key]):
                    defined_class_name = key
                    break

            if not defined_class_name:
                raise AssertionError("No callable class found in the executed locals.")

            cls = exec_locals[defined_class_name]

            # Test with valid percentage values
            valid_values = [0.1, 0.5, 0.9]
            for val in valid_values:
                try:
                    instance = cls(val)
                    assert instance.percentage == val, f"Failed to set percentage correctly for {val}."
                    passed_count += 1
                except Exception as e:
                    failed_count += 1
                    results.append({
                        "function_name": "__init__",
                        "code": code,
                        "result": "failed",
                        "error": str(e)
                    })
                    continue

            # Test with invalid percentage values
            invalid_values = [-0.1, 0, 1, 1.1]
            for val in invalid_values:
                try:
                    cls(val)
                    raise AssertionError(f"No exception raised for invalid percentage value {val}.")
                except ValueError as e:
                    assert str(e) == "Percentage value not in (0, 1).", f"Unexpected error message: {e}."
                    passed_count += 1
                except Exception as e:
                    failed_count += 1
                    results.append({
                        "function_name": "__init__",
                        "code": code,
                        "result": "failed",
                        "error": str(e)
                    })
                    continue

        except Exception as e:
            failed_count += 1
            results.append({
                "function_name": "__init__",
                "code": code,
                "result": "failed",
                "error": str(e)
            })

        print(f"\nTest Summary: {passed_count} passed, {failed_count} failed.\n")
        self.assertEqual(passed_count + failed_count, len(valid_values) + len(invalid_values), "Test count mismatch!")

        # ============= Write the test results to test_result.jsonl =============
        # Read existing test_result.jsonl (if it exists)
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

if __name__ == "__main__":
    unittest.main()
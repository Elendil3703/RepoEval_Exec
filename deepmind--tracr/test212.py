import unittest
import json
import os

TEST_RESULT_JSONL = "test_result.jsonl"


class TestStrMethod(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippet = data[211]  # Get the 212th JSON element (index 211)
        if not cls.code_snippet:
            raise ValueError("Expected a code snippet at the 212th position")

    def test_str_method(self):
        """Dynamically test the __str__ method implementation."""
        passed_count = 0  # Counter for passed tests
        failed_count = 0  # Counter for failed tests
        results = []  # Collect results to be written to JSONL

        code = self.code_snippet
        exec_globals = {}
        exec_locals = {}

        try:
            exec(code, exec_globals, exec_locals)

            # Assuming the class with __str__ method is the only one defined
            class_name = next(
                (name for name, obj in exec_locals.items() if isinstance(obj, type)), None
            )
            assert class_name is not None, "No class found in the code snippet."

            # Create a dynamic instance of the class to test __str__ method
            _class = exec_locals[class_name]
            instance_with_value = _class()
            instance_with_value.name = "name"
            instance_with_value.value = "value"

            instance_without_value = _class()
            instance_without_value.name = "name"
            instance_without_value.value = None

            # Test when value is not None
            self.assertEqual(
                str(instance_with_value),
                "name:value",
                "Failed to correctly convert to string with value.",
            )

            # Test when value is None
            self.assertEqual(
                str(instance_without_value),
                "name",
                "Failed to correctly convert to string without value.",
            )

            print(f"Code snippet: PASSED all assertions.\n")
            passed_count += 1
            results.append({
                "function_name": "__str__",
                "code": code,
                "result": "passed"
            })
        except Exception as e:
            print(f"Code snippet: FAILED with error: {e}\n")
            failed_count += 1
            results.append({
                "function_name": "__str__",
                "code": code,
                "result": "failed"
            })

        # Final statistics
        print(f"\nTest Summary: {passed_count} passed, {failed_count} failed, total {passed_count + failed_count}\n")
        
        # ====== Write results to test_result.jsonl ======
        existing_records = []
        if os.path.exists(TEST_RESULT_JSONL):
            with open(TEST_RESULT_JSONL, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    existing_records.append(json.loads(line))

        # Remove old records for function __str__
        existing_records = [rec for rec in existing_records if rec.get("function_name") != "__str__"]

        # Append new results
        existing_records.extend(results)

        # Rewrite test_result.jsonl
        with open(TEST_RESULT_JSONL, "w", encoding="utf-8") as f:
            for record in existing_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        print("Results have been written to test_result.jsonl")

if __name__ == "__main__":
    unittest.main()
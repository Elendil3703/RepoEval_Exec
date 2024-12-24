import unittest
import json
import os

TEST_RESULT_JSONL = "test_result.jsonl"

class TestInitFunction(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippet = data[410]  # Get the 411th JSON element from a zero-based index

        if len(cls.code_snippet) < 1:
            raise ValueError("Expected at least one code snippet in the 411th JSON array")

    def test_init_function(self):
        """Dynamically test the __init__ method within the given code snippet."""
        passed_count = 0  # Counter for passed tests
        failed_count = 0  # Counter for failed tests
        results = []      # To collect the test results to write to JSONL

        code = self.code_snippet

        # Define a mock optimizer and scheduler for testing
        class MockOptimizer:
            def __init__(self):
                pass

        class MockScheduler:
            def __init__(self):
                pass

        # Setup for dynamic execution
        exec_globals = {}
        exec_locals = {
            'MockOptimizer': MockOptimizer,
            'MockScheduler': MockScheduler,
        }

        try:
            # Dynamically execute the code snippet
            exec(code, exec_globals, exec_locals)

            # Check if the __init__ method and the class containing it are correctly defined
            # Assuming the class is named MyTestableClass for this example
            class_name = [key for key in exec_globals if not key.startswith('__')][0]
            my_class = exec_globals[class_name]

            # Create an instance using the __init__ method
            instance = my_class(optimizer=MockOptimizer(), schedulers=MockScheduler())

            # Perform assertions
            self.assertTrue(hasattr(instance, 'optimizer'), "Instance does not have 'optimizer' attribute.")
            self.assertTrue(hasattr(instance, 'schedulers'), "Instance does not have 'schedulers' attribute.")

            print(f"Code snippet 410: PASSED all assertions.\n")
            passed_count += 1
            results.append({
                "function_name": "__init__",
                "code": code,
                "result": "passed"
            })
        except Exception as e:
            print(f"Code snippet 410: FAILED with error: {e}\n")
            failed_count += 1
            results.append({
                "function_name": "__init__",
                "code": code,
                "result": "failed"
            })

        # Test summary and assertions
        self.assertEqual(passed_count + failed_count, 1, "Test count mismatch!")  # Only one snippet is tested

        # ============= Write test results to test_result.jsonl =============
        # Read existing test_result.jsonl (if it exists)
        existing_records = []
        if os.path.exists(TEST_RESULT_JSONL):
            with open(TEST_RESULT_JSONL, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    existing_records.append(json.loads(line))

        # Remove old records for function_name == "__init__"
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
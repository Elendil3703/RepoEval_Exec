import unittest
import json
import logging
import os

TEST_RESULT_JSONL = "test_result.jsonl"

class TestFunWrapped(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippet = data[232]  # Get the 233rd JSON element

    def test_fun_wrapped(self):
        """Test the function fun_wrapped with dynamic code execution and checks."""
        # Prepare to capture logs
        logger = logging.getLogger()
        logger.setLevel(logging.WARNING)
        log_stream = logging.StreamHandler()
        logger.addHandler(log_stream)

        passed_count = 0  # Counter for passed tests
        failed_count = 0  # Counter for failed tests
        results = []      # Collect the results to be written in JSONL

        code = self.code_snippet
        exec_globals = {}
        exec_locals = {}

        try:
            # Define a mock function 'fun' to simulate 'fun_wrapped' calls
            def fun(*args):
                # Simple arithmetic operation, simulate regular and error cases
                if args and args[0] == 0:
                    raise ArithmeticError("Mock Error")
                return sum(args)

            exec_globals['fun'] = fun  # Inject mock fun
            exec_globals['logging'] = logging

            # Execute the provided code
            exec(code, exec_globals, exec_locals)

            # Check if 'fun_wrapped' is defined after execution
            if 'fun_wrapped' not in exec_locals:
                print("Failed: 'fun_wrapped' not found in exec_locals.")
                failed_count += 1
            else:
                # Test cases for fun_wrapped
                fun_wrapped = exec_locals['fun_wrapped']
                
                # Test case 1: Regular input
                result = fun_wrapped(1, 2, 3)
                if result == 6:
                    print("Test case 1 passed.")
                    passed_count += 1
                    results.append({
                        "function_name": "fun_wrapped",
                        "code": code,
                        "result": "passed"
                    })
                else:
                    print("Test case 1 failed.")
                    failed_count += 1
                    results.append({
                        "function_name": "fun_wrapped",
                        "code": code,
                        "result": "failed"
                    })

                # Test case 2: Input causing ArithmeticError
                log_stream.stream = []
                result = fun_wrapped(0)
                if result is None and len(log_stream.stream) > 0:
                    print("Test case 2 passed.")
                    passed_count += 1
                    results.append({
                        "function_name": "fun_wrapped",
                        "code": code,
                        "result": "passed"
                    })
                else:
                    print("Test case 2 failed.")
                    failed_count += 1
                    results.append({
                        "function_name": "fun_wrapped",
                        "code": code,
                        "result": "failed"
                    })

        except Exception as e:
            print(f"Failed with unexpected error: {e}")
            failed_count += 1
            results.append({
                "function_name": "fun_wrapped",
                "code": code,
                "result": "failed"
            })

        # Final statistics
        print(f"\nTest Summary: {passed_count} passed, {failed_count} failed, total {passed_count + failed_count}\n")
        self.assertEqual(passed_count + failed_count, 2, "Test count mismatch!")

        # Write the test results to test_result.jsonl
        existing_records = []
        if os.path.exists(TEST_RESULT_JSONL):
            with open(TEST_RESULT_JSONL, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    existing_records.append(json.loads(line))

        # Remove old records for 'fun_wrapped'
        existing_records = [
            rec for rec in existing_records
            if rec.get("function_name") != "fun_wrapped"
        ]

        # Append new results
        existing_records.extend(results)

        # Rewrite test_result.jsonl
        with open(TEST_RESULT_JSONL, "w", encoding="utf-8") as f:
            for record in existing_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    unittest.main()
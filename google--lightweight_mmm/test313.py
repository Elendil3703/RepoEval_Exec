import unittest
import json
import os
from typing import Any  # 确保注入的环境中有 Any

TEST_RESULT_JSONL = "test_result.jsonl"

class TestSimpleInterceptResult(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        # Get the 313th JSON element (index 312)
        cls.code_snippet = data[312]
        if not cls.code_snippet:
            raise ValueError("Expected at least one piece of code in the 313th JSON element")

    def test_simple_intercept(self):
        """Dynamically test the simple_intercept function from the code snippet."""
        passed_count = 0  # Counter for passed tests
        failed_count = 0  # Counter for failed tests
        results = []      # Collect test results to write to JSONL

        # Extract the code snippet
        code = self.code_snippet
        print(f"Running test for the code snippet...")

        # ------------------- Static checks -------------------
        # Ensure the necessary function definition "simple_intercept" is in the code.
        if "def simple_intercept" not in code:
            print(f"FAILED: Function 'simple_intercept' not found.\n")
            failed_count += 1
            # Log failure
            results.append({
                "function_name": "simple_intercept",
                "code": code,
                "result": "failed"
            })
            return

        # ------------------- Dynamic execution and logical testing -------------------
        exec_globals = { 'numpyro': None, 'priors': None, 'core_utils': None, 'custom_priors': None }
        exec_locals = {}

        try:
            # Dynamic execution of code snippet
            exec(code, exec_globals, exec_locals)

            # Ensure 'simple_intercept' is defined
            if 'simple_intercept' not in exec_locals:
                print(f"FAILED: 'simple_intercept' not found in exec_locals.\n")
                failed_count += 1
                results.append({
                    "function_name": "simple_intercept",
                    "code": code,
                    "result": "failed"
                })
                return

            # Substitute mock values and functions for the dependencies to run test
            class MockPriors:
                INTERCEPT = "intercept"
                @staticmethod
                def get_default_priors():
                    return { "intercept": lambda: "default" }

            class MockUtils:
                @staticmethod
                def get_number_geos(data):
                    return len(data)

            exec_globals['priors'] = MockPriors
            exec_globals['core_utils'] = MockUtils
            exec_globals['custom_priors'] = { 'intercept': None }  # Simulate empty custom priors
            exec_globals['numpyro'] = type('MockNumPyro', (), {
                'plate': lambda name, size: range(size),
                'sample': lambda name, fn: "sampled_value"
            })

            # Run the function and check the result
            intercept = exec_locals['simple_intercept']([])
            self.assertEqual(intercept, "sampled_value", "Failed to get the correct sampled value.")

            print(f"Code snippet: PASSED all assertions.\n")
            passed_count += 1
            results.append({
                "function_name": "simple_intercept",
                "code": code,
                "result": "passed"
            })
        except Exception as e:
            print(f"FAILED with error: {e}\n")
            failed_count += 1
            results.append({
                "function_name": "simple_intercept",
                "code": code,
                "result": "failed"
            })

        # ============= Write the test results to test_result.jsonl =============
        # Load existing test_result.jsonl (ignore if not present)
        existing_records = []
        if os.path.exists(TEST_RESULT_JSONL):
            with open(TEST_RESULT_JSONL, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    existing_records.append(json.loads(line))

        # Remove old records with function_name == "simple_intercept"
        existing_records = [
            rec for rec in existing_records
            if rec.get("function_name") != "simple_intercept"
        ]

        # Add new results
        existing_records.extend(results)

        # Write back to test_result.jsonl
        with open(TEST_RESULT_JSONL, "w", encoding="utf-8") as f:
            for record in existing_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        print("Results have been written to test_result.jsonl")

if __name__ == "__main__":
    unittest.main()
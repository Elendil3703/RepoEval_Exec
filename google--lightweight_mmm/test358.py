import unittest
import json
import os
import sys
from typing import Any

TEST_RESULT_JSONL = "test_result.jsonl"

class TestPredictiveFunction(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippet = data[357]  # Get the 358th JSON element (index 357)

    def test_predictive_function(self):
        """Test the infer.Predictive function in the provided code snippet."""
        passed_count = 0
        failed_count = 0
        results = []

        code = self.code_snippet

        # Check static presence of infer.Predictive
        if "infer.Predictive" not in code:
            print("Code snippet: FAILED, 'infer.Predictive' not found in code.\n")
            failed_count += 1
            results.append({
                "function_name": "_predict", 
                "code": code,
                "result": "failed"
            })
            self.fail("'infer.Predictive' not found in code.")
            return

        # Prepare execution environment
        exec_globals = {
            'infer': type('infer', (), {'Predictive': lambda *args, **kwargs: lambda **calls: "mock_result"}),
            'model': None,
            'posterior_samples': None,
            'rng_key': None,
            'media_data': None,
            'extra_features': None,
            'media_prior': None,
            'degrees_seasonality': None,
            'frequency': None,
            'transform_function': None,
            'custom_priors': None,
            'weekday_seasonality': None,
            'sys': sys,
            'Any': Any,
        }
        exec_locals = {}

        try:
            # Execute the code snippet dynamically
            exec(code, exec_globals, exec_locals)

            # Run the _predict function
            result = exec_locals['_predict']()
            self.assertEqual(result, "mock_result", "Code snippet did not return the expected mock result.")

            print("Code snippet: PASSED all assertions.\n")
            passed_count += 1
            results.append({
                "function_name": "_predict",
                "code": code,
                "result": "passed"
            })

        except Exception as e:
            print(f"Code snippet: FAILED with error: {e}\n")
            failed_count += 1
            results.append({
                "function_name": "_predict",
                "code": code,
                "result": "failed"
            })

        finally:
            # Final statistics
            print(f"\nTest Summary: {passed_count} passed, {failed_count} failed, total 1")

            # Write results to test_result.jsonl like reference code
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
                if rec.get("function_name") != "_predict"
            ]

            existing_records.extend(results)

            with open(TEST_RESULT_JSONL, "w", encoding="utf-8") as f:
                for record in existing_records:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")

            print("Results have been written to test_result.jsonl")

if __name__ == "__main__":
    unittest.main()
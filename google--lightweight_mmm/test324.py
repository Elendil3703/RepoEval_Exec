import unittest
import json
import os
from typing import Dict
from immutabledict import immutabledict
import torch.distributions as dist

TEST_RESULT_JSONL = "test_result.jsonl"

class TestGetDefaultPriors(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippet = data[323]  # Get the 324th JSON element
  
    def test_get_default_priors(self):
        """Dynamically test `get_default_priors` in the JSON with additional checks."""
        passed = False
        results = []

        code = self.code_snippet

        # Create a dictionary of exec_globals to pass to `exec`
        exec_globals = {
            'immutabledict': immutabledict,
            'dist': dist
        }

        exec_locals = {}
  
        try:
            # Dynamically execute the code snippet
            exec(code, exec_globals, exec_locals)

            # Check if 'get_default_priors' is defined
            if 'get_default_priors' not in exec_locals:
                raise AssertionError("'get_default_priors' not found in exec_locals after execution.")

            get_default_priors = exec_locals['get_default_priors']

            # Call the function and verify its output
            priors = get_default_priors()

            # Expected keys that should be in the priors
            expected_keys = [
                '_INTERCEPT', '_COEF_TREND', '_EXPO_TREND', '_SIGMA',
                '_GAMMA_SEASONALITY', '_WEEKDAY', '_COEF_EXTRA_FEATURES', '_COEF_SEASONALITY'
            ]

            # Check that all expected keys are in the resulting priors
            for key in expected_keys:
                self.assertIn(key, priors, f"Missing expected key: {key}")

            # Check that each key returns the correct distribution type
            self.assertIsInstance(priors['_INTERCEPT'], dist.HalfNormal)
            self.assertIsInstance(priors['_COEF_TREND'], dist.Normal)
            self.assertIsInstance(priors['_EXPO_TREND'], dist.Uniform)
            self.assertIsInstance(priors['_SIGMA'], dist.Gamma)

            print(f"Code snippet: PASSED all assertions.\n")
            passed = True

        except Exception as e:
            print(f"Code snippet: FAILED with error: {e}\n")
  
        if passed:
            results.append({
                "function_name": "_get_default_priors",
                "code": code,
                "result": "passed"
            })
        else:
            results.append({
                "function_name": "_get_default_priors",
                "code": code,
                "result": "failed"
            })

        # ============= Save results to test_result.jsonl =============
        existing_records = []
        if os.path.exists(TEST_RESULT_JSONL):
            with open(TEST_RESULT_JSONL, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    existing_records.append(json.loads(line))

        # Remove old records for "_get_default_priors"
        existing_records = [
            rec for rec in existing_records
            if rec.get("function_name") != "_get_default_priors"
        ]

        # Append the new results
        existing_records.extend(results)

        # Write back to test_result.jsonl
        with open(TEST_RESULT_JSONL, "w", encoding="utf-8") as f:
            for record in existing_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        print("Results have been written to test_result.jsonl")

if __name__ == "__main__":
    unittest.main()
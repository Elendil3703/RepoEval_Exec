import unittest
import json
import os
from typing import Tuple
from scipy import optimize

TEST_RESULT_JSONL = "test_result.jsonl"

class TestGetBetaParamsFromMuSigma(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippet = data[339]  # Get the 340th JSON element (index 339)
        if len(cls.code_snippet) < 1:
            raise ValueError("Expected at least one code snippet in the selected JSON array")

    def test_get_beta_params_from_mu_sigma(self):
        """Dynamically test 'get_beta_params_from_mu_sigma' against various scenarios."""
        results = []  # Collect test results to write to JSONL

        # Prepare the environment for execution
        exec_globals = {
            'optimize': optimize,
            'Tuple': Tuple,
        }

        # Execute the code snippet
        exec(self.code_snippet, exec_globals)

        # Validate the function exists
        func_name = 'get_beta_params_from_mu_sigma'
        if func_name not in exec_globals:
            print(f"FAILED: Function '{func_name}' not found in executed code snippet.\n")
            results.append({
                "function_name": func_name,
                "code": self.code_snippet,
                "result": "failed"
            })
            self.fail(f"Function '{func_name}' was not found in the executed code snippet.")

        get_beta_params_from_mu_sigma = exec_globals[func_name]

        # Test cases: (mu, sigma, expected_a, expected_b)
        test_cases = [
            (0.5, 0.1, 4, 4),  # Symmetric beta distribution
            (0.2, 0.05, 1, 4),  # Skewed beta distribution
            (0.8, 0.05, 4, 1),  # Skewed beta distribution
        ]

        for i, (mu, sigma, expected_a, expected_b) in enumerate(test_cases):
            with self.subTest(mu=mu, sigma=sigma):
                try:
                    a, b = get_beta_params_from_mu_sigma(mu, sigma)
                    self.assertAlmostEqual(a, expected_a, places=0, msg=f"Test case {i} failed for parameter 'a'.")
                    self.assertAlmostEqual(b, expected_b, places=0, msg=f"Test case {i} failed for parameter 'b'.")
                    print(f"Test case {i}: PASSED with a={a}, b={b}.\n")
                    results.append({
                        "function_name": func_name,
                        "code": self.code_snippet,
                        "result": "passed"
                    })
                except Exception as e:
                    print(f"Test case {i}: FAILED with error: {e}\n")
                    results.append({
                        "function_name": func_name,
                        "code": self.code_snippet,
                        "result": "failed"
                    })

        # ============= Write test results to test_result.jsonl =============
        # Read existing test_result.jsonl (if exists)
        existing_records = []
        if os.path.exists(TEST_RESULT_JSONL):
            with open(TEST_RESULT_JSONL, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    existing_records.append(json.loads(line))

        # Delete old records for this function
        existing_records = [
            rec for rec in existing_records
            if rec.get("function_name") != func_name
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
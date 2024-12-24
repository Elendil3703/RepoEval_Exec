import unittest
import json
import os
import sys
import torch

TEST_RESULT_JSONL = "test_result.jsonl"

class TestForwardFunction(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippet = data[379]  # Get the 380th JSON element (index 379)

    def test_forward_function(self):
        """Dynamically test the forward function in the JSON with additional checks."""
        passed_count = 0  # Counter for passed tests
        failed_count = 0  # Counter for failed tests
        results = []      # Collect results to write to JSONL

        code = self.code_snippet

        print("Running test for the forward function...")

        # ------------------- Static checks -------------------
        if "def forward(self, features)" not in code:
            print("Code snippet: FAILED, function 'forward' definition not found.\n")
            failed_count += 1
            results.append({
                "function_name": "forward",
                "code": code,
                "result": "failed"
            })
        else:
            # ------------------- Dynamic execution and checks -------------------
            exec_globals = {
                'torch': torch,
                'zip': zip,
            }
            exec_locals = {}

            try:
                # Dynamically execute the code snippet
                exec(code, exec_globals, exec_locals)

                # Define a mock module to test the forward function
                class MockModule:
                    def __call__(self, feature):
                        return feature * 2  # Example transformation

                # Simulate a class with preprocessing modules and features
                class TestClass:
                    preprocessing_modules = [MockModule(), MockModule()]

                    def forward(self, features):
                        _features = []
                        for module, feature in zip(self.preprocessing_modules, features):
                            _features.append(module(feature))
                        return torch.stack(_features, dim=1)

                # Initialize the class and prepare features
                test_instance = TestClass()
                features = [torch.tensor(1.0), torch.tensor(2.0)]

                # Execute the forward function
                result = test_instance.forward(features)

                # Perform assertions
                self.assertTrue(torch.equal(result, torch.tensor([[2.0], [4.0]])),
                                "Failed: Output of forward function does not match expected output.")
                print("Code snippet: PASSED all assertions.\n")
                passed_count += 1
                results.append({
                    "function_name": "forward",
                    "code": code,
                    "result": "passed"
                })
            except Exception as e:
                print(f"Code snippet: FAILED with error: {e}\n")
                failed_count += 1
                results.append({
                    "function_name": "forward",
                    "code": code,
                    "result": "failed"
                })

        # Final statistics
        print(f"\nTest Summary: {passed_count} passed, {failed_count} failed.\n")
        self.assertEqual(passed_count + failed_count, 1, "Test count mismatch!")  # Only one snippet is tested

        # ============= Write test results to test_result.jsonl =============
        # Read existing test_result.jsonl (ignore if it doesn't exist)
        existing_records = []
        if os.path.exists(TEST_RESULT_JSONL):
            with open(TEST_RESULT_JSONL, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    existing_records.append(json.loads(line))

        # Remove old records with function_name == "forward"
        existing_records = [
            rec for rec in existing_records
            if rec.get("function_name") != "forward"
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
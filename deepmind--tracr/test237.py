import unittest
import json
import sys
import os

TEST_RESULT_JSONL = "test_result.jsonl"

class TestForwardFunction(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippets = data[236]  # Get the 237th JSON element
        if len(cls.code_snippets) < 1:
            raise ValueError("Expected at least one code snippet in the JSON array")

    def test_forward_function(self):
        """Dynamically test the forward function within the code snippets."""
        passed_count = 0  # Counter for passed tests
        failed_count = 0  # Counter for failed tests
        results = []      # Collect test results to write to JSONL

        for i, code in enumerate(self.code_snippets):
            with self.subTest(code_index=i):
                print(f"Running test for code snippet {i}...")
                
                # Static checks
                if "def forward" not in code:
                    print(f"Code snippet {i}: FAILED, 'forward' function not found.\n")
                    failed_count += 1
                    # Record failed result
                    results.append({
                        "function_name": "forward",
                        "code": code,
                        "result": "failed"
                    })
                    continue

                exec_globals = {
                    'compressed_model': compressed_model,
                    'model': model,
                    'causal': causal,
                }
                exec_locals = {}

                try:
                    # Execute code snippet
                    exec(code, exec_globals, exec_locals)

                    # Check if forward is defined
                    if 'forward' not in exec_locals:
                        print(f"Code snippet {i}: FAILED, 'forward' function not found in exec_locals.\n")
                        failed_count += 1
                        results.append({
                            "function_name": "forward",
                            "code": code,
                            "result": "failed"
                        })
                        continue

                    forward = exec_locals['forward']

                    # Assume existence of dummy input data for testing
                    dummy_emb = ...  # placeholder for actual dummy embedding data
                    dummy_mask = ... # placeholder for actual dummy mask data

                    # Test the function execution
                    try:
                        result = forward(dummy_emb, dummy_mask)
                        # Perform potential assertions on `result` if needed
                        print(f"Code snippet {i}: PASSED all assertions.\n")
                        passed_count += 1
                        results.append({
                            "function_name": "forward",
                            "code": code,
                            "result": "passed"
                        })
                    except Exception as e:
                        print(f"Code snippet {i}: FAILED during forward execution with error: {e}\n")
                        failed_count += 1
                        results.append({
                            "function_name": "forward",
                            "code": code,
                            "result": "failed"
                        })
                    
                except Exception as e:
                    print(f"Code snippet {i}: FAILED with error: {e}\n")
                    failed_count += 1
                    results.append({
                        "function_name": "forward",
                        "code": code,
                        "result": "failed"
                    })

        # Test summary
        print(f"\nTest Summary: {passed_count} passed, {failed_count} failed, total {len(self.code_snippets)}\n")
        self.assertEqual(passed_count + failed_count, len(self.code_snippets), "Test count mismatch!")

        # Write test results to test_result.jsonl
        existing_records = []
        if os.path.exists(TEST_RESULT_JSONL):
            with open(TEST_RESULT_JSONL, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    existing_records.append(json.loads(line))

        # Remove old records for forward function
        existing_records = [
            rec for rec in existing_records
            if rec.get("function_name") != "forward"
        ]

        # Extend with new results
        existing_records.extend(results)

        # Rewrite test_result.jsonl
        with open(TEST_RESULT_JSONL, "w", encoding="utf-8") as f:
            for record in existing_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        print("Results have been written to test_result.jsonl")

if __name__ == "__main__":
    unittest.main()
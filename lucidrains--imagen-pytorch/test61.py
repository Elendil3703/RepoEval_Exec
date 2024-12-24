import unittest
import json
import os

TEST_RESULT_JSONL = "test_result.jsonl"

class TestGroundTruthInitializer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippets = data[60]  # Get the 61st JSON element
        
        if len(cls.code_snippets) < 1:
            raise ValueError("Expected at least one code snippet in the 61st JSON array")
    
    def test_initializers(self):
        """Test all initializer code snippets with specific checks."""
        passed_count = 0  # Counter for passed tests
        failed_count = 0  # Counter for failed tests
        results = []      # Collect results for JSONL

        for i, code in enumerate(self.code_snippets):
            with self.subTest(code_index=i):
                print(f"Running test for code snippet {i}...")
                
                # Check for '__init__' presence
                if "def __init__(" not in code:
                    print(f"Code snippet {i}: FAILED, '__init__' not found in code.\n")
                    failed_count += 1
                    results.append({
                        "function_name": "__init__",
                        "code": code,
                        "result": "failed"
                    })
                    continue

                exec_globals = {}
                exec_locals = {}

                try:
                    # Execute the snippet to check for errors
                    exec(code, exec_globals, exec_locals)

                    # Instantiate objects with different noise schedules
                    for noise_schedule in ["linear", "cosine"]:
                        instance = exec_locals['MyClass'](noise_schedule=noise_schedule)
                        self.assertTrue(
                            hasattr(instance, 'log_snr'),
                            f"Code snippet {i}: object does not have attribute 'log_snr'."
                        )
                        self.assertTrue(
                            hasattr(instance, 'num_timesteps'),
                            f"Code snippet {i}: object does not have 'num_timesteps'."
                        )
                        self.assertEqual(
                            instance.num_timesteps, 1000,
                            f"Code snippet {i}: 'num_timesteps' is not correctly set to 1000."
                        )

                    try:
                        # Test with invalid noise schedule
                        exec_locals['MyClass'](noise_schedule="invalid")
                        print(f"Code snippet {i}: FAILED, exception not raised for invalid noise schedule.\n")
                        failed_count += 1
                        results.append({
                            "function_name": "__init__",
                            "code": code,
                            "result": "failed"
                        })
                        continue
                    except ValueError as e:
                        if "invalid noise schedule" not in str(e):
                            raise

                    print(f"Code snippet {i}: PASSED all assertions.\n")
                    passed_count += 1
                    results.append({
                        "function_name": "__init__",
                        "code": code,
                        "result": "passed"
                    })
                except Exception as e:
                    print(f"Code snippet {i}: FAILED with error: {e}\n")
                    failed_count += 1
                    results.append({
                        "function_name": "__init__",
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

        existing_records = [
            rec for rec in existing_records
            if rec.get("function_name") != "__init__"
        ]

        existing_records.extend(results)

        with open(TEST_RESULT_JSONL, "w", encoding="utf-8") as f:
            for record in existing_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        print("Results have been written to test_result.jsonl")

if __name__ == "__main__":
    unittest.main()
import unittest
import json
import os

TEST_RESULT_JSONL = "test_result.jsonl"

class TestIsDefaultFp16Result(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippets = data[270]  # Get the 271st JSON element (index 270)
        if len(cls.code_snippets) < 1:
            raise ValueError("Expected at least one code snippet in the JSON")

    def test_is_default_fp16(self):
        """Test is_default_fp16 in each code snippet."""
        passed_count = 0  # Counter for passed tests
        failed_count = 0  # Counter for failed tests
        results = []      # Store results to write to JSONL

        for i, code in enumerate(self.code_snippets):
            with self.subTest(code_index=i):
                print(f"Running test for code snippet {i}...")
                
                # Check if the function definition exists
                if "def is_default_fp16" not in code:
                    print(f"Code snippet {i}: FAILED, 'is_default_fp16' not found in code.\n")
                    failed_count += 1
                    results.append({
                        "function_name": "is_default_fp16",
                        "code": code,
                        "result": "failed"
                    })
                    continue

                # Dynamic execution
                exec_globals = {}
                exec_locals = {}

                try:
                    exec(code, exec_globals, exec_locals)

                    # Check if 'is_default_fp16' exists in the executed code
                    if 'is_default_fp16' not in exec_locals:
                        print(f"Code snippet {i}: FAILED, 'is_default_fp16' not found in exec_locals.\n")
                        failed_count += 1
                        results.append({
                            "function_name": "is_default_fp16",
                            "code": code,
                            "result": "failed"
                        })
                        continue

                    # Test function behavior
                    class TestFP16:
                        def __init__(self, fp16, strategy):
                            self._fp16 = fp16
                            self._strategy = strategy

                        is_default_fp16 = exec_locals['is_default_fp16']
                    
                    # Test cases
                    instance1 = TestFP16(False, 'none')
                    instance2 = TestFP16(True, 'none')
                    instance3 = TestFP16(True, 'accelerate')
                    
                    # Assertions
                    self.assertFalse(instance1.is_default_fp16(), f"Code snippet {i} failed case 1")
                    self.assertTrue(instance2.is_default_fp16(), f"Code snippet {i} failed case 2")
                    self.assertFalse(instance3.is_default_fp16(), f"Code snippet {i} failed case 3")

                    print(f"Code snippet {i}: PASSED all assertions.\n")
                    passed_count += 1
                    results.append({
                        "function_name": "is_default_fp16",
                        "code": code,
                        "result": "passed"
                    })

                except Exception as e:
                    print(f"Code snippet {i}: FAILED with error: {e}\n")
                    failed_count += 1
                    results.append({
                        "function_name": "is_default_fp16",
                        "code": code,
                        "result": "failed"
                    })

        # Summary
        print(f"\nTest Summary: {passed_count} passed, {failed_count} failed, total {len(self.code_snippets)}\n")
        self.assertEqual(passed_count + failed_count, len(self.code_snippets), "Test count mismatch!")

        # Write results to test_result.jsonl
        existing_records = []
        if os.path.exists(TEST_RESULT_JSONL):
            with open(TEST_RESULT_JSONL, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    existing_records.append(json.loads(line))

        # Remove old entries with function_name == "is_default_fp16"
        existing_records = [
            rec for rec in existing_records
            if rec.get("function_name") != "is_default_fp16"
        ]

        # Append the new results
        existing_records.extend(results)

        # Rewrite the test_result.jsonl
        with open(TEST_RESULT_JSONL, "w", encoding="utf-8") as f:
            for record in existing_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        print("Results have been written to test_result.jsonl")

if __name__ == "__main__":
    unittest.main()
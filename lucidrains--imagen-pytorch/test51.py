import unittest
import json
import os

TEST_RESULT_JSONL = "test_result.jsonl"

class TestInnerFunctionality(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippets = data[50]  # Get the 51st JSON element
        if len(cls.code_snippets) < 1:
            raise ValueError("Expected at least one code snippet in the corresponding JSON array")

    def test_inner_functionality(self):
        """Dynamically test all code snippets for 'inner' function logic."""
        passed_count = 0  # Counter for passed tests
        failed_count = 0  # Counter for failed tests
        results = []      # To collect test results for JSONL writing

        for i, code in enumerate(self.code_snippets):
            with self.subTest(code_index=i):
                print(f"Running test for code snippet {i}...")
                
                # Check that "def inner" is in the code snippet
                if "def inner" not in code:
                    print(f"Code snippet {i}: FAILED, 'inner' function not found.\n")
                    failed_count += 1
                    results.append({
                        "function_name": "inner",
                        "code": code,
                        "result": "failed"
                    })
                    continue

                # Dynamic execution preparation
                exec_globals = {}
                exec_locals = {}

                try:
                    # Prepare dummy environment
                    called = False

                    def dummy_fn(x):
                        return f"Processed {x}"

                    exec_globals.update({
                        'called': called,
                        'fn': dummy_fn,
                    })
                    
                    # Dynamically execute code snippet
                    exec(code, exec_globals, exec_locals)

                    # Check if 'inner' is in the executed locals
                    if 'inner' not in exec_locals:
                        print(f"Code snippet {i}: FAILED, 'inner' not found in exec_locals.\n")
                        failed_count += 1
                        results.append({
                            "function_name": "inner",
                            "code": code,
                            "result": "failed"
                        })
                        continue

                    # Test the 'inner' function
                    inner_func = exec_locals['inner']

                    self.assertFalse(exec_globals['called'], 
                                     "Precondition 'called' should be False")
                    
                    result = inner_func('input data')

                    self.assertTrue(exec_globals['called'], 
                                    "Postcondition 'called' should be True after first call")
                    
                    self.assertEqual(result, "Processed input data",
                                     f"Code snippet {i} failed to process input correctly.")

                    # Call again and should return None
                    second_result = inner_func('input data 2')
                    self.assertIsNone(second_result, 
                                      "Subsequent calls should return None.")

                    print(f"Code snippet {i}: PASSED all assertions.\n")
                    passed_count += 1
                    results.append({
                        "function_name": "inner",
                        "code": code,
                        "result": "passed"
                    })
                except Exception as e:
                    print(f"Code snippet {i}: FAILED with error: {e}\n")
                    failed_count += 1
                    results.append({
                        "function_name": "inner",
                        "code": code,
                        "result": "failed"
                    })

        # Summary of tests
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

        existing_records = [
            rec for rec in existing_records
            if rec.get("function_name") != "inner"
        ]

        existing_records.extend(results)

        with open(TEST_RESULT_JSONL, "w", encoding="utf-8") as f:
            for record in existing_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        print("Results have been written to test_result.jsonl")


if __name__ == "__main__":
    unittest.main()
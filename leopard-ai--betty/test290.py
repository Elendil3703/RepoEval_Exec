import unittest
import json
import os

TEST_RESULT_JSONL = "test_result.jsonl"

class TestMakeDataLoader(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        # Get the 290th JSON element (index 289)
        cls.code_snippets = data[289]
        if len(cls.code_snippets) < 1:
            raise ValueError("Expected at least one code snippet in the 290th JSON array")

    def test_make_data_loader(self):
        """Dynamically test all code snippets for make_data_loader function."""
        passed_count = 0  # Counter for passed tests
        failed_count = 0  # Counter for failed tests
        results = []      # Collect results to write to JSONL

        for i, code in enumerate(self.code_snippets):
            with self.subTest(code_index=i):
                print(f"Running test for code snippet {i}...")

                # ------------------- Static Check -------------------
                if "def make_data_loader" not in code:
                    print(f"Code snippet {i}: FAILED, function 'make_data_loader' not found.\n")
                    failed_count += 1
                    results.append({
                        "function_name": "make_data_loader",
                        "code": code,
                        "result": "failed"
                    })
                    continue

                # ------------------- Dynamic Execution -------------------
                exec_globals = {}
                exec_locals = {}

                try:
                    # Execute the code snippet
                    exec(code, exec_globals, exec_locals)

                    # Ensure the make_data_loader function exists
                    if "make_data_loader" not in exec_locals:
                        print(f"Code snippet {i}: FAILED, 'make_data_loader' not found in exec_locals.\n")
                        failed_count += 1
                        results.append({
                            "function_name": "make_data_loader",
                            "code": code,
                            "result": "failed"
                        })
                        continue

                    # Retrieve the function
                    make_data_loader = exec_locals["make_data_loader"]

                    # Test the function with sample input
                    xs = [1, 2, 3]
                    ys = [4, 5, 6]
                    expected_output = [(xs, ys)]
                    output = make_data_loader(xs, ys)

                    # Assertions
                    self.assertEqual(output, expected_output, f"Code snippet {i}: Output mismatch.")

                    print(f"Code snippet {i}: PASSED all assertions.\n")
                    passed_count += 1
                    results.append({
                        "function_name": "make_data_loader",
                        "code": code,
                        "result": "passed"
                    })
                except Exception as e:
                    print(f"Code snippet {i}: FAILED with error: {e}\n")
                    failed_count += 1
                    results.append({
                        "function_name": "make_data_loader",
                        "code": code,
                        "result": "failed"
                    })

        # Summary
        print(f"\nTest Summary: {passed_count} passed, {failed_count} failed, total {len(self.code_snippets)}\n")
        self.assertEqual(passed_count + failed_count, len(self.code_snippets), "Test count mismatch!")

        # ============= Write Results to test_result.jsonl =============
        existing_records = []

        if os.path.exists(TEST_RESULT_JSONL):
            with open(TEST_RESULT_JSONL, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    existing_records.append(json.loads(line))

        existing_records = [rec for rec in existing_records if rec.get("function_name") != "make_data_loader"]
        existing_records.extend(results)

        with open(TEST_RESULT_JSONL, "w", encoding="utf-8") as f:
            for record in existing_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        print("Results have been written to test_result.jsonl")

if __name__ == "__main__":
    unittest.main()
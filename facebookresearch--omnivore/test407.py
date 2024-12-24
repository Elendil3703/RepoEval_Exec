import unittest
import json
import os

TEST_RESULT_JSONL = "test_result.jsonl"

class TestGetFullParameterName(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippets = data[406]  # Get the 407th JSON element

        if len(cls.code_snippets) < 1:
            raise ValueError("Expected at least one code snippet in the 407th JSON array")

    def test_code_snippets(self):
        """Dynamically test all code snippets in the JSON to verify `get_full_parameter_name`."""
        passed_count = 0  # Counter for passed tests
        failed_count = 0  # Counter for failed tests
        results = []      # Collect test results to write into JSONL

        for i, code in enumerate(self.code_snippets):
            with self.subTest(code_index=i):
                print(f"Running test for code snippet {i}...")

                # Check if 'get_full_parameter_name' is in the code
                if "def get_full_parameter_name" not in code:
                    print(f"Code snippet {i}: FAILED, function 'get_full_parameter_name' not found.\n")
                    failed_count += 1
                    results.append({
                        "function_name": "get_full_parameter_name",
                        "code": code,
                        "result": "failed"
                    })
                    continue

                exec_globals = {}
                exec_locals = {}

                try:
                    # Dynamically execute code snippet
                    exec(code, exec_globals, exec_locals)

                    # Check if get_full_parameter_name is actually defined
                    if 'get_full_parameter_name' not in exec_locals:
                        print(f"Code snippet {i}: FAILED, 'get_full_parameter_name' not found in exec_locals.\n")
                        failed_count += 1
                        results.append({
                            "function_name": "get_full_parameter_name",
                            "code": code,
                            "result": "failed"
                        })
                        continue

                    # Retrieve the function for testing
                    get_full_parameter_name = exec_locals['get_full_parameter_name']

                    # Test cases
                    self.assertEqual(get_full_parameter_name("", "weight"), "weight")
                    self.assertEqual(get_full_parameter_name("module", "weight"), "module.weight")
                    self.assertEqual(get_full_parameter_name("layer1.block2", "bias"), "layer1.block2.bias")
                    self.assertEqual(get_full_parameter_name("net", ""), "net.")
                
                    print(f"Code snippet {i}: PASSED all assertions.\n")
                    passed_count += 1
                    results.append({
                        "function_name": "get_full_parameter_name",
                        "code": code,
                        "result": "passed"
                    })
                except Exception as e:
                    print(f"Code snippet {i}: FAILED with error: {e}\n")
                    failed_count += 1
                    results.append({
                        "function_name": "get_full_parameter_name",
                        "code": code,
                        "result": "failed"
                    })

        # Final summary
        print(f"\nTest Summary: {passed_count} passed, {failed_count} failed, total {len(self.code_snippets)}\n")
        self.assertEqual(passed_count + failed_count, len(self.code_snippets), "Test count mismatch!")

        # Writing test results to test_result.jsonl
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
            if rec.get("function_name") != "get_full_parameter_name"
        ]

        existing_records.extend(results)

        with open(TEST_RESULT_JSONL, "w", encoding="utf-8") as f:
            for record in existing_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        print("Results have been written to test_result.jsonl")

if __name__ == "__main__":
    unittest.main()
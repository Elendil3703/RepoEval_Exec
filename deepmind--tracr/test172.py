import unittest
import json
import sys
import os

TEST_RESULT_JSONL = "test_result.jsonl"

class TestCarperAITrlxResult(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippets = data[171]  # Get the 172nd JSON element (index 171)
        if len(cls.code_snippets) < 1:
            raise ValueError("Expected at least one code snippet in the 172nd JSON array")

    def test_code_snippets(self):
        """Test the make_length function in all code snippets."""
        passed_count = 0  # Counter for passed tests
        failed_count = 0  # Counter for failed tests
        results = []      # Collect results to write to JSONL

        for i, code in enumerate(self.code_snippets):
            with self.subTest(code_index=i):
                print(f"Running test for code snippet {i}...")

                # Perform some static checks on the code snippet.
                if "def make_length(" not in code:
                    print(f"Code snippet {i}: FAILED, 'make_length' function not defined correctly.\n")
                    failed_count += 1
                    # Writing failure record
                    results.append({
                        "function_name": "make_length",
                        "code": code,
                        "result": "failed"
                    })
                    continue

                # Execute and test the snippet dynamically.
                exec_globals = {
                    'rasp': sys.modules.get('rasp', MockRaspModule())
                }
                exec_locals = {}

                try:
                    # Dynamically execute the code snippet
                    exec(code, exec_globals, exec_locals)

                    # Check if make_length was defined
                    if 'make_length' not in exec_locals:
                        print(f"Code snippet {i}: FAILED, 'make_length' not found in exec_locals.\n")
                        failed_count += 1
                        results.append({
                            "function_name": "make_length",
                            "code": code,
                            "result": "failed"
                        })
                        continue

                    # Test make_length assuming it returns an object with a 'named' method
                    result = exec_locals['make_length']()
                    self.assertTrue(callable(getattr(result, "named", None)),
                                    f"Code snippet {i}: 'make_length' did not return an object with a 'named' method.\n")

                    print(f"Code snippet {i}: PASSED all assertions.\n")
                    passed_count += 1
                    results.append({
                        "function_name": "make_length",
                        "code": code,
                        "result": "passed"
                    })
                except Exception as e:
                    print(f"Code snippet {i}: FAILED with error: {e}\n")
                    failed_count += 1
                    results.append({
                        "function_name": "make_length",
                        "code": code,
                        "result": "failed"
                    })

        # Print test summary
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
            if rec.get("function_name") != "make_length"
        ]

        existing_records.extend(results)

        with open(TEST_RESULT_JSONL, "w", encoding="utf-8") as f:
            for record in existing_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        print("Results have been written to test_result.jsonl")

class MockRaspModule:
    def Select(self, a, b, c):
        return MockSelector()

    def Comparison(self):
        class TRUE:
            pass
        return TRUE()

class MockSelector:
    def named(self, name):
        return MockSelectorWidth()

    def SelectorWidth(self, selector):
        return MockSelectorWidth()

class MockSelectorWidth:
    def named(self, name):
        return self

if __name__ == "__main__":
    unittest.main()
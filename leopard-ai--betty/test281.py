import unittest
import json
import os

TEST_RESULT_JSONL = "test_result.jsonl"

class TestDFSFunction(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippet = data[280]  # Get the 281st JSON element (index 280)
        if not cls.code_snippet:
            raise ValueError("Expected to find code snippet at index 280")

    def test_dfs_function(self):
        """Test the dfs function extracted from the code snippet."""
        code = self.code_snippet
        passed_count = 0  # Counter for passed tests
        failed_count = 0  # Counter for failed tests
        results = []      # Collecting results for JSONL output

        # Setting up a dummy class to contain the dfs method
        class Dummy:
            def __init__(self, dependencies):
                self.dependencies = dependencies
        
        # Exec the code snippet to register the dfs function in the Dummy class
        exec_locals = {}
        exec(code, {}, exec_locals)
        Dummy.dfs = exec_locals['dfs']

        # Test cases for the dfs method
        test_cases = [
            {
                "desc": "Direct path",
                "dependencies": {"l2u": {1: [2]}},
                "src": 1,
                "dst": 2,
                "expected": [[1, 2]]
            },
            {
                "desc": "No path available",
                "dependencies": {"l2u": {1: [3]}},
                "src": 1,
                "dst": 2,
                "expected": []
            },
            {
                "desc": "Cyclic path",
                "dependencies": {"l2u": {1: [2], 2: [1, 3]}},
                "src": 1,
                "dst": 3,
                "expected": [[1, 2, 3]]
            },
            {
                "desc": "Multiple paths",
                "dependencies": {"l2u": {1: [2, 3], 2: [3]}},
                "src": 1,
                "dst": 3,
                "expected": [[1, 2, 3], [1, 3]]
            }
        ]

        for i, case in enumerate(test_cases):
            with self.subTest(test_case=case["desc"]):
                dummy = Dummy(case["dependencies"])
                path, results_list = [case["src"]], []
                dummy.dfs(case["src"], case["dst"], path, results_list)
                try:
                    self.assertEqual(results_list, case["expected"], case["desc"])
                    print(f"Test case '{case['desc']}': PASSED.")
                    passed_count += 1
                    results.append({
                        "function_name": "dfs",
                        "case": case["desc"],
                        "result": "passed"
                    })
                except AssertionError as e:
                    print(f"Test case '{case['desc']}': FAILED with error: {e}")
                    failed_count += 1
                    results.append({
                        "function_name": "dfs",
                        "case": case["desc"],
                        "result": "failed"
                    })

        # Final statistics
        print(f"\nTest Summary: {passed_count} passed, {failed_count} failed, total {len(test_cases)} tests\n")

        # Write the results to test_result.jsonl
        self._write_results_to_jsonl(results)

    def _write_results_to_jsonl(self, new_results):
        # Read existing test_result.jsonl (if it exists)
        existing_records = []
        if os.path.exists(TEST_RESULT_JSONL):
            with open(TEST_RESULT_JSONL, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        existing_records.append(json.loads(line))

        # Remove old records with function_name == "dfs"
        existing_records = [
            rec for rec in existing_records
            if rec.get("function_name") != "dfs"
        ]

        # Append new results
        existing_records.extend(new_results)

        # Write results back to test_result.jsonl
        with open(TEST_RESULT_JSONL, "w", encoding="utf-8") as f:
            for record in existing_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    unittest.main()
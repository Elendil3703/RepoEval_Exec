import unittest
import json
import os

TEST_RESULT_JSONL = "test_result.jsonl"

class MockProblem:
    def __init__(self, name):
        self.name = name

class TestFindPathsResult(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippets = data[279]
        if len(cls.code_snippets) < 1:
            raise ValueError("Expected at least one code snippet in the JSON array")

    def test_find_paths(self):
        """Dynamically test all code snippets in the JSON for find_paths function."""
        passed_count = 0  # Counter for passed tests
        failed_count = 0  # Counter for failed tests
        results = []      # To collect results for JSONL

        for i, code in enumerate(self.code_snippets):
            with self.subTest(code_index=i):
                print(f"Running test for code snippet {i}...")

                # ------------------- Static Checks for 'find_paths' -------------------
                if "def find_paths" not in code:
                    print(f"Code snippet {i}: FAILED, function 'find_paths' not found in code.\n")
                    failed_count += 1
                    results.append({
                        "function_name": "find_paths",
                        "code": code,
                        "result": "failed"
                    })
                    continue

                # We'll assume 'dfs' method is required since it's used in the find_paths function

                # ------------------- Dynamic Tests -------------------
                exec_globals = {
                    'MockProblem': MockProblem
                }
                exec_locals = {}

                try:
                    # Dynamically execute the code snippet
                    exec(code, exec_globals, exec_locals)

                    if 'find_paths' not in exec_locals:
                        print(f"Code snippet {i}: FAILED, 'find_paths' not defined after exec.\n")
                        failed_count += 1
                        results.append({
                            "function_name": "find_paths",
                            "code": code,
                            "result": "failed"
                        })
                        continue

                    # Setup a mock class with the required dfs method
                    class MockGraph:
                        def dfs(self, src, dst, path, results):
                            # Simulate a trivial path for testing
                            if src.name == dst.name:
                                results.append(path[:])
                            else:
                                # Simply simulate path finding by adding a direct path
                                path.append(dst)
                                results.append(path[:])
                                path.pop()

                        find_paths = exec_locals['find_paths']

                    # Define test cases
                    src = MockProblem('A')
                    dst = MockProblem('B')

                    # Instantiate and call the function
                    graph = MockGraph()
                    paths = graph.find_paths(src, dst)

                    # Assert that the paths returned are non-empty
                    self.assertTrue(paths, f"Code snippet {i} returned no paths.")
                    
                    # Example simplistic check: The first path should end with the destination node
                    self.assertEqual(paths[0][-1].name, dst.name)
                    
                    print(f"Code snippet {i}: PASSED all assertions.\n")
                    passed_count += 1
                    results.append({
                        "function_name": "find_paths",
                        "code": code,
                        "result": "passed"
                    })
                except Exception as e:
                    print(f"Code snippet {i}: FAILED with error: {e}\n")
                    failed_count += 1
                    results.append({
                        "function_name": "find_paths",
                        "code": code,
                        "result": "failed"
                    })

        # Test Summary
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

        # Remove old records with function_name == "find_paths"
        existing_records = [
            rec for rec in existing_records
            if rec.get("function_name") != "find_paths"
        ]

        # Add new results
        existing_records.extend(results)

        # Rewrite the JSONL file
        with open(TEST_RESULT_JSONL, "w", encoding="utf-8") as f:
            for record in existing_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        print("Results have been written to test_result.jsonl")

if __name__ == "__main__":
    unittest.main()
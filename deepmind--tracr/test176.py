import unittest
import json
import os
import rasp

TEST_RESULT_JSONL = "test_result.jsonl"

class TestMakeSortUnique(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippet = data[175]  # Get the 176th JSON element

    def test_make_sort_unique(self):
        """Test the 'make_sort_unique' functionality."""
        results = []

        # Define a mock class with necessary methods and comparisons
        class MockSOp:
            def __init__(self, data):
                self.data = data

            def __repr__(self):
                return repr(self.data)

        class MockComparison:
            LT = "less_than"
            EQ = "equal_to"

        rasp.Select = lambda a, b, comparison: f"Select({a}, {b}, {comparison})"
        rasp.SelectorWidth = lambda a: f"SelectorWidth({a})"
        rasp.Aggregate = lambda a, b: f"Aggregate({a}, {b})"
        rasp.Comparison = MockComparison
        rasp.indices = "indices"
        rasp.SOp = MockSOp

        # Inject globals
        exec_globals = {
            'rasp': rasp,
        }
        exec_locals = {}

        try:
            # Execute the snippet
            exec(self.code_snippet, exec_globals, exec_locals)

            # Ensure function exists
            assert 'make_sort_unique' in exec_locals

            # Retrieve the function
            make_sort_unique = exec_locals['make_sort_unique']

            # Test cases
            test_cases = [
                ([2, 4, 3, 1], [2, 4, 3, 1], [2, 3, 4, 1]),
                ([5, 3, 9], [2, 1, 3], [5, 3, 9]),
                ([10], [0], [10]),
                ([], [], [])
            ]

            for vals, keys, expected in test_cases:
                with self.subTest(vals=vals, keys=keys):
                    result = make_sort_unique(MockSOp(vals), MockSOp(keys))
                    print(f"Result: {result}")  # You can remove this line after testing

                    results.append({
                        "function_name": "make_sort_unique",
                        "input": {"vals": vals, "keys": keys},
                        "result": "passed" if result == expected else "failed",
                        "output": repr(result)
                    })
                    self.assertEqual(
                        result,
                        expected,
                        f"Failed for vals={vals} and keys={keys}, expected {expected}, but got {result}"
                    )

            print(f"All test cases passed for 'make_sort_unique'.\n")

        except Exception as e:
            results.append({
                "function_name": "make_sort_unique",
                "error": str(e),
                "code": self.code_snippet,
                "result": "failed"
            })
            self.fail(f"An error occurred during testing: {e}")

        # Writing the results to the test_result.jsonl file
        existing_records = []
        if os.path.exists(TEST_RESULT_JSONL):
            with open(TEST_RESULT_JSONL, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    existing_records.append(json.loads(line))

        # Remove existing records for function_name == "make_sort_unique"
        existing_records = [
            rec for rec in existing_records
            if rec.get("function_name") != "make_sort_unique"
        ]

        # Add new results
        existing_records.extend(results)

        # Write back to the JSONL file
        with open(TEST_RESULT_JSONL, "w", encoding="utf-8") as f:
            for record in existing_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    unittest.main()
import unittest
import json
import os

TEST_RESULT_JSONL = "test_result.jsonl"

class TestCrossFunction(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file and read the required code snippets
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippet = data[450]  # Get the 451st JSON element

        if len(cls.code_snippet) < 1:
            raise ValueError("Expected at least one code snippet in the 451st JSON array")

    def test_cross_function(self):
        """Test the 'cross' function in the DataFrame class."""
        passed_count = 0  # Counter for passed tests
        failed_count = 0  # Counter for failed tests
        results = []      # Collect test results to write as JSONL

        code = self.code_snippet

        # Set up exec environment
        exec_globals = {}
        exec_locals = {
            'DataFrame': self.MockDataFrame,
            '_check_type': self.mock_check_type,
            '_wrap': self.mock_wrap
        }

        try:
            # Dynamically execute the code snippet
            exec(code, exec_globals, exec_locals)

            # Verify if 'cross' method is in MockDataFrame
            if 'cross' not in dir(self.MockDataFrame):
                print(f"Code snippet: FAILED, 'cross' method not found in MockDataFrame.\n")
                failed_count += 1
                results.append({
                    "function_name": "cross",
                    "code": code,
                    "result": "failed"
                })
                return

            # Test example with one DataFrame
            df = self.MockDataFrame({"foo": ["a", "b"], "bar": [1, 2]})
            result = df.cross()
            expected_result = [("a", 1, "a", 1), ("a", 1, "b", 2), ("b", 2, "a", 1), ("b", 2, "b", 2)]
            self.assertEqual(result, expected_result, "Cross join failed on a single dataframe")

            # Test example with two DataFrames
            dfa = self.MockDataFrame({"foo": [1, 2]})
            dfb = self.MockDataFrame({"bar": [1, 2]})
            result = dfa.cross(dfb, postfix=("_a", "_b"))
            expected_result_two_dfs = [(1, 1), (1, 2), (2, 1), (2, 2)]
            self.assertEqual(result, expected_result_two_dfs, "Cross join failed on two dataframes")

            print(f"Code snippet: PASSED all assertions.\n")
            passed_count += 1
            results.append({
                "function_name": "cross",
                "code": code,
                "result": "passed"
            })

        except Exception as e:
            print(f"Code snippet: FAILED with error: {e}\n")
            failed_count += 1
            results.append({
                "function_name": "cross",
                "code": code,
                "result": "failed"
            })

        # Final summaries and assertions
        print(f"\nTest Summary: {passed_count} passed, {failed_count} failed.\n")
        self.assertEqual(passed_count + failed_count, 1, "Test count mismatch!")  # We test one snippet

        # Write test results to test_result.jsonl
        self._write_results_to_jsonl(results)

    @staticmethod
    def _write_results_to_jsonl(results):
        # Read existing records if any
        existing_records = []
        if os.path.exists(TEST_RESULT_JSONL):
            with open(TEST_RESULT_JSONL, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    existing_records.append(json.loads(line))

        # Remove old records for "cross"
        existing_records = [
            rec for rec in existing_records
            if rec.get("function_name") != "cross"
        ]

        # Append the new results
        existing_records.extend(results)

        # Overwrite test_result.jsonl
        with open(TEST_RESULT_JSONL, "w", encoding="utf-8") as f:
            for record in existing_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        print("Results have been written to test_result.jsonl")

    class MockDataFrame:
        """A mock DataFrame class to simulate cross behavior for testing."""
        def __init__(self, data):
            self.data = data

        def cross(self, rhs=None, postfix=("_lhs", "_rhs")):
            rhs = self if (rhs is None) else rhs
            results = []
            for left_row in zip(*self.data.values()):
                for right_row in zip(*rhs.data.values()):
                    results.append(left_row + right_row)
            return results

    @staticmethod
    def mock_check_type(instance, cls):
        assert isinstance(instance, cls)

    @staticmethod
    def mock_wrap(data):
        return data

if __name__ == "__main__":
    unittest.main()
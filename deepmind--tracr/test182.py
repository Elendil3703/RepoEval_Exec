import unittest
import json
import os

TEST_RESULT_JSONL = "test_result.jsonl"

class TestMakeCountLessFreq(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the specific JSON code snippet
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippet = data[181]  # Get the 182nd JSON element (index 181)
        if not cls.code_snippet:
            raise ValueError("Expected code snippet at position 181")

    def test_make_count_less_freq(self):
        """Dynamically test the make_count_less_freq function."""
        passed_count = 0
        failed_count = 0
        results = []

        code = self.code_snippet  # The code from the JSON

        # ------------------- Dynamically execute and test the function -------------------
        exec_globals = {
            'rasp': rasp,  # Assuming rasp is a known module in the test context
        }
        exec_locals = {}

        try:
            # Execute the code snippet
            exec(code, exec_globals, exec_locals)

            # Check if make_count_less_freq is defined
            if 'make_count_less_freq' not in exec_locals:
                self.fail("Function 'make_count_less_freq' not defined in the executed code.")

            make_count_less_freq = exec_locals['make_count_less_freq']

            # Examples to test make_count_less_freq
            examples = [
                {
                    'input': (2, ["a", "a", "a", "b", "b", "c"]),
                    'expected': [3, 3, 3, 3, 3, 3]
                },
                {
                    'input': (2, ["a", "a", "c", "b", "b", "c"]),
                    'expected': [6, 6, 6, 6, 6, 6]
                }
            ]

            for i, example in enumerate(examples):
                with self.subTest(example_index=i):
                    n, tokens = example['input']
                    expected_output = example['expected']

                    # Invoke the function
                    result = make_count_less_freq(n).apply(tokens)

                    # Check the result
                    self.assertEqual(result, expected_output, f"Test case {i} failed")
                    passed_count += 1
                    results.append({
                        "function_name": "make_count_less_freq",
                        "input": example['input'],
                        "expected": example['expected'],
                        "result": "passed"
                    })

        except Exception as e:
            print(f"Execution failed with error: {e}")
            failed_count += 1
            results.append({
                "function_name": "make_count_less_freq",
                "code": code,
                "result": "failed"
            })

        # Final assertions
        self.assertEqual(passed_count + failed_count, len(examples), "Test count mismatch!")

        # ============= Write the test results to test_result.jsonl =============
        existing_records = []
        if os.path.exists(TEST_RESULT_JSONL):
            with open(TEST_RESULT_JSONL, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    existing_records.append(json.loads(line))

        # Remove previous records for make_count_less_freq
        existing_records = [
            rec for rec in existing_records
            if rec.get("function_name") != "make_count_less_freq"
        ]

        # Append new results
        existing_records.extend(results)

        # Rewrite test_result.jsonl
        with open(TEST_RESULT_JSONL, "w", encoding="utf-8") as f:
            for record in existing_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    unittest.main()
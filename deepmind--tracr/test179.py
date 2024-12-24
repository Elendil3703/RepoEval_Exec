import unittest
import json
import os
import rasp  # Assuming rasp is a module in the environment

TEST_RESULT_JSONL = "test_result.jsonl"

class TestMakeFracPrevs(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippet = data[178]  # Get the 179th JSON element

    def test_make_frac_prevs(self):
        """"Test the make_frac_prevs function."""
        code = self.code_snippet
        results = []  # Collect results for writing to JSONL

        exec_globals = {
            'rasp': rasp
        }
        exec_locals = {}

        try:
            # Dynamically execute the code snippet
            exec(code, exec_globals, exec_locals)

            # Check if make_frac_prevs is defined
            if 'make_frac_prevs' not in exec_locals:
                raise AssertionError("Function 'make_frac_prevs' not found in the executed code.")

            make_frac_prevs = exec_locals['make_frac_prevs']

            # Test cases for make_frac_prevs function
            input_bool = rasp.tokens == "l"
            expected_output = [0, 0, 1/3, 1/2, 2/5]
            num_l = make_frac_prevs(input_bool)
            result_output = num_l("hello")
            
            self.assertEqual(result_output, expected_output, "Output mismatch.")

            results.append({
                "function_name": "make_frac_prevs",
                "code": code,
                "result": "passed"
            })
            print("Code snippet PASSED all assertions.\n")
        except Exception as e:
            results.append({
                "function_name": "make_frac_prevs",
                "code": code,
                "result": "failed",
                "error": str(e)
            })
            print(f"Code snippet FAILED with error: {e}\n")

        # Write test results into test_result.jsonl
        existing_records = []
        if os.path.exists(TEST_RESULT_JSONL):
            with open(TEST_RESULT_JSONL, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    existing_records.append(json.loads(line))

        # Remove old records for make_frac_prevs
        existing_records = [
            rec for rec in existing_records
            if rec.get("function_name") != "make_frac_prevs"
        ]

        # Append new results
        existing_records.extend(results)

        # Rewrite test_result.jsonl
        with open(TEST_RESULT_JSONL, "w", encoding="utf-8") as f:
            for record in existing_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        print("Results have been written to test_result.jsonl")


if __name__ == "__main__":
    unittest.main()
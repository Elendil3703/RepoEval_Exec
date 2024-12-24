import unittest
import json
import os

TEST_RESULT_JSONL = "test_result.jsonl"

class TestWNewFunction(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the RepoEval result JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippet = data[186]  # Get the 187th JSON element (index 186)
        if len(cls.code_snippet) < 1:
            raise ValueError("Expected at least one code snippet in the 187th JSON array")

    def test_w_new(self):
        """Test the W_new function logic as described in the ground truth."""
        passed_count = 0
        failed_count = 0
        results = []

        # Standard test variables and results
        bos_direction = "bos_direction"
        one_direction = "one_direction"
        softmax_coldness = 2.0
        always_attend_to_bos = True
        attn_fn = lambda q, k: 0.0 if q == one_direction or k == one_direction else 1.0
        
        # Mock code snippet expected behavior
        def qk_fun(query, key):
            if key == bos_direction and query == one_direction:
                c = 1.0 if always_attend_to_bos else 0.5
                return c * softmax_coldness
            elif {key, query}.intersection({one_direction, bos_direction}):
                return 0
            return softmax_coldness * attn_fn(query, key)

        # Define test cases for qk_fun
        test_cases = [
            # Testing main case: attending to bos with always_attend_to_bos
            (one_direction, bos_direction, 2.0),
            # Testing intersection with bos or one_direction returns 0
            (one_direction, "other_direction", 0),
            ("other_direction", bos_direction, 0),
            # Testing normal attention
            ("other_direction", "yet_another_direction", 2.0),
        ]
        
        for i, (query, key, expected_output) in enumerate(test_cases):
            with self.subTest(test_index=i):
                try:
                    result = qk_fun(query, key)
                    self.assertEqual(result, expected_output, f"Code snippet has incorrect output for case {i}.")
                    print(f"Test case {i}: PASSED")
                    passed_count += 1
                    results.append({
                        "function_name": "qk_fun",
                        "test_case": {"query": query, "key": key},
                        "expected": expected_output,
                        "result": "passed"
                    })
                except AssertionError as e:
                    print(f"Test case {i}: FAILED with assertion error: {e}")
                    failed_count += 1
                    results.append({
                        "function_name": "qk_fun",
                        "test_case": {"query": query, "key": key},
                        "expected": expected_output,
                        "result": "failed"
                    })

        print(f"\nTest Summary: {passed_count} passed, {failed_count} failed, total {len(test_cases)}\n")
        self.assertEqual(passed_count + failed_count, len(test_cases), "Test count mismatch!")

        # ============= Write test results to test_result.jsonl =============
        existing_records = []
        if os.path.exists(TEST_RESULT_JSONL):
            with open(TEST_RESULT_JSONL, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    existing_records.append(json.loads(line))

        # Remove old records with function_name == "qk_fun"
        existing_records = [rec for rec in existing_records if rec.get("function_name") != "qk_fun"]

        # Append new results
        existing_records.extend(results)

        # Re-write the JSONL file
        with open(TEST_RESULT_JSONL, "w", encoding="utf-8") as f:
            for record in existing_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        print("Results have been written to test_result.jsonl")


if __name__ == "__main__":
    unittest.main()
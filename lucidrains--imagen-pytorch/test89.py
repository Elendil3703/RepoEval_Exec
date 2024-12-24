import unittest
import json
import os
from typing import Any  # 确保注入的环境中有 Any

TEST_RESULT_JSONL = "test_result.jsonl"

class TestCheckImageSizes(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load and parse the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippets = data[88]  # Get the 89th JSON element (index 88)
        if len(cls.code_snippets) < 1:
            raise ValueError("Expected at least one code snippet in the JSON data")

    def test_check_image_sizes(self):
        """Dynamically test check_image_sizes function from JSON code snippets."""
        passed_count = 0
        failed_count = 0
        results = []

        for i, code in enumerate(self.code_snippets):
            with self.subTest(code_index=i):
                print(f"Running test for code snippet {i}...")

                if "def check_image_sizes" not in code:
                    print(f"Code snippet {i}: FAILED, function 'check_image_sizes' not found.\n")
                    failed_count += 1
                    results.append({
                        "function_name": "check_image_sizes",
                        "code": code,
                        "result": "failed"
                    })
                    continue

                exec_globals = {
                    'ValueError': ValueError,
                }
                try:
                    exec(code, exec_globals)

                    if 'check_image_sizes' not in exec_globals:
                        print(f"Code snippet {i}: FAILED, 'check_image_sizes' not in exec_globals.\n")
                        failed_count += 1
                        results.append({
                            "function_name": "check_image_sizes",
                            "code": code,
                            "result": "failed"
                        })
                        continue

                    check_image_sizes = exec_globals['check_image_sizes']

                    # Test with matching lengths
                    try:
                        result = check_image_sizes(None, [256, 512], {'unets': ['unet1', 'unet2']})
                        self.assertEqual(result, [256, 512], "check_image_sizes returned unexpected result.")
                        passed_count += 1
                        results.append({
                            "function_name": "check_image_sizes",
                            "code": code,
                            "result": "passed"
                        })
                    except Exception as e:
                        print(f"Code snippet {i}: FAILED on matching lengths with error: {e}\n")
                        failed_count += 1
                        results.append({
                            "function_name": "check_image_sizes",
                            "code": code,
                            "result": "failed"
                        })

                    # Test with mismatching lengths
                    with self.assertRaises(ValueError, msg="Expected ValueError for mismatching lengths."):
                        check_image_sizes(None, [256], {'unets': ['unet1', 'unet2']})

                except Exception as e:
                    print(f"Code snippet {i}: FAILED with error: {e}\n")
                    failed_count += 1
                    results.append({
                        "function_name": "check_image_sizes",
                        "code": code,
                        "result": "failed"
                    })

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
            if rec.get("function_name") != "check_image_sizes"
        ]

        existing_records.extend(results)

        with open(TEST_RESULT_JSONL, "w", encoding="utf-8") as f:
            for record in existing_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        print("Results have been written to test_result.jsonl.")

if __name__ == "__main__":
    unittest.main()
import unittest
import json
import os

TEST_RESULT_JSONL = "test_result.jsonl"


class TestSaveFunction(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippets = data[390]  # Get the 391st JSON element
        if len(cls.code_snippets) < 1:
            raise ValueError("Expected at least one code snippet in this JSON array")

    def test_code_snippets(self):
        """Dynamically test all code snippets and verify the save function logic."""
        passed_count = 0  # Counter for passed tests
        failed_count = 0  # Counter for failed tests
        results = []      # Gather test results to write to JSONL

        for i, code in enumerate(self.code_snippets):
            with self.subTest(code_index=i):
                print(f"Running test for code snippet {i}...")

                # Check for expected function signature
                if "def save(" not in code:
                    print(f"Code snippet {i}: FAILED, no 'save' function found.\n")
                    failed_count += 1
                    results.append({
                        "function_name": "save",
                        "code": code,
                        "result": "failed"
                    })
                    continue

                exec_globals = {}
                exec_locals = {}

                try:
                    exec(code, exec_globals, exec_locals)

                    if 'save' not in exec_locals:
                        print(f"Code snippet {i}: FAILED, 'save' not found in exec_locals.\n")
                        failed_count += 1
                        results.append({
                            "function_name": "save",
                            "code": code,
                            "result": "failed"
                        })
                        continue

                    # Create a mock class implementing necessary methods used in the save function
                    class Mock(nn.Module):
                        def __init__(self):
                            super().__init__()

                        def save(self, file_path):
                            return f"Saved model to {file_path}"

                    class MockDetectionMethod:
                        def __init__(self):
                            self.save = Mock().save

                        def _index_file(self, folder, prepend):
                            return f"{folder}/{prepend}_index.file"

                        def _detection_file(self, folder, prepend):
                            return f"{folder}/{prepend}_detection.file"

                        def _save(self, file_path, features):
                            return f"Saved features to {file_path}"

                        def __init__(self):
                            self.nn_method = Mock()
                            self.detection_features = "detection features"

                    mock_instance = MockDetectionMethod()
                    exec_locals['save'](mock_instance, "test_folder", True, "test")

                    print(f"Code snippet {i}: PASSED all checks.\n")
                    passed_count += 1
                    results.append({
                        "function_name": "save",
                        "code": code,
                        "result": "passed"
                    })
                except Exception as e:
                    print(f"Code snippet {i}: FAILED with error: {e}\n")
                    failed_count += 1
                    results.append({
                        "function_name": "save",
                        "code": code,
                        "result": "failed"
                    })

        # Final statistics
        print(f"\nTest Summary: {passed_count} passed, {failed_count} failed, total {len(self.code_snippets)}\n")
        self.assertEqual(passed_count + failed_count, len(self.code_snippets), "Test count mismatch!")

        # Write test results to test_result.jsonl
        existing_records = []
        if os.path.exists(TEST_RESULT_JSONL):
            with open(TEST_RESULT_JSONL, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    existing_records.append(json.loads(line))

        # Remove old records for the 'save' function
        existing_records = [
            rec for rec in existing_records
            if rec.get("function_name") != "save"
        ]

        # Add new results
        existing_records.extend(results)

        # Rewrite test_result.jsonl
        with open(TEST_RESULT_JSONL, "w", encoding="utf-8") as f:
            for record in existing_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        print("Results have been written to test_result.jsonl")

if __name__ == "__main__":
    unittest.main()
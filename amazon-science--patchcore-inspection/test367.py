import unittest
import json
import sys
import re
import os
import pickle
from unittest.mock import MagicMock, patch
from typing import Any

TEST_RESULT_JSONL = "test_result.jsonl"

class TestPatchCoreSaveToPath(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippet = data[366]  # Get the 367th JSON element (index 366)
        if not cls.code_snippet:
            raise ValueError("Expected code snippet data at index 366")

    def test_save_to_path(self):
        """Test the save_to_path function of PatchCore instance."""
        passed_count = 0
        failed_count = 0
        results = []

        code = self.code_snippet
        for i, class_code in enumerate(code):
            with self.subTest(class_index=i):
                print(f"Running test for code snippet {i}...")
                if "def save_to_path" not in class_code:
                    print(f"Code snippet {i}: FAILED, function 'save_to_path' not found.\n")
                    failed_count += 1
                    results.append({
                        "function_name": "save_to_path",
                        "code": class_code,
                        "result": "failed"
                    })
                    continue

                exec_globals = {
                    'MagicMock': MagicMock,
                    'pickle': pickle,
                    'patch': patch
                }
                exec_locals = {}

                try:
                    exec(class_code, exec_globals, exec_locals)

                    # Ensure the class containing save_to_path is defined
                    containing_class = [cls for cls in exec_locals.values() if hasattr(cls, "save_to_path")]
                    if not containing_class:
                        raise AssertionError("No class with save_to_path found.")

                    # Create a mock instance of the class
                    instance = containing_class[0]()
                    
                    # Mock required attributes and methods
                    instance.anomaly_scorer = MagicMock()
                    instance.backbone = MagicMock()
                    instance.patch_maker = MagicMock()
                    instance.forward_modules = {
                        "preprocessing": MagicMock(),
                        "preadapt_aggregator": MagicMock(),
                    }
                    instance._params_file = MagicMock(return_value="/mock/path/to/params.pkl")
                    
                    with patch("builtins.open", new_callable=MagicMock()) as mock_open:
                        instance.save_to_path("mock_path", "mock_prepend")
                        # Assert the file was opened correctly and pickle was called
                        mock_open.assert_called_with("/mock/path/to/params.pkl", "wb")
                        args, kwargs = instance.anomaly_scorer.save.call_args
                        self.assertEqual(args[0], "mock_path")
                        self.assertEqual(kwargs['prepend'], "mock_prepend")

                    print(f"Code snippet {i}: PASSED all assertions.\n")
                    passed_count += 1
                    results.append({
                        "function_name": "save_to_path",
                        "code": class_code,
                        "result": "passed"
                    })

                except Exception as e:
                    print(f"Code snippet {i}: FAILED with error: {e}\n")
                    failed_count += 1
                    results.append({
                        "function_name": "save_to_path",
                        "code": class_code,
                        "result": "failed"
                    })

        print(f"\nTest Summary: {passed_count} passed, {failed_count} failed, total {len(code)}\n")
        self.assertEqual(passed_count + failed_count, len(code), "Test count mismatch!")

        existing_records = []
        if os.path.exists(TEST_RESULT_JSONL):
            with open(TEST_RESULT_JSONL, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    existing_records.append(json.loads(line))

        existing_records = [rec for rec in existing_records if rec.get("function_name") != "save_to_path"]
        existing_records.extend(results)

        with open(TEST_RESULT_JSONL, "w", encoding="utf-8") as f:
            for record in existing_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
    
if __name__ == "__main__":
    unittest.main()
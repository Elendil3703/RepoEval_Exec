import unittest
import json
import os
import sys
from typing import Any

TEST_RESULT_JSONL = "test_result.jsonl"

class MockDataLoader:
    def __init__(self, data):
        self.data = data
        self.index = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.index >= len(self.data):
            raise StopIteration
        result = self.data[self.index]
        self.index += 1
        return result

    def set_epoch(self, epoch):
        self.index = 0  # Simplified for testing purpose

def convert_tensor(value, device, is_fp16):
    """Mock function for convert_tensor."""
    return value  # Simply return the input value for testing

class TestGetBatchSingleLoader(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        
        # Select the 266th group.
        cls.code_snippets = data[265]
        if len(cls.code_snippets) < 1:
            raise ValueError("Expected at least one code snippet in the item 265")

    def test_get_batch_single_loader(self):
        passed_count = 0
        failed_count = 0
        results = []

        for i, code in enumerate(self.code_snippets):
            with self.subTest(code_index=i):
                print(f"Running test for code snippet {i}...")
                exec_globals = {
                    'Any': Any,
                    'MockDataLoader': MockDataLoader,
                    'convert_tensor': convert_tensor
                }
                exec_locals = {}

                try:
                    exec(code, exec_globals, exec_locals)
                    
                    # Ensure get_batch_single_loader is defined
                    if 'get_batch_single_loader' not in exec_locals:
                        print(f"Code snippet {i}: FAILED, 'get_batch_single_loader' not found in exec_locals.\n")
                        failed_count += 1
                        results.append({
                            "function_name": "get_batch_single_loader",
                            "code": code,
                            "result": "failed"
                        })
                        continue
                    
                    # Mock an object to simulate the environment
                    class MockObj:
                        def __init__(self):
                            self.device = 'cpu'
                            self.epoch_callback_exec = lambda: None
                            self.train_data_loader = [
                                MockDataLoader([{"input": i} for i in range(3)])
                            ]
                            self.train_data_iterator = [iter(self.train_data_loader[0])]
                            self.epoch_counter = [0]
                            self._strategy = ""
                            self._is_default_fp16 = lambda: False
                   
                    obj = MockObj()
                    batch = exec_locals['get_batch_single_loader'](obj, 0)
                    
                    # Verify that the function returns the correct format
                    self.assertTrue(
                        isinstance(batch, (dict, tuple)),
                        f"Code snippet {i} did not return a correct batch type."
                    )

                    # Verify if epoch_callback_exec is called
                    obj.epoch_callback_exec = unittest.mock.Mock()
                    for _ in range(4):  # Iterate more than the loader to force StopIteration
                        exec_locals['get_batch_single_loader'](obj, 0)
                    obj.epoch_callback_exec.assert_called()                    

                    print(f"Code snippet {i}: PASSED all assertions.\n")
                    passed_count += 1
                    results.append({
                        "function_name": "get_batch_single_loader",
                        "code": code,
                        "result": "passed"
                    })

                except Exception as e:
                    print(f"Code snippet {i}: FAILED with error: {e}\n")
                    failed_count += 1
                    results.append({
                        "function_name": "get_batch_single_loader",
                        "code": code,
                        "result": "failed"
                    })

        print(f"\nTest Summary: {passed_count} passed, {failed_count} failed, total {len(self.code_snippets)}\n")
        self.assertEqual(passed_count + failed_count, len(self.code_snippets), "Test count mismatch!")

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
            if rec.get("function_name") != "get_batch_single_loader"
        ]

        existing_records.extend(results)

        with open(TEST_RESULT_JSONL, "w", encoding="utf-8") as f:
            for record in existing_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        print("Results have been written to test_result.jsonl")

if __name__ == "__main__":
    unittest.main()
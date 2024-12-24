import unittest
import json
import os
import torch

TEST_RESULT_JSONL = "test_result.jsonl"

def _detach(features, detach=True):
    if detach:
        return [x.detach().cpu().numpy() for x in features]
    return features

class TestDetachFunction(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippet = data[361]  # Get the 362nd JSON element

    def test_detach_function(self):
        """Test the _detach function."""
        passed_count = 0
        failed_count = 0
        results = []

        for i, code in enumerate(self.code_snippet):
            with self.subTest(code_index=i):
                print(f"Running test for code snippet {i}...")

                # Simulate features with PyTorch tensors
                features = [torch.tensor([1.0, 2.0], requires_grad=True), torch.tensor([3.0, 4.0], requires_grad=True)]
                
                # ------------------- 动态执行并测试逻辑 -------------------
                exec_globals = {
                    'torch': torch,
                    '_detach': _detach
                }
                exec_locals = {}

                try:
                    # 动态执行代码片段
                    exec(code, exec_globals, exec_locals)

                    # 检查 _detach 的行为
                    detached_features = exec_locals['_detach'](features, detach=True)
                    
                    # Assert all detached_features are numpy arrays and have no grad
                    for orig, detached in zip(features, detached_features):
                        self.assertFalse(orig.is_leaf and orig.requires_grad, f"Original tensor should have gradients.")
                        self.assertIsInstance(detached, (list, tuple), f"Detached feature is not a list or tuple.")
                        self.assertIsInstance(detached[0], float, f"First element of detached feature is not a float.")
                    
                    print(f"Code snippet {i}: PASSED all assertions.\n")
                    passed_count += 1
                    results.append({
                        "function_name": "_detach",
                        "code": code,
                        "result": "passed"
                    })
                except Exception as e:
                    print(f"Code snippet {i}: FAILED with error: {e}\n")
                    failed_count += 1
                    results.append({
                        "function_name": "_detach",
                        "code": code,
                        "result": "failed"
                    })

        # 最终统计信息
        print(f"\nTest Summary: {passed_count} passed, {failed_count} failed, total {len(self.code_snippet)}\n")

        # ============= 将测试结果写入 test_result.jsonl =============
        existing_records = []
        if os.path.exists(TEST_RESULT_JSONL):
            with open(TEST_RESULT_JSONL, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    existing_records.append(json.loads(line))

        # 删除 function_name == "_detach" 的旧记录
        existing_records = [
            rec for rec in existing_records
            if rec.get("function_name") != "_detach"
        ]

        # 将新结果附加
        existing_records.extend(results)

        # 重写 test_result.jsonl
        with open(TEST_RESULT_JSONL, "w", encoding="utf-8") as f:
            for record in existing_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        print("Results have been written to test_result.jsonl")

if __name__ == "__main__":
    unittest.main()
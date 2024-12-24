import unittest
import json
import os
import numpy as np
from typing import Callable

TEST_RESULT_JSONL = "test_result.jsonl"

class VectorSpaceWithBasis:
    def __init__(self, num_dims, basis):
        self.num_dims = num_dims
        self.basis = basis

class BasisDirection:
    def __init__(self, index):
        self.index = index

class ScalarBilinear:
    def __init__(self, left_space, right_space, matrix):
        self.left_space = left_space
        self.right_space = right_space
        self.matrix = matrix

class TestFromAction(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippets = data[210]  # Get the 211th JSON element

    def test_from_action(self):
        """Test the `from_action` function with given logic."""
        passed_count = 0  # Counter for passed tests
        failed_count = 0  # Counter for failed tests
        results = []      # 收集要写入 JSONL 的测试结果

        for i, code in enumerate(self.code_snippets):
            with self.subTest(code_index=i):
                print(f"Running test for code snippet {i}...")
                
                # Set up sample VectorSpaceWithBasis and action function
                left_space = VectorSpaceWithBasis(
                    3, [BasisDirection(k) for k in range(3)]
                )
                right_space = VectorSpaceWithBasis(
                    3, [BasisDirection(k) for k in range(3)]
                )

                def sample_action(ld, rd):
                    # Sample simple action: multiplying indices
                    return ld.index * rd.index

                # ------------------- 动态执行并测试逻辑 -------------------
                exec_globals = {
                    'VectorSpaceWithBasis': VectorSpaceWithBasis,
                    'BasisDirection': BasisDirection,
                    'ScalarBilinear': ScalarBilinear,
                    'np': np,
                }
                exec_locals = {}

                try:
                    # 动态执行代码片段，确保有 from_action 函数定义
                    exec(code, exec_globals, exec_locals)

                    # 检查 from_action 是否真的存在
                    if 'from_action' not in exec_locals:
                        print(f"Code snippet {i}: FAILED, 'from_action' not found in exec_locals.\n")
                        failed_count += 1
                        results.append({
                            "function_name": "from_action",
                            "code": code,
                            "result": "failed"
                        })
                        continue

                    from_action = exec_locals['from_action']
                    result_bilinear = from_action(None, left_space, right_space, sample_action)

                    expected_matrix = np.array([[i * j for j in range(right_space.num_dims)]
                                                for i in range(left_space.num_dims)])

                    # Assert the created matrix is as expected
                    np.testing.assert_array_equal(
                        result_bilinear.matrix,
                        expected_matrix,
                        f"Code snippet {i} did not produce the expected matrix."
                    )

                    print(f"Code snippet {i}: PASSED all assertions.\n")
                    passed_count += 1
                    results.append({
                        "function_name": "from_action",
                        "code": code,
                        "result": "passed"
                    })
                except Exception as e:
                    print(f"Code snippet {i}: FAILED with error: {e}\n")
                    failed_count += 1
                    results.append({
                        "function_name": "from_action",
                        "code": code,
                        "result": "failed"
                    })

        # 最终统计信息
        print(f"\nTest Summary: {passed_count} passed, {failed_count} failed, total {len(self.code_snippets)}\n")
        self.assertEqual(passed_count + failed_count, len(self.code_snippets), "Test count mismatch!")

        # ============= 将测试结果写入 test_result.jsonl =============
        # 读取现有 test_result.jsonl（若不存在则忽略）
        existing_records = []
        if os.path.exists(TEST_RESULT_JSONL):
            with open(TEST_RESULT_JSONL, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    existing_records.append(json.loads(line))

        # 删除 function_name == "from_action" 的旧记录
        existing_records = [
            rec for rec in existing_records
            if rec.get("function_name") != "from_action"
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
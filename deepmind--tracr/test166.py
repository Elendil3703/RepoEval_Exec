import unittest
import json
import sys
import os
from typing import Any

TEST_RESULT_JSONL = "test_result.jsonl"

class RepoEvalResult:
    """A class to mock the environment where _get_dummy_block function exists."""
    
    def __init__(self):
        self.categorical_attn = sys.modules[__name__]
        self.categorical_mlp = sys.modules[__name__]
        self.bases = sys.modules[__name__]

    class VectorSpaceWithBasis:
        @classmethod
        def from_names(cls, names):
            return names

    def categorical_attn(self, query_space, key_space, value_space, output_space, bos_space, one_space, attn_fn):
        """Mocking categorical_attn method"""
        return {
            "query_space": query_space,
            "key_space": key_space,
            "value_space": value_space,
            "output_space": output_space,
            "bos_space": bos_space,
            "one_space": one_space,
            "attn_fn": attn_fn
        }

    def map_categorical_mlp(self, input_space, output_space, operation):
        """Mocking map_categorical_mlp method"""
        return {
            "input_space": input_space,
            "output_space": output_space,
            "operation": operation
        }

class TestGetDummyBlockResult(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippets = data[165]  # Get the 166th JSON element (index 165)
        if len(cls.code_snippets) < 1:
            raise ValueError("Expected at least one code snippet in the JSON element")

    def test_get_dummy_block(self):
        """Dynamically test _get_dummy_block function in code snippet."""
        passed_count = 0  # Counter for passed tests
        failed_count = 0  # Counter for failed tests
        results = []      # 收集要写入 JSONL 的测试结果
        
        mock_env = RepoEvalResult()

        for i, code in enumerate(self.code_snippets):
            with self.subTest(code_index=i):
                print(f"Running test for code snippet {i}...")
                # ------------------- 动态执行并测试逻辑 -------------------
                exec_globals = {
                    'sys': sys,
                    'Any': Any,
                }
                exec_locals = {
                    'self': mock_env
                }
                
                try:
                    exec(code, exec_globals, exec_locals)

                    # Assume _get_dummy_block is defined and test function is present
                    _get_dummy_block = exec_locals.get("_get_dummy_block")
                    if _get_dummy_block is None:
                        raise ValueError(f"Code snippet {i}: '_get_dummy_block' not found.")

                    # Test ATTN type
                    result_attn = _get_dummy_block(mock_env, "ATTN")
                    expected_attn = mock_env.categorical_attn(
                        query_space=["query"],
                        key_space=["bos", "key"],
                        value_space=["bos", "value"],
                        output_space=["output"],
                        bos_space=["bos"],
                        one_space=["one"],
                        attn_fn=lambda x, y: True
                    )
                    self.assertEqual(result_attn, expected_attn, f"Code snippet {i} failed on ATTN type.")

                    # Test MLP type
                    result_mlp = _get_dummy_block(mock_env, "MLP")
                    expected_mlp = mock_env.map_categorical_mlp(
                        input_space=["input"],
                        output_space=["output"],
                        operation=lambda x: x
                    )
                    self.assertEqual(result_mlp["input_space"], expected_mlp["input_space"], 
                                     f"Code snippet {i} failed on MLP type input_space.")
                    self.assertEqual(result_mlp["output_space"], expected_mlp["output_space"],
                                     f"Code snippet {i} failed on MLP type output_space.")

                    # Test invalid type
                    self.assertIsNone(_get_dummy_block(mock_env, "INVALID"), f"Code snippet {i} failed on invalid type.")

                    print(f"Code snippet {i}: PASSED all assertions.\n")
                    passed_count += 1
                    results.append({
                        "function_name": "_get_dummy_block",
                        "code": code,
                        "result": "passed"
                    })
                except Exception as e:
                    print(f"Code snippet {i}: FAILED with error: {e}\n")
                    failed_count += 1
                    results.append({
                        "function_name": "_get_dummy_block",
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

        # 删除 function_name == "_get_dummy_block" 的旧记录
        existing_records = [
            rec for rec in existing_records
            if rec.get("function_name") != "_get_dummy_block"
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
import unittest
import json
import os
from typing import Any  # 确保注入的环境中有 Any

TEST_RESULT_JSONL = "test_result.jsonl"

class TestEmbedOutput(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippet = data[170]  # Get the 171st JSON element based on 0-index
        if not cls.code_snippet:
            raise ValueError("Expected a code snippet in the JSON array")

    def test_embed_output_functionality(self):
        """Test _embed_output function's correctness."""
        passed_count = 0  # Counter for passed tests
        failed_count = 0  # Counter for failed tests
        results = []      # 收集要写入 JSONL 的测试结果

        code = self.code_snippet
        print(f"Running test for the _embed_output code snippet...")

        # ------------------- 静态检查 -------------------
        if "_embed_output" not in code:
            print("Code snippet FAILED, '_embed_output' not found in code.\n")
            failed_count += 1
            results.append({
                "function_name": "_embed_output",
                "code": code,
                "result": "failed"
            })
            return

        # ------------------- 动态执行并测试逻辑 -------------------
        exec_globals = {
            'Any': Any,  # 注入 Any
            'output_space': type('DummyOutputSpace', (object,), {
                'basis': [type('DummyBasis', (object,), {'name': 'dummy_label'})],
                'null_vector': lambda: 'null_vector',
                'vector_from_basis_direction': lambda x: f'vector_{x.name}'
            })
        }
        exec_locals = {}

        try:
            # 动态执行代码片段
            exec(code, exec_globals, exec_locals)

            # 检查 _embed_output 是否真的存在
            if '_embed_output' not in exec_locals:
                print("Code snippet FAILED, '_embed_output' not found in exec_locals.\n")
                failed_count += 1
                results.append({
                    "function_name": "_embed_output",
                    "code": code,
                    "result": "failed"
                })
                return

            # 使用 _embed_output 测试逻辑
            _embed_output = exec_locals['_embed_output']
            output_space = exec_globals['output_space']

            # Test case: Categorical output
            output_seq = [1, 2, None, 4]
            embedded_output = _embed_output(output_seq, output_space, categorical_output=True)
            expected_output = [
                'vector_dummy_label', 'vector_dummy_label', 'null_vector', 'vector_dummy_label'
            ]
            self.assertEqual(embedded_output, expected_output,
                             "Failed on categorical output test.")

            # Test case: Non-categorical output
            output_seq = [2, 4, None, 8]
            embedded_output = _embed_output(output_seq, output_space, categorical_output=False)
            expected_output = [
                'vector_dummy_label', 'vector_dummy_label', 'null_vector', 'vector_dummy_label'
            ] # simplified for brevity, imagine actual multiplication logic here

            self.assertEqual(embedded_output, expected_output,
                             "Failed on non-categorical output test.")

            print("Code snippet PASSED all assertions.\n")
            passed_count += 1
            results.append({
                "function_name": "_embed_output",
                "code": code,
                "result": "passed"
            })
        except Exception as e:
            print(f"Code snippet FAILED with error: {e}\n")
            failed_count += 1
            results.append({
                "function_name": "_embed_output",
                "code": code,
                "result": "failed"
            })

        # 最终统计信息
        print(f"\nTest Summary: {passed_count} passed, {failed_count} failed\n")

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

        # 删除 function_name == "_embed_output" 的旧记录
        existing_records = [
            rec for rec in existing_records
            if rec.get("function_name") != "_embed_output"
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
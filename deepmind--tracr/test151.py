import unittest
import json
import os
from typing import List

TEST_RESULT_JSONL = "test_result.jsonl"

class TestApplyFunction(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippets = data[150]  # Get the 151st JSON element
        if len(cls.code_snippets) < 1:
            raise ValueError("Expected at least one code snippet in the 151st JSON array")

    def test_apply_function(self):
        """Dynamically test all code snippets for the 'apply' function."""
        passed_count = 0  # Counter for passed tests
        failed_count = 0  # Counter for failed tests
        results = []      # 收集要写入 JSONL 的测试结果

        for i, code in enumerate(self.code_snippets):
            with self.subTest(code_index=i):
                print(f"Running test for code snippet {i}...")
                
                # Check for 'apply' function definition
                if "def apply" not in code:
                    print(f"Code snippet {i}: FAILED, 'apply' function not found in code.\n")
                    failed_count += 1
                    # 写入失败记录
                    results.append({
                        "function_name": "apply",
                        "code": code,
                        "result": "failed"
                    })
                    continue

                exec_globals = {
                    'jnp': __import__('jax.numpy'),  # Example: imported jax.numpy for jnp.array
                    'AssembledTransformerModelOutput': self.mock_AssembledTransformerModelOutput,
                    'bases': self.mock_bases,
                }
                exec_locals = {}

                try:
                    # Dynamically execute the code snippet
                    exec(code, exec_globals, exec_locals)

                    # Check if 'apply' function exists
                    if 'apply' not in exec_locals:
                        print(f"Code snippet {i}: FAILED, 'apply' not found in exec_locals.\n")
                        failed_count += 1
                        results.append({
                            "function_name": "apply",
                            "code": code,
                            "result": "failed"
                        })
                        continue

                    # Test the 'apply' function
                    class MockModel:
                        def __init__(self):
                            self.input_encoder = None  # Mock input_encoder
                            self.output_encoder = None  # Mock output_encoder
                            self.params = {}

                        def forward(self, params, tokens):
                            class MockOutput:
                                def __init__(self):
                                    self.unembedded_output = [[1, 2, 3]]
                                    self.transformer_output = self
                                    self.layer_outputs = "layer_outputs"
                                    self.residuals = "residuals"
                                    self.attn_logits = "attn_logits"
                                    self.output = "output"
                                    self.input_embeddings = "input_embeddings"

                            return MockOutput()

                    model = MockModel()

                    # Call 'apply' function and check the output
                    output = exec_locals['apply'](model, [10, 20, 30])
                    self.assertEqual(output.decoded, [1, 2, 3])

                    print(f"Code snippet {i}: PASSED all assertions.\n")
                    passed_count += 1
                    results.append({
                        "function_name": "apply",
                        "code": code,
                        "result": "passed"
                    })
                except Exception as e:
                    print(f"Code snippet {i}: FAILED with error: {e}\n")
                    failed_count += 1
                    results.append({
                        "function_name": "apply",
                        "code": code,
                        "result": "failed"
                    })

        # 最终统计信息
        print(f"Test Summary: {passed_count} passed, {failed_count} failed, total {len(self.code_snippets)}\n")
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

        # 删除 function_name == "apply" 的旧记录
        existing_records = [
            rec for rec in existing_records
            if rec.get("function_name") != "apply"
        ]

        # 将新结果附加
        existing_records.extend(results)

        # 重写 test_result.jsonl
        with open(TEST_RESULT_JSONL, "w", encoding="utf-8") as f:
            for record in existing_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        print("Results have been written to test_result.jsonl")

    class mock_AssembledTransformerModelOutput:
        def __init__(self, decoded, unembedded, layer_outputs, residuals, attn_logits, transformer_output, input_embeddings):
            self.decoded = decoded
            self.unembedded = unembedded
            self.layer_outputs = layer_outputs
            self.residuals = residuals
            self.attn_logits = attn_logits
            self.transformer_output = transformer_output
            self.input_embeddings = input_embeddings

    class mock_bases:
        class Value:
            pass

if __name__ == "__main__":
    unittest.main()
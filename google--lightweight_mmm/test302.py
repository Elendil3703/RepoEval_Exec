import unittest
import json
import os
from unittest.mock import MagicMock, patch
import matplotlib.figure

TEST_RESULT_JSONL = "test_result.jsonl"
REPOEVAL_RESULT_JSON = "RepoEval_result.json"

class TestPlotModelFitFunction(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open(REPOEVAL_RESULT_JSON, "r") as f:
            data = json.load(f)
        cls.code_snippet = data[301]  # Get the 302nd JSON element

    @patch('lightweight_mmm.LightweightMMM', autospec=True)
    @patch('preprocessing.CustomScaler', autospec=True)
    def test_plot_model_fit(self, MockedLightweightMMM, MockedCustomScaler):
        """Test the function plot_model_fit."""
        passed_count = 0  # Counter for passed tests
        failed_count = 0  # Counter for failed tests
        results = []      # Collect test results to write to JSONL
        
        with self.subTest():
            # Mock model and scaler
            mock_model = MockedLightweightMMM()
            mock_scaler = MockedCustomScaler()
            
            # Setting expected attributes
            mock_model.trace = {"mu": [0.1, 0.2, 0.3]}
            mock_model._target = [1.0, 1.5, 2.0]
            mock_scaler.inverse_transform.side_effect = lambda x: x  # Identity function for testing

            exec_globals = {
                'matplotlib': matplotlib,
                'lightweight_mmm': lightweight_mmm,
                '_call_fit_plotter': MagicMock(),  # Replace plot call with mock
            }
            
            try:
                # Execute code snippet in an isolated namespace
                exec(self.code_snippet, exec_globals)

                # Get the function
                plot_model_fit = exec_globals.get('plot_model_fit', None)
                if not plot_model_fit:
                    raise ValueError("Function plot_model_fit not found in code snippet.")

                # Call the function with mocked objects
                result = plot_model_fit(
                    media_mix_model=mock_model,
                    target_scaler=mock_scaler,
                    interval_mid_range=0.9,
                    digits=3
                )
                
                # Assertions to verify the behavior
                self.assertTrue(isinstance(result, matplotlib.figure.Figure))
                exec_globals['_call_fit_plotter'].assert_called_once()

                print("Code snippet: PASSED all assertions.\n")
                passed_count += 1
                results.append({
                    "function_name": "plot_model_fit",
                    "code": self.code_snippet,
                    "result": "passed"
                })

            except Exception as e:
                print(f"Code snippet: FAILED with error: {e}\n")
                failed_count += 1
                results.append({
                    "function_name": "plot_model_fit",
                    "code": self.code_snippet,
                    "result": "failed"
                })

        # 最终统计信息
        print(f"\nTest Summary: {passed_count} passed, {failed_count} failed.\n")
        self.assertEqual(passed_count + failed_count, 1, "Test count mismatch!")

        # ============= 将测试结果写入 test_result.jsonl =============
        existing_records = []
        if os.path.exists(TEST_RESULT_JSONL):
            with open(TEST_RESULT_JSONL, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    existing_records.append(json.loads(line))

        # 删除 function_name == "plot_model_fit" 的旧记录
        existing_records = [
            rec for rec in existing_records
            if rec.get("function_name") != "plot_model_fit"
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
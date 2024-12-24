import unittest
import json
import os
from unittest import mock

TEST_RESULT_JSONL = "test_result.jsonl"

class TestSetUpFunction(unittest.TestCase):
    def setUp(self):
        """Set up mocks for plotting functions."""
        self.mock_ax_scatter = self.enter_context(
            mock.patch.object(plot.plt.Axes, "scatter", autospec=True))
        self.mock_sns_lineplot = self.enter_context(
            mock.patch.object(plot.sns, "lineplot", autospec=True))
        self.mock_plt_plot = self.enter_context(
            mock.patch.object(plot.plt.Axes, "plot", autospec=True))
        self.mock_plt_barplot = self.enter_context(
            mock.patch.object(plot.plt.Axes, "bar", autospec=True))
        self.mock_pd_area_plot = self.enter_context(
            mock.patch.object(plot.pd.DataFrame.plot, "area", autospec=True))
        self.mock_sns_kdeplot = self.enter_context(
            mock.patch.object(plot.sns, "kdeplot", autospec=True))
        self.mock_plt_ax_legend = self.enter_context(
            mock.patch.object(plot.plt.Axes, "legend", autospec=True))

    def test_setup_mocks(self):
        """Test that all mocks are set up correctly."""
        # Check that all mocks are created and are mock objects
        mock_objects = [
            self.mock_ax_scatter,
            self.mock_sns_lineplot,
            self.mock_plt_plot,
            self.mock_plt_barplot,
            self.mock_pd_area_plot,
            self.mock_sns_kdeplot,
            self.mock_plt_ax_legend
        ]
        
        for i, mock_obj in enumerate(mock_objects):
            with self.subTest(mock_index=i):
                self.assertIsInstance(mock_obj, mock.MagicMock, f"Mock at index {i} is not a MagicMock instance.")

    @classmethod
    def tearDownClass(cls):
        # Collect test results
        results = []

        for test, result in cls._outcome.result.failures + cls._outcome.result.errors:
            results.append({
                "function_name": "setUp",
                "test_name": test.id(),
                "outcome": "failed",
                "details": str(result)
            })
        
        for test in cls._outcome.result.successes:
            results.append({
                "function_name": "setUp",
                "test_name": test.id(),
                "outcome": "passed",
                "details": ""
            })
        
        # Read existing test results if they exist
        existing_records = []
        if os.path.exists(TEST_RESULT_JSONL):
            with open(TEST_RESULT_JSONL, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    existing_records.append(json.loads(line))
        
        # Remove old results for setUp
        existing_records = [
            rec for rec in existing_records
            if rec.get("function_name") != "setUp"
        ]
        
        # Append new results
        existing_records.extend(results)
        
        # Write updated results to the file
        with open(TEST_RESULT_JSONL, "w", encoding="utf-8") as f:
            for record in existing_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        
        print("Results have been written to test_result.jsonl")

if __name__ == "__main__":
    unittest.main()
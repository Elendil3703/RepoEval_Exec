import unittest
import json
import os
import torch
from typing import Any, Iterable, List, Dict, Tuple

TEST_RESULT_JSONL = "test_result.jsonl"

def map_scheduler_cfgs_to_param_groups(
    scheduler_cfgs_per_param_group: Iterable[List[Dict]], model: torch.nn.Module
) -> Tuple[List[Dict[Any, Any]], List[Dict[str, List[torch.nn.Parameter]]]]:
    schedulers = []
    param_groups = []
    for scheduler_cfgs in scheduler_cfgs_per_param_group:
        param_constraints = [
            scheduler_cfg["parameter_names"] for scheduler_cfg in scheduler_cfgs
        ]
        matching_parameters = name_constraints_to_parameters(param_constraints, model)
        if len(matching_parameters) == 0:  # If no overlap of parameters, skip
            continue
        schedulers_for_group = {
            scheduler_cfg["option"]: scheduler_cfg["scheduler"]
            for scheduler_cfg in scheduler_cfgs
            if "option" in scheduler_cfg
        }
        schedulers.append(schedulers_for_group)
        param_groups.append({"params": matching_parameters})
    return schedulers, param_groups

def name_constraints_to_parameters(constraints, model):
    # Mock implementation to simulate matching parameters based on constraints.
    params = []
    for name, param in model.named_parameters():
        if any(name in constraint for constraint in constraints):
            params.append(param)
    return params

class SimpleMockModel(torch.nn.Module):
    def __init__(self):
        super(SimpleMockModel, self).__init__()
        self.linear = torch.nn.Linear(10, 1)

    def forward(self, x):
        return self.linear(x)

class TestMapSchedulerCfgs(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippets = data[409]  # Get the 410th JSON element
        if len(cls.code_snippets) < 1:
            raise ValueError("Expected at least one code snippet in the specified JSON array")

    def test_map_scheduler_cfgs_to_param_groups(self):
        passed_count = 0
        failed_count = 0
        results = []

        for i, code in enumerate(self.code_snippets):
            with self.subTest(code_index=i):
                print(f"Running test for code snippet {i}...")

                global_vars = {
                    "torch": torch,
                    "map_scheduler_cfgs_to_param_groups": map_scheduler_cfgs_to_param_groups,
                    "name_constraints_to_parameters": name_constraints_to_parameters,
                    "SimpleMockModel": SimpleMockModel,
                }
                local_vars = {}

                exec(code, global_vars, local_vars)

                # Create a mock model
                mock_model = SimpleMockModel()

                # Define test scheduler configurations
                scheduler_cfgs_per_param_group = [
                    [
                        {"parameter_names": ["linear.weight"], "option": "opt1", "scheduler": "SchedulerA"},
                        {"parameter_names": ["linear.bias"], "option": "opt2", "scheduler": "SchedulerB"}
                    ],
                    [
                        {"parameter_names": ["linear.weight"], "option": "opt3", "scheduler": "SchedulerC"}
                    ]
                ]

                try:
                    schedulers, param_groups = map_scheduler_cfgs_to_param_groups(
                        scheduler_cfgs_per_param_group, mock_model
                    )

                    expected_schedulers = [
                        {"opt1": "SchedulerA", "opt2": "SchedulerB"},
                        {"opt3": "SchedulerC"}
                    ]

                    self.assertEqual(schedulers, expected_schedulers, f"Code snippet {i} failed: incorrect schedulers output.")
                    self.assertEqual(len(param_groups), 2, f"Code snippet {i} failed: incorrect parameter groups length.")
                    self.assertEqual(len(param_groups[0]["params"]), 2, f"Code snippet {i} failed: incorrect parameter match count for first group.")

                    print(f"Code snippet {i}: PASSED all assertions.\n")
                    passed_count += 1
                    results.append({
                        "function_name": "map_scheduler_cfgs_to_param_groups",
                        "code": code,
                        "result": "passed"
                    })

                except Exception as e:
                    print(f"Code snippet {i}: FAILED with error: {e}\n")
                    failed_count += 1
                    results.append({
                        "function_name": "map_scheduler_cfgs_to_param_groups",
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
            if rec.get("function_name") != "map_scheduler_cfgs_to_param_groups"
        ]

        existing_records.extend(results)

        with open(TEST_RESULT_JSONL, "w", encoding="utf-8") as f:
            for record in existing_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        print("Results have been written to test_result.jsonl")

if __name__ == "__main__":
    unittest.main()
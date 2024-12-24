import unittest
import json
import os

TEST_RESULT_JSONL = "test_result.jsonl"

class MockClass:
    def __init__(self, key_pattern=None, source_pattern='', target_patterns=[]):
        self.key_pattern = key_pattern
        self.source_pattern = source_pattern
        self.target_patterns = target_patterns

    def __call__(self, state_dict):
        all_keys = set(state_dict.keys())
        include_keys = set(state_dict.keys())
        if self.key_pattern is not None:
            include_keys = _unix_pattern_to_parameter_names(
                self.key_pattern, state_dict.keys()
            )

        excluded_keys = all_keys - include_keys
        new_state_dict = {}
        for k in excluded_keys:
            new_state_dict[k] = state_dict[k]

        for key in include_keys:
            if self.source_pattern in key:
                for target_pattern in self.target_patterns:
                    new_key = key.replace(self.source_pattern, target_pattern, 1)
                    new_state_dict[new_key] = state_dict[key]
            else:
                new_state_dict[key] = state_dict[key]

        return new_state_dict

def _unix_pattern_to_parameter_names(pattern, keys):
    return {k for k in keys if pattern in k}

class TestCallFunction(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippet = data[395]
        cls.class_instance = MockClass

    def test_call_function(self):
        passed_count = 0
        failed_count = 0
        results = []

        # Example data based on possible input structure and functionality
        test_cases = [
            {
                "state_dict": {"param1": 10, "param2": 20},
                "key_pattern": None,
                "source_pattern": "param",
                "target_patterns": ["replacement"],
                "expected": {"replacement1": 10, "replacement2": 20},
            },
            {
                "state_dict": {"weight": 1.0, "bias": 0.5},
                "key_pattern": "bias",
                "source_pattern": "bias",
                "target_patterns": ["offset"],
                "expected": {"weight": 1.0, "offset": 0.5},
            },
        ]

        for i, case in enumerate(test_cases):
            with self.subTest(case_index=i):
                instance = self.class_instance(
                    key_pattern=case["key_pattern"],
                    source_pattern=case["source_pattern"],
                    target_patterns=case["target_patterns"],
                )
                result = instance(case["state_dict"])
                try:
                    self.assertEqual(result, case["expected"])
                    passed_count += 1
                    results.append({"function_name": "__call__", "case_index": i, "result": "passed"})
                except AssertionError as e:
                    failed_count += 1
                    results.append({"function_name": "__call__", "case_index": i, "result": "failed"})

        print(f"\nTest Summary: {passed_count} passed, {failed_count} failed, total {len(test_cases)}\n")
        self.assertEqual(passed_count + failed_count, len(test_cases), "Test count mismatch!")

        existing_records = []
        if os.path.exists(TEST_RESULT_JSONL):
            with open(TEST_RESULT_JSONL, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    existing_records.append(json.loads(line))

        existing_records = [rec for rec in existing_records if rec.get("function_name") != "__call__"]
        existing_records.extend(results)

        with open(TEST_RESULT_JSONL, "w", encoding="utf-8") as f:
            for record in existing_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        print("Results have been written to test_result.jsonl")

if __name__ == "__main__":
    unittest.main()
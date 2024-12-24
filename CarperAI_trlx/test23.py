import unittest
import json
import os
from typing import List
import transformers

TEST_RESULT_JSONL = "test_result.jsonl"

def generate_layer_regex(config, num_layers_unfrozen):
    """Mock function for layer regex generation."""
    total_layers = config.num_hidden_layers
    if num_layers_unfrozen == -1:  # Default to fully unfreezing all layers
        return r"[0-9]+"
    return r"[0-9]{1,%d}" % len(str(total_layers))

class MockPretrainedConfig(transformers.PretrainedConfig):
    """Mock config class for simulating transformers PretrainedConfig."""
    def __init__(self, num_hidden_layers, is_encoder_decoder):
        super().__init__()
        self.num_hidden_layers = num_hidden_layers
        self.is_encoder_decoder = is_encoder_decoder

class TestGetDeltaModifiedModules(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the specific code snippet
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippets = data[22]  # Adjust for zero-based index

        if len(cls.code_snippets) < 1:
            raise ValueError("Expected at least one code snippet in the specified JSON array")

        cls.code = "\n".join(cls.code_snippets)

    def test_get_delta_modified_modules(self):
        """Test get_delta_modified_modules function from code snippet."""
        passed_count = 0  # Counter for passed tests
        failed_count = 0  # Counter for failed tests
        results = []      # Collect results for JSONL

        exec_globals = {
            "transformers": transformers,
            "List": List,
            "generate_layer_regex": generate_layer_regex
        }
        exec(self.code, exec_globals)

        get_delta_modified_modules = exec_globals['get_delta_modified_modules']

        # Prepare mock data for testing
        config = MockPretrainedConfig(num_hidden_layers=12, is_encoder_decoder=False)
        modified_modules = ['attention', 'output']
        num_layers_unfrozen = 4

        try:
            # Perform a function test for a specific case
            result = get_delta_modified_modules(config, modified_modules, num_layers_unfrozen)

            # Expected results considering mock setup
            expected_pattern = r"[r][0-9]{1,2}(attention|output)"
            decoder_prefix = "decoder.block." if config.is_encoder_decoder else ""
            expected = [f"[r]{decoder_prefix}[0-9]{1,2}{mod}" for mod in modified_modules]

            # Assert the outcome
            self.assertEqual(result, expected, "Function did not return expected modified modules list.")
            print("Test passed.")
            passed_count += 1

            results.append({
                "function_name": "get_delta_modified_modules",
                "code": self.code,
                "result": "passed"
            })
        except Exception as e:
            print(f"Test failed with error: {e}")
            failed_count += 1
            results.append({
                "function_name": "get_delta_modified_modules",
                "code": self.code,
                "result": "failed"
            })

        # Write results to test_result.jsonl
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
            if rec.get("function_name") != "get_delta_modified_modules"
        ]

        existing_records.extend(results)

        with open(TEST_RESULT_JSONL, "w", encoding="utf-8") as f:
            for record in existing_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        print("Results have been written to test_result.jsonl")


if __name__ == "__main__":
    unittest.main()
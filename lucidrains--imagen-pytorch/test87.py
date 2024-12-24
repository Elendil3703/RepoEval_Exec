import unittest
import json
import os

TEST_RESULT_JSONL = "test_result.jsonl"

class MockModel:
    # A mock model class to test cast_model_parameters
    def __init__(self, lowres_cond, text_embed_dim, channels, channels_out, cond_on_text):
        self.lowres_cond = lowres_cond
        self._locals = {'text_embed_dim': text_embed_dim}
        self.channels = channels
        self.channels_out = channels_out
        self.cond_on_text = cond_on_text
        
    def cast_model_parameters(self, *, lowres_cond, text_embed_dim, channels, channels_out, cond_on_text):
        if (lowres_cond == self.lowres_cond and
            channels == self.channels and
            cond_on_text == self.cond_on_text and
            text_embed_dim == self._locals['text_embed_dim'] and
            channels_out == self.channels_out):
            return self

        updated_kwargs = dict(
            lowres_cond = lowres_cond,
            text_embed_dim = text_embed_dim,
            channels = channels,
            channels_out = channels_out,
            cond_on_text = cond_on_text
        )

        return self.__class__(**{**self._locals, **updated_kwargs})

class TestCastModelParametersFunction(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        
        # Get the 87th code snippet (index 86)
        cls.code_snippet = data[86]

    def test_cast_model_parameters(self):
        """Test the cast_model_parameters function."""
        passed_count = 0  # Counter for passed tests
        failed_count = 0  # Counter for failed tests
        results = []      # Collect results to write to JSONL

        exec_globals = {
            'MockModel': MockModel,
        }
        exec_locals = {}

        # Execute the 87th code snippet to test it
        code = self.code_snippet
        try:
            exec(code, exec_globals, exec_locals)
            print(f"Executing code snippet:\n{code}\n")
            
            # Retrieve the function from execution locals
            if 'cast_model_parameters' not in exec_locals:
                raise RuntimeError("Function 'cast_model_parameters' not found in executed code.")

            # Create an instance of the mock class
            model_instance = MockModel(
                lowres_cond=True,
                text_embed_dim=512,
                channels=3,
                channels_out=3,
                cond_on_text=True
            )

            # Test case: No change in parameters
            result = model_instance.cast_model_parameters(
                lowres_cond=True,
                text_embed_dim=512,
                channels=3,
                channels_out=3,
                cond_on_text=True
            )
            self.assertIs(result, model_instance, "Model instance should be returned.")

            # Test case: Change in parameters
            new_result = model_instance.cast_model_parameters(
                lowres_cond=False,
                text_embed_dim=256,
                channels=3,
                channels_out=3,
                cond_on_text=False
            )
            self.assertIsNot(new_result, model_instance, "New instance should be returned for changed parameters.")
            self.assertEqual(new_result.lowres_cond, False, "Parameter lowres_cond should be updated.")
            self.assertEqual(new_result._locals['text_embed_dim'], 256, "Parameter text_embed_dim should be updated.")
            self.assertEqual(new_result.cond_on_text, False, "Parameter cond_on_text should be updated.")

            print("Code snippet: PASSED all assertions.\n")
            passed_count += 1
            results.append({
                "function_name": "cast_model_parameters",
                "code": code,
                "result": "passed"
            })
        except Exception as e:
            print(f"Code snippet FAILED with error: {e}\n")
            failed_count += 1
            results.append({
                "function_name": "cast_model_parameters",
                "code": code,
                "result": "failed"
            })

        # ============= Write results to test_result.jsonl =============
        existing_records = []
        if os.path.exists(TEST_RESULT_JSONL):
            with open(TEST_RESULT_JSONL, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    existing_records.append(json.loads(line))

        # Remove old records for function_name == "cast_model_parameters"
        existing_records = [
            rec for rec in existing_records
            if rec.get("function_name") != "cast_model_parameters"
        ]

        # Append new results
        existing_records.extend(results)

        # Rewrite test_result.jsonl
        with open(TEST_RESULT_JSONL, "w", encoding="utf-8") as f:
            for record in existing_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        print("Results have been written to test_result.jsonl")

if __name__ == "__main__":
    unittest.main()
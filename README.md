# RepoEval_Exec: An automated testing framework for function-level code completion in RepoEval

## Introduction

[RepoEval](https://github.com/microsoft/CodeT/tree/main/RepoCoder) is highlighted in the SOTA(state of the art) method of repository-level code completion, [RepoCoder](https://arxiv.org/abs/2303.12570). However, the function-level completion dataset in RepoEval, published on their GitHub repository, lacks an automated testing framework. This limitation prevents experiments from being conducted based on the pass rate. To address this issue, we developed **RepoEval_Exec**, an automated testing framework specifically for function-level completion in RepoEval.

## Quick Start

### Install requirements

Change repo_name to the repo you want to test.

```bash
conda create -n myenv python=3.9
conda activate myenv
pip install -r {repo_name}/requirements.txt
```

### Prepare input

Input json should be named `RepoEval_result.json` and follow the following format.
func_n represents different target functions in sequential order as defined in function_level_completion_2k_context_codex.test.jsonl.
code_n represents different solutions to a target function.

```json
[
    [
        "func1_code1",
        "func1_code2",
        "func1_code3"
    ],
    [
        "func2_code1",
        "func2_code2",
        "func2_code3"
    ],
    [
        "func3_code1",
        "func3_code2",
        "func3_code3"
    ]
]
```

### Run testing

You can edit `run_pipeline.py` to run tests for specific repositories. Run `run_pipeline.py` to execute all tests.

```shell
python run_pipeline.py
```

### Show results

results will be written into test_result.jsonl

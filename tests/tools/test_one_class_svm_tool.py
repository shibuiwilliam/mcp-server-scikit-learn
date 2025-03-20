"""Unit tests for the one-class SVM tool."""

import json
from typing import Any, Dict

import pytest

from mcp_server_scikit_learn.tools.one_class_svm_tool import (
    handle_one_class_svm_tool,
)


@pytest.mark.asyncio
@pytest.mark.usefixtures("scope_function")
@pytest.mark.parametrize(
    ("input_args", "expected_keys"),
    [
        (
            {
                "training_dataset": [
                    {"feature1": 1.0, "feature2": 2.0},
                    {"feature1": 2.0, "feature2": 3.0},
                ],
                "prediction_dataset": [
                    {"feature1": 1.5, "feature2": 2.5},
                    {"feature1": 2.5, "feature2": 3.5},
                ],
                "kernel": "rbf",
                "nu": 0.1,
            },
            ["description", "predictions"],
        ),
        (
            {
                "training_dataset": [
                    {"feature1": 1.0, "feature2": 2.0},
                    {"feature1": 2.0, "feature2": 3.0},
                ],
                "prediction_dataset": [
                    {"feature1": 1.5, "feature2": 2.5},
                    {"feature1": 2.5, "feature2": 3.5},
                ],
                "kernel": "linear",
                "nu": 0.2,
                "gamma": "scale",
                "tol": 1e-3,
                "shrinking": True,
                "cache_size": 200,
                "verbose": False,
                "max_iter": -1,
            },
            ["description", "predictions"],
        ),
        (
            {
                "training_dataset": [
                    {"feature1": 1.0, "feature2": 2.0},
                    {"feature1": 2.0, "feature2": 3.0},
                ],
                "prediction_dataset": [
                    {"feature1": 1.5, "feature2": 2.5},
                    {"feature1": 2.5, "feature2": 3.5},
                ],
                "column_to_ignore": ["feature2"],
                "kernel": "poly",
                "nu": 0.1,
                "degree": 3,
                "coef0": 0.0,
            },
            ["description", "predictions"],
        ),
    ],
)
async def test_handle_one_class_svm_tool(
    mocker,
    scope_function,
    input_args: Dict[str, Any],
    expected_keys: list[str],
) -> None:
    """Test the one-class SVM tool handler.

    Args:
        input_args: Input arguments for the tool.
        expected_keys: Expected keys in the response.
    """
    # Convert training and prediction datasets to strings
    input_args["training_dataset"] = json.dumps(input_args["training_dataset"])
    input_args["prediction_dataset"] = json.dumps(input_args["prediction_dataset"])

    # Call the handler
    result = await handle_one_class_svm_tool(input_args)

    # Check that we got a TextContent result
    assert len(result) == 1
    assert result[0].type == "text"

    # Parse the result
    result_dict = json.loads(result[0].text)

    # Check that all expected keys are present
    for key in expected_keys:
        assert key in result_dict

    # Check that predictions is a list
    assert isinstance(result_dict["predictions"], list)

    # Check that the number of predictions matches the prediction dataset
    assert len(result_dict["predictions"]) == len(json.loads(input_args["prediction_dataset"]))

    # Check that predictions are integers (-1 for outliers, 1 for inliers)
    for prediction in result_dict["predictions"]:
        assert isinstance(prediction, int)
        assert prediction in [-1, 1]

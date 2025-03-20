"""Unit tests for the isolation forest tool."""

import json
from typing import Any, Dict

import pytest

from mcp_server_scikit_learn.tools.isolation_forest_tool import (
    handle_isolation_forest_tool,
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
                "n_estimators": 100,
                "max_samples": "auto",
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
                "n_estimators": 50,
                "max_samples": 0.8,
                "contamination": 0.1,
                "max_features": 1.0,
                "bootstrap": True,
                "random_state": 42,
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
                "n_estimators": 100,
                "max_samples": 0.7,
                "contamination": "auto",
            },
            ["description", "predictions"],
        ),
    ],
)
async def test_handle_isolation_forest_tool(
    mocker,
    scope_function,
    input_args: Dict[str, Any],
    expected_keys: list[str],
) -> None:
    """Test the isolation forest tool handler.

    Args:
        input_args: Input arguments for the tool.
        expected_keys: Expected keys in the response.
    """
    # Convert training and prediction datasets to strings
    input_args["training_dataset"] = json.dumps(input_args["training_dataset"])
    input_args["prediction_dataset"] = json.dumps(input_args["prediction_dataset"])

    # Call the handler
    result = await handle_isolation_forest_tool(input_args)

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

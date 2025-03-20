"""Unit tests for the decision tree regressor tool."""

import json
from typing import Any, Dict

import pytest

from mcp_server_scikit_learn.tools.decision_tree_regressor_tool import (
    handle_decision_tree_regressor_tool,
)


@pytest.mark.asyncio
@pytest.mark.usefixtures("scope_function")
@pytest.mark.parametrize(
    ("input_args", "expected_keys"),
    [
        (
            {
                "training_dataset": [
                    {"feature1": 1.0, "feature2": 2.0, "target": 0.5},
                    {"feature1": 2.0, "feature2": 3.0, "target": 1.5},
                ],
                "prediction_dataset": [
                    {"feature1": 1.5, "feature2": 2.5},
                    {"feature1": 2.5, "feature2": 3.5},
                ],
                "target_column": "target",
                "max_depth": 5,
            },
            ["description", "predictions"],
        ),
        (
            {
                "training_dataset": [
                    {"feature1": 1.0, "feature2": 2.0, "target": 0.5},
                    {"feature1": 2.0, "feature2": 3.0, "target": 1.5},
                ],
                "prediction_dataset": [
                    {"feature1": 1.5, "feature2": 2.5},
                    {"feature1": 2.5, "feature2": 3.5},
                ],
                "target_column": "target",
                "criterion": "absolute_error",
                "splitter": "random",
                "max_depth": 3,
                "min_samples_split": 2,
                "min_samples_leaf": 1,
                "max_features": "sqrt",
            },
            ["description", "predictions"],
        ),
        (
            {
                "training_dataset": [
                    {"feature1": 1.0, "feature2": 2.0, "target": 0.5},
                    {"feature1": 2.0, "feature2": 3.0, "target": 1.5},
                ],
                "prediction_dataset": [
                    {"feature1": 1.5, "feature2": 2.5},
                    {"feature1": 2.5, "feature2": 3.5},
                ],
                "target_column": "target",
                "column_to_ignore": ["feature2"],
                "criterion": "poisson",
                "max_leaf_nodes": 5,
            },
            ["description", "predictions"],
        ),
    ],
)
async def test_handle_decision_tree_regressor_tool(
    mocker,
    scope_function,
    input_args: Dict[str, Any],
    expected_keys: list[str],
) -> None:
    """Test the decision tree regressor tool handler.

    Args:
        input_args: Input arguments for the tool.
        expected_keys: Expected keys in the response.
    """
    # Convert training and prediction datasets to strings
    input_args["training_dataset"] = json.dumps(input_args["training_dataset"])
    input_args["prediction_dataset"] = json.dumps(input_args["prediction_dataset"])

    # Call the handler
    result = await handle_decision_tree_regressor_tool(input_args)

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

    # Check that predictions are floats (since this is a regressor)
    for prediction in result_dict["predictions"]:
        assert isinstance(prediction, float)

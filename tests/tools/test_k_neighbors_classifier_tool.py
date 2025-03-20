"""Unit tests for the k-neighbors classifier tool."""

import json
import random
from typing import Any, Dict

import pytest

from mcp_server_scikit_learn.tools.k_neighbors_classifier_tool import (
    handle_k_neighbors_classifier_tool,
)


@pytest.mark.asyncio
@pytest.mark.usefixtures("scope_function")
@pytest.mark.parametrize(
    ("input_args", "expected_keys"),
    [
        (
            {
                "training_dataset": [
                    {"feature1": random.randint(1, 10), "feature2": random.randint(1, 20), "target": i % 2}
                    for i in range(20)
                ],
                "prediction_dataset": [
                    {"feature1": random.randint(1, 10), "feature2": random.randint(1, 20)} for _ in range(10)
                ],
                "target_column": "target",
                "n_neighbors": 3,
            },
            ["description", "predictions"],
        ),
        (
            {
                "training_dataset": [
                    {"feature1": random.randint(1, 10), "feature2": random.randint(1, 20), "target": i % 2}
                    for i in range(20)
                ],
                "prediction_dataset": [
                    {"feature1": random.randint(1, 10), "feature2": random.randint(1, 20)} for _ in range(10)
                ],
                "target_column": "target",
                "n_neighbors": 5,
                "weights": "distance",
                "algorithm": "ball_tree",
                "leaf_size": 30,
                "p": 2,
                "metric": "minkowski",
            },
            ["description", "predictions"],
        ),
        (
            {
                "training_dataset": [
                    {"feature1": random.randint(1, 10), "feature2": random.randint(1, 20), "target": i % 2}
                    for i in range(20)
                ],
                "prediction_dataset": [
                    {"feature1": random.randint(1, 10), "feature2": random.randint(1, 20)} for _ in range(10)
                ],
                "target_column": "target",
                "column_to_ignore": ["feature2"],
                "n_neighbors": 3,
                "weights": "uniform",
                "algorithm": "kd_tree",
            },
            ["description", "predictions"],
        ),
    ],
)
async def test_handle_k_neighbors_classifier_tool(
    mocker,
    scope_function,
    input_args: Dict[str, Any],
    expected_keys: list[str],
) -> None:
    """Test the k-neighbors classifier tool handler.

    Args:
        input_args: Input arguments for the tool.
        expected_keys: Expected keys in the response.
    """
    # Convert training and prediction datasets to strings
    input_args["training_dataset"] = json.dumps(input_args["training_dataset"])
    input_args["prediction_dataset"] = json.dumps(input_args["prediction_dataset"])

    # Call the handler
    result = await handle_k_neighbors_classifier_tool(input_args)

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

    # Check that predictions are integers (0 or 1 for binary classification)
    for prediction in result_dict["predictions"]:
        assert isinstance(prediction, int)
        assert prediction in [0, 1]

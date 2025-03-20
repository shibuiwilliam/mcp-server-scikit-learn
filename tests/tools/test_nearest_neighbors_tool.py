"""Unit tests for the nearest neighbors tool."""

import json
import random
from typing import Any, Dict

import pytest

from mcp_server_scikit_learn.tools.nearest_neighbors_tool import (
    handle_nearest_neighbors_tool,
)


@pytest.mark.asyncio
@pytest.mark.usefixtures("scope_function")
@pytest.mark.parametrize(
    ("input_args", "expected_keys"),
    [
        (
            {
                "training_dataset": [
                    {"feature1": random.randint(1, 10), "feature2": random.randint(1, 20)} for _ in range(20)
                ],
                "prediction_dataset": [
                    {"feature1": random.randint(1, 10), "feature2": random.randint(1, 20)} for _ in range(10)
                ],
                "n_neighbors": 2,
            },
            ["description", "distances", "indices"],
        ),
        (
            {
                "training_dataset": [
                    {"feature1": random.randint(1, 10), "feature2": random.randint(1, 20)} for _ in range(20)
                ],
                "prediction_dataset": [
                    {"feature1": random.randint(1, 10), "feature2": random.randint(1, 20)} for _ in range(10)
                ],
                "n_neighbors": 3,
                "algorithm": "ball_tree",
                "leaf_size": 30,
                "p": 2,
                "metric": "minkowski",
                "return_distance": True,
            },
            ["description", "distances", "indices"],
        ),
        (
            {
                "training_dataset": [
                    {"feature1": random.randint(1, 10), "feature2": random.randint(1, 20)} for _ in range(20)
                ],
                "prediction_dataset": [
                    {"feature1": random.randint(1, 10), "feature2": random.randint(1, 20)} for _ in range(10)
                ],
                "column_to_ignore": ["feature2"],
                "n_neighbors": 2,
                "algorithm": "kd_tree",
                "return_distance": False,
            },
            ["description", "indices"],
        ),
    ],
)
async def test_handle_nearest_neighbors_tool(
    mocker,
    scope_function,
    input_args: Dict[str, Any],
    expected_keys: list[str],
) -> None:
    """Test the nearest neighbors tool handler.

    Args:
        input_args: Input arguments for the tool.
        expected_keys: Expected keys in the response.
    """
    # Convert training and prediction datasets to strings
    input_args["training_dataset"] = json.dumps(input_args["training_dataset"])
    input_args["prediction_dataset"] = json.dumps(input_args["prediction_dataset"])

    # Call the handler
    result = await handle_nearest_neighbors_tool(input_args)

    # Check that we got a TextContent result
    assert len(result) == 1
    assert result[0].type == "text"

    # Parse the result
    result_dict = json.loads(result[0].text)

    # Check that all expected keys are present
    for key in expected_keys:
        assert key in result_dict

    # Check that indices is a list
    assert isinstance(result_dict["indices"], list)

    # Check that the number of predictions matches the prediction dataset
    assert len(result_dict["indices"]) == len(json.loads(input_args["prediction_dataset"]))

    # Check that each prediction has the correct number of neighbors
    for indices in result_dict["indices"]:
        assert isinstance(indices, list)
        assert len(indices) == input_args["n_neighbors"]
        assert all(isinstance(idx, int) for idx in indices)
        assert all(0 <= idx < len(json.loads(input_args["training_dataset"])) for idx in indices)

    # If distances are returned, check their format
    if "distances" in result_dict:
        assert isinstance(result_dict["distances"], list)
        assert len(result_dict["distances"]) == len(json.loads(input_args["prediction_dataset"]))
        for distances in result_dict["distances"]:
            assert isinstance(distances, list)
            assert len(distances) == input_args["n_neighbors"]
            assert all(isinstance(d, float) for d in distances)
            assert all(d >= 0 for d in distances)

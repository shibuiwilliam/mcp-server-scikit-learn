"""Unit tests for the k-means tool."""

import json
from typing import Any, Dict

import pytest

from mcp_server_scikit_learn.tools.k_means_tool import (
    handle_k_means_tool,
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
                    {"feature1": 10.0, "feature2": 11.0},
                ],
                "prediction_dataset": [
                    {"feature1": 1.5, "feature2": 2.5},
                    {"feature1": 2.5, "feature2": 3.5},
                    {"feature1": 10.5, "feature2": 11.5},
                ],
                "n_clusters": 2,
                "init": "k-means++",
            },
            ["description", "labels", "distances"],
        ),
        (
            {
                "training_dataset": [
                    {"feature1": 1.0, "feature2": 2.0},
                    {"feature1": 2.0, "feature2": 3.0},
                    {"feature1": 10.0, "feature2": 11.0},
                ],
                "prediction_dataset": [
                    {"feature1": 1.5, "feature2": 2.5},
                    {"feature1": 2.5, "feature2": 3.5},
                    {"feature1": 10.5, "feature2": 11.5},
                ],
                "n_clusters": 3,
                "init": "random",
                "n_init": 10,
                "max_iter": 300,
                "tol": 1e-4,
                "random_state": 42,
            },
            ["description", "labels", "distances"],
        ),
        (
            {
                "training_dataset": [
                    {"feature1": 1.0, "feature2": 2.0},
                    {"feature1": 2.0, "feature2": 3.0},
                    {"feature1": 10.0, "feature2": 11.0},
                ],
                "prediction_dataset": [
                    {"feature1": 1.5, "feature2": 2.5},
                    {"feature1": 2.5, "feature2": 3.5},
                    {"feature1": 10.5, "feature2": 11.5},
                ],
                "column_to_ignore": ["feature2"],
                "n_clusters": 2,
                "init": "k-means++",
                "algorithm": "elkan",
            },
            ["description", "labels", "distances"],
        ),
    ],
)
async def test_handle_k_means_tool(
    mocker,
    scope_function,
    input_args: Dict[str, Any],
    expected_keys: list[str],
) -> None:
    """Test the k-means tool handler.

    Args:
        input_args: Input arguments for the tool.
        expected_keys: Expected keys in the response.
    """
    # Convert training and prediction datasets to strings
    input_args["training_dataset"] = json.dumps(input_args["training_dataset"])
    input_args["prediction_dataset"] = json.dumps(input_args["prediction_dataset"])

    # Call the handler
    result = await handle_k_means_tool(input_args)

    # Check that we got a TextContent result
    assert len(result) == 1
    assert result[0].type == "text"

    # Parse the result
    result_dict = json.loads(result[0].text)

    # Check that all expected keys are present
    for key in expected_keys:
        assert key in result_dict

    # Check that labels is a list
    assert isinstance(result_dict["labels"], list)

    # Check that the number of labels matches the prediction dataset
    assert len(result_dict["labels"]) == len(json.loads(input_args["prediction_dataset"]))

    # Check that labels are integers (cluster labels)
    for label in result_dict["labels"]:
        assert isinstance(label, int)
        assert label >= 0
        assert label < input_args["n_clusters"]

    # Check that distances is a list
    assert isinstance(result_dict["distances"], list)

    # Check that the number of distance lists matches the prediction dataset
    assert len(result_dict["distances"]) == len(json.loads(input_args["prediction_dataset"]))

    # Check that each distance list has the correct number of distances (one per cluster)
    for distances in result_dict["distances"]:
        assert isinstance(distances, list)
        assert len(distances) == input_args["n_clusters"]
        assert all(isinstance(d, float) for d in distances)
        assert all(d >= 0 for d in distances)

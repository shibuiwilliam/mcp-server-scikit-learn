"""Tool for finding nearest neighbors using scikit-learn's NearestNeighbors."""

import json
from typing import Any, Optional

import mcp.types as types
import polars as pl
from pydantic import BaseModel, ConfigDict
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

from ..configure_logger import make_logger
from ..utils.string_util import string_to_list_dict
from .tools import MCPServerScikitLearnTools

logger = make_logger(__name__)


class NearestNeighborsInputSchema(BaseModel):
    """Input schema for the Nearest Neighbors tool.

    Defines the structure and validation rules for input data and hyperparameters.
    """

    model_config = ConfigDict(
        validate_assignment=True,
        frozen=True,
        extra="forbid",
        arbitrary_types_allowed=True,
    )

    training_dataset: list[dict[str, str | int | float | bool | None]]
    prediction_dataset: list[dict[str, str | int | float | bool | None]]
    column_to_ignore: Optional[list[str]] = None
    # Nearest Neighbors hyperparameters
    n_neighbors: Optional[int] = 5  # Number of neighbors to use
    radius: Optional[float] = 1.0  # Range of parameter space to use by default for radius_neighbors queries
    algorithm: Optional[str] = "auto"  # Options: "auto", "ball_tree", "kd_tree", "brute"
    leaf_size: Optional[int] = 30  # Leaf size passed to BallTree or KDTree
    p: Optional[int] = 2  # Power parameter for the Minkowski metric
    metric: Optional[str] = "minkowski"  # Distance metric to use
    metric_params: Optional[dict[str, Any]] = None  # Additional keyword arguments for the metric function
    n_jobs: Optional[int] = None  # Number of parallel jobs to run for neighbor search

    @staticmethod
    def input_schema() -> dict:
        """Get the JSON schema for input validation.

        Returns:
            A dictionary containing the JSON schema for input validation.
        """
        return {
            "type": "object",
            "properties": {
                "training_dataset": {
                    "type": "string",
                    "description": "The dataset to train in Base64 encoded string or dumped JSON.",
                },
                "prediction_dataset": {
                    "type": "string",
                    "description": "The dataset to predict in Base64 encoded string or dumped JSON.",
                },
                "column_to_ignore": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "The columns to ignore in training and prediction.",
                },
                "n_neighbors": {
                    "type": "integer",
                    "description": "Number of neighbors to use by default for kneighbors queries.",
                },
                "radius": {
                    "type": "number",
                    "description": "Range of parameter space to use by default for radius_neighbors queries.",
                },
                "algorithm": {
                    "type": "string",
                    "enum": ["auto", "ball_tree", "kd_tree", "brute"],
                    "description": "Algorithm used to compute the nearest neighbors.",
                },
                "leaf_size": {
                    "type": "integer",
                    "description": "Leaf size passed to BallTree or KDTree.",
                },
                "p": {
                    "type": "integer",
                    "description": "Power parameter for the Minkowski metric.",
                },
                "metric": {
                    "type": "string",
                    "description": "Distance metric to use for the tree.",
                },
                "metric_params": {
                    "type": "object",
                    "description": "Additional keyword arguments for the metric function.",
                },
                "n_jobs": {
                    "type": ["integer", "null"],
                    "description": "Number of parallel jobs to run for neighbor search.",
                },
            },
        }

    @staticmethod
    def from_str(
        training_dataset: str,
        prediction_dataset: str,
        column_to_ignore: Optional[list[str]] = None,
        n_neighbors: Optional[int] = 5,
        radius: Optional[float] = 1.0,
        algorithm: Optional[str] = "auto",
        leaf_size: Optional[int] = 30,
        p: Optional[int] = 2,
        metric: Optional[str] = "minkowski",
        metric_params: Optional[dict[str, Any]] = None,
        n_jobs: Optional[int] = None,
    ) -> "NearestNeighborsInputSchema":
        """Create a schema instance from string inputs.

        Args:
            training_dataset: Training data as a string.
            prediction_dataset: Prediction data as a string.
            column_to_ignore: List of columns to ignore.
            n_neighbors: Number of neighbors to use.
            radius: Range of parameter space for radius_neighbors queries.
            algorithm: Algorithm used to compute nearest neighbors.
            leaf_size: Leaf size for tree algorithms.
            p: Power parameter for Minkowski metric.
            metric: Distance metric to use.
            metric_params: Additional parameters for the metric function.
            n_jobs: Number of parallel jobs.

        Returns:
            A NearestNeighborsInputSchema instance.
        """
        structured_training_dataset = string_to_list_dict(training_dataset)
        structured_prediction_dataset = string_to_list_dict(prediction_dataset)

        return NearestNeighborsInputSchema(
            training_dataset=structured_training_dataset,
            prediction_dataset=structured_prediction_dataset,
            column_to_ignore=column_to_ignore,
            n_neighbors=n_neighbors,
            radius=radius,
            algorithm=algorithm,
            leaf_size=leaf_size,
            p=p,
            metric=metric,
            metric_params=metric_params,
            n_jobs=n_jobs,
        )


nearest_neighbors_tool = types.Tool(
    name=MCPServerScikitLearnTools.NEAREST_NEIGHBORS.value,
    description="Find nearest neighbors using scikit-learn's NearestNeighbors.",
    inputSchema=NearestNeighborsInputSchema.input_schema(),
)


async def handle_nearest_neighbors_tool(
    arguments: dict[str, Any],
) -> list[types.TextContent | types.ImageContent | types.EmbeddedResource]:
    """Handle requests for the Nearest Neighbors tool.

    Args:
        arguments: Dictionary containing input data and hyperparameters.

    Returns:
        List containing the nearest neighbors indices and distances.
    """
    logger.info("Handling nearest neighbors tool")

    # Extract input data and hyperparameters
    training_dataset = arguments["training_dataset"]
    prediction_dataset = arguments["prediction_dataset"]
    column_to_ignore = arguments.get("column_to_ignore", None)

    # Get hyperparameters from arguments
    n_neighbors = arguments.get("n_neighbors", 5)
    radius = arguments.get("radius", 1.0)
    algorithm = arguments.get("algorithm", "auto")
    leaf_size = arguments.get("leaf_size", 30)
    p = arguments.get("p", 2)
    metric = arguments.get("metric", "minkowski")
    metric_params = arguments.get("metric_params", None)
    n_jobs = arguments.get("n_jobs", None)

    # Create input schema and validate data
    nearest_neighbors_input = NearestNeighborsInputSchema.from_str(
        training_dataset,
        prediction_dataset,
        column_to_ignore,
        n_neighbors=n_neighbors,
        radius=radius,
        algorithm=algorithm,
        leaf_size=leaf_size,
        p=p,
        metric=metric,
        metric_params=metric_params,
        n_jobs=n_jobs,
    )

    # Convert input data to DataFrames
    train_df = pl.DataFrame(nearest_neighbors_input.training_dataset)
    prediction_df = pl.DataFrame(nearest_neighbors_input.prediction_dataset)
    training_columns = []
    numerical_columns_scalars = {}
    categorical_columns_encoders = {}

    # Preprocess training data
    for col in train_df.columns:
        if nearest_neighbors_input.column_to_ignore and col in nearest_neighbors_input.column_to_ignore:
            continue

        training_columns.append(col)
        if train_df[col].dtype in [pl.Int64, pl.Float64]:
            scaler = MinMaxScaler()
            train_df = train_df.with_columns(
                [pl.Series(col, scaler.fit_transform(train_df[col].to_numpy().reshape(-1, 1)).flatten())]
            )
            numerical_columns_scalars[col] = scaler
        if train_df[col].dtype == pl.Utf8:
            label_encoder = LabelEncoder()
            train_df = train_df.with_columns(pl.Series(col, label_encoder.fit_transform(train_df[col].to_list())))
            categorical_columns_encoders[col] = label_encoder

    # Preprocess prediction data
    for col in prediction_df.columns:
        if nearest_neighbors_input.column_to_ignore and col in nearest_neighbors_input.column_to_ignore:
            continue
        if col not in training_columns:
            continue
        if col in numerical_columns_scalars.keys():
            prediction_df = prediction_df.with_columns(
                [
                    pl.Series(
                        col,
                        numerical_columns_scalars[col]
                        .transform(prediction_df[col].to_numpy().reshape(-1, 1))
                        .flatten(),
                    )
                ]
            )
        if col in categorical_columns_encoders.keys():
            prediction_df = prediction_df.with_columns(
                pl.Series(col, categorical_columns_encoders[col].transform(prediction_df[col].to_list()))
            )

    # Prepare features for training and prediction
    x = train_df
    if nearest_neighbors_input.column_to_ignore:
        x = x.drop(*nearest_neighbors_input.column_to_ignore)
    x = x.to_numpy()

    prediction_x = prediction_df
    if nearest_neighbors_input.column_to_ignore:
        prediction_x = prediction_x.drop(*nearest_neighbors_input.column_to_ignore)
    prediction_x = prediction_x.to_numpy()

    # Train the model
    estimator = NearestNeighbors(
        n_neighbors=nearest_neighbors_input.n_neighbors,
        radius=nearest_neighbors_input.radius,
        algorithm=nearest_neighbors_input.algorithm,
        leaf_size=nearest_neighbors_input.leaf_size,
        p=nearest_neighbors_input.p,
        metric=nearest_neighbors_input.metric,
        metric_params=nearest_neighbors_input.metric_params,
        n_jobs=nearest_neighbors_input.n_jobs,
    )
    estimator.fit(x)

    # Find nearest neighbors
    distances, indices = estimator.kneighbors(prediction_x)

    return [
        types.TextContent(
            type="text",
            text=json.dumps(
                {
                    "description": "Result of Nearest Neighbors search.",
                    "distances": distances.tolist(),
                    "indices": indices.tolist(),
                }
            ),
        )
    ]

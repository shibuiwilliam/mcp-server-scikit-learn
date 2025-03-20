"""Tool for training and predicting with scikit-learn's KNeighborsRegressor."""

import json
from typing import Any, Optional

import mcp.types as types
import polars as pl
from pydantic import BaseModel, ConfigDict
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

from ..configure_logger import make_logger
from ..utils.string_util import string_to_list_dict
from .tools import MCPServerScikitLearnTools

logger = make_logger(__name__)


class KNeighborsRegressorInputSchema(BaseModel):
    """Input schema for the K-Neighbors Regressor tool.

    Defines the structure and validation rules for input data and hyperparameters.
    """

    model_config = ConfigDict(
        validate_assignment=True,
        frozen=True,
        extra="forbid",
        arbitrary_types_allowed=True,
    )

    training_dataset: list[dict[str, str | int | float | bool | None]]
    target_column: str
    prediction_dataset: list[dict[str, str | int | float | bool | None]]
    column_to_ignore: Optional[list[str]] = None
    # K-Neighbors hyperparameters
    n_neighbors: Optional[int] = 5  # Number of neighbors to use
    weights: Optional[str] = "uniform"  # Options: "uniform", "distance"
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
                "target_column": {
                    "type": "string",
                    "description": "The target column to predict from training_dataset.",
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
                "weights": {
                    "type": "string",
                    "enum": ["uniform", "distance"],
                    "description": "Weight function used in prediction.",
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
        target_column: str,
        prediction_dataset: str,
        column_to_ignore: Optional[list[str]] = None,
        n_neighbors: Optional[int] = 5,
        weights: Optional[str] = "uniform",
        algorithm: Optional[str] = "auto",
        leaf_size: Optional[int] = 30,
        p: Optional[int] = 2,
        metric: Optional[str] = "minkowski",
        metric_params: Optional[dict[str, Any]] = None,
        n_jobs: Optional[int] = None,
    ) -> "KNeighborsRegressorInputSchema":
        """Create a schema instance from string inputs.

        Args:
            training_dataset: Training data as a string.
            target_column: Name of the target column.
            prediction_dataset: Prediction data as a string.
            column_to_ignore: List of columns to ignore.
            n_neighbors: Number of neighbors to use.
            weights: Weight function used in prediction.
            algorithm: Algorithm used to compute nearest neighbors.
            leaf_size: Leaf size for tree algorithms.
            p: Power parameter for Minkowski metric.
            metric: Distance metric to use.
            metric_params: Additional parameters for the metric function.
            n_jobs: Number of parallel jobs.

        Returns:
            A KNeighborsRegressorInputSchema instance.
        """
        structured_training_dataset = string_to_list_dict(training_dataset)
        structured_prediction_dataset = string_to_list_dict(prediction_dataset)

        return KNeighborsRegressorInputSchema(
            training_dataset=structured_training_dataset,
            target_column=target_column,
            prediction_dataset=structured_prediction_dataset,
            column_to_ignore=column_to_ignore,
            n_neighbors=n_neighbors,
            weights=weights,
            algorithm=algorithm,
            leaf_size=leaf_size,
            p=p,
            metric=metric,
            metric_params=metric_params,
            n_jobs=n_jobs,
        )


k_neighbors_regressor_tool = types.Tool(
    name=MCPServerScikitLearnTools.K_NEIGHBORS_REGRESSOR.value,
    description="Train and predict with K-Neighbors Regressor.",
    inputSchema=KNeighborsRegressorInputSchema.input_schema(),
)


async def handle_k_neighbors_regressor_tool(
    arguments: dict[str, Any],
) -> list[types.TextContent | types.ImageContent | types.EmbeddedResource]:
    """Handle requests for the K-Neighbors Regressor tool.

    Args:
        arguments: Dictionary containing input data and hyperparameters.

    Returns:
        List containing the model's predictions.
    """
    logger.info("Handling k-neighbors regressor tool")

    # Extract input data and hyperparameters
    training_dataset = arguments["training_dataset"]
    target_column = arguments["target_column"]
    prediction_dataset = arguments["prediction_dataset"]
    column_to_ignore = arguments.get("column_to_ignore", None)

    # Get hyperparameters from arguments
    n_neighbors = arguments.get("n_neighbors", 5)
    weights = arguments.get("weights", "uniform")
    algorithm = arguments.get("algorithm", "auto")
    leaf_size = arguments.get("leaf_size", 30)
    p = arguments.get("p", 2)
    metric = arguments.get("metric", "minkowski")
    metric_params = arguments.get("metric_params", None)
    n_jobs = arguments.get("n_jobs", None)

    # Create input schema and validate data
    k_neighbors_regressor_input = KNeighborsRegressorInputSchema.from_str(
        training_dataset,
        target_column,
        prediction_dataset,
        column_to_ignore,
        n_neighbors=n_neighbors,
        weights=weights,
        algorithm=algorithm,
        leaf_size=leaf_size,
        p=p,
        metric=metric,
        metric_params=metric_params,
        n_jobs=n_jobs,
    )

    # Convert input data to DataFrames
    train_df = pl.DataFrame(k_neighbors_regressor_input.training_dataset)
    prediction_df = pl.DataFrame(k_neighbors_regressor_input.prediction_dataset)
    training_columns = []
    numerical_columns_scalars = {}
    categorical_columns_encoders = {}

    # Preprocess training data
    for col in train_df.columns:
        if col == k_neighbors_regressor_input.target_column:
            continue
        if k_neighbors_regressor_input.column_to_ignore and col in k_neighbors_regressor_input.column_to_ignore:
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
        if col == k_neighbors_regressor_input.target_column:
            continue
        if k_neighbors_regressor_input.column_to_ignore and col in k_neighbors_regressor_input.column_to_ignore:
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

    # Prepare features and target for training
    x = train_df.drop(k_neighbors_regressor_input.target_column)
    if k_neighbors_regressor_input.column_to_ignore:
        x = x.drop(*k_neighbors_regressor_input.column_to_ignore)
    x = x.to_numpy()
    y = train_df[k_neighbors_regressor_input.target_column].to_numpy()

    # Train the model
    estimator = KNeighborsRegressor(
        n_neighbors=k_neighbors_regressor_input.n_neighbors,
        weights=k_neighbors_regressor_input.weights,
        algorithm=k_neighbors_regressor_input.algorithm,
        leaf_size=k_neighbors_regressor_input.leaf_size,
        p=k_neighbors_regressor_input.p,
        metric=k_neighbors_regressor_input.metric,
        metric_params=k_neighbors_regressor_input.metric_params,
        n_jobs=k_neighbors_regressor_input.n_jobs,
    )
    estimator.fit(x, y)

    # Prepare features for prediction
    prediction_x = (
        prediction_df.drop(k_neighbors_regressor_input.target_column)
        if k_neighbors_regressor_input.target_column in prediction_df.columns
        else prediction_df
    )
    if k_neighbors_regressor_input.column_to_ignore:
        prediction_x = prediction_x.drop(*k_neighbors_regressor_input.column_to_ignore)
    prediction_x = prediction_x.to_numpy()

    # Generate predictions
    predictions = estimator.predict(prediction_x).tolist()

    return [
        types.TextContent(
            type="text",
            text=json.dumps(
                {
                    "description": "Result of K-Neighbors Regressor.",
                    "predictions": predictions,
                }
            ),
        )
    ]

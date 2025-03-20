"""Tool for clustering with scikit-learn's KMeans."""

import json
from typing import Any, Optional

import mcp.types as types
import polars as pl
from pydantic import BaseModel, ConfigDict
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

from ..configure_logger import make_logger
from ..utils.string_util import string_to_list_dict
from .tools import MCPServerScikitLearnTools

logger = make_logger(__name__)


class KMeansInputSchema(BaseModel):
    """Input schema for the K-Means clustering tool.

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
    # K-Means hyperparameters
    n_clusters: int = 8  # Number of clusters to form
    init: Optional[str] = "k-means++"  # Options: "k-means++", "random", or array-like
    n_init: Optional[int] = 10  # Number of times the k-means algorithm will be run
    max_iter: Optional[int] = 300  # Maximum number of iterations
    tol: Optional[float] = 1e-4  # Relative tolerance with regards to Frobenius norm
    algorithm: Optional[str] = "lloyd"  # Options: "lloyd", "elkan"

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
                "n_clusters": {
                    "type": "integer",
                    "description": "Number of clusters to form.",
                },
                "init": {
                    "type": "string",
                    "enum": ["k-means++", "random"],
                    "description": "Method for initialization.",
                },
                "n_init": {
                    "type": "integer",
                    "description": "Number of times the k-means algorithm will be run.",
                },
                "max_iter": {
                    "type": "integer",
                    "description": "Maximum number of iterations.",
                },
                "tol": {
                    "type": "number",
                    "description": "Relative tolerance with regards to Frobenius norm.",
                },
                "algorithm": {
                    "type": "string",
                    "enum": ["lloyd", "elkan"],
                    "description": "K-means algorithm to use.",
                },
            },
        }

    @staticmethod
    def from_str(
        training_dataset: str,
        prediction_dataset: str,
        column_to_ignore: Optional[list[str]] = None,
        n_clusters: int = 8,
        init: Optional[str] = "k-means++",
        n_init: Optional[int] = 10,
        max_iter: Optional[int] = 300,
        tol: Optional[float] = 1e-4,
        algorithm: Optional[str] = "lloyd",
    ) -> "KMeansInputSchema":
        """Create a schema instance from string inputs.

        Args:
            training_dataset: Training data as a string.
            prediction_dataset: Prediction data as a string.
            column_to_ignore: List of columns to ignore.
            n_clusters: Number of clusters to form.
            init: Method for initialization.
            n_init: Number of times the k-means algorithm will be run.
            max_iter: Maximum number of iterations.
            tol: Relative tolerance with regards to Frobenius norm.
            algorithm: K-means algorithm to use.

        Returns:
            A KMeansInputSchema instance.
        """
        structured_training_dataset = string_to_list_dict(training_dataset)
        structured_prediction_dataset = string_to_list_dict(prediction_dataset)

        return KMeansInputSchema(
            training_dataset=structured_training_dataset,
            prediction_dataset=structured_prediction_dataset,
            column_to_ignore=column_to_ignore,
            n_clusters=n_clusters,
            init=init,
            n_init=n_init,
            max_iter=max_iter,
            tol=tol,
            algorithm=algorithm,
        )


k_means_tool = types.Tool(
    name=MCPServerScikitLearnTools.K_MEANS.value,
    description="Perform K-Means clustering using scikit-learn's KMeans.",
    inputSchema=KMeansInputSchema.input_schema(),
)


async def handle_k_means_tool(
    arguments: dict[str, Any],
) -> list[types.TextContent | types.ImageContent | types.EmbeddedResource]:
    """Handle requests for the K-Means clustering tool.

    Args:
        arguments: Dictionary containing input data and hyperparameters.

    Returns:
        List containing the cluster assignments and distances to cluster centers.
    """
    logger.info("Handling k-means clustering tool")

    # Extract input data and hyperparameters
    training_dataset = arguments["training_dataset"]
    prediction_dataset = arguments["prediction_dataset"]
    column_to_ignore = arguments.get("column_to_ignore", None)

    # Get hyperparameters from arguments
    n_clusters = arguments.get("n_clusters", 8)
    init = arguments.get("init", "k-means++")
    n_init = arguments.get("n_init", 10)
    max_iter = arguments.get("max_iter", 300)
    tol = arguments.get("tol", 1e-4)
    algorithm = arguments.get("algorithm", "lloyd")

    # Create input schema and validate data
    k_means_input = KMeansInputSchema.from_str(
        training_dataset,
        prediction_dataset,
        column_to_ignore,
        n_clusters=n_clusters,
        init=init,
        n_init=n_init,
        max_iter=max_iter,
        tol=tol,
        algorithm=algorithm,
    )

    # Convert input data to DataFrames
    train_df = pl.DataFrame(k_means_input.training_dataset)
    prediction_df = pl.DataFrame(k_means_input.prediction_dataset)
    training_columns = []
    numerical_columns_scalars = {}
    categorical_columns_encoders = {}

    # Preprocess training data
    for col in train_df.columns:
        if k_means_input.column_to_ignore and col in k_means_input.column_to_ignore:
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
        if k_means_input.column_to_ignore and col in k_means_input.column_to_ignore:
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
    if k_means_input.column_to_ignore:
        x = x.drop(*k_means_input.column_to_ignore)
    x = x.to_numpy()

    prediction_x = prediction_df
    if k_means_input.column_to_ignore:
        prediction_x = prediction_x.drop(*k_means_input.column_to_ignore)
    prediction_x = prediction_x.to_numpy()

    # Train the model with internal-only parameters
    estimator = KMeans(
        n_clusters=k_means_input.n_clusters,
        init=k_means_input.init,
        n_init=k_means_input.n_init,
        max_iter=k_means_input.max_iter,
        tol=k_means_input.tol,
        verbose=0,  # Internal-only parameter
        random_state=42,  # Internal-only parameter
        algorithm=k_means_input.algorithm,
    )
    estimator.fit(x)

    # Get cluster assignments and distances
    labels = estimator.predict(prediction_x).tolist()
    distances = estimator.transform(prediction_x).tolist()

    return [
        types.TextContent(
            type="text",
            text=json.dumps(
                {
                    "description": "Result of K-Means clustering.",
                    "labels": labels,
                    "distances": distances,
                }
            ),
        )
    ]

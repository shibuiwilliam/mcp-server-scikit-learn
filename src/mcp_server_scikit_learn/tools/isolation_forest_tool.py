import json
from typing import Any, Optional

import mcp.types as types
import polars as pl
from pydantic import BaseModel, ConfigDict
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

from ..configure_logger import make_logger
from ..utils.string_util import string_to_list_dict
from .tools import MCPServerScikitLearnTools

logger = make_logger(__name__)


class IsolationForestInputSchema(BaseModel):
    model_config = ConfigDict(
        validate_assignment=True,
        frozen=True,
        extra="forbid",
        arbitrary_types_allowed=True,
    )

    training_dataset: list[dict[str, str | int | float | bool | None]]
    prediction_dataset: list[dict[str, str | int | float | bool | None]]
    column_to_ignore: Optional[list[str]] = None
    # Isolation Forest hyperparameters
    n_estimators: Optional[int] = 100
    max_samples: Optional[str | int | float] = "auto"
    contamination: Optional[str | float] = "auto"
    max_features: Optional[str | int | float] = 1.0
    bootstrap: Optional[bool] = False
    verbose: Optional[int] = 0

    @staticmethod
    def input_schema() -> dict:
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
                "n_estimators": {
                    "type": "integer",
                    "description": "The number of base estimators in the ensemble.",
                },
                "max_samples": {
                    "type": ["string", "integer", "number"],
                    "description": "The number of samples to draw from X to train each base estimator.",
                },
                "contamination": {
                    "type": ["string", "number"],
                    "description": "The amount of contamination of the data set, i.e. the proportion of outliers in the data set.",
                },
                "max_features": {
                    "type": ["string", "integer", "number"],
                    "description": "The number of features to draw from X to train each base estimator.",
                },
                "bootstrap": {
                    "type": "boolean",
                    "description": "If True, individual trees are fit on random subsets of the training data sampled with replacement.",
                },
                "verbose": {
                    "type": "integer",
                    "description": "Controls the verbosity of the tree building process.",
                },
            },
        }

    @staticmethod
    def from_str(
        training_dataset: str,
        prediction_dataset: str,
        column_to_ignore: Optional[list[str]] = None,
        n_estimators: Optional[int] = 100,
        max_samples: Optional[str | int | float] = "auto",
        contamination: Optional[str | float] = "auto",
        max_features: Optional[str | int | float] = 1.0,
        bootstrap: Optional[bool] = False,
        verbose: Optional[int] = 0,
    ) -> "IsolationForestInputSchema":
        structured_training_dataset = string_to_list_dict(training_dataset)
        structured_prediction_dataset = string_to_list_dict(prediction_dataset)

        return IsolationForestInputSchema(
            training_dataset=structured_training_dataset,
            prediction_dataset=structured_prediction_dataset,
            column_to_ignore=column_to_ignore,
            n_estimators=n_estimators,
            max_samples=max_samples,
            contamination=contamination,
            max_features=max_features,
            bootstrap=bootstrap,
            verbose=verbose,
        )


isolation_forest_tool = types.Tool(
    name=MCPServerScikitLearnTools.ISOLATION_FOREST.value,
    description="Train and predict with Isolation Forest for anomaly detection.",
    inputSchema=IsolationForestInputSchema.input_schema(),
)


async def handle_isolation_forest_tool(
    arguments: dict[str, Any],
) -> list[types.TextContent | types.ImageContent | types.EmbeddedResource]:
    logger.info("Handling isolation forest tool")

    training_dataset = arguments["training_dataset"]
    prediction_dataset = arguments["prediction_dataset"]
    column_to_ignore = arguments.get("column_to_ignore", None)

    # Get hyperparameters from arguments
    n_estimators = arguments.get("n_estimators", 100)
    max_samples = arguments.get("max_samples", "auto")
    contamination = arguments.get("contamination", "auto")
    max_features = arguments.get("max_features", 1.0)
    bootstrap = arguments.get("bootstrap", False)
    verbose = arguments.get("verbose", 0)

    isolation_forest_input = IsolationForestInputSchema.from_str(
        training_dataset,
        prediction_dataset,
        column_to_ignore,
        n_estimators=n_estimators,
        max_samples=max_samples,
        contamination=contamination,
        max_features=max_features,
        bootstrap=bootstrap,
        verbose=verbose,
    )

    train_df = pl.DataFrame(isolation_forest_input.training_dataset)
    prediction_df = pl.DataFrame(isolation_forest_input.prediction_dataset)
    training_columns = []
    numerical_columns_scalars = {}
    categorical_columns_encoders = {}

    for col in train_df.columns:
        if isolation_forest_input.column_to_ignore and col in isolation_forest_input.column_to_ignore:
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

    for col in prediction_df.columns:
        if isolation_forest_input.column_to_ignore and col in isolation_forest_input.column_to_ignore:
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

    x = train_df
    if isolation_forest_input.column_to_ignore:
        x = x.drop(*isolation_forest_input.column_to_ignore)
    x = x.to_numpy()

    estimator = IsolationForest(
        n_estimators=isolation_forest_input.n_estimators,
        max_samples=isolation_forest_input.max_samples,
        contamination=isolation_forest_input.contamination,
        max_features=isolation_forest_input.max_features,
        bootstrap=isolation_forest_input.bootstrap,
        n_jobs=-1,
        random_state=42,
        verbose=isolation_forest_input.verbose,
    )
    estimator.fit(x)

    prediction_x = prediction_df
    if isolation_forest_input.column_to_ignore:
        prediction_x = prediction_x.drop(*isolation_forest_input.column_to_ignore)
    prediction_x = prediction_x.to_numpy()

    # Get both predictions and scores
    predictions = estimator.predict(prediction_x).tolist()
    scores = estimator.score_samples(prediction_x).tolist()

    return [
        types.TextContent(
            type="text",
            text=json.dumps(
                {
                    "description": "Result of Isolation Forest.",
                    "predictions": predictions,  # 1 for inliers, -1 for outliers
                    "scores": scores,  # The anomaly score of the input samples. The lower, the more abnormal.
                }
            ),
        )
    ]

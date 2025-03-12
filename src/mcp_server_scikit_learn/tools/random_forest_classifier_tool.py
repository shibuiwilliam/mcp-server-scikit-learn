import json
from typing import Any, Optional

import mcp.types as types
import polars as pl
from pydantic import BaseModel, ConfigDict
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

from ..configure_logger import make_logger
from ..utils.string_util import string_to_list_dict
from .tools import MCPServerScikitLearnTools

logger = make_logger(__name__)


class RandomForestClassifierInputSchema(BaseModel):
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
    predict_proba: bool = False

    @staticmethod
    def input_schema() -> dict:
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
                "predict_proba": {
                    "type": "bool",
                    "description": "Whether to predict probabilities or not.",
                },
            },
        }

    @staticmethod
    def from_str(
        training_dataset: str,
        target_column: str,
        prediction_dataset: str,
        column_to_ignore: Optional[list[str]] = None,
        predict_proba=False,
    ) -> "RandomForestClassifierInputSchema":
        structured_training_dataset = string_to_list_dict(training_dataset)
        structured_prediction_dataset = string_to_list_dict(prediction_dataset)

        return RandomForestClassifierInputSchema(
            training_dataset=structured_training_dataset,
            target_column=target_column,
            prediction_dataset=structured_prediction_dataset,
            column_to_ignore=column_to_ignore,
            predict_proba=predict_proba,
        )


random_forest_classifier_tool = types.Tool(
    name=MCPServerScikitLearnTools.RANDOM_FOREST_CLASSIFIER.value,
    description="Train and predict with Random Forest Classifier.",
    inputSchema=RandomForestClassifierInputSchema.input_schema(),
)


async def handle_random_forest_classifier_tool(
    arguments: dict[str, Any],
) -> list[types.TextContent | types.ImageContent | types.EmbeddedResource]:
    logger.info("Handling random forest classifier tool")

    training_dataset = arguments["training_dataset"]
    target_column = arguments["target_column"]
    prediction_dataset = arguments["prediction_dataset"]
    column_to_ignore = arguments.get("column_to_ignore", None)
    predict_proba = arguments.get("predict_proba", False)
    random_forest_classifier_input = RandomForestClassifierInputSchema.from_str(
        training_dataset, target_column, prediction_dataset, column_to_ignore, predict_proba
    )

    train_df = pl.DataFrame(random_forest_classifier_input.training_dataset)
    prediction_df = pl.DataFrame(random_forest_classifier_input.prediction_dataset)
    training_columns = []
    numerical_columns_scalars = {}
    categorical_columns_encoders = {}

    for col in train_df.columns:
        if col == random_forest_classifier_input.target_column:
            continue
        if random_forest_classifier_input.column_to_ignore and col in random_forest_classifier_input.column_to_ignore:
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
        if col == random_forest_classifier_input.target_column:
            continue
        if random_forest_classifier_input.column_to_ignore and col in random_forest_classifier_input.column_to_ignore:
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

    x = train_df.drop(random_forest_classifier_input.target_column)
    if random_forest_classifier_input.column_to_ignore:
        x = x.drop(*random_forest_classifier_input.column_to_ignore)
    x = x.to_numpy()
    y = train_df[random_forest_classifier_input.target_column].to_numpy()

    estimator = RandomForestClassifier(
        n_jobs=-1,
    )
    estimator.fit(x, y)

    prediction_x = prediction_df.drop(random_forest_classifier_input.target_column)
    if random_forest_classifier_input.column_to_ignore:
        prediction_x = prediction_x.drop(*random_forest_classifier_input.column_to_ignore)
    prediction_x = prediction_x.to_numpy()

    if random_forest_classifier_input.predict_proba:
        proba = estimator.predict_proba(prediction_x)
        classes = estimator.classes_.tolist()
        result = [{classes[i]: proba[j][i] for i in range(len(classes))} for j in range(len(proba))]
    else:
        result = estimator.predict(prediction_x).tolist()

    return [
        types.TextContent(
            type="text",
            text=json.dumps(
                {
                    "description": "Result of Random Forest Classifier.",
                    "predictions": result,
                }
            ),
        )
    ]

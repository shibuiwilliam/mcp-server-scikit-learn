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

    dataset: list[dict[str, str | int | float | bool | None]]
    column_to_ignore: Optional[list[str]] = None

    @staticmethod
    def input_schema() -> dict:
        return {
            "type": "object",
            "properties": {
                "dataset": {
                    "type": "string",
                    "description": "The dataset to train and predict in Base64 encoded string or dumped JSON.",
                },
                "column_to_ignore": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "The columns to ignore in training and prediction.",
                },
            },
        }

    @staticmethod
    def from_str(dataset: str, column_to_ignore: Optional[list[str]] = None) -> "IsolationForestInputSchema":
        structured_dataset = string_to_list_dict(dataset)
        logger.info(f"Dataset: {structured_dataset}")

        return IsolationForestInputSchema(dataset=structured_dataset, column_to_ignore=column_to_ignore)


isolation_forest_tool = types.Tool(
    name=MCPServerScikitLearnTools.ISOLATION_FOREST.value,
    description="Train and predict anomaly with Isolation Forest.",
    inputSchema=IsolationForestInputSchema.input_schema(),
)


async def handle_isolation_forest_tool(
    arguments: dict[str, Any],
) -> list[types.TextContent | types.ImageContent | types.EmbeddedResource]:
    logger.info("Handling isolation forest tool")

    dataset = arguments["dataset"]
    column_to_ignore = arguments.get("column_to_ignore", None)
    isolation_forest_input = IsolationForestInputSchema.from_str(dataset, column_to_ignore)

    scaler = MinMaxScaler()
    label_encoder = LabelEncoder()

    df = pl.DataFrame(isolation_forest_input.dataset)
    numerical_columns = [
        col
        for col in df.columns
        if df[col].dtype in [pl.Int64, pl.Float64]
        and (isolation_forest_input.column_to_ignore and col not in isolation_forest_input.column_to_ignore)
    ]
    df = df.with_columns(
        [pl.Series(col, scaler.fit_transform(df[col].to_numpy().reshape(-1, 1)).flatten()) for col in numerical_columns]
    )

    category_columns = [
        col
        for col in df.columns
        if df[col].dtype == pl.Utf8
        and (isolation_forest_input.column_to_ignore and col not in isolation_forest_input.column_to_ignore)
    ]
    for col in category_columns:
        df = df.with_columns(pl.Series(col, label_encoder.fit_transform(df[col].to_list())))

    estimator = IsolationForest(
        n_estimators=100,
        max_samples="auto",
        contamination="auto",
        max_features=1.0,
        bootstrap=False,
        n_jobs=-1,
    )
    result = estimator.fit_predict(df.to_numpy())

    return [
        types.TextContent(
            type="text",
            text=json.dumps(
                {
                    "description": "-1 corresponds to outliers and 1 corresponds to inliers.",
                    "predictions": result.tolist(),
                }
            ),
        )
    ]

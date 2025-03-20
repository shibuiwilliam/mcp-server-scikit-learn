"""Tool for training and predicting with scikit-learn's DecisionTreeRegressor."""

import json
from typing import Any, Optional

import mcp.types as types
import polars as pl
from pydantic import BaseModel, ConfigDict
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.tree import DecisionTreeRegressor

from ..configure_logger import make_logger
from ..utils.string_util import string_to_list_dict
from .tools import MCPServerScikitLearnTools

logger = make_logger(__name__)


class DecisionTreeRegressorInputSchema(BaseModel):
    """Input schema for the Decision Tree Regressor tool.

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
    # Decision Tree hyperparameters
    criterion: Optional[str] = "squared_error"  # Options: "squared_error", "friedman_mse", "absolute_error", "poisson"
    splitter: Optional[str] = "best"  # Options: "best", "random"
    max_depth: Optional[int | None] = None
    min_samples_split: Optional[int | float] = 2
    min_samples_leaf: Optional[int | float] = 1
    min_weight_fraction_leaf: Optional[float] = 0.0
    max_features: Optional[str | int | float] = None
    max_leaf_nodes: Optional[int | None] = None
    min_impurity_decrease: Optional[float] = 0.0
    ccp_alpha: Optional[float] = 0.0

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
                "criterion": {
                    "type": "string",
                    "enum": ["squared_error", "friedman_mse", "absolute_error", "poisson"],
                    "description": "The function to measure the quality of a split.",
                },
                "splitter": {
                    "type": "string",
                    "enum": ["best", "random"],
                    "description": "The strategy used to choose the split at each node.",
                },
                "max_depth": {
                    "type": ["integer", "null"],
                    "description": "The maximum depth of the tree.",
                },
                "min_samples_split": {
                    "type": ["integer", "number"],
                    "description": "The minimum number of samples required to split an internal node.",
                },
                "min_samples_leaf": {
                    "type": ["integer", "number"],
                    "description": "The minimum number of samples required to be at a leaf node.",
                },
                "min_weight_fraction_leaf": {
                    "type": "number",
                    "description": "The minimum weighted fraction of the sum total of weights required to be at a leaf node.",
                },
                "max_features": {
                    "type": ["string", "integer", "number", "null"],
                    "description": "The number of features to consider when looking for the best split.",
                },
                "max_leaf_nodes": {
                    "type": ["integer", "null"],
                    "description": "Grow trees with max_leaf_nodes in best-first fashion.",
                },
                "min_impurity_decrease": {
                    "type": "number",
                    "description": "A node will be split if this split induces a decrease of the impurity greater than or equal to this value.",
                },
                "ccp_alpha": {
                    "type": "number",
                    "description": "Complexity parameter used for Minimal Cost-Complexity Pruning.",
                },
            },
        }

    @staticmethod
    def from_str(
        training_dataset: str,
        target_column: str,
        prediction_dataset: str,
        column_to_ignore: Optional[list[str]] = None,
        criterion: Optional[str] = "squared_error",
        splitter: Optional[str] = "best",
        max_depth: Optional[int | None] = None,
        min_samples_split: Optional[int | float] = 2,
        min_samples_leaf: Optional[int | float] = 1,
        min_weight_fraction_leaf: Optional[float] = 0.0,
        max_features: Optional[str | int | float] = None,
        max_leaf_nodes: Optional[int | None] = None,
        min_impurity_decrease: Optional[float] = 0.0,
        ccp_alpha: Optional[float] = 0.0,
    ) -> "DecisionTreeRegressorInputSchema":
        """Create a schema instance from string inputs.

        Args:
            training_dataset: Training data as a string.
            target_column: Name of the target column.
            prediction_dataset: Prediction data as a string.
            column_to_ignore: List of columns to ignore.
            criterion: Split quality measurement function.
            splitter: Node splitting strategy.
            max_depth: Maximum tree depth.
            min_samples_split: Minimum samples for splitting.
            min_samples_leaf: Minimum samples at leaf nodes.
            min_weight_fraction_leaf: Minimum weighted fraction at leaf nodes.
            max_features: Number of features to consider for splits.
            max_leaf_nodes: Maximum number of leaf nodes.
            min_impurity_decrease: Minimum impurity decrease for splitting.
            ccp_alpha: Cost-complexity pruning parameter.

        Returns:
            A DecisionTreeRegressorInputSchema instance.
        """
        structured_training_dataset = string_to_list_dict(training_dataset)
        structured_prediction_dataset = string_to_list_dict(prediction_dataset)

        return DecisionTreeRegressorInputSchema(
            training_dataset=structured_training_dataset,
            target_column=target_column,
            prediction_dataset=structured_prediction_dataset,
            column_to_ignore=column_to_ignore,
            criterion=criterion,
            splitter=splitter,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_features=max_features,
            max_leaf_nodes=max_leaf_nodes,
            min_impurity_decrease=min_impurity_decrease,
            ccp_alpha=ccp_alpha,
        )


decision_tree_regressor_tool = types.Tool(
    name=MCPServerScikitLearnTools.DECISION_TREE_REGRESSOR.value,
    description="Train and predict with Decision Tree Regressor.",
    inputSchema=DecisionTreeRegressorInputSchema.input_schema(),
)


async def handle_decision_tree_regressor_tool(
    arguments: dict[str, Any],
) -> list[types.TextContent | types.ImageContent | types.EmbeddedResource]:
    """Handle requests for the Decision Tree Regressor tool.

    Args:
        arguments: Dictionary containing input data and hyperparameters.

    Returns:
        List containing the model's predictions.
    """
    logger.info("Handling decision tree regressor tool")

    # Extract input data and hyperparameters
    training_dataset = arguments["training_dataset"]
    target_column = arguments["target_column"]
    prediction_dataset = arguments["prediction_dataset"]
    column_to_ignore = arguments.get("column_to_ignore", None)

    # Get hyperparameters from arguments
    criterion = arguments.get("criterion", "squared_error")
    splitter = arguments.get("splitter", "best")
    max_depth = arguments.get("max_depth", None)
    min_samples_split = arguments.get("min_samples_split", 2)
    min_samples_leaf = arguments.get("min_samples_leaf", 1)
    min_weight_fraction_leaf = arguments.get("min_weight_fraction_leaf", 0.0)
    max_features = arguments.get("max_features", None)
    max_leaf_nodes = arguments.get("max_leaf_nodes", None)
    min_impurity_decrease = arguments.get("min_impurity_decrease", 0.0)
    ccp_alpha = arguments.get("ccp_alpha", 0.0)

    # Create input schema and validate data
    decision_tree_regressor_input = DecisionTreeRegressorInputSchema.from_str(
        training_dataset,
        target_column,
        prediction_dataset,
        column_to_ignore,
        criterion=criterion,
        splitter=splitter,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        min_weight_fraction_leaf=min_weight_fraction_leaf,
        max_features=max_features,
        max_leaf_nodes=max_leaf_nodes,
        min_impurity_decrease=min_impurity_decrease,
        ccp_alpha=ccp_alpha,
    )

    # Convert input data to DataFrames
    train_df = pl.DataFrame(decision_tree_regressor_input.training_dataset)
    prediction_df = pl.DataFrame(decision_tree_regressor_input.prediction_dataset)
    training_columns = []
    numerical_columns_scalars = {}
    categorical_columns_encoders = {}

    # Preprocess training data
    for col in train_df.columns:
        if col == decision_tree_regressor_input.target_column:
            continue
        if decision_tree_regressor_input.column_to_ignore and col in decision_tree_regressor_input.column_to_ignore:
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
        if col == decision_tree_regressor_input.target_column:
            continue
        if decision_tree_regressor_input.column_to_ignore and col in decision_tree_regressor_input.column_to_ignore:
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
    x = train_df.drop(decision_tree_regressor_input.target_column)
    if decision_tree_regressor_input.column_to_ignore:
        x = x.drop(*decision_tree_regressor_input.column_to_ignore)
    x = x.to_numpy()
    y = train_df[decision_tree_regressor_input.target_column].to_numpy()

    # Train the model
    estimator = DecisionTreeRegressor(
        criterion=decision_tree_regressor_input.criterion,
        splitter=decision_tree_regressor_input.splitter,
        max_depth=decision_tree_regressor_input.max_depth,
        min_samples_split=decision_tree_regressor_input.min_samples_split,
        min_samples_leaf=decision_tree_regressor_input.min_samples_leaf,
        min_weight_fraction_leaf=decision_tree_regressor_input.min_weight_fraction_leaf,
        max_features=decision_tree_regressor_input.max_features,
        max_leaf_nodes=decision_tree_regressor_input.max_leaf_nodes,
        min_impurity_decrease=decision_tree_regressor_input.min_impurity_decrease,
        random_state=42,  # Set internally for reproducibility
        ccp_alpha=decision_tree_regressor_input.ccp_alpha,
    )
    estimator.fit(x, y)

    # Prepare features for prediction
    prediction_x = (
        prediction_df.drop(decision_tree_regressor_input.target_column)
        if decision_tree_regressor_input.target_column in prediction_df.columns
        else prediction_df
    )
    if decision_tree_regressor_input.column_to_ignore:
        prediction_x = prediction_x.drop(*decision_tree_regressor_input.column_to_ignore)
    prediction_x = prediction_x.to_numpy()

    # Generate predictions
    predictions = estimator.predict(prediction_x).tolist()

    return [
        types.TextContent(
            type="text",
            text=json.dumps(
                {
                    "description": "Result of Decision Tree Regressor.",
                    "predictions": predictions,
                }
            ),
        )
    ]

"""Tool for anomaly detection with scikit-learn's OneClassSVM."""

import json
from typing import Any, Optional

import mcp.types as types
import polars as pl
from pydantic import BaseModel, ConfigDict
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.svm import OneClassSVM

from ..configure_logger import make_logger
from ..utils.string_util import string_to_list_dict
from .tools import MCPServerScikitLearnTools

logger = make_logger(__name__)


class OneClassSVMInputSchema(BaseModel):
    """Input schema for the One-Class SVM tool.

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
    # One-Class SVM hyperparameters
    kernel: Optional[str] = "rbf"  # Options: "linear", "poly", "rbf", "sigmoid"
    degree: Optional[int] = 3  # Degree of the polynomial kernel function
    gamma: Optional[str] = "scale"  # Options: "scale", "auto", or float
    coef0: Optional[float] = 0.0  # Independent term in kernel function
    tol: Optional[float] = 1e-3  # Stopping criterion
    nu: Optional[float] = 0.5  # An upper bound on the fraction of training errors
    shrinking: Optional[bool] = True  # Whether to use the shrinking heuristic
    max_iter: Optional[int] = -1  # Hard limit on iterations within solver

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
                "kernel": {
                    "type": "string",
                    "enum": ["linear", "poly", "rbf", "sigmoid"],
                    "description": "Specifies the kernel type to be used in the algorithm.",
                },
                "degree": {
                    "type": "integer",
                    "description": "Degree of the polynomial kernel function.",
                },
                "gamma": {
                    "oneOf": [
                        {"type": "string", "enum": ["scale", "auto"]},
                        {"type": "number"},
                    ],
                    "description": "Kernel coefficient for 'rbf', 'poly' and 'sigmoid'.",
                },
                "coef0": {
                    "type": "number",
                    "description": "Independent term in kernel function.",
                },
                "tol": {
                    "type": "number",
                    "description": "Stopping criterion.",
                },
                "nu": {
                    "type": "number",
                    "description": "An upper bound on the fraction of training errors.",
                },
                "shrinking": {
                    "type": "boolean",
                    "description": "Whether to use the shrinking heuristic.",
                },
                "max_iter": {
                    "type": "integer",
                    "description": "Hard limit on iterations within solver.",
                },
            },
        }

    @staticmethod
    def from_str(
        training_dataset: str,
        prediction_dataset: str,
        column_to_ignore: Optional[list[str]] = None,
        kernel: Optional[str] = "rbf",
        degree: Optional[int] = 3,
        gamma: Optional[str] = "scale",
        coef0: Optional[float] = 0.0,
        tol: Optional[float] = 1e-3,
        nu: Optional[float] = 0.5,
        shrinking: Optional[bool] = True,
        max_iter: Optional[int] = -1,
    ) -> "OneClassSVMInputSchema":
        """Create a schema instance from string inputs.

        Args:
            training_dataset: Training data as a string.
            prediction_dataset: Prediction data as a string.
            column_to_ignore: List of columns to ignore.
            kernel: Specifies the kernel type to be used.
            degree: Degree of the polynomial kernel function.
            gamma: Kernel coefficient for 'rbf', 'poly' and 'sigmoid'.
            coef0: Independent term in kernel function.
            tol: Stopping criterion.
            nu: An upper bound on the fraction of training errors.
            shrinking: Whether to use the shrinking heuristic.
            max_iter: Hard limit on iterations within solver.

        Returns:
            A OneClassSVMInputSchema instance.
        """
        structured_training_dataset = string_to_list_dict(training_dataset)
        structured_prediction_dataset = string_to_list_dict(prediction_dataset)

        return OneClassSVMInputSchema(
            training_dataset=structured_training_dataset,
            prediction_dataset=structured_prediction_dataset,
            column_to_ignore=column_to_ignore,
            kernel=kernel,
            degree=degree,
            gamma=gamma,
            coef0=coef0,
            tol=tol,
            nu=nu,
            shrinking=shrinking,
            max_iter=max_iter,
        )


one_class_svm_tool = types.Tool(
    name=MCPServerScikitLearnTools.ONE_CLASS_SVM.value,
    description="Perform anomaly detection using scikit-learn's OneClassSVM.",
    inputSchema=OneClassSVMInputSchema.input_schema(),
)


async def handle_one_class_svm_tool(
    arguments: dict[str, Any],
) -> list[types.TextContent | types.ImageContent | types.EmbeddedResource]:
    """Handle requests for the One-Class SVM tool.

    Args:
        arguments: Dictionary containing input data and hyperparameters.

    Returns:
        List containing the anomaly predictions and scores.
    """
    logger.info("Handling one-class SVM tool")

    # Extract input data and hyperparameters
    training_dataset = arguments["training_dataset"]
    prediction_dataset = arguments["prediction_dataset"]
    column_to_ignore = arguments.get("column_to_ignore", None)

    # Get hyperparameters from arguments
    kernel = arguments.get("kernel", "rbf")
    degree = arguments.get("degree", 3)
    gamma = arguments.get("gamma", "scale")
    coef0 = arguments.get("coef0", 0.0)
    tol = arguments.get("tol", 1e-3)
    nu = arguments.get("nu", 0.5)
    shrinking = arguments.get("shrinking", True)
    max_iter = arguments.get("max_iter", -1)

    # Create input schema and validate data
    one_class_svm_input = OneClassSVMInputSchema.from_str(
        training_dataset,
        prediction_dataset,
        column_to_ignore,
        kernel=kernel,
        degree=degree,
        gamma=gamma,
        coef0=coef0,
        tol=tol,
        nu=nu,
        shrinking=shrinking,
        max_iter=max_iter,
    )

    # Convert input data to DataFrames
    train_df = pl.DataFrame(one_class_svm_input.training_dataset)
    prediction_df = pl.DataFrame(one_class_svm_input.prediction_dataset)
    training_columns = []
    numerical_columns_scalars = {}
    categorical_columns_encoders = {}

    # Preprocess training data
    for col in train_df.columns:
        if one_class_svm_input.column_to_ignore and col in one_class_svm_input.column_to_ignore:
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
        if one_class_svm_input.column_to_ignore and col in one_class_svm_input.column_to_ignore:
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
    if one_class_svm_input.column_to_ignore:
        x = x.drop(*one_class_svm_input.column_to_ignore)
    x = x.to_numpy()

    prediction_x = prediction_df
    if one_class_svm_input.column_to_ignore:
        prediction_x = prediction_x.drop(*one_class_svm_input.column_to_ignore)
    prediction_x = prediction_x.to_numpy()

    # Train the model with internal-only parameters
    estimator = OneClassSVM(
        kernel=one_class_svm_input.kernel,
        degree=one_class_svm_input.degree,
        gamma=one_class_svm_input.gamma,
        coef0=one_class_svm_input.coef0,
        tol=one_class_svm_input.tol,
        nu=one_class_svm_input.nu,
        shrinking=one_class_svm_input.shrinking,
        cache_size=200,  # Internal-only parameter
        verbose=False,  # Internal-only parameter
        max_iter=one_class_svm_input.max_iter,
    )
    estimator.fit(x)

    # Get predictions and scores
    predictions = estimator.predict(prediction_x).tolist()
    scores = estimator.score_samples(prediction_x).tolist()

    return [
        types.TextContent(
            type="text",
            text=json.dumps(
                {
                    "description": "Result of One-Class SVM anomaly detection.",
                    "predictions": predictions,
                    "scores": scores,
                }
            ),
        )
    ]

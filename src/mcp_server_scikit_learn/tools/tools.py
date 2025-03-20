"""Enumeration of available scikit-learn tools in the MCP server."""

from enum import Enum


class MCPServerScikitLearnTools(str, Enum):
    """Enumeration of available scikit-learn tools.

    This enum represents the different scikit-learn models available for training
    and prediction in the MCP server.
    """

    RANDOM_FOREST_CLASSIFIER = "random_forest_classifier"
    RANDOM_FOREST_REGRESSOR = "random_forest_regressor"
    ISOLATION_FOREST = "isolation_forest"
    DECISION_TREE_CLASSIFIER = "decision_tree_classifier"
    DECISION_TREE_REGRESSOR = "decision_tree_regressor"
    K_NEIGHBORS_CLASSIFIER = "k_neighbors_classifier"
    K_NEIGHBORS_REGRESSOR = "k_neighbors_regressor"
    NEAREST_NEIGHBORS = "nearest_neighbors"
    K_MEANS = "k_means"
    ONE_CLASS_SVM = "one_class_svm"

    @classmethod
    def from_str(cls, value: str) -> "MCPServerScikitLearnTools":
        """Convert a string to a tool enum value.

        Args:
            value: String value to convert.

        Returns:
            The corresponding MCPServerScikitLearnTools enum value.

        Raises:
            ValueError: If the string value is not a valid tool name.
        """
        try:
            return cls(value)
        except ValueError:
            raise ValueError(f"Unknown tool: {value}")

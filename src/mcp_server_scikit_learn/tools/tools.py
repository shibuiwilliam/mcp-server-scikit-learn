from enum import Enum


class MCPServerScikitLearnTools(Enum):
    ISOLATION_FOREST = "isolation_forest"
    RANDOM_FOREST_CLASSIFIER = "random_forest_classifier"

    @staticmethod
    def from_str(table: str) -> "MCPServerScikitLearnTools":
        for t in MCPServerScikitLearnTools:
            if t.value == table:
                return t
        raise ValueError(f"Invalid table: {table}")

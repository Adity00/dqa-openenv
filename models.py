"""
Models for the DQA-OpenEnv environment.

This module defines the core data structures used in the Data Quality Assurance
environment, including actions, observations, and environment state. These models
extend the base OpenEnv types and are strictly validated using Pydantic v2.
"""

from typing import Any, Dict, List, Literal, Optional

from pydantic import Field, field_validator, model_validator, ConfigDict
from openenv.core.env_server.types import Action, Observation, State


class DQAAction(Action):
    """
    Represents an action taken by an agent in the DQA environment.
    
    This model captures the specific operation to be performed, the target
    column (if applicable), and any required parameters for the operation.
    """

    action_type: Literal[
        "fill_nulls",
        "drop_duplicates",
        "cast_type",
        "normalize_category",
        "clip_outliers",
        "drop_column",
        "filter_rows",
        "merge_categories",
        "submit",
        "noop"
    ]
    """The type of data cleaning action to perform."""

    column: Optional[str] = None
    """The name of the target column for column-specific actions."""

    parameters: Dict[str, Any] = {}
    """Additional parameters required for the action (e.g., strategy, target_type)."""

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "action_type": "fill_nulls",
                    "column": "age",
                    "parameters": {"strategy": "mean"}
                },
                {
                    "action_type": "cast_type",
                    "column": "id",
                    "parameters": {"target_type": "int"}
                },
                {
                    "action_type": "submit",
                    "column": None,
                    "parameters": {}
                }
            ]
        }
    )

    @model_validator(mode="after")
    def validate_column_for_actions(self) -> "DQAAction":
        """
        Validates that a column is provided for actions that require one.
        
        Raises:
            ValueError: If a column-specific action is missing the target column.
            
        Returns:
            The validated action instance.
        """
        requires_column = {
            "fill_nulls",
            "cast_type",
            "normalize_category",
            "clip_outliers",
            "drop_column",
            "merge_categories"
        }
        
        if self.action_type in requires_column and self.column is None:
            raise ValueError(f"Action '{self.action_type}' requires a target column.")
            
        return self


class DQAObservation(Observation):
    """
    Represents the observation returned by the environment after a step.
    
    This includes task details, dataset previews, metadata about the data's
    quality, and the outcomes of previous actions.
    """

    task_id: str = ""
    """The identifier for the active task tier (e.g., 'easy', 'medium', 'hard')."""

    task_description: str = ""
    """A natural language description of the goal for the current episode."""

    dataset_preview: List[Dict[str, Any]] = []
    """A preview of the dataset, represented as a list of dictionaries (first 10 rows)."""

    column_stats: Dict[str, Any] = {}
    """
    Per-column statistics dictionary. Expected fields per column:
    null_count, null_pct, dtype, unique_count, sample_values, min_val, max_val.
    """

    issue_hints: List[str] = []
    """A list of plain English strings describing automatically detected data quality issues."""

    quality_scores: Dict[str, float] = {}
    """
    Live quality scores representing the state of the dataset.
    Keys usually include: 'completeness', 'consistency', 'uniqueness', 'validity', 'overall'.
    Contains values from 0.0 to 1.0.
    """

    step_count: int = 0
    """The current step number in the episode."""

    max_steps: int = 20
    """The maximum number of steps allowed in the episode."""

    action_history: List[str] = []
    """Descriptions of the previously taken actions in the episode."""

    last_action_result: str = ""
    """A message indicating the success or failure of the most recent action."""

    last_action_success: bool = True
    """Boolean flag indicating if the last action was successfully applied."""


class DQAState(State):
    """
    Represents the internal state of the environment.
    
    This includes tracking the current quality scores, accumulated reward,
    and completion status.
    """

    task_id: str = ""
    """The unique identifier of the currently active task."""

    current_quality_scores: Dict[str, float] = {}
    """The latest evaluation metrics for the dataset quality."""

    actions_taken: int = 0
    """The total number of actions the agent has successfully applied."""

    episode_reward: float = 0.0
    """The cumulative reward obtained during the current episode."""

    is_submitted: bool = False
    """Flag indicating if the agent has chosen to submit the final dataset."""

    task_completed: bool = False
    """Flag indicating whether the episode has reached a terminal state."""


if __name__ == "__main__":
    # Instantiate models to verify they work correctly
    test_action = DQAAction(
        action_type="fill_nulls",
        column="age",
        parameters={"strategy": "mean"}
    )
    
    test_observation = DQAObservation(
        task_id="easy",
        task_description="Clean the dataset.",
        dataset_preview=[{"id": 1, "age": None}],
        column_stats={"age": {"null_count": 1, "null_pct": 1.0, "dtype": "float"}},
        issue_hints=["Column 'age' has missing values."],
        quality_scores={"completeness": 0.5},
        step_count=1,
        max_steps=20,
        action_history=["noop"],
        last_action_result="success",
        last_action_success=True
    )
    
    test_state = DQAState(
        episode_id="test_ep_001",
        step_count=1,
        task_id="easy",
        current_quality_scores={"completeness": 0.5},
        actions_taken=1,
        episode_reward=10.0,
        is_submitted=False,
        task_completed=False
    )
    
    print("Action:")
    print(test_action.model_dump_json(indent=2))
    
    print("\nObservation:")
    print(test_observation.model_dump_json(indent=2))
    
    print("\nState:")
    print(test_state.model_dump_json(indent=2))

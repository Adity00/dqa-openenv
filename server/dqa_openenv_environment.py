"""
DQA-OpenEnv Environment — main environment class.

This module implements the core Environment interface for the Data Quality
Assurance OpenEnv competition. It wires together DatasetFactory, RewardEngine,
DQAGrader, and the Pydantic models to provide a complete reset/step loop.
"""

import pandas as pd
import numpy as np
from uuid import uuid4
from typing import Optional, Dict, Any, List, Tuple
from copy import deepcopy

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

import sys
import os

try:
    from ..models import DQAAction, DQAObservation, DQAState
except ImportError:
    try:
        from models import DQAAction, DQAObservation, DQAState
    except ImportError:
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from models import DQAAction, DQAObservation, DQAState

try:
    from .datasets.generators import DatasetFactory
    from .rewards.reward_engine import RewardEngine, QualityScorer
    from .graders.grader import DQAGrader
except ImportError:
    from server.datasets.generators import DatasetFactory
    from server.rewards.reward_engine import RewardEngine, QualityScorer
    from server.graders.grader import DQAGrader


class DqaOpenenvEnvironment(Environment):
    """
    Core OpenEnv environment for Data Quality Assurance.

    Manages the full episode lifecycle: reset → step → done,
    applying agent actions to a dirty DataFrame and scoring
    quality improvement against a hidden ground truth.
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self) -> None:
        """Initializes the environment with default empty state."""
        self._episode_id: str = str(uuid4())
        self._step_count: int = 0
        self._task_id: str = "easy"
        self._dirty_df: Optional[pd.DataFrame] = None
        self._clean_df: Optional[pd.DataFrame] = None
        self._agent_df: Optional[pd.DataFrame] = None
        self._reward_engine: Optional[RewardEngine] = None
        self._grader: DQAGrader = DQAGrader()
        self._current_scores: Dict[str, float] = {}
        self._action_history: List[str] = []
        self._episode_reward: float = 0.0
        self._is_submitted: bool = False
        self._max_steps: int = 20
        self._task_config: Dict[str, Any] = {}

    def reset(self, task_id: str = "easy") -> DQAObservation:
        """
        Resets the environment for a new episode.

        Args:
            task_id: The difficulty tier — "easy", "medium", or "hard".

        Returns:
            The initial DQAObservation for the new episode.
        """
        self._task_id = task_id
        self._task_config = DatasetFactory.get_task_config(task_id)
        self._max_steps = self._task_config["max_steps"]

        self._dirty_df, self._clean_df = DatasetFactory.get_task_data(task_id)
        self._agent_df = deepcopy(self._dirty_df)

        self._episode_id = str(uuid4())
        self._step_count = 0
        self._action_history = []
        self._episode_reward = 0.0
        self._is_submitted = False
        self._reward_engine = RewardEngine(task_id)

        self._current_scores = QualityScorer.score(
            self._agent_df, self._clean_df, task_id
        )

        return self._build_observation(
            last_action_result="Environment reset. Task started.",
            last_action_success=True
        )

    def step(self, action: DQAAction) -> DQAObservation:
        """
        Executes a single agent action and returns the resulting observation.

        Args:
            action: The DQAAction to execute.

        Returns:
            A DQAObservation reflecting the post-action state.

        Raises:
            RuntimeError: If reset() has not been called yet.
        """
        if self._agent_df is None:
            raise RuntimeError("Call reset() before step()")

        if self._is_submitted or self._step_count >= self._max_steps:
            return self._build_observation(
                last_action_result="Episode already finished.",
                last_action_success=False,
                done=True
            )

        self._step_count += 1

        action_success, action_result_msg = self._apply_action(action)

        if action.action_type == "submit":
            self._is_submitted = True

        step_reward, new_scores = self._reward_engine.calculate_step_reward(
            action_type=action.action_type,
            action_success=action_success,
            agent_df=self._agent_df,
            clean_df=self._clean_df,
            previous_scores=self._current_scores,
            action_history=self._action_history
        )
        self._current_scores = new_scores
        self._episode_reward += step_reward

        self._action_history.append(action.action_type)

        done = self._is_submitted or (self._step_count >= self._max_steps)

        return self._build_observation(
            reward=step_reward,
            last_action_result=action_result_msg,
            last_action_success=action_success,
            done=done
        )

    def _apply_action(self, action: DQAAction) -> Tuple[bool, str]:
        """
        Applies the given action to self._agent_df in place.

        Args:
            action: The DQAAction to apply.

        Returns:
            A tuple of (success: bool, message: str).
        """
        try:
            action_type = action.action_type
            col = action.column
            params = action.parameters or {}

            if action_type == "noop":
                return (True, "No operation performed.")

            if action_type == "submit":
                return (True, "Agent submitted the dataset for final grading.")

            if action_type == "fill_nulls":
                strategy = params.get("strategy", "mean")
                if col not in self._agent_df.columns:
                    return (False, f"Column '{col}' not found.")
                if strategy == "mean":
                    val = self._agent_df[col].mean()
                    self._agent_df[col] = self._agent_df[col].fillna(val)
                elif strategy == "median":
                    val = self._agent_df[col].median()
                    self._agent_df[col] = self._agent_df[col].fillna(val)
                elif strategy == "mode":
                    val = self._agent_df[col].mode()[0]
                    self._agent_df[col] = self._agent_df[col].fillna(val)
                elif strategy == "constant":
                    val = params.get("value", 0)
                    self._agent_df[col] = self._agent_df[col].fillna(val)
                elif strategy == "drop":
                    self._agent_df.dropna(subset=[col], inplace=True)
                    self._agent_df.reset_index(drop=True, inplace=True)
                else:
                    return (False, f"Unknown fill strategy: {strategy}")
                return (True, f"Filled nulls in '{col}' using strategy '{strategy}'.")

            if action_type == "drop_duplicates":
                subset = params.get("subset", None)
                before = len(self._agent_df)
                self._agent_df.drop_duplicates(subset=subset, inplace=True)
                self._agent_df.reset_index(drop=True, inplace=True)
                after = len(self._agent_df)
                return (True, f"Removed {before - after} duplicate rows.")

            if action_type == "cast_type":
                target_type = params.get("target_type", "str")
                if col not in self._agent_df.columns:
                    return (False, f"Column '{col}' not found.")
                type_map = {"int": "Int64", "float": "float64",
                            "str": "str", "bool": "bool"}
                dtype = type_map.get(target_type, target_type)
                self._agent_df[col] = self._agent_df[col].astype(dtype)
                return (True, f"Cast column '{col}' to type '{target_type}'.")

            if action_type == "normalize_category":
                mapping = params.get("mapping", {})
                if col not in self._agent_df.columns:
                    return (False, f"Column '{col}' not found.")
                self._agent_df[col] = self._agent_df[col].replace(mapping)
                return (True, f"Normalized {len(mapping)} category values in '{col}'.")

            if action_type == "clip_outliers":
                method = params.get("method", "iqr")
                if col not in self._agent_df.columns:
                    return (False, f"Column '{col}' not found.")
                if method == "iqr":
                    Q1 = self._agent_df[col].quantile(0.25)
                    Q3 = self._agent_df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower = Q1 - 1.5 * IQR
                    upper = Q3 + 1.5 * IQR
                    self._agent_df[col] = self._agent_df[col].clip(lower, upper)
                elif method == "zscore":
                    mean = self._agent_df[col].mean()
                    std = self._agent_df[col].std()
                    lower = mean - 3 * std
                    upper = mean + 3 * std
                    self._agent_df[col] = self._agent_df[col].clip(lower, upper)
                else:
                    return (False, f"Unknown clip method: {method}")
                return (True, f"Clipped outliers in '{col}' using '{method}' method.")

            if action_type == "drop_column":
                if col not in self._agent_df.columns:
                    return (False, f"Column '{col}' not found.")
                self._agent_df.drop(columns=[col], inplace=True)
                return (True, f"Dropped column '{col}'.")

            if action_type == "filter_rows":
                condition = params.get("condition", "")
                if not condition:
                    return (False, "No condition provided for filter_rows.")
                before = len(self._agent_df)
                self._agent_df.query(condition, inplace=True)
                self._agent_df.reset_index(drop=True, inplace=True)
                after = len(self._agent_df)
                return (True, f"Filtered {before - after} rows matching '{condition}'.")

            if action_type == "merge_categories":
                from_values = params.get("from_values", [])
                to_value = params.get("to_value", "")
                if col not in self._agent_df.columns:
                    return (False, f"Column '{col}' not found.")
                mapping = {v: to_value for v in from_values}
                self._agent_df[col] = self._agent_df[col].replace(mapping)
                return (True, f"Merged {len(from_values)} categories into '{to_value}' in '{col}'.")

            return (False, f"Unknown action_type: {action_type}")

        except Exception as e:
            return (False, f"Action failed with error: {str(e)}")

    def _build_observation(
        self,
        reward: float = 0.0,
        last_action_result: str = "",
        last_action_success: bool = True,
        done: bool = False
    ) -> DQAObservation:
        """
        Constructs a DQAObservation from the current environment state.

        Args:
            reward:              The step reward to include.
            last_action_result:  Human-readable result message.
            last_action_success: Whether the last action succeeded.
            done:                Whether the episode is over.

        Returns:
            A fully populated DQAObservation.
        """
        # Dataset preview
        if self._agent_df is not None:
            preview_df = self._agent_df.head(10)
            dataset_preview = preview_df.where(
                pd.notnull(preview_df), None
            ).to_dict(orient="records")
        else:
            dataset_preview = []

        # Column stats
        if self._agent_df is not None:
            column_stats: Dict[str, Any] = {}
            for col in self._agent_df.columns:
                s = self._agent_df[col]
                null_count = int(s.isnull().sum())
                null_pct = round(float(s.isnull().mean()), 4)
                dtype = str(s.dtype)
                unique_count = int(s.nunique())
                sample_vals = s.dropna().head(3).tolist()
                try:
                    min_val = float(s.min()) if pd.api.types.is_numeric_dtype(s) else None
                    max_val = float(s.max()) if pd.api.types.is_numeric_dtype(s) else None
                except Exception:
                    min_val, max_val = None, None
                column_stats[col] = {
                    "null_count": null_count,
                    "null_pct": null_pct,
                    "dtype": dtype,
                    "unique_count": unique_count,
                    "sample_values": [str(v) for v in sample_vals],
                    "min_val": min_val,
                    "max_val": max_val,
                }
        else:
            column_stats = {}

        # Issue hints
        issue_hints: List[str] = []
        if self._agent_df is not None and self._clean_df is not None:
            for col in self._agent_df.columns:
                null_pct = self._agent_df[col].isnull().mean()
                if null_pct > 0.01:
                    issue_hints.append(
                        f"Column '{col}' has {null_pct:.1%} missing values"
                    )

            dup_count = int(self._agent_df.duplicated().sum())
            if dup_count > 0:
                issue_hints.append(f"Dataset has {dup_count} duplicate rows")

            for col in self._agent_df.select_dtypes(include=["object", "str"]).columns:
                if col in self._clean_df.columns:
                    clean_vals = set(
                        self._clean_df[col].dropna().astype(str).str.strip()
                    )
                    agent_vals = set(
                        self._agent_df[col].dropna().astype(str).str.strip()
                    )
                    extra_vals = agent_vals - clean_vals
                    if extra_vals:
                        sample = list(extra_vals)[:3]
                        issue_hints.append(
                            f"Column '{col}' has unexpected values: {sample}"
                        )

        task_desc = self._task_config.get("description", "Clean the dataset.")

        return DQAObservation(
            done=done,
            reward=reward,
            task_id=self._task_id,
            task_description=task_desc,
            dataset_preview=dataset_preview,
            column_stats=column_stats,
            issue_hints=issue_hints,
            quality_scores=self._current_scores,
            step_count=self._step_count,
            max_steps=self._max_steps,
            action_history=self._action_history.copy(),
            last_action_result=last_action_result,
            last_action_success=last_action_success,
        )

    @property
    def state(self) -> State:
        """Returns the current internal DQAState for this episode."""
        return DQAState(
            episode_id=self._episode_id,
            step_count=self._step_count,
            task_id=self._task_id,
            current_quality_scores=self._current_scores,
            actions_taken=self._step_count,
            episode_reward=round(self._episode_reward, 4),
            is_submitted=self._is_submitted,
            task_completed=self._is_submitted,
        )


if __name__ == "__main__":
    env = DqaOpenenvEnvironment()

    print("=== ENVIRONMENT SMOKE TEST ===\n")

    for task_id in ["easy", "medium", "hard"]:
        print(f"--- Testing task: {task_id} ---")
        obs = env.reset(task_id)
        print(f"Reset OK: task={obs.task_id}, max_steps={obs.max_steps}")
        print(f"Issues detected: {len(obs.issue_hints)}")
        print(f"Initial scores: {obs.quality_scores}")

        # Step 1: noop
        action = DQAAction(action_type="noop")
        obs = env.step(action)
        print(f"After noop: reward={obs.reward}, done={obs.done}")

        # Step 2: fill_nulls on first null column
        null_cols = [k for k, v in obs.column_stats.items()
                     if v["null_count"] > 0]
        if null_cols:
            action = DQAAction(
                action_type="fill_nulls",
                column=null_cols[0],
                parameters={"strategy": "mean"}
            )
            obs = env.step(action)
            print(f"After fill_nulls on '{null_cols[0]}': reward={obs.reward}")

        # Final: submit
        action = DQAAction(action_type="submit")
        obs = env.step(action)
        print(f"After submit: done={obs.done}, reward={obs.reward}")

        # Check state
        state = env.state
        print(f"State: steps={state.step_count}, submitted={state.is_submitted}")
        print(f"Episode reward total: {state.episode_reward}")
        print()

    print("=== SMOKE TEST COMPLETE ===")

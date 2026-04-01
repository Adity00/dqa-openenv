"""
Reward calculation engine for the DQA-OpenEnv environment.

This module houses the QualityScorer and RewardEngine classes, which compute
dataset quality dimensions precisely and evaluate agent actions by computing
the delta (change) in quality scores over time.
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional
from dataclasses import dataclass


class QualityScorer:
    """Calculates multidimensional data quality scores."""

    @staticmethod
    def score(agent_df: pd.DataFrame, clean_df: pd.DataFrame, task_id: str) -> Dict[str, float]:
        """
        Computes 4 data quality dimensions for agent_df compared against the clean_df reference
        and returns a weighted overall score.
        """
        # --- COMPLETENESS ---
        if clean_df.empty or len(clean_df.columns) == 0:
            completeness = 1.0
        else:
            col_scores = []
            for col in clean_df.columns:
                if col not in agent_df.columns:
                    col_scores.append(0.0)
                else:
                    clean_null_pct = clean_df[col].isnull().mean()
                    agent_null_pct = agent_df[col].isnull().mean()
                    if agent_null_pct > clean_null_pct:
                        extra_null_pct = agent_null_pct - clean_null_pct
                        col_score = 1.0 - float(extra_null_pct)
                        col_scores.append(max(0.0, min(1.0, col_score)))
            
            if not col_scores:
                completeness = 1.0
            else:
                completeness = float(np.mean(col_scores))

        # --- CONSISTENCY ---
        string_cols = [
            c for c in clean_df.columns 
            if pd.api.types.is_string_dtype(clean_df[c]) or pd.api.types.is_object_dtype(clean_df[c])
        ]
        if not string_cols:
            consistency = 1.0
        else:
            consist_scores = []
            for col in string_cols:
                if col not in agent_df.columns:
                    consist_scores.append(0.0)
                else:
                    clean_values = set(clean_df[col].dropna().astype(str).str.strip())
                    agent_values = agent_df[col].dropna()
                    if len(agent_values) > 0:
                        matching = agent_values.apply(
                            lambda x: str(x).strip() in clean_values
                        ).mean()
                        col_score = float(matching)
                    else:
                        col_score = 1.0
                    consist_scores.append(col_score)
            consistency = float(np.mean(consist_scores)) if consist_scores else 1.0

        # --- UNIQUENESS ---
        if len(clean_df) > 0:
            clean_duplicate_rate = clean_df.duplicated().mean()
        else:
            clean_duplicate_rate = 0.0
            
        if len(agent_df) > 0:
            agent_duplicate_rate = agent_df.duplicated().mean()
        else:
            agent_duplicate_rate = 0.0
            
        uniqueness = 1.0 - max(0.0, float(agent_duplicate_rate - clean_duplicate_rate))
        uniqueness = max(0.0, min(1.0, uniqueness))

        # --- VALIDITY ---
        numeric_cols = [c for c in clean_df.columns if pd.api.types.is_numeric_dtype(clean_df[c])]
        if not numeric_cols:
            validity = 1.0
        else:
            val_scores = []
            for col in numeric_cols:
                if col not in agent_df.columns:
                    val_scores.append(0.0)
                else:
                    clean_min = clean_df[col].min()
                    clean_max = clean_df[col].max()
                    agent_col = agent_df[col].dropna()
                    
                    if len(agent_col) == 0:
                        col_score = 1.0
                    else:
                        in_range = ((agent_col >= clean_min * 0.99) & 
                                    (agent_col <= clean_max * 1.01)).mean()
                        col_score = float(in_range)
                    val_scores.append(col_score)
            validity = float(np.mean(val_scores)) if val_scores else 1.0

        # --- OVERALL (Weighted Average) ---
        if task_id == "easy":
            weights = {"completeness": 0.40, "consistency": 0.20, "uniqueness": 0.20, "validity": 0.20}
        elif task_id == "medium":
            weights = {"completeness": 0.25, "consistency": 0.30, "uniqueness": 0.30, "validity": 0.15}
        elif task_id == "hard":
            weights = {"completeness": 0.20, "consistency": 0.25, "uniqueness": 0.20, "validity": 0.35}
        else:
            weights = {"completeness": 0.25, "consistency": 0.25, "uniqueness": 0.25, "validity": 0.25}
            
        overall = (
            completeness * weights["completeness"] + 
            consistency * weights["consistency"] + 
            uniqueness * weights["uniqueness"] + 
            validity * weights["validity"]
        )
        
        return {
            "completeness": round(max(0.0, min(1.0, completeness)), 4),
            "consistency": round(max(0.0, min(1.0, consistency)), 4),
            "uniqueness": round(max(0.0, min(1.0, uniqueness)), 4),
            "validity": round(max(0.0, min(1.0, validity)), 4),
            "overall": round(max(0.0, min(1.0, overall)), 4)
        }


class RewardEngine:
    """Computes specific agent rewards based on change (delta) in quality scores."""

    def __init__(self, task_id: str):
        """Initializes the reward engine for a given task execution depth."""
        self.task_id = task_id
        self.previous_scores: Dict[str, float] = {}
        self.step_count: int = 0

    def calculate_step_reward(
        self,
        action_type: str,
        action_success: bool,
        agent_df: pd.DataFrame,
        clean_df: pd.DataFrame,
        previous_scores: Dict[str, float],
        action_history: list
    ) -> Tuple[float, Dict[str, float]]:
        """
        Calculates a state transition reward primarily driven by delta progression
        and strategic agent action properties.
        
        Returns:
            Tuple containing the resultant float reward and the newly evaluated scores.
        """
        # Step 1 - Compute new quality scores
        new_scores = QualityScorer.score(agent_df, clean_df, self.task_id)
        
        # Step 2 - If action failed (action_success=False)
        if not action_success:
            return (-0.05, new_scores)
            
        # Step 3 - Calculate delta from previous scores
        if not previous_scores:
            delta = 0.0
        else:
            delta = new_scores.get("overall", 0.0) - previous_scores.get("overall", 0.0)
            
        # Step 4 - Base reward from delta
        reward = delta * 2.0
        
        # Step 5 - Action-specific adjustments
        if action_type == "noop":
            reward -= 0.02
        
        if action_type == "submit":
            reward += 0.05
            
        if action_type in ["drop_column", "filter_rows"]:
            if delta < -0.05:
                reward -= 0.15
                
        # Step 6 - Loop detection penalty
        if len(action_history) >= 3:
            last_3 = action_history[-3:]
            if len(set(last_3)) == 1 and action_type == action_history[-1]:
                reward -= 0.10
                
        # Step 7 - Clamp final reward
        reward = max(-1.0, min(1.0, round(reward, 4)))
        
        return float(reward), new_scores

    def calculate_terminal_reward(
        self,
        final_scores: Dict[str, float],
        steps_used: int,
        max_steps: int,
        submitted: bool
    ) -> float:
        """Calculates final episode completion reward and efficiency bounds."""
        base = final_scores.get("overall", 0.0)
        
        if submitted:
            efficiency_bonus = 0.05 * (1.0 - steps_used / max_steps)
            return round(min(1.0, base + efficiency_bonus), 4)
        else:
            return round(base - 0.10, 4)


if __name__ == "__main__":
    import sys
    import os
    
    try:
        from server.datasets.generators import DatasetFactory
    except ImportError:
        try:
            from datasets.generators import DatasetFactory
        except ImportError:
            sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            from datasets.generators import DatasetFactory

    for task_id in ["easy", "medium", "hard"]:
        dirty_df, clean_df = DatasetFactory.get_task_data(task_id)
        engine = RewardEngine(task_id)
        
        # Simulate: score the DIRTY df first (baseline)
        baseline = QualityScorer.score(dirty_df, clean_df, task_id)
        print(f"\n=== {task_id.upper()} TASK ===")
        print(f"Baseline quality (dirty vs clean):")
        for k, v in baseline.items():
            print(f"  {k}: {v:.4f}")
        
        # Simulate: score the CLEAN df (perfect score)
        perfect = QualityScorer.score(clean_df, clean_df, task_id)
        print(f"Perfect quality (clean vs clean):")
        for k, v in perfect.items():
            print(f"  {k}: {v:.4f}")
        
        # Simulate a noop action
        reward, scores = engine.calculate_step_reward(
            action_type="noop",
            action_success=True,
            agent_df=dirty_df,
            clean_df=clean_df,
            previous_scores={},
            action_history=[]
        )
        print(f"Reward for noop on dirty data: {reward}")

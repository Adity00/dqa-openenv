"""
Deterministic grading system for the DQA-OpenEnv competition environment.

This module provides the DQAGrader class, which evaluates an agent's final
cleaned dataset against the hidden ground truth at episode end (on submit or timeout).
Grading is 100% deterministic — same inputs always produce the same score.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
import sys
import os

try:
    from server.rewards.reward_engine import QualityScorer
except ImportError:
    try:
        from rewards.reward_engine import QualityScorer
    except ImportError:
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from rewards.reward_engine import QualityScorer


class DQAGrader:
    """
    Deterministic grader for DQA-OpenEnv episodes.

    Evaluates the agent's final dataset state against the ground truth
    and returns a structured result dict containing a final score (0.0–1.0),
    letter grade, dimension-level quality scores, feedback, and efficiency.
    """

    def grade(
        self,
        agent_df: pd.DataFrame,
        clean_df: pd.DataFrame,
        task_id: str,
        steps_used: int,
        max_steps: int,
        submitted: bool
    ) -> Dict[str, Any]:
        """
        Grade the agent's final dataset submission for a single episode.

        Args:
            agent_df:   The agent's final cleaned dataset.
            clean_df:   The hidden ground truth dataset.
            task_id:    One of "easy", "medium", "hard".
            steps_used: Number of steps used by the agent in this episode.
            max_steps:  Total step budget for this episode.
            submitted:  True if the agent called submit(); False if it timed out.

        Returns:
            A dict with keys: final_score, quality_scores, passed, grade,
            efficiency, submitted, steps_used, feedback, dimension_gaps.
        """
        # Step 1: Get quality scores
        scores = QualityScorer.score(agent_df, clean_df, task_id)

        # Step 2: Calculate final score with efficiency adjustment
        base_score = scores["overall"]
        if submitted:
            efficiency_bonus = 0.05 * max(0.0, 1.0 - steps_used / max_steps)
            final_score = min(1.0, base_score + efficiency_bonus)
        else:
            final_score = max(0.0, base_score - 0.10)
        final_score = round(final_score, 4)

        # Step 3: Calculate grade letter
        if final_score >= 0.85:
            grade = "A"
        elif final_score >= 0.70:
            grade = "B"
        elif final_score >= 0.55:
            grade = "C"
        elif final_score >= 0.40:
            grade = "D"
        else:
            grade = "F"

        # Step 4: Calculate dimension gaps
        dimension_gaps = {
            dim: round(1.0 - scores[dim], 4)
            for dim in ["completeness", "consistency", "uniqueness", "validity"]
        }

        # Step 5: Generate feedback strings
        feedback: List[str] = []

        if scores["completeness"] < 0.95:
            gap = round(1.0 - scores["completeness"], 2)
            feedback.append(
                f"Completeness gap of {gap:.0%}: some null values were not handled correctly"
            )

        if scores["consistency"] < 0.95:
            gap = round(1.0 - scores["consistency"], 2)
            feedback.append(
                f"Consistency gap of {gap:.0%}: categorical values may still have variants"
            )

        if scores["uniqueness"] < 0.95:
            gap = round(1.0 - scores["uniqueness"], 2)
            feedback.append(
                f"Uniqueness gap of {gap:.0%}: duplicate rows may not be fully removed"
            )

        if scores["validity"] < 0.95:
            gap = round(1.0 - scores["validity"], 2)
            feedback.append(
                f"Validity gap of {gap:.0%}: some values may be outside expected ranges"
            )

        if not submitted:
            feedback.append("Episode timed out — agent did not call submit() action")

        if final_score >= 0.85:
            feedback.append("Excellent cleaning — dataset closely matches ground truth")
        elif final_score >= 0.70:
            feedback.append("Good cleaning — most major issues resolved")
        elif final_score >= 0.55:
            feedback.append("Partial cleaning — several issues remain unresolved")
        else:
            feedback.append("Insufficient cleaning — major quality issues remain")

        if len(feedback) == 0:
            feedback.append("Perfect score — dataset matches ground truth exactly")

        # Step 6: Return full result dict
        return {
            "final_score": final_score,
            "quality_scores": scores,
            "passed": final_score >= 0.6,
            "grade": grade,
            "efficiency": round(steps_used / max_steps, 4),
            "submitted": submitted,
            "steps_used": steps_used,
            "feedback": feedback,
            "dimension_gaps": dimension_gaps,
        }

    def grade_all_tasks(
        self,
        dirty_dfs: List[pd.DataFrame],
        clean_dfs: List[pd.DataFrame],
        task_ids: List[str],
        steps_used_list: List[int],
        max_steps_list: List[int],
        submitted_list: List[bool]
    ) -> Dict[str, Any]:
        """
        Grade multiple tasks in a single batch call.

        Args:
            dirty_dfs:       List of agent output DataFrames (one per task).
            clean_dfs:       List of ground truth DataFrames (one per task).
            task_ids:        List of task identifier strings.
            steps_used_list: List of step counts used per task.
            max_steps_list:  List of maximum step budgets per task.
            submitted_list:  List of submission flags per task.

        Returns:
            A dict with keys: task_results, mean_score, tasks_passed,
            total_tasks, overall_grade.
        """
        task_results = []
        for agent_df, clean_df, task_id, steps_used, max_steps, submitted in zip(
            dirty_dfs, clean_dfs, task_ids, steps_used_list, max_steps_list, submitted_list
        ):
            result = self.grade(
                agent_df=agent_df,
                clean_df=clean_df,
                task_id=task_id,
                steps_used=steps_used,
                max_steps=max_steps,
                submitted=submitted
            )
            task_results.append(result)

        final_scores = [r["final_score"] for r in task_results]
        mean_score = round(float(np.mean(final_scores)), 4) if final_scores else 0.0
        tasks_passed = sum(1 for r in task_results if r["passed"])

        if mean_score >= 0.85:
            overall_grade = "A"
        elif mean_score >= 0.70:
            overall_grade = "B"
        elif mean_score >= 0.55:
            overall_grade = "C"
        elif mean_score >= 0.40:
            overall_grade = "D"
        else:
            overall_grade = "F"

        return {
            "task_results": task_results,
            "mean_score": mean_score,
            "tasks_passed": tasks_passed,
            "total_tasks": len(task_ids),
            "overall_grade": overall_grade,
        }


if __name__ == "__main__":
    try:
        from server.datasets.generators import DatasetFactory
    except ImportError:
        try:
            from datasets.generators import DatasetFactory
        except ImportError:
            sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            from datasets.generators import DatasetFactory

    grader = DQAGrader()

    print("=== GRADER VERIFICATION ===\n")

    for task_id in ["easy", "medium", "hard"]:
        dirty_df, clean_df = DatasetFactory.get_task_data(task_id)
        config = DatasetFactory.get_task_config(task_id)

        print(f"--- {task_id.upper()} TASK ---")

        # Test 1: Grade the dirty df (worst case — agent did nothing)
        result_worst = grader.grade(
            agent_df=dirty_df,
            clean_df=clean_df,
            task_id=task_id,
            steps_used=config["max_steps"],
            max_steps=config["max_steps"],
            submitted=False
        )
        print(f"Worst case (dirty, timeout): score={result_worst['final_score']}, grade={result_worst['grade']}")

        # Test 2: Grade the clean df (best case — perfect agent)
        result_best = grader.grade(
            agent_df=clean_df,
            clean_df=clean_df,
            task_id=task_id,
            steps_used=5,
            max_steps=config["max_steps"],
            submitted=True
        )
        print(f"Best case (clean, step 5): score={result_best['final_score']}, grade={result_best['grade']}")
        print(f"Feedback: {result_best['feedback']}")
        print(f"Gaps: {result_best['dimension_gaps']}")
        print()

    print("=== ALL TASKS BATCH GRADE ===")
    dirty_dfs, clean_dfs, configs = [], [], []
    for t in ["easy", "medium", "hard"]:
        d, c = DatasetFactory.get_task_data(t)
        cfg = DatasetFactory.get_task_config(t)
        dirty_dfs.append(d)
        clean_dfs.append(c)
        configs.append(cfg)

    batch = grader.grade_all_tasks(
        dirty_dfs=clean_dfs,  # using clean as agent output = perfect scenario
        clean_dfs=clean_dfs,
        task_ids=["easy", "medium", "hard"],
        steps_used_list=[5, 8, 10],
        max_steps_list=[c["max_steps"] for c in configs],
        submitted_list=[True, True, True]
    )
    print(f"Mean score: {batch['mean_score']}")
    print(f"Tasks passed: {batch['tasks_passed']}/{batch['total_tasks']}")
    print(f"Overall grade: {batch['overall_grade']}")

import os
import re
import json
import time
import textwrap
from typing import Optional
from openai import OpenAI

# Read mandatory env vars (competition requirement)
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
API_KEY = os.getenv("OPENAI_API_KEY") or os.getenv("HF_TOKEN")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-3.5-turbo")

# Episode config — kept small for 20-minute runtime constraint
MAX_STEPS_PER_TASK = 8
TEMPERATURE = 0.0  # deterministic for reproducibility
MAX_TOKENS = 300

# Environment server URL — can be local or HF Space
ENV_BASE_URL = os.getenv(
    "ENV_BASE_URL",
    "http://localhost:8000"
)

# For local execution, import environment directly
# This avoids needing a running server during inference
import sys
import os as _os
_os.chdir(_os.path.dirname(_os.path.abspath(__file__))
          if "__file__" in dir() else _os.getcwd())

try:
    from server.dqa_openenv_environment import DqaOpenenvEnvironment
    from models import DQAAction
    DIRECT_MODE = True
except ImportError:
    DIRECT_MODE = False


def parse_action(response_text: str) -> dict:
    """
    Parse LLM response text into a valid DQA action dict.

    Tries JSON parsing first. Falls back to noop on failure.
    Validates action_type is in allowed list.
    """
    VALID_ACTIONS = {
        "fill_nulls", "drop_duplicates", "cast_type",
        "normalize_category", "clip_outliers", "drop_column",
        "filter_rows", "merge_categories", "submit", "noop"
    }

    # Strip markdown code fences if present
    text = response_text.strip()
    text = re.sub(r"```(?:json)?", "", text).strip()
    text = text.strip("`").strip()

    # Find JSON object in response
    try:
        # Try direct parse first
        data = json.loads(text)
    except json.JSONDecodeError:
        # Try finding JSON object within text
        match = re.search(r'\{[^{}]*\}', text, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group())
            except json.JSONDecodeError:
                return {"action_type": "noop", "column": None, "parameters": {}}
        else:
            return {"action_type": "noop", "column": None, "parameters": {}}

    # Validate action_type
    action_type = data.get("action_type", "noop")
    if action_type not in VALID_ACTIONS:
        action_type = "noop"

    return {
        "action_type": action_type,
        "column": data.get("column", None),
        "parameters": data.get("parameters", {})
    }


def build_prompt(obs_dict: dict, task_id: str, step: int,
                 max_steps: int, system_prompt: str) -> list:
    """
    Build the messages list for the OpenAI API call.

    Formats the current observation into a clear prompt
    the LLM can reason about.
    """
    # Format issue hints
    hints = obs_dict.get("issue_hints", [])
    hints_text = "\n".join(f"  - {h}" for h in hints) if hints else "  - None detected"

    # Format quality scores
    scores = obs_dict.get("quality_scores", {})
    scores_text = "\n".join(
        f"  {k}: {v:.3f}" for k, v in scores.items()
    )

    # Format column null info
    col_stats = obs_dict.get("column_stats", {})
    null_cols = {
        col: stats["null_count"]
        for col, stats in col_stats.items()
        if stats.get("null_count", 0) > 0
    }
    null_text = "\n".join(
        f"  {col}: {count} nulls"
        for col, count in null_cols.items()
    ) if null_cols else "  None"

    # Format action history
    history = obs_dict.get("action_history", [])
    history_text = ", ".join(history[-5:]) if history else "none"

    last_result = obs_dict.get("last_action_result", "")
    last_success = obs_dict.get("last_action_success", True)

    user_content = textwrap.dedent(f"""
    TASK: {obs_dict.get("task_description", "")}

    STEP: {step} of {max_steps}

    CURRENT DATA QUALITY SCORES:
    {scores_text}

    DETECTED ISSUES:
    {hints_text}

    COLUMNS WITH MISSING VALUES:
    {null_text}

    AVAILABLE COLUMNS: {list(col_stats.keys())}

    RECENT ACTIONS: {history_text}
    LAST ACTION RESULT: {"SUCCESS" if last_success else "FAILED"} — {last_result}

    What is your next action? Reply with a single JSON object only.
    If no more issues, reply with: {{"action_type": "submit", "column": null, "parameters": {{}}}}
    """).strip()

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content}
    ]


def run_task(client: OpenAI, task_id: str) -> dict:
    """
    Run a full episode on a single task and return results.

    Returns dict with:
        task_id, final_score, steps_used, submitted,
        action_history, episode_reward
    """
    from server.tasks.task_easy import SYSTEM_PROMPT as EASY_PROMPT
    from server.tasks.task_medium import SYSTEM_PROMPT as MED_PROMPT
    from server.tasks.task_hard import SYSTEM_PROMPT as HARD_PROMPT

    system_prompts = {
        "easy": EASY_PROMPT,
        "medium": MED_PROMPT,
        "hard": HARD_PROMPT
    }
    system_prompt = system_prompts[task_id]

    # Initialize environment directly (no HTTP server needed)
    env = DqaOpenenvEnvironment()
    obs = env.reset(task_id=task_id)

    # Convert observation to dict for prompt building
    obs_dict = json.loads(obs.model_dump_json())

    episode_actions = []
    episode_reward = 0.0
    final_score = 0.0
    submitted = False

    print(f"\n{'='*50}")
    print(f"TASK: {task_id.upper()}")
    print(f"Initial quality score: {obs_dict.get('quality_scores', {}).get('overall', 0):.3f}")
    print(f"Issues detected: {len(obs_dict.get('issue_hints', []))}")
    print(f"{'='*50}")

    for step in range(1, MAX_STEPS_PER_TASK + 1):
        if obs_dict.get("done", False):
            break

        # Build prompt and call LLM
        messages = build_prompt(
            obs_dict, task_id, step, MAX_STEPS_PER_TASK, system_prompt
        )

        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
            )
            response_text = response.choices[0].message.content or ""
        except Exception as e:
            print(f"  Step {step}: LLM call failed ({e}), using noop")
            response_text = '{"action_type": "noop"}'

        # Parse action
        action_dict = parse_action(response_text)
        action_type = action_dict["action_type"]

        print(f"  Step {step}: {action_type}", end="")
        if action_dict.get("column"):
            print(f" on '{action_dict['column']}'", end="")

        # Execute action
        try:
            action = DQAAction(**action_dict)
            obs = env.step(action)
            obs_dict = json.loads(obs.model_dump_json())
            reward = obs_dict.get("reward", 0.0) or 0.0
            episode_reward += reward
            print(f" → reward: {reward:+.4f}")
        except Exception as e:
            print(f" → ERROR: {e}")
            obs_dict["done"] = False

        episode_actions.append(action_type)

        if action_type == "submit":
            submitted = True
            break

    # Get final score from grader
    from server.graders.grader import DQAGrader
    from server.rewards.reward_engine import QualityScorer

    grader = DQAGrader()
    _, clean_df = __import__(
        'server.datasets.generators', fromlist=['DatasetFactory']
    ).DatasetFactory.get_task_data(task_id)

    final_quality = QualityScorer.score(env._agent_df, clean_df, task_id)
    grader_result = grader.grade(
        agent_df=env._agent_df,
        clean_df=clean_df,
        task_id=task_id,
        steps_used=len(episode_actions),
        max_steps=MAX_STEPS_PER_TASK,
        submitted=submitted
    )
    final_score = grader_result["final_score"]

    print(f"\nFinal quality scores:")
    for k, v in final_quality.items():
        print(f"  {k}: {v:.4f}")
    print(f"FINAL SCORE: {final_score:.4f} (grade: {grader_result['grade']})")
    print(f"Feedback: {grader_result['feedback']}")

    return {
        "task_id": task_id,
        "final_score": final_score,
        "grade": grader_result["grade"],
        "steps_used": len(episode_actions),
        "submitted": submitted,
        "action_history": episode_actions,
        "episode_reward": round(episode_reward, 4),
        "quality_scores": final_quality
    }


def main():
    """
    Run baseline inference on all 3 DQA tasks.

    This is the competition's mandatory baseline script.
    Reads API config from environment variables.
    Prints scores for all 3 tasks.
    """
    print("=" * 60)
    print("DQA-OpenEnv Baseline Inference Script")
    print("=" * 60)
    print(f"Model: {MODEL_NAME}")
    print(f"API Base: {API_BASE_URL}")
    print(f"Max steps per task: {MAX_STEPS_PER_TASK}")
    print(f"Mode: {'Direct (no server)' if DIRECT_MODE else 'HTTP client'}")

    if not API_KEY:
        print("\nERROR: No API key found.")
        print("Set OPENAI_API_KEY or HF_TOKEN environment variable.")
        return

    if not DIRECT_MODE:
        print("\nERROR: Could not import environment directly.")
        print("Run from project root: python inference.py")
        return

    # Initialize OpenAI client (mandatory per competition spec)
    client = OpenAI(
        base_url=API_BASE_URL,
        api_key=API_KEY
    )

    start_time = time.time()
    results = []

    for task_id in ["easy", "medium", "hard"]:
        task_result = run_task(client, task_id)
        results.append(task_result)
        time.sleep(1)  # brief pause between tasks

    elapsed = time.time() - start_time

    # Final summary — judges look for this output
    print("\n" + "=" * 60)
    print("BASELINE INFERENCE RESULTS SUMMARY")
    print("=" * 60)
    for r in results:
        print(f"  {r['task_id'].upper():8s} | score: {r['final_score']:.4f} "
              f"| grade: {r['grade']} "
              f"| steps: {r['steps_used']} "
              f"| submitted: {r['submitted']}")

    mean_score = sum(r["final_score"] for r in results) / len(results)
    print(f"\n  MEAN SCORE: {mean_score:.4f}")
    print(f"  TOTAL TIME: {elapsed:.1f}s")
    print(f"  STATUS: {'PASS' if mean_score >= 0.5 else 'FAIL'}")
    print("=" * 60)


if __name__ == "__main__":
    main()

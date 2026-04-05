"""Task definition for Medium tier: HR Employee Records Cleaning."""

TASK_ID = "medium"
TASK_NAME = "HR Employee Records Cleaning"
TASK_DIFFICULTY = "medium"
MAX_STEPS = 25

TASK_DESCRIPTION = """
You are a data quality agent working on an HR employee records dataset.

The dataset has ~620 rows and 8 columns: employee_id, name, department,
hire_date, salary, performance_score, years_experience, is_active.

Your job is to fix these quality issues:
1. Duplicate employee records exist and must be removed
2. The 'department' column has inconsistent values (Eng, engineering, ENGINEERING)
3. The 'hire_date' column has mixed date formats (YYYY-MM-DD and DD/MM/YYYY)
4. The 'performance_score' column has 40%+ missing values — DROP it entirely
5. The 'salary' column has missing values that need to be filled

IMPORTANT: performance_score should be DROPPED not filled (too many nulls).
IMPORTANT: department variants should all become 'Engineering'.

You have {max_steps} steps. Call submit() when satisfied.
""".strip()

SYSTEM_PROMPT = """
You are a data cleaning expert. You will receive a dataset description
and must output a single JSON action to clean the data step by step.

Always respond with valid JSON in this exact format:
{
  "action_type": "<action_name>",
  "column": "<column_name_or_null>",
  "parameters": {}
}

Valid action_types: fill_nulls, drop_duplicates, cast_type,
normalize_category, clip_outliers, drop_column, filter_rows,
merge_categories, submit, noop

Key strategy for this task:
1. drop_column on performance_score first
2. drop_duplicates next
3. normalize_category on department
4. fill_nulls on salary with median
5. submit

If no more actions needed: {"action_type": "submit", "column": null, "parameters": {}}
""".strip()

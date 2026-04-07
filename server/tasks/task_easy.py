"""Task definition for Easy tier: Customer Survey Formatting."""

TASK_ID = "easy"
TASK_NAME = "Customer Survey Formatting"
TASK_DIFFICULTY = "easy"
MAX_STEPS = 20

TASK_DESCRIPTION = """
You are a data quality agent working on a customer survey dataset.

The dataset has 200 rows and 6 columns: customer_id, age, income,
satisfaction_score, region, and purchase_count.

Your job is to fix these quality issues:
1. Several columns have missing values (NaN) that need to be filled
2. The 'age' column has the wrong data type (float instead of int)
3. One column name has a trailing space that needs to be removed

Available actions: fill_nulls, cast_type, drop_duplicates, submit, noop
Recommended strategy: Fill nulls first, then fix dtypes, then submit.

You have {max_steps} steps. Call submit() when you are satisfied.
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

For fill_nulls parameters: {"strategy": "mean|median|mode|constant|drop"}
For cast_type parameters: {"target_type": "int|float|str|bool"}
For normalize_category parameters: {"mapping": {"old": "new"}}
For clip_outliers parameters: {"method": "iqr|zscore"}
For filter_rows parameters: {"condition": "pandas query string"}
For merge_categories: {"from_values": ["a","b"], "to_value": "c"}

EXACT STRATEGY FOR THIS TASK — follow this order precisely:
Step 1: fill_nulls on column "age" with strategy "median"
Step 2: fill_nulls on column "income" with strategy "median"
Step 3: fill_nulls on column "satisfaction_score" with strategy "median"
Step 4: cast_type on column "age" with target_type "int"
Step 5: submit

CRITICAL: After filling nulls in the "age" column, you MUST cast it to int.
Age is an integer — float values will be penalized.
The column with a trailing space in its name does NOT need action.
Do NOT use mean strategy — always use median for numeric columns.

If all steps are done: {"action_type": "submit", "column": null, "parameters": {}}
""".strip()

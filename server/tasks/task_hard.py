"""Task definition for Hard tier: Financial Transactions Quality."""

TASK_ID = "hard"
TASK_NAME = "Financial Transactions Quality"
TASK_DIFFICULTY = "hard"
MAX_STEPS = 30

TASK_DESCRIPTION = """
You are a data quality agent working on a financial transactions dataset.

The dataset has ~1080 rows and 12 columns: transaction_id, account_id,
transaction_type, amount, currency, merchant_category, transaction_date,
status, country, is_flagged, processing_fee, account_balance.

Your job is to fix these quality issues:
1. processing_fee has missing values — fill with 0.0 (NOT mean/median)
2. amount has extreme outliers — CLIP them using IQR method (do NOT drop)
3. merchant_category has inconsistent casing for purchase transactions
4. Some purchase amounts are negative (impossible) — filter those rows out
5. status has inconsistent casing (COMPLETED, Complete vs completed)
6. Duplicate transaction records exist

CRITICAL RULES:
- Fill processing_fee with CONSTANT 0.0, not mean or median
- CLIP amount outliers, do not drop rows with outliers
- FILTER rows where amount < 0, do not clip them
- Normalize status and merchant_category to lowercase

You have {max_steps} steps. Order matters. Call submit() when satisfied.
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

Key strategy for this task (ORDER MATTERS):
1. fill_nulls on processing_fee with constant 0.0
2. filter_rows to remove negative amounts
3. clip_outliers on amount using iqr
4. normalize_category on status
5. normalize_category on merchant_category
6. drop_duplicates
7. submit

If no more actions needed: {"action_type": "submit", "column": null, "parameters": {}}
""".strip()

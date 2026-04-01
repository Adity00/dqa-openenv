"""
Data generators for DQA-OpenEnv tests.

This module provides data generation routines for three difficulty tiers:
Easy (Customer Survey), Medium (HR Employee Records), and Hard (Financial Transactions).
Each task generates a paired tuple of (dirty_df, clean_df) where the dirty
dataframe is handed to the agent and the clean dataframe acts as hidden ground truth.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any
from copy import deepcopy


def generate_easy_task(seed: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generates the Easy DQA task dataset (Customer Survey).
    
    Generates 200 rows of valid customer survey data and introduces
    basic corruptions like missing values, incorrect types, and trailing spaces in columns.
    
    Returns:
        A tuple of (dirty_df, clean_df).
    """
    np.random.seed(seed)
    n_rows = 200
    
    clean_df = pd.DataFrame({
        "customer_id": np.arange(1, n_rows + 1, dtype=int),
        "age": np.random.randint(18, 76, size=n_rows),
        "income": np.round(np.random.uniform(20000, 150000, size=n_rows), 2),
        "satisfaction_score": np.round(np.random.uniform(1.0, 5.0, size=n_rows), 1),
        "region": np.random.choice(["North", "South", "East", "West"], size=n_rows),
        "purchase_count": np.random.randint(0, 51, size=n_rows)
    })
    
    dirty_df = deepcopy(clean_df)
    
    # C1: Nulls in age: set 15% of rows to NaN
    idx_age = np.random.choice(n_rows, size=int(n_rows * 0.15), replace=False)
    # C4: Convert age column to float64 to support np.nan
    dirty_df["age"] = dirty_df["age"].astype(float)
    dirty_df.loc[idx_age, "age"] = np.nan
    
    # C2: Nulls in income: set 12% of rows to NaN
    idx_income = np.random.choice(n_rows, size=int(n_rows * 0.12), replace=False)
    dirty_df.loc[idx_income, "income"] = np.nan
    
    # C3: Nulls in satisfaction_score: set 10% of rows to NaN
    idx_sat = np.random.choice(n_rows, size=int(n_rows * 0.10), replace=False)
    dirty_df.loc[idx_sat, "satisfaction_score"] = np.nan
    
    # C5: Column name with trailing space
    dirty_df.rename(columns={"purchase_count": "purchase_count "}, inplace=True)
    
    return dirty_df, clean_df


def generate_medium_task(seed: int = 123) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generates the Medium DQA task dataset (HR Employee Records).
    
    Generates 500 rows of HR data and introduces intermediate corruptions
    including duplicate rows, string inconsistencies, mixed date formats, and structurally missing columns.
    
    Returns:
        A tuple of (dirty_df, clean_df).
    """
    np.random.seed(seed)
    n_rows = 500
    
    names_pool = [
        "James", "Mary", "John", "Patricia", "Robert", "Jennifer", 
        "Michael", "Linda", "William", "Elizabeth", "David", "Barbara", 
        "Richard", "Susan", "Joseph", "Jessica", "Thomas", "Sarah", 
        "Charles", "Karen"
    ]
                  
    # Generate random dates
    start_date = np.datetime64('2015-01-01')
    end_date = np.datetime64('2024-01-01')
    days_range = (end_date - start_date).astype(int)
    random_days = np.random.randint(0, days_range, size=n_rows)
    dates = (start_date + random_days).astype(str)
    
    clean_df = pd.DataFrame({
        "employee_id": [f"EMP{i:03d}" for i in range(1, n_rows + 1)],
        "name": np.random.choice(names_pool, size=n_rows),
        "department": np.random.choice(["Engineering", "Marketing", "Sales", "HR", "Finance"], size=n_rows),
        "hire_date": dates,
        "salary": np.round(np.random.uniform(35000, 120000, size=n_rows), 2),
        "performance_score": np.round(np.random.uniform(1.0, 5.0, size=n_rows), 1),
        "years_experience": np.random.randint(0, 21, size=n_rows),
        "is_active": np.random.choice([True, False], size=n_rows, p=[0.9, 0.1])
    })
    
    dirty_df = deepcopy(clean_df)
    
    # C4: High null column: set 40% of performance_score to NaN
    idx_perf = np.random.choice(n_rows, size=int(n_rows * 0.40), replace=False)
    dirty_df.loc[idx_perf, "performance_score"] = np.nan
    # Expected outcome is dropping the column completely
    clean_df.drop(columns=["performance_score"], inplace=True)
    
    # C2: Category inconsistency in department column
    eng_indices = dirty_df[dirty_df["department"] == "Engineering"].index.tolist()
    num_to_corrupt = int(len(eng_indices) * 0.90)
    idx_corrupt_eng = np.random.choice(eng_indices, size=num_to_corrupt, replace=False)
    dirty_df.loc[idx_corrupt_eng, "department"] = np.random.choice(["engineering", "Eng", "ENGINEERING"], size=num_to_corrupt)
    
    # C3: Mixed date formats
    idx_date = np.random.choice(n_rows, size=int(n_rows * 0.65), replace=False)
    def reformat_date(d_str: str) -> str:
        parts = d_str.split("-")
        return f"{parts[2]}/{parts[1]}/{parts[0]}"
    dirty_df.loc[idx_date, "hire_date"] = dirty_df.loc[idx_date, "hire_date"].apply(reformat_date)
    
    # C5: Salary nulls
    idx_sal = np.random.choice(n_rows, size=int(n_rows * 0.28), replace=False)
    dirty_df.loc[idx_sal, "salary"] = np.nan
    
    # C1: Duplicate rows: duplicate 40 random rows
    idx_dup = np.random.choice(n_rows, size=120, replace=False)
    duplicates = dirty_df.iloc[idx_dup].copy()
    dirty_df = pd.concat([dirty_df, duplicates], ignore_index=True)
    
    return dirty_df, clean_df


def generate_hard_task(seed: int = 777) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generates the Hard DQA task dataset (Financial Transactions).
    
    Generates 1000 rows of transaction data with intensive corruptions including
    domain-aware null solutions, statistical outliers needing IQR clipping, 
    impossible values ensuring rows are dropped, and hierarchy/status inconsistencies.
    
    Returns:
        A tuple of (dirty_df, clean_df).
    """
    np.random.seed(seed)
    n_rows = 1000
    
    account_ids = [f"ACC{i:03d}" for i in range(1, 51)]
    
    start_date = np.datetime64('2023-01-01')
    end_date = np.datetime64('2025-01-01')
    days_range = (end_date - start_date).astype(int)
    random_days = np.random.randint(0, days_range, size=n_rows)
    dates = (start_date + random_days).astype(str)
    
    amounts = np.round(np.random.uniform(1.0, 5000.0, size=n_rows), 2)
    fees = np.round(np.random.uniform(0.01, 0.05, size=n_rows) * amounts, 4)
    
    clean_df = pd.DataFrame({
        "transaction_id": [f"TXN{i:04d}" for i in range(1, n_rows + 1)],
        "account_id": np.random.choice(account_ids, size=n_rows),
        "transaction_type": np.random.choice(["purchase", "refund", "transfer", "withdrawal"], size=n_rows),
        "amount": amounts,
        "currency": np.random.choice(["USD", "EUR", "GBP"], size=n_rows, p=[0.7, 0.2, 0.1]),
        "merchant_category": np.random.choice(["retail", "food", "travel", "utilities", "healthcare"], size=n_rows),
        "transaction_date": dates,
        "status": np.random.choice(["completed", "pending", "failed"], size=n_rows, p=[0.85, 0.10, 0.05]),
        "country": np.random.choice(["US", "UK", "DE", "FR", "JP"], size=n_rows),
        "is_flagged": np.random.choice([True, False], size=n_rows, p=[0.05, 0.95]),
        "processing_fee": fees,
        "account_balance": np.round(np.random.uniform(100.0, 50000.0, size=n_rows), 2)
    })
    
    dirty_df = deepcopy(clean_df)
    
    # C1 & C5: Domain-aware nulls in processing_fee
    idx_fee = np.random.choice(n_rows, size=int(n_rows * 0.30), replace=False)
    dirty_df.loc[idx_fee, "processing_fee"] = np.nan
    clean_df.loc[idx_fee, "processing_fee"] = 0.0
    
    # C2: Statistical outliers in amount
    idx_outliers = np.random.choice(n_rows, size=int(n_rows * 0.12), replace=False)
    dirty_df.loc[idx_outliers, "amount"] = dirty_df.loc[idx_outliers, "amount"] * 100.0
    
    # Expected outcome for clean_df computes IQR bounds over the dirty distribution
    Q1 = dirty_df["amount"].quantile(0.25)
    Q3 = dirty_df["amount"].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    clean_df["amount"] = dirty_df["amount"].clip(lower_bound, upper_bound)
    
    # C3: Category hierarchy inconsistency
    purchase_idx = dirty_df[dirty_df["transaction_type"] == "purchase"].index.tolist()
    num_cat_err = int(len(purchase_idx) * 0.75)
    cat_err_idx = np.random.choice(purchase_idx, size=num_cat_err, replace=False)
    dirty_df.loc[cat_err_idx, "merchant_category"] = np.random.choice(["RETAIL", "Food"], size=num_cat_err)
    
    # C4: Impossible values
    num_neg = int(len(purchase_idx) * 0.15)
    neg_idx = np.random.choice(purchase_idx, size=num_neg, replace=False)
    dirty_df.loc[neg_idx, "amount"] = -np.abs(dirty_df.loc[neg_idx, "amount"])
    # Clean matches dropping these negative occurrences entirely
    clean_df.drop(index=neg_idx, inplace=True)
    
    # C6: Status inconsistency
    completed_idx = dirty_df[dirty_df["status"] == "completed"].index.tolist()
    num_status_err = int(len(completed_idx) * 0.60)
    status_err_idx = np.random.choice(completed_idx, size=num_status_err, replace=False)
    dirty_df.loc[status_err_idx, "status"] = np.random.choice(["COMPLETED", "Complete"], size=num_status_err)
    
    # C7: Add duplicate rows to hard task
    idx_hard_dup = np.random.choice(len(dirty_df), size=80, replace=False)
    hard_duplicates = dirty_df.iloc[idx_hard_dup].copy()
    dirty_df = pd.concat([dirty_df, hard_duplicates], ignore_index=True)

    # Standardize clean copy
    clean_df.reset_index(drop=True, inplace=True)
    
    return dirty_df, clean_df

class DatasetFactory:
    """Factory for generating requested DQA task datasets based on tier."""
    
    @staticmethod
    def get_task_data(task_id: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Provides the pre-generated pairs for the targeted task depth."""
        if task_id == "easy":
            return generate_easy_task()
        elif task_id == "medium":
            return generate_medium_task()
        elif task_id == "hard":
            return generate_hard_task()
        else:
            raise ValueError(f"Unknown task_id: {task_id}")

    @staticmethod
    def get_task_config(task_id: str) -> Dict[str, Any]:
        """Returns foundational metadata properties tied to each task tier."""
        if task_id == "easy":
            return {
                "task_id": "easy",
                "name": "Customer Survey Formatting",
                "description": "Clean standard issues around null imputation and types.",
                "max_steps": 20,
                "num_rows": 200,
                "num_columns": 6,
                "issues_count": 5,
                "difficulty": "easy"
            }
        elif task_id == "medium":
            return {
                "task_id": "medium",
                "name": "HR Employee Records Cleaning",
                "description": "Scale through duplicates, null boundaries, and string variants.",
                "max_steps": 25,
                "num_rows": 540,
                "num_columns": 8,
                "issues_count": 5,
                "difficulty": "medium"
            }
        elif task_id == "hard":
            return {
                "task_id": "hard",
                "name": "Financial Transactions Quality",
                "description": "Aggressive scaling tests covering outliers, specific zero-filling bounds, and logical impossibilities.",
                "max_steps": 30,
                "num_rows": 1000,
                "num_columns": 12,
                "issues_count": 6,
                "difficulty": "hard"
            }
        else:
            raise ValueError(f"Unknown task_id: {task_id}")


if __name__ == "__main__":
    for t_id in ["easy", "medium", "hard"]:
        dirty, clean = DatasetFactory.get_task_data(t_id)
        config = DatasetFactory.get_task_config(t_id)
        
        print(f"Task: {config['name']} ({t_id})")
        print(f"Dirty shape: {dirty.shape}, Clean shape: {clean.shape}")
        
        print("Dirty dtypes:")
        print(dirty.dtypes)
        
        print("Dirty isnull sum:")
        print(dirty.isnull().sum())
        
        print(f"First 3 rows of dirty:")
        print(dirty.head(3))
        print("---")

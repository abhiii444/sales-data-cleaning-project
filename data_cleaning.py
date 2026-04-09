"""
Task 1: Data Cleaning and Preprocessing
Dataset: Sales Data (custom raw dataset)
Tools: Python (Pandas)
Author: Data Analyst Intern
"""

import pandas as pd
import numpy as np

# ─────────────────────────────────────────
# 1. LOAD RAW DATA
# ─────────────────────────────────────────
df = pd.read_csv('raw_sales_data.csv')

print("=" * 60)
print("STEP 1: INITIAL DATA OVERVIEW")
print("=" * 60)
print(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns\n")
print("Column Data Types:")
print(df.dtypes)
print("\nFirst 5 rows:")
print(df.head())
print("\nMissing Values:")
print(df.isnull().sum())
print("\nBasic Statistics:")
print(df.describe(include='all'))

summary_log = []
initial_shape = df.shape

# ─────────────────────────────────────────
# 2. RENAME COLUMNS (uniform, lowercase, no spaces)
# ─────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 2: RENAME COLUMNS")
print("=" * 60)

df.columns = (
    df.columns
    .str.strip()
    .str.lower()
    .str.replace(' ', '_', regex=False)
)
print("Renamed columns:", list(df.columns))
summary_log.append("✔ Renamed all column headers to lowercase with underscores.")

# ─────────────────────────────────────────
# 3. REMOVE DUPLICATE ROWS
# ─────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 3: REMOVE DUPLICATES")
print("=" * 60)

before = len(df)
df.drop_duplicates(inplace=True)
after = len(df)
removed = before - after
print(f"Duplicates removed: {removed} rows ({before} → {after})")
summary_log.append(f"✔ Removed {removed} duplicate rows ({before} → {after} rows).")

# ─────────────────────────────────────────
# 4. STANDARDIZE TEXT COLUMNS
# ─────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 4: STANDARDIZE TEXT COLUMNS")
print("=" * 60)

# --- Gender ---
gender_map = {
    'male': 'Male', 'm': 'Male',
    'female': 'Female', 'f': 'Female',
}
df['gender'] = df['gender'].str.strip().str.lower().map(
    lambda x: gender_map.get(x, x.capitalize() if isinstance(x, str) else x)
)
print("Gender unique values:", df['gender'].unique())
summary_log.append("✔ Standardized 'gender' column (M/male/MALE → Male, F/female/FEMALE → Female).")

# --- Country ---
country_map = {
    'usa': 'USA', 'united states': 'USA', 'us': 'USA',
    'india': 'India', 'ind': 'India',
    'uk': 'UK', 'united kingdom': 'UK',
    'germany': 'Germany', 'ger': 'Germany',
}
df['country'] = df['country'].str.strip().str.lower().map(
    lambda x: country_map.get(x, x.title() if isinstance(x, str) else x)
)
print("Country unique values:", df['country'].unique())
summary_log.append("✔ Standardized 'country' column (abbrev/mixed-case → consistent names).")

# --- Product Category ---
df['product_category'] = df['product_category'].str.strip().str.title()
print("Product category unique values:", df['product_category'].unique())
summary_log.append("✔ Standardized 'product_category' to Title Case.")

# --- Payment Method ---
df['payment_method'] = df['payment_method'].str.strip().str.title()
print("Payment method unique values:", df['payment_method'].unique())
summary_log.append("✔ Standardized 'payment_method' to Title Case.")

# ─────────────────────────────────────────
# 5. FIX DATE FORMAT
# ─────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 5: FIX DATE FORMAT")
print("=" * 60)

def parse_dates(date_str):
    if pd.isna(date_str):
        return pd.NaT
    for fmt in ('%d-%m-%Y', '%m/%d/%Y', '%Y-%m-%d'):
        try:
            return pd.to_datetime(date_str, format=fmt)
        except (ValueError, TypeError):
            continue
    return pd.NaT

df['purchase_date'] = df['purchase_date'].apply(parse_dates)
df['purchase_date'] = pd.to_datetime(df['purchase_date'], errors='coerce')
# Standardize to dd-mm-yyyy string
df['purchase_date'] = df['purchase_date'].dt.strftime('%d-%m-%Y')
print("Sample dates after fix:", df['purchase_date'].dropna().head().tolist())
summary_log.append("✔ Standardized all date formats to dd-mm-yyyy (parsed dd-mm-yyyy, mm/dd/yyyy, yyyy-mm-dd).")

# ─────────────────────────────────────────
# 6. FIX DATA TYPES
# ─────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 6: FIX DATA TYPES")
print("=" * 60)

# Age → int (handle invalid values first)
df['age'] = pd.to_numeric(df['age'], errors='coerce')
invalid_age = ((df['age'] < 0) | (df['age'] > 120)).sum()
df.loc[(df['age'] < 0) | (df['age'] > 120), 'age'] = np.nan
print(f"Invalid age values fixed (out of 0–120 range): {invalid_age}")
summary_log.append(f"✔ Fixed {invalid_age} invalid 'age' values (negatives / >120 → NaN).")

# Amount Spent → float (handle negatives/zeros)
df['amount_spent'] = pd.to_numeric(df['amount_spent'], errors='coerce')
invalid_amt = (df['amount_spent'] <= 0).sum()
df.loc[df['amount_spent'] <= 0, 'amount_spent'] = np.nan
print(f"Invalid amount_spent values fixed (<=0): {invalid_amt}")
summary_log.append(f"✔ Fixed {invalid_amt} invalid 'amount_spent' values (≤0 → NaN).")

# Rating → int (1–5 valid range)
df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
invalid_rating = ((df['rating'] < 1) | (df['rating'] > 5)).sum()
df.loc[(df['rating'] < 1) | (df['rating'] > 5), 'rating'] = np.nan
print(f"Invalid rating values fixed (outside 1–5): {invalid_rating}")
summary_log.append(f"✔ Fixed {invalid_rating} invalid 'rating' values (outside 1–5 → NaN).")

# ─────────────────────────────────────────
# 7. HANDLE MISSING VALUES
# ─────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 7: HANDLE MISSING VALUES")
print("=" * 60)

print("Missing values before treatment:")
print(df.isnull().sum())

# Numeric columns: fill with median
df['age'] = df['age'].fillna(df['age'].median()).round().astype('Int64')
df['amount_spent'] = df['amount_spent'].fillna(df['amount_spent'].median())
df['rating'] = df['rating'].fillna(df['rating'].median()).round().astype('Int64')

# Categorical columns: fill with mode
for col in ['gender', 'country', 'product_category', 'payment_method']:
    mode_val = df[col].mode()[0]
    null_count = df[col].isnull().sum()
    df[col] = df[col].fillna(mode_val)
    if null_count > 0:
        summary_log.append(f"✔ Filled {null_count} missing '{col}' values with mode ('{mode_val}').")

# Dates: fill with 'Unknown'
date_nulls = df['purchase_date'].isnull().sum()
df['purchase_date'] = df['purchase_date'].fillna('Unknown')

print("\nMissing values after treatment:")
print(df.isnull().sum())
summary_log.append("✔ Filled missing numeric values (age, amount_spent, rating) with column median.")
summary_log.append(f"✔ Filled {date_nulls} missing 'purchase_date' values with 'Unknown'.")

# ─────────────────────────────────────────
# 8. OUTLIER TREATMENT (Amount Spent)
# ─────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 8: OUTLIER TREATMENT (IQR Method)")
print("=" * 60)

Q1 = df['amount_spent'].quantile(0.25)
Q3 = df['amount_spent'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
outliers = df[(df['amount_spent'] < lower_bound) | (df['amount_spent'] > upper_bound)]
print(f"IQR: {IQR:.2f} | Bounds: [{lower_bound:.2f}, {upper_bound:.2f}]")
print(f"Outliers detected: {len(outliers)}")
# Cap outliers instead of dropping
df['amount_spent'] = df['amount_spent'].clip(lower=lower_bound, upper=upper_bound)
summary_log.append(f"✔ Capped {len(outliers)} outliers in 'amount_spent' using IQR method "
                   f"(bounds: [{lower_bound:.2f}, {upper_bound:.2f}]).")

# ─────────────────────────────────────────
# 9. FINAL CHECK & SAVE
# ─────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 9: FINAL CLEANED DATASET")
print("=" * 60)
print(f"Final shape: {df.shape[0]} rows × {df.shape[1]} columns")
print("\nData Types:")
print(df.dtypes)
print("\nSample of cleaned data:")
print(df.head(10).to_string())

df.to_csv('cleaned_sales_data.csv', index=False)
print("\n✅ Cleaned dataset saved as 'cleaned_sales_data.csv'")

# ─────────────────────────────────────────
# 10. PRINT SUMMARY REPORT
# ─────────────────────────────────────────
print("\n" + "=" * 60)
print("CLEANING SUMMARY REPORT")
print("=" * 60)
print(f"Initial shape : {initial_shape[0]} rows × {initial_shape[1]} columns")
print(f"Final shape   : {df.shape[0]} rows × {df.shape[1]} columns")
print("\nChanges Made:")
for i, item in enumerate(summary_log, 1):
    print(f"  {i}. {item}")

import pandas as pd
import numpy as np

np.random.seed(42)
n = 200

data = {
    'CustomerID': list(range(1001, 1001 + n)),
    'customer_name': [
        np.random.choice(['Alice Johnson', 'Bob Smith', 'Carol White', 'David Brown',
                          'Eve Davis', 'Frank Miller', 'Grace Wilson', 'Henry Moore',
                          'Ivy Taylor', 'Jack Anderson']) for _ in range(n)
    ],
    'Gender': [np.random.choice(['Male', 'Female', 'male', 'female', 'M', 'F', 'MALE', 'FEMALE', None])
               for _ in range(n)],
    'Age': [np.random.choice([np.random.randint(18, 70), None, -5, 150]) for _ in range(n)],
    'Country': [np.random.choice(['USA', 'United States', 'US', 'India', 'IND', 'india',
                                   'UK', 'United Kingdom', 'Germany', 'GER', None]) for _ in range(n)],
    'Purchase_Date': [np.random.choice([
        f'{np.random.randint(1,28):02d}-{np.random.randint(1,12):02d}-2023',
        f'{np.random.randint(1,12):02d}/{np.random.randint(1,28):02d}/2023',
        f'2023-{np.random.randint(1,12):02d}-{np.random.randint(1,28):02d}',
        None
    ]) for _ in range(n)],
    'Product Category': [np.random.choice(['Electronics', 'electronics', 'ELECTRONICS',
                                             'Clothing', 'clothing', 'Food', 'FOOD',
                                             'Sports', 'sports', None]) for _ in range(n)],
    'Amount Spent': [np.random.choice([round(np.random.uniform(10, 5000), 2), None, -100, 0])
                     for _ in range(n)],
    'Payment_Method': [np.random.choice(['Credit Card', 'credit card', 'CREDIT CARD',
                                          'Debit Card', 'debit card', 'Cash', 'CASH',
                                          'PayPal', 'paypal', None]) for _ in range(n)],
    'Rating': [np.random.choice([np.random.randint(1, 6), None, 0, 10]) for _ in range(n)],
}

df = pd.DataFrame(data)

# Add duplicates (20 duplicate rows)
duplicate_rows = df.sample(20, random_state=42)
df = pd.concat([df, duplicate_rows], ignore_index=True)

# Shuffle
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

df.to_csv('/home/claude/raw_sales_data.csv', index=False)
print(f"Raw dataset created: {df.shape[0]} rows, {df.shape[1]} columns")
print(df.head())

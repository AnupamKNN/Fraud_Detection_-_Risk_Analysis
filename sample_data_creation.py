import pandas as pd
import numpy as np

# Columns based on your training data excluding 'Class'
columns = [
    "Time", "V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9",
    "V10", "V11", "V12", "V13", "V14", "V15", "V16", "V17", "V18", "V19",
    "V20", "V21", "V22", "V23", "V24", "V25", "V26", "V27", "V28", "Amount"
]

# Set random seed for reproducibility
np.random.seed(42)

# Generate synthetic data with similar value ranges as your dataset
data = {
    "Time": np.random.uniform(0, 172800, 5000),  # up to 2 days in seconds
    "V1": np.random.normal(0, 2, 5000),
    "V2": np.random.normal(0, 2, 5000),
    "V3": np.random.normal(0, 2, 5000),
    "V4": np.random.normal(0, 2, 5000),
    "V5": np.random.normal(0, 2, 5000),
    "V6": np.random.normal(0, 2, 5000),
    "V7": np.random.normal(0, 2, 5000),
    "V8": np.random.normal(0, 2, 5000),
    "V9": np.random.normal(0, 2, 5000),
    "V10": np.random.normal(0, 2, 5000),
    "V11": np.random.normal(0, 2, 5000),
    "V12": np.random.normal(0, 2, 5000),
    "V13": np.random.normal(0, 2, 5000),
    "V14": np.random.normal(0, 2, 5000),
    "V15": np.random.normal(0, 2, 5000),
    "V16": np.random.normal(0, 2, 5000),
    "V17": np.random.normal(0, 2, 5000),
    "V18": np.random.normal(0, 2, 5000),
    "V19": np.random.normal(0, 2, 5000),
    "V20": np.random.normal(0, 2, 5000),
    "V21": np.random.normal(0, 2, 5000),
    "V22": np.random.normal(0, 2, 5000),
    "V23": np.random.normal(0, 2, 5000),
    "V24": np.random.normal(0, 2, 5000),
    "V25": np.random.normal(0, 2, 5000),
    "V26": np.random.normal(0, 2, 5000),
    "V27": np.random.normal(0, 2, 5000),
    "V28": np.random.normal(0, 2, 5000),
    "Amount": np.random.uniform(0.0, 5000.0, 5000)  # realistic amount range
}

df = pd.DataFrame(data, columns=columns)

# Save to CSV
df.to_csv("sample_transactions.csv", index=False)

print("✅ sample_transactions.csv with 5000 rows created successfully!")

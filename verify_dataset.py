import pandas as pd
import os

for file in os.listdir():
    if file.endswith("_data.csv"):
        df = pd.read_csv(file)
        print(f"\n✅ {file} - Shape: {df.shape}")
        print(df['label'].value_counts())
        if df.shape[1] != 64:
            print("⚠️  Error: Wrong column count (expected 64 incl. label)")

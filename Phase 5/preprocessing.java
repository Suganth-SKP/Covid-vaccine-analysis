import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
df=pd.read_csv('preprocessed_data.csv')
# Check for missing values
missing_values = df.isnull().sum()
# Check for duplicates
duplicate_rows = df[df.duplicated(keep='first')]
print("Missing Values:")
print(missing_values)
print("Duplicate Rows:")
print(duplicate_rows)

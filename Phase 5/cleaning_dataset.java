import pandas as pd
import numpy as np

# Load your dataset into a Pandas DataFrame
data = pd.read_csv("country_vaccinations.csv")
# Handling Missing Data
# Check for missing values in the dataset
data.isnull().sum()
# Depending on your analysis, you can either drop rows with missing values or fill them with appropriate values (e.g., zeros or the mean of the column).
# Drop rows with missing values
data.dropna(inplace=True)
#  Data Type Conversion
# Ensure the data types of columns are appropriate for analysis
data['date'] = pd.to_datetime(data['date'])
#  Feature Engineering
# Create new columns or features if needed
data['Vaccination Rate'] = data['daily_vaccinations'] / data['total_vaccinations_per_hundred']
#  Data Scaling and Normalization
# Scale and normalize numeric columns if necessary
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
data[['total_vaccinations_per_hundred', 'people_vaccinated_per_hundred','people_fully_vaccinated_per_hundred']] = scaler.fit_transform(data[['total_vaccinations_per_hundred', 'people_vaccinated_per_hundred','people_fully_vaccinated_per_hundred']])
#  Encoding Categorical Data
# If you have categorical data like 'Country', you can encode it into numerical values.
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
data['country'] = encoder.fit_transform(data['country'])
#  Save Preprocessed Data
# Save the preprocessed data to a new CSV file for further analysis
data.to_csv("preprocessed_data.csv", index=False)

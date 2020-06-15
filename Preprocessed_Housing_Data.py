import pandas as pd
import numpy as np

# Set the option of displaying all columns (this will be a lot but good for scanning over the information)
pd.set_option('display.max_columns', None)

# Import csv file
train = pd.read_csv('/Users/Jonas/Desktop/DataScience/Kaggle/HousePrices/CSVs/train.csv')

# Inspect general information and get some statistics
# print(train.info())
# print(train.describe())

# Get a sense of the target value
# print(train.SalePrice.describe())
# Median sales price is 163k and IQR is between 129k to 214k

# Save the first and third quartile as lower quartile and upper quartile
lqr = np.percentile(train.SalePrice, 25)
uqr = np.percentile(train.SalePrice, 75)

# Now it's about inspecting some of the features to see if any seem more relevant than others
# First feature to inspect is the neighborhood
# print(train.groupby('Neighborhood').SalePrice.mean())
# There seem to be big differences in sales prices across neighborhoods

# Separate the neighborhoods into three groups: below IQR, within IQR and above IQR
print(train.isna().any())


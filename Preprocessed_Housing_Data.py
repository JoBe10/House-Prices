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

# Save the first, second and third quartile as lower quartile and upper quartile
sp_lqr = np.percentile(train.SalePrice, 25)
sp_med = np.percentile(train.SalePrice, 50)
sp_uqr = np.percentile(train.SalePrice, 75)

# Now it's about inspecting some of the features to see if any seem more relevant than others
# First feature to inspect is the neighborhood
# print(train.groupby('Neighborhood').SalePrice.mean())
# There seem to be big differences in sales prices across neighborhoods

# Separate the neighborhoods into four groups: below IQR, below median, below 75th percentile and above IQR
# Create empty lists that will be filled with neighborhoods belonging to the respective groups
below_lqr = []
below_med = []
below_uqr = []
above_uqr = []

# Loop through the neighborhoods and append to the correct group based on average sales price
neighborhoods = train.groupby('Neighborhood').SalePrice.mean()
for i in neighborhoods.keys():
    if neighborhoods[i] < sp_lqr:
        below_lqr.append(i)
    elif neighborhoods[i] < sp_med:
        below_med.append(i)
    elif neighborhoods[i] < sp_uqr:
        below_uqr.append(i)
    else:
        above_uqr.append(i)

# Create new column for the neighborhood group
train['neighborhood_group'] = train.Neighborhood.apply(lambda x: 0 if x in below_lqr else (1 if x in below_med else (2 if x in below_uqr else 3)))


# Inspect statistics about years built
# print(train.YearBuilt.describe())

# Let's create 4 groups based on year built: After 2000, between 1973 and 2000, between 1954 and 1973 and before 1954
# Obtain the quartiles of YearBuilt
yb_uqr = np.percentile(train.YearBuilt, 75)
yb_med = np.percentile(train.YearBuilt, 50)
yb_lqr = np.percentile(train.YearBuilt, 25)

# Create new column for the year group
train['year_group'] = train.YearBuilt.apply(lambda x: 0 if x < yb_lqr else (1 if x < yb_med else(2 if x < yb_uqr else 3)))

# Check to see whether there is a difference in sales prices between the groups
# print(train.groupby('year_group').SalePrice.mean())
# It seems as though the newer the house the higher the sales price

# Another feature that could have a big impact on sales price is whether the house has a pool or not




import pandas as pd
import numpy as np

# Set the option of displaying all columns (this will be a lot but good for scanning over the information)
pd.set_option('display.max_columns', None)

# Import csv file
train = pd.read_csv('/Users/Jonas/Desktop/DataScience/Kaggle/HousePrices/CSVs/train.csv')

# Inspect general information and get some statistics
# print(train.info())
# print(train.describe())

# Get list of all columns that have NaNs
nas = train.columns[train.isna().any()].tolist()
print(nas)
# There is too many to worry about them now
# Later, if a feature is considered in the regression and it appears in this list we will deal with the NaNs

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
train['has_pool'] = train.PoolArea.apply(lambda x: 1 if x > 0 else 0)
#print(train.has_pool.value_counts())
# Unfortunately there are only few houses with pools, limiting the "power" of thsi feature

# Let's inspect  differences in type of dwelling
# print(train.groupby('MSSubClass').SalePrice.mean())
# 20, 60, 75 and 120 seem to be sought after, 30, 180, 190 and 45 not so much
# Group them according to average sale price with the average being either > 190k, < 130k or in between

# Store the average sales prices per dwelling type
dwellings = train.groupby('MSSubClass').SalePrice.mean()

# Create empty lists to be filled with the dwelling types that fall in the different categories
high_dw = []
med_dw = []
low_dw = []

# Loop through the dwelling keys and fill the above lists
for i in dwellings.keys():
    if dwellings[i] < 130000:
        low_dw.append(i)
    elif dwellings[i] < 190000:
        med_dw.append(i)
    else:
        high_dw.append(i)

# Create new column for the different groups of dwellings
train['dwelling'] = train.MSSubClass.apply(lambda x: 0 if x in low_dw else (1 if x in med_dw else 2))

# Check to make sure there is a difference in average sales price between the groups
# print(train.groupby('dwelling').SalePrice.mean())
# print(train.dwelling.value_counts())
# So far so good

# Inspect differences in average sales prices by zone
# print(train.groupby('MSZoning').SalePrice.mean())
# print(train.MSZoning.value_counts())
# C has an extremely low average sales price, FV and RL a really high one and the rest is somewhere in the middle

# Create new column for groups of the zones
train['zone_group'] = train.MSZoning.apply(lambda x: 3 if x == 'FV' else (2 if x == 'RL' else(0 if x == 'C (all)' else 1)))

# Inspect the different building types
print(train.groupby('BldgType').SalePrice.mean())
print(train.BldgType.value_counts())
# Single-family and Townhouse end units seem to be selling for more

# Create a column with 1s for 1Fam and TwnhsE and 0 for all others
train['type_group'] = train.BldgType.apply(lambda x: 1 if x == ('1Fam' or 'TwnhsE') else 0)
print(train.groupby('type_group').SalePrice.mean())

# Inspect style of building
# print(train.groupby('HouseStyle').SalePrice.mean())
# print(train.HouseStyle.value_counts())
# 2 Story and 2.5Fin have substantially higher sales prices than most others

# Split the styles into three groups, one for two story with the second story finished, one for one story and one for everything else
train['style_group'] = train.HouseStyle.apply(lambda x: 2 if x == ('2Story' or '2. 5Fin') else (1 if x == '1Story' else 0))

# Just out of interest, what would the combination of 1Fam and 2Story look like
# Because we are basing this column on two other columns we need to define a function that we can feed into apply
def type_style(x):
    if (x['BldgType'] == '1Fam' and x['HouseStyle'] == '2Story'):
        return 1
    else:
        return 0
train['1f2s'] = train.apply(type_style, axis=1)
print(train['1f2s'].value_counts())




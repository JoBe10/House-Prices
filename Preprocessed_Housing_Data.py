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
# print(train.groupby('BldgType').SalePrice.mean())
# print(train.BldgType.value_counts())
# Single-family and Townhouse end units seem to be selling for more

# Create a column with 1s for 1Fam and TwnhsE and 0 for all others
train['type_group'] = train.BldgType.apply(lambda x: 1 if x == ('1Fam' or 'TwnhsE') else 0)

# Inspect style of building
# print(train.groupby('HouseStyle').SalePrice.mean())
# print(train.HouseStyle.value_counts())
# 2 Story and 2.5Fin have substantially higher sales prices than most others

# Split the styles into three groups, one for two story with the second story finished, one for one story and one for everything else
train['style_group'] = train.HouseStyle.apply(lambda x: 2 if x in ['2Story', '2. 5Fin'] else (1 if x == '1Story' else 0))

# Just out of interest, what would the combination of 1Fam and 2Story look like
# Because we are basing this column on two other columns we need to define a function that we can feed into apply
def type_style(x):
    if (x['BldgType'] == '1Fam' and x['HouseStyle'] == '2Story'):
        return 1
    else:
        return 0

# Using the above function and axis=1 (to pass the Series row_wise) create the interaction column
train['one_fam_two_story'] = train.apply(type_style, axis=1)
# print(train['one_fam_two_story'].value_counts())
# print(train.groupby('one_fam_two_story').SalePrice.mean())
# There seems to be a big difference and sufficient counts in each group

# Let's say that overall quality and condition matter but the combination is what's most important
# Create a column that calculates the average of the quality and condition scores
train['qu_co'] = round((train.OverallQual + train.OverallCond) / 2)

# Create a column that multiplies the two scores (just to compare later on
train['qual_cond'] = round((train.OverallQual * train.OverallCond) / 10)

# Inspect differences in average slaes price for different roof styles
# print(train.groupby('RoofStyle').SalePrice.mean())
# print(train['RoofStyle'].value_counts())
# Shed and hip seem to increase sale prices most, followed by flat and then everything else

# Create a column with groups of roof styles where Shed and Hip have value 2, Flat has 1 and everything else a 0
train['roof_style'] = train.RoofStyle.apply(lambda x: 2 if x in ['Shed', 'Hip'] else (1 if x == 'Flat' else 0))

# Inspect differences in average slaes price for different roof materials
# print(train.groupby('RoofMatl').SalePrice.mean())
# print(train['RoofMatl'].value_counts())
# Wood and Membrane as roof material seem to drastically increase the sales price

# Create a column with 1s for wood or membrane and 0 for everything else
train['wood_membrane'] = train.RoofMatl.apply(lambda x: 1 if x in ['Membran', 'WdShake', 'WdShngl'] else 0)

# Inspect both exterior coverings of houses and the average sale prices related to then
# print(train.groupby('Exterior1st').SalePrice.mean())
# print(train['Exterior1st'].value_counts())
# print(train.groupby('Exterior2nd').SalePrice.mean())
# print(train['Exterior2nd'].value_counts())
# Vinyl Siding is not only common but also seems to increase sales prices
# Other exterior coverings that seem to increase sales prices are: CemntBd, Imstucc, Stone (only when 1st) and BrkFace

# Create a binary column for any VinylSd
# train['vinyl_side'] = train.apply(lambda x: 1 if (x['Exterior1st'] or x['Exterior2nd']) == 'VinylSd' else 0, axis=1)
train['vinyl_side'] = train.apply(lambda x: 1 if 'VinylSd' in [x['Exterior1st'], x['Exterior2nd']] else 0, axis=1)


# List of exterior coverings to attach value 1 to
ideal_exterior = ['CemntBd', 'ImStucc', 'BrkFace']

# Function for attaching 1 if either exterior 1 or exterior 2 are in the above list
def in_ie(x):
    if (x['Exterior1st'] or x['Exterior2nd']) in ideal_exterior:
        return 1
    else:
        return 0

# Create a binary column for any CemntBd, ImStucc or BrkFace
train['ideal_ext'] = train.apply(in_ie, axis=1)

# Do the typical inspection for ExterQual
# print(train.groupby('ExterQual').SalePrice.mean())
# print(train['ExterQual'].value_counts())
# Excellent has a really high average sales price and Good has a decently high average as well

# Create a column with 2 for Ex, 1 for Gd and 0 for everything else
train['exter_qual'] = train.ExterQual.apply(lambda x: 2 if x == 'Ex' else (1 if x == 'Gd' else 0))

# Do the typical inspection for ExterCond
# print(train.groupby('ExterCond').SalePrice.mean())
# print(train['ExterCond'].value_counts())
# The differences here aren't too great so we'll leave this one out for now

# Do the typical inspection for Foundation
# print(train.groupby('Foundation').SalePrice.mean())
# print(train['Foundation'].value_counts())
# PConc has the highest average sales price and Slab and BrkTil have quite low averages

# Create a column with 2 for PConc, 0 for Slab and BrkTil and 1 otherwise
train['foundation'] = train.Foundation.apply(lambda x: 2 if x == 'PConc' else (0 if x in ['Slab', 'BrkTil'] else 1))

# Inspect the statistics of TotalBsmtSF
# print(train['TotalBsmtSF'].describe())

# Get quartiles for basement size
bs_uqr = np.percentile(train.TotalBsmtSF, 75)
bs_med = np.percentile(train.TotalBsmtSF, 50)
bs_lqr = np.percentile(train.TotalBsmtSF, 25)

# Group houses according to the quartile of total basement size they fall into
train['bsmt_group'] = train.TotalBsmtSF.apply(lambda x: 0 if x < bs_lqr else (1 if x < bs_med else(2 if x < bs_uqr else 3)))

import pandas as pd
import numpy as np

"""
Script to load and process data
"""

# //////////////////////
# Load data
# //////////////////////
print('Loading data...')
path = '/Users/Bing/Documents/DS/Zillow_Kaggle/'
train = pd.read_csv(path + 'train_2016.csv')
print('Training set loaded')
prop = pd.read_csv(path + 'properties_2016.csv')
print('Properties data set loaded.')
sample = pd.read_csv(path + 'sample_submission.csv')
print('Submission sample loaded.')

print('Binding to float32...')

for c, dtype in zip(prop.columns, prop.dtypes):
    if dtype == np.float64:
        prop[c] = prop[c].astype(np.float32)

# ///////////////////
# Feature Engineering
# ///////////////////

print('Engineering features...')
ulimit = np.percentile(train.logerror.values, 99)
llimit = np.percentile(train.logerror.values, 1)
train['logerror'].ix[train['logerror'] > ulimit] = ulimit
train['logerror'].ix[train['logerror'] < llimit] = llimit

# apply log
prop['structuretaxvaluedollarcnt'] = prop['structuretaxvaluedollarcnt'].apply(np.log10)
prop['calculatedfinishedsquarefeet'] = prop['calculatedfinishedsquarefeet'].apply(np.log10)
prop['lotsizesquarefeet'] = prop['lotsizesquarefeet'].apply(np.log10)

# ==== storytypeid and basementsqft ==============
# if basementsqft is NaN, assume no basement
prop['basementsqft'] = prop['basementsqft'].fillna(0.0)
# if basementsqft not null, storytypeid = 7.0
# since storytypeid has only 2 unique values, 7.0 and NaN, replace with True/False
prop['storytypeid'] = prop['storytypeid'].fillna(False)
prop['storytypeid'] = prop['storytypeid'].replace(7.0, True)

# ==== fireplace ======================
prop['fireplacecnt'] = prop['fireplacecnt'].fillna(0.0)
prop['fireplaceflag'] = prop['fireplaceflag'].fillna(False)

# ==== finishedsquarefeet columns =================
# drop correlated columns, keep calculatedfinishedsquarefeet which has most of the values
prop['calculatedfinishedsquarefeet'] = prop['calculatedfinishedsquarefeet'].fillna(0.0)
prop = prop.drop(['finishedsquarefeet6', 'finishedsquarefeet13', 'finishedsquarefeet12', 'finishedsquarefeet15', 'finishedsquarefeet50'], axis = 1)
# group finishedfloor1squarefeet by bedroom count and impute mean sq foot of the different bedroomcounts into NaN
prop['finishedfloor1squarefeet'] = prop['finishedfloor1squarefeet'].groupby(prop['bedroomcnt']).transform(lambda x: x.fillna(x.mean()))

# ==== garage =========================
prop = prop.drop(['garagetotalsqft'], axis = 1)

# ==== pools ==========================
prop['poolcnt'] = prop['poolcnt'].fillna(0.0)


# ==== make categorical data ===================
for c in prop.dtypes[prop.dtypes == object].index.values:
    prop[c] = (prop[c] == True)

# ==== property county ===========
# ['propertycountylandusecode'] propertylandusetypeid, propertyzoningdesc
# TODO: one hot encode this

# === rawcensustractandblock =======
# rawcensustractandblock censustractandblock
# TODO three classes, convert them into classes, one hot encode

# === yearbuilt ===================


# === time ===================
# TODO: logerror changes over time. 
# for each column in submission, there is a different month. 
# extract months and use as a feature. In the prediction, use the column as the month for prediction
train['month'] = train['transactiondate'].apply(lambda x: int(x.split('-')[1]))
# train['day'] = train['transactiondate'].apply(lambda x: int(x.split('-')[2]))

#//////////////////////
# Training set
#//////////////////////
# Merge prop and train to form the training set

print('Creating training set...')
df_train = train.merge(prop, how = 'left', on='parcelid')

x_train = df_train.drop(['parcelid', 'logerror', 'transactiondate', 'propertyzoningdesc', 'propertycountylandusecode'], axis = 1)
y_train = df_train['logerror']

x_train.to_csv('train_features.csv', index = False)
y_train.to_csv('train_target.csv', index = False, header = 'logerror') # add header so that to_csv doesn't make the first value the header

#//////////////////////
# Test
#//////////////////////
# Merge prop and sample to form test set

print('Building test set ...')

sample['parcelid'] = sample['ParcelId'] 
df_test = sample.merge(prop, on = 'parcelid', how = 'left')

df_test_Oct = df_test
df_test_Oct['month'] = 10
df_test_Oct = df_test_Oct[x_train.columns]
df_test_Oct.to_csv('OctTest.csv', index = False)

df_test_Nov = df_test
df_test_Nov = df_test_Nov['month'] = 11
df_test_Nov = df_test_Nov[x_train.columns]
df_test_Nov.to_csv('NovTest.csv', index = False)

df_test_Dec = df_test
df_test_Dec = df_test_Dec['month'] = 12
df_test_Dec = df_test_Dec[x_train.columns]
df_test_Dec.to_csv('DecTest.csv', index = False)

#for c in df_test.dtypes[df_test.dtypes == object].index.values:
#    df_test[c] = (df_test[c] == True)

# df_test.to_csv('test.csv', index = False)

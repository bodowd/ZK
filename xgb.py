import pandas as pd
import xgboost as xgb
import gc
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
"""
xgboost training and predict
"""

date = '6_10_v1'

path = '/Users/Bing/Documents/DS/Zillow_Kaggle/'

# //////////////////////
# Load data
# //////////////////////
print('Loading training set...')

df_train = pd.read_csv('train_features.csv')
df_target = pd.read_csv('train_target.csv').values.ravel() # get values and ravel to get into the right shape for input to xgb
print(df_train.shape, df_target.shape)

#//////////////////////
# Train
#//////////////////////
# split = 80000
# x_train, y_train, x_valid, y_valid = x_train[:split], y_train[:split], x_train[split:], y_train[split:]

x_train, x_valid, y_train, y_valid = train_test_split(df_train, df_target, test_size = 0.1, random_state = 0)

print('Building DMatrix...')

d_train = xgb.DMatrix(x_train, label = y_train)
d_valid = xgb.DMatrix(x_valid, label = y_valid)

del x_train, x_valid; gc.collect()

print('Training...')

params = {}
params['eta'] = 0.02
params['objective'] = 'reg:linear'
params['eval_metric'] = 'mae'
params['max_depth'] = 4
params['silent'] = 1

watchlist = [(d_train, 'train'), (d_valid, 'valid')]
clf = xgb.train(params, d_train, 10000, watchlist, early_stopping_rounds = 100, verbose_eval = 10)

## plot feature importances and save the figure
importance = sorted(clf.get_fscore().items())
feature_df = pd.DataFrame(importance, columns = ['feature', 'fscore'])
feature_df['fscore'] = feature_df['fscore'] / feature_df['fscore'].sum()

feature_df.plot(kind = 'barh', x = 'feature', y = 'fscore', legend = False)
plt.xlabel('relative importance')
plt.savefig(path + 'Plots/feature_importance' + date + '.png')

del d_train, d_valid; gc.collect()

#//////////////////////
# Test
#//////////////////////

print('Loading test sets')

# df_test = pd.read_csv('test.csv')
df_test_Oct = pd.read_csv('OctTest.csv')
df_test_Nov = pd.read_csv('NovTest.csv')
df_test_Dec = pd.read_csv('DecTest.csv')

d_test_Oct = xgb.DMatrix(df_test_Oct)
d_test_Nov = xgb.DMatrix(df_test_Nov)
d_test_Dec = xgb.DMatrix(df_test_Dec)


del df_test_Oct, df_test_Nov, df_test_Dec; gc.collect()

#//////////////////////
# Prediction
#//////////////////////

print('Predicting on test set...')

p_test_Oct = clf.predict(d_test_Oct)
p_test_Nov = clf.predict(d_test_Nov)
p_test_Dec = clf.predict(d_test_Dec)
# p_test = 0.95*p_test + 0.09*0.011

del d_test_Oct, d_test_Nov, d_test_Dec; gc.collect() 

sub = pd.read_csv(path + 'sample_submission.csv')
# for c in sub.columns[sub.columns != 'ParcelId']:
    # sub[c] = p_test
sub[['201610', '201710']] = p_test_Oct
sub[['201611', '201711']] = p_test_Nov
sub[['201612', '201712']] = p_test_Dec

print('Writing csv...')
sub.to_csv(path + 'Submissions/xgb_' + date + '.csv', index = False, float_format = '%.4g') 

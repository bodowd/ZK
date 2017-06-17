import sys

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import Imputer
from sklearn.ensemble import RandomForestRegressor

class CV:
    """
    Cross validation class
    """
    def __init__(self, df_train, df_target, n_splits, model, **params):
        """

        :param params: model parameters
        :param df_train: training set
        :param df_target: vector of targets
        :param n_splits: number of KFolds
        :param model: model to get cross validation for

        """
        self.params = params
        self.df_train = df_train
        self.df_target = df_target
        self.n_splits = n_splits
        self.model = model

    def cross_validate(self, scoring = 'neg_mean_absolute_error'):
        """
        Kfold cross validation
        """
        kf = KFold(n_splits = self.n_splits, random_state = 0)
        self.model.fit(self.df_train, self.df_target)
        results = cross_val_score(self.model, self.df_train, self.df_target, cv = kf, scoring = scoring)
        return results.mean(), results.std()

    def plot_importances(self, filename = None):
        """
        Plot feature importances
        """
        features = self.model.feature_importances_
        cols = self.df_train.columns
        feature_dataframe = pd.DataFrame({'features': cols,
        'Random Forest feature importances': features})
        # # Feature importances bar plot
        ax = feature_dataframe.plot(kind = 'barh', legend = None)
        ax.set_yticklabels(feature_dataframe['features'].values)
        plt.tight_layout()
        plt.savefig('{}Plots/{}.png'.format(path, filename))

    def report(self, model, date, mae_mean, mae_std):
        """
        print report to stdout as well as store in a file
        """
        original = sys.stdout
        sys.stdout = open('../Logs/{}.txt'.format(date), 'a')
        print('{} MAE : {} ({})\n{}'.format(model, mae_mean, mae_std, self.params))
        sys.stdout = original
        print('{} MAE : {} ({})'.format(model, mae_mean, mae_std))

# -------------------------------------------------------
# tests
if __name__ == '__main__':
    print('Loading data...')
    path = '/Users/Bing/Documents/DS/Zillow_Kaggle/'
    df_train = pd.read_csv('train_features.csv')
    df_target = pd.read_csv('train_target.csv').values.ravel()
    imp = Imputer()
    df_train_imp = pd.DataFrame(imp.fit_transform(df_train), columns = df_train.columns)

    rf_params = {
        'n_jobs': -1,
        'n_estimators': 10,
        'warm_start': True,
        'max_depth' : 6,
        'min_samples_leaf': 2,
        'max_features' : 'sqrt',
        'verbose': 0
        }
    rf = RandomForestRegressor(random_state = 0, **rf_params) 

    rf_CV = CV(df_train = df_train_imp, df_target = df_target, n_splits = 10, model = rf, params = rf_params) 
    mae_mean, mae_std = rf_CV.cross_validate()
    print('RF MAE: {} ({})'.format(mae_mean, mae_std))
    rf_CV.plot_importances('rf_6_17_test')

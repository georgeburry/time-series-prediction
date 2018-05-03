'''2018-05-03
George Burry

XGBoost is a powerful and versatile tool, which has enabled many Kaggle competition participants to achieve winning scores. How well does XGBoost perform when used to predict future values of a time-series? This was put to the test by aggregating datasets containing time-series from three Kaggle competitions. Random samples were extracted from each time-series, with lags of t-10 and a target value (forecast horizon) of t+5. Up until now, the results have been interesting and warrant further work.'''

# Standard imports
import datetime
import re
import numpy as np
import pandas as pd
import matplotlib.pylab as plt
from math import sqrt

# Modifying and splitting the data
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# XGboost model
import xgboost as xgb

# Model selection tools
from sklearn.model_selection import cross_val_score, KFold
from sklearn.model_selection import GridSearchCV

# Saving models
from sklearn.externals import joblib

# Error metrics
from sklearn.metrics import mean_squared_error, r2_score


'''Takes specified number of time-series from dataset and then randomly creates specified number of training
examples (each with a target value) from each time-series. For example, if 1000 time-series are specified,
where 10 examples are taken from each time-series, the X-matrix will have shape: (10000, 10).

sample_num = number of sequences taken from each time-series
lag = number of preceding points to use for the prediction
lead = the forecast horizon (target value)'''

class create_inputs:
    def __init__(self, sample_num, lag, lead):
        self.sample_num = sample_num
        self.lag = lag
        self.lead = lead

    def fit(self, X, y=None):
        self.X = X

    # X must be an numpy matrix or array
    def transform(self, X, y=None):
        X_matrix = []
        y = []

        for row in range(len(X)):

            ts = self.X[row]

            for i in range(self.sample_num):
                np.random.seed(i)
                start_point = np.random.randint(1, len(ts) - self.lag - self.lead)

                sample = []
                for n in range(self.lag + 1):
                    sample.append(ts[start_point + n])

                X_matrix.append(sample)
                y.append(ts[start_point + self.lag + self.lead])

        self.X = np.array(X_matrix)
        self.y = np.array(y)
        return self.X, self.y

    def fit_transform(self, X, y=None):
        self.fit(X)
        return self.transform(X, y=None)


# creates training examples for supervised learning from a time-series (i.e. X: t-n, t-1, t; y: t+1)
# lag is the number of preceding points used for the prediction (X), and lead is the forecast horizon (y)
def timeseries_to_supervised(data, lag=1, lead=1):
    df = pd.DataFrame(data)
    columns = [df.shift(-i) for i in range(0, lag+1)]
    df_X = pd.concat(columns, axis=1)
    df_all = pd.concat([df_X, df.shift(-(lead + lag))], axis=1)
    df_all = df_all.iloc[:-(lead + lag), :]
    df_all.fillna(0, inplace=True)
    return df_all

# saving a model
def save_model(model, filename):
    return joblib.dump(model, filename)

# loading a model
def load_model(filename):
    return joblib.load(filename)

def calculate_SMAPE(y, y_hat):
    print('SMAPE score: ', 100 / len(y) * np.sum(np.abs(y_hat - y) / ((np.abs(y) + np.abs(y_hat))/2)))


views_df = pd.read_csv('./datasets/wikipedia-views_train.csv', index_col=0)
print(views_df.head())

sales_df = pd.read_csv('./datasets/supermarket-sales_train.csv', index_col=0)
print(sales_df.head())

visitors_df = pd.read_csv('./datasets/restaurant-visits_train.csv', index_col=0)
print(visitors_df.head())


# create a list of datasets to iterate through
datasets = [views_df, sales_df, visitors_df]

Xs = pd.DataFrame()
ys = pd.DataFrame()

# set input creation parameters
sample_num, lag, lead = 10, 10, 5

# iterate through datasets, create random supervised learning examples and combine into one training set
for dataset in datasets:
    input_setup = create_inputs(sample_num, lag, lead)
    data = dataset.as_matrix()
    X, y = input_setup.fit_transform(data)
    Xs  = Xs.append(pd.DataFrame(X))
    ys = ys.append(pd.DataFrame(y))

X_train = Xs.as_matrix()
y_train = ys.as_matrix()

'''
# setup regressor
xgb_model = xgb.XGBRegressor()

# perform a grid search
tweaked_model = GridSearchCV(
    xgb_model,
    {
        'max_depth': [1, 2, 5, 10, 20],
        'n_estimators': [20, 30, 50, 70, 100],
        'learning_rate': [0.1, 0.2, 0.3, 0.4, 0.5]
    },
    cv=10,
    verbose=1,
    n_jobs=-1,
    scoring='neg_median_absolute_error'
)

tweaked_model.fit(X_train, y_train)

# summarize results
print("Best: %f using %s" % (tweaked_model.best_score_, tweaked_model.best_params_))
'''

# used to convert time-series index values to datetime format
def parser(x):
    return pd.datetime.strptime(x, '%Y-%m-%d')

# wikipedia examples saved in separate files
wikipedia_examples = {
    'DE':'de-wikipedia_series(Rubinrot).csv',
    'EN':'en-wikipedia_series(Ellen_DeGeneres).csv',
    'FR':'fr-wikipedia_series(Crue_de_la_Seine).csv',
    'JP':'jp-wikipedia_series(メジャー_(アニメ)).csv',
    'RU':'ru-wikipedia_series(Ломоносов).csv',
    'CN':'zh-wikipedia_series(蕭正楠).csv'
}

path = './datasets/test_examples/wikipedia-pages/'

# append each time-series to list
views_test_samples = []
for key in wikipedia_examples:
    series = pd.read_csv(path + wikipedia_examples[key], index_col=0, header=None, parse_dates=[0], squeeze=True, date_parser=parser)
    views_test_samples.append(series)

path = './datasets/test_examples/supermarket-items/'
sales_test_df = pd.read_csv(path + 'supermarket-sales_test.csv', index_col=0)

# create random indices for collecting time-series at random
np.random.seed(42)
rand_idx = np.random.randint(len(sales_test_df), size=6)

# append each time-series to list
sales_test_samples = []
for idx in rand_idx:
    series = pd.Series(sales_test_df.iloc[idx], index = pd.to_datetime(sales_test_df.iloc[idx].index))
    sales_test_samples.append(series)

path = './datasets/test_examples/restaurant-stores/'
visitors_test_df = pd.read_csv(path + 'restaurant-visits-test.csv', index_col=0)

# append each time-series to list
visitors_test_samples = []
for idx in range(len(visitors_test_df)):
    series = pd.Series(visitors_test_df.iloc[idx], index = pd.to_datetime(visitors_test_df.iloc[idx].index))
    visitors_test_samples.append(series)

test_collection = {
    'Wikipedia views': views_test_samples,
    'Supermarket sales': sales_test_samples,
    'Restaurant visitors': visitors_test_samples
}

model = load_model('./models/XGB-model_lag=10_lead=5.pkl')

# set prediction parameters
lag, lead = 10, 5  # lead = number of previous points to use, and lead is the forecast horizon

for case in test_collection:
    test_samples = test_collection[case]
    all_targets, all_predictions = [], []
    for series in test_samples:
        series = series[:105] # roughly first 90 days

        # transform data to be supervised learning
        supervised = timeseries_to_supervised(series.values, lag, lead).as_matrix()
        predictors, targets = supervised[:, :-1], supervised[:, -1]
        predictions = model.predict(predictors)
        all_targets.append(targets)
        all_predictions.append(predictions)

    # plot
    f, ax = plt.subplots(2, 3, figsize=(20,10))
    f.suptitle(case, fontsize=20)

    count = 0
    for i in range(2):
        for j in range(3):
            ax[i, j].plot(all_targets[count], label='ACTUAL')
            ax[i, j].plot(all_predictions[count], label='PREDICTED')
            ax[i, j].set_title(' RMSE: {:10.2f}'.format(np.sqrt(mean_squared_error(all_targets[count], all_predictions[count]))))

            count += 1

    ax[0, 0].legend()
    ax[0, 0].set_ylabel('Views')
    ax[1, 0].set_ylabel('Views')
    ax[1, 0].set_xlabel('Days')
    ax[1, 1].set_xlabel('Days')
    ax[1, 2].set_xlabel('Days')

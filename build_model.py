import pandas as pd
# import glob
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import sklearn
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, precision_recall_curve, auc, roc_curve
from sklearn.metrics import f1_score, accuracy_score, log_loss, precision_score, recall_score
from sklearn.metrics import make_scorer
from sklearn.model_selection import StratifiedShuffleSplit
# import matplotlib as mpl
# mpl.use('Agg')
# import matplotlib.pyplot as plt
# %matplotlib inline
import math
import time
from scipy import stats
import sklearn.preprocessing as sp
import pickle
import logging
import logging.config

target='label'

# Set logger
logging.config.fileConfig('./config/logging.conf', disable_existing_loggers=False)
logger = logging.getLogger('root')


def create_dataset():
    # Load feature file
    # feature_filelist = glob.glob('./dataset/agg_features/part-*.csv')#modify!!!
    # agg_feature_data = [pd.read_table(feature_filelist[i], parse_dates=[0]) for i in range(len(feature_filelist))]
    # df_agg_features = pd.concat(agg_feature_data, ignore_index=True)
    df_agg_features = pd.read_csv('./dataset/agg_features/part-00000-bfdd70d2-ea5c-4c55-9f92-e7b21de30624-c000.csv')

    # Load label file
    # label_filelist = glob.glob('./labels/part-*.csv')
    # agg_label_data = [pd.read_table(label_filelist[i], parse_dates=[0]) for i in range(len(label_filelist))]
    # df_label = pd.concat(agg_label_data, ignore_index=True)
    df_label = pd.read_csv('./labels/part-00000-e9445087-aa0a-433b-a7f6-7f4c19d78ad6-c000.csv')


    # Exclude bookingIDs whose label value is not unique
    df_label = df_label[~(df_label.bookingID.duplicated())]

    # Join feature data and label
    df = df_agg_features.merge(df_label, how='inner', on='bookingID')
    df.to_csv('./modelling_dataset.csv', index=False)
    return df


def preprocess_dataset(_df):
    # Remove ID information
    df_merge = _df.drop('bookingID', axis=1)
    X = df_merge.iloc[:,0:len(df_merge.columns)-1]
    y = df_merge[target]
    
    # Use 70% of data for training and 30% of data for hold-out
    sss = StratifiedShuffleSplit(n_splits=2, test_size=0.3, random_state=1021)
    for train_index, test_index in sss.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y[train_index], y[test_index]

    # transform DMatrix type
    dtrain = xgb.DMatrix(X_train,label=y_train, feature_names=X_train.columns)
#     dtest = xgb.DMatrix(X_test.as_matrix(),label=y_test.tolist(), feature_names=X_train.columns)
    return dtrain, X_train, X_test, y_train, y_test


def select_model(_X_train, _y_train, _scoring_metric='roc_auc', _num_fold=5, ):
    print('Metric: ', _scoring_metric)
    xgb_model = xgb.XGBClassifier()
    xgb_model.get_params().keys()
    params_cv={'objective': ['binary:logistic'],
            'learning_rate': [0.001], #eta
    #                'reg_alpha': [0.1], # alpha(L1)
    #                'reg_lambda': [0.1], #lambda(L2)
                'gamma': [0], # min_split_loss
                'max_depth': [3], # max depth of a tree [default=6]
#                 'max_depth': [3, 7], # max depth of a tree [default=6]
                'min_child_weight': [1], # [default=1]
                'max_delta_step' : [1],
                'subsample': [0.3], # [default=1]
#                 'subsample': [0.3, 0.5], # [default=1]
                'colsample_bytree': [0.3], # [default=1]
#                 'colsample_bytree': [0.3, 0.5], # [default=1]
                'colsample_bylevel': [0.3], # [default=1]
#                 'colsample_bylevel': [0.3, 0.5], # [default=1]
                'nthread': [8],
                'scale_pos_weight': [1],
                'n_estimators': [500], # n_estimators = num_boost_round
                'seed': [1021]
        }
    stratified_gscv = StratifiedKFold(n_splits=_num_fold, random_state=1021, shuffle=False)
    gscv = GridSearchCV(xgb_model, params_cv, scoring=_scoring_metric, cv=stratified_gscv.split(_X_train, _y_train))
    gscv.fit(_X_train, _y_train)

    print('CV', _scoring_metric,':', gscv.best_score_)
    print(gscv.best_estimator_.get_params())
    return gscv.best_estimator_.get_params()


def build_model(_selected_model, _dtrain):
    # train model by using tuned hyper-parameter
    model_fulldata=xgb.train(_selected_model, _dtrain, num_boost_round=500) # n_estimators = num_boost_round
    # Save XGBoost model as a pickle file
    pickle.dump(model_fulldata, open('./model/xgb_model_fulldata.pkl', 'wb'))


if __name__ == '__main__':
    logger.info('build_model.py start!')
    logger.info('create_dataset() start')
    df = create_dataset()
    logger.info('preprocess_dataset() start')
    dtrain, X_train, X_test, y_train, y_test = preprocess_dataset(_df=df)
    logger.info('select_model() start')
    selected_model = select_model(_X_train=X_train, _y_train=y_train)
    logger.info('build_model() start')
    build_model(_selected_model=selected_model, _dtrain=dtrain)
    logger.info('build_model.py completed!')
    logger.info('')

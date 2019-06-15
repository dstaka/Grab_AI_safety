import pandas as pd
import glob
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
import math
import time
from scipy import stats
import sklearn.preprocessing as sp
import pickle
import logging
import logging.config

target='label'
feature_file_path = './dataset/train/part-*.csv'
label_file_path = './labels/train/part-*.csv'
scoring_metric = 'roc_auc'
num_fold = 5 # 5-fold cross-validation

# Set logger
logging.config.fileConfig('./config/logging.conf', disable_existing_loggers=False)
logger = logging.getLogger('root')



def create_dataset(_feature_file_path, _label_file_path):
    # Load feature file
    try:
        feature_filelist = glob.glob(_feature_file_path)
        tmp_list = []
        for _filename in feature_filelist:
            tmp_df = pd.read_csv(_filename, index_col=None, header=0)
            tmp_list.append(tmp_df)
        df_agg_features = pd.concat(tmp_list)
        length_df_agg_features = len(df_agg_features)
        logger.info('Total No. of bookingIDs in feature files: ' + str(length_df_agg_features))
    except Exception as e:
        logger.error(
            'build_model.py: Failed to read aggregated feature files!')
        logger.error('Exception on create_dataset(): '+str(e))
        raise

    # Load label file
    try:
        label_filelist = glob.glob(_label_file_path)
        tmp_list = []
        for _filename in label_filelist:
            tmp_df = pd.read_csv(_filename, index_col=None, header=0)
            tmp_list.append(tmp_df)
        df_label = pd.concat(tmp_list)
        # Exclude bookingIDs whose label value is not unique
        df_label = df_label[~(df_label.bookingID.duplicated())]
        length_df_label = len(df_label)
        logger.info('Total No. of bookingIDs in label files: ' + str(length_df_label))
    except Exception as e:
        logger.error(
            'build_model.py: Failed to read label files!')
        logger.error('Exception on create_dataset(): '+str(e))
        raise

    if(length_df_agg_features != length_df_label):
        logger.warn('No. of bookingIDs is not matched between feature files and label files!')
    
    # Merge feature data and label
    df = df_agg_features.merge(df_label, how='inner', on='bookingID')
    length_df = len(df)
    logger.info('Total No. of bookingIDs in modelling dataset: ' + str(length_df))
    df.to_csv('./dataset/train/modelling_dataset.csv', index=False)
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
        
    logger.info('No. of bookingIDs in training dataset: ' + str(len(X_train)))
    logger.info('No. of bookingIDs in testing dataset: ' + str(len(y_train)))
    
    # Transform DMatrix type
    dmatrix_fulldata = xgb.DMatrix(X, label=y, feature_names=X_train.columns)
    return dmatrix_fulldata, X_train, X_test, y_train, y_test


def select_model(_X_train, _y_train, _scoring_metric, _num_fold):
    logger.info('Run ' + str(_num_fold) + '-fold cross-validation')
    logger.info('Metric is ' + _scoring_metric)
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

    logger.info('CV score: ' + _scoring_metric + ' = ' + str(gscv.best_score_))
    logger.info('Hyperparameters of selected model:')
    logger.info(gscv.best_estimator_.get_params())
    return gscv.best_estimator_.get_params()


def build_fulldata_model(_selected_model, _dtrain):
    # Train model by using tuned hyper-parameter
    model_fulldata=xgb.train(_selected_model, _dtrain, num_boost_round=500) # n_estimators = num_boost_round
    # Save XGBoost model as a pickle file
    pickle.dump(model_fulldata, open('./model/xgb_model_fulldata.pkl', 'wb'))


if __name__ == '__main__':
    logger.info('build_model.py start!')
    logger.info('create_dataset() start')
    df = create_dataset(_feature_file_path=feature_file_path, _label_file_path=label_file_path)
    logger.info('preprocess_dataset() start')
    dmatrix_fulldata, X_train, X_test, y_train, y_test = preprocess_dataset(_df=df)
    logger.info('select_model() start')
    selected_model = select_model(_X_train=X_train, _y_train=y_train, _scoring_metric=scoring_metric, _num_fold=num_fold)
    logger.info('build_fulldata_model() start')
    build_fulldata_model(_selected_model=selected_model, _dtrain=dmatrix_fulldata)
    logger.info('build_model.py completed!')

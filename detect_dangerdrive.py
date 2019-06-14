import pandas as pd
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


def create_dataset():# can be in common with build_model.py
    # Load feature file
    # feature_filelist = glob.glob('./dataset/agg_features/part-*.csv')#modify!!!
    # agg_feature_data = [pd.read_table(feature_filelist[i], parse_dates=[0]) for i in range(len(feature_filelist))]
    # df_agg_features = pd.concat(agg_feature_data, ignore_index=True)
    df_agg_features = pd.read_csv('./dataset/test/part-00000-bfdd70d2-ea5c-4c55-9f92-e7b21de30624-c000.csv')

    # Load label file
    # label_filelist = glob.glob('./labels/part-*.csv')
    # agg_label_data = [pd.read_table(label_filelist[i], parse_dates=[0]) for i in range(len(label_filelist))]
    # df_label = pd.concat(agg_label_data, ignore_index=True)
    df_label = pd.read_csv('./labels/test/part-00000-e9445087-aa0a-433b-a7f6-7f4c19d78ad6-c000.csv')

    # Exclude bookingIDs whose label value is not unique
    df_label = df_label[~(df_label.bookingID.duplicated())]

    # Join feature data and label
    df_merge = df_agg_features.merge(df_label, how='inner', on='bookingID')
    # Remove ID information
    df_merge = df_merge.drop('bookingID', axis=1)
    df_merge.to_csv('./dataset/test/modelling_dataset.csv', index=False)
    
    X_test = df_merge.iloc[:,0:len(df_merge.columns)-1]
    y_test = df_merge[target]
    
    # Transform DMatrix type
    dtest = xgb.DMatrix(X_test.as_matrix(),label=y_test.tolist(), feature_names=X_test.columns)
    return dtest, X_test, y_test


def detect_dangerdrive(_dtest, _y_test):
    # Load model
    model_fulldata = pickle.load(open('./model/xgb_model_fulldata.pkl', 'rb'))
    # Create 
    predicted_prob = model_fulldata.predict(_dtest)
    fpr_test, tpr_test, thresholds_test= roc_curve(_y_test, predicted_prob)

    # Calculate AUC
    roc_auc_tmp = roc_auc_score(_y_test, predicted_prob)
    print('Test AUC: ', roc_auc_tmp)
    # Calculate F1_score
    y_test_val = _y_test.get_values()
    y_bin = [1. if _y_test > 0.5 else 0. for y_cont in predicted_prob] # binaryzing output
    f1_tmp = f1_score(y_test_val, y_bin)
    print('Test F1-score: ', f1_tmp)

    # Calculate neg_log_loss
    log_loss_tmp = log_loss(_y_test, predicted_prob)
    print('Test log_loss: ', log_loss_tmp)

    # Calculate accuracy
    accuracy_tmp = accuracy_score(y_test_val, y_bin)
    print('Test accuracy: ', accuracy_tmp)


if __name__ == '__main__':
    logger.info('detect_dangerdrive.py start!')
    logger.info('create_dataset() start')
    dtest, X_test, y_test = create_dataset()
    logger.info('detect_dangerdrive() start')
    detect_dangerdrive(_dtest=dtest, _y_test=dtest)
    logger.info('build_model.py completed!')

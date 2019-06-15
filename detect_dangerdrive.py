from python_module import create_dataset
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

target = 'label'
feature_file_path = './dataset/test/part-*.csv'
label_file_path = './labels/test/part-*.csv'
output_file_path = './dataset/test/modelling_dataset.csv'
scoring_metric = 'roc_auc'

# Set logger
logging.config.fileConfig('./config/logging.conf', disable_existing_loggers=False)
logger = logging.getLogger('root')

def score_dangerdrive(_df):
    # Transform DMatrix type
    X_test = _df.iloc[:,0:len(_df.columns)-1]
    y_test = _df[target]
    dtest = xgb.DMatrix(X_test.as_matrix(),label=y_test.tolist(), feature_names=X_test.columns)

    # Load model
    model_fulldata = pickle.load(open('./model/xgb_model_fulldata.pkl', 'rb'))
    # Create 
    predicted_prob = model_fulldata.predict(dtest)
    fpr_test, tpr_test, thresholds_test= roc_curve(y_test, predicted_prob)

    # Calculate AUC
    roc_auc_score = roc_auc_score(y_test, predicted_prob)
    logger.info('Test AUC: ' + str(roc_auc_score))
    # Calculate F1_score
    y_test_val = y_test.get_values()
    y_bin = [1. if y_test > 0.5 else 0. for y_cont in predicted_prob] # binaryzing output
    f1_score = f1_score(y_test_val, y_bin)
    logger.info('Test F1-score: ' + str(f1_score))
    # Calculate neg_log_loss
    log_loss_score = log_loss(y_test, predicted_prob)
    logger.info('Test log_loss: ' + str(log_loss_score))
    # Calculate accuracy
    accuracy_score = accuracy_score(y_test_val, y_bin)
    logger.info('Test accuracy: ' + accuracy_score)


if __name__ == '__main__':
    logger.info('detect_dangerdrive.py start!')
    logger.info('create_dataset.merge_feature_and_label() start')
    df = create_dataset.merge_feature_and_label(_feature_file_path=feature_file_path, _label_file_path=label_file_path, _output_file_path=output_file_path)
    logger.info('score_dangerdrive() start')
    try:
        score_dangerdrive(_df=df)
    except Exception as e:
        logger.error(
            'detect_dangerdrive.py: Failed to score danger drive!')
        logger.error('Exception on score_dangerdrive(): '+str(e))
        raise
    logger.info('build_model.py completed!')
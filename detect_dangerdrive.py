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


def create_dataset():# can be in common with build_model.py
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

predicted_prob = model_fulldata.predict(dtest)
fpr_test, tpr_test, thresholds_test= roc_curve(y_test, predicted_prob)


#calculate AUC
roc_auc_tmp = roc_auc_score(y_test, predicted_prob)
print('Test AUC: ', roc_auc_tmp)

#calculate F1_score
y_test_val = y_test.get_values()
y_bin = [1. if y_cont > 0.5 else 0. for y_cont in predicted_prob] # binaryzing output
f1_tmp = f1_score(y_test_val, y_bin)
print('Test F1-score: ', f1_tmp)

#calculate neg_log_loss
log_loss_tmp = log_loss(y_test, predicted_prob)
print('Test log_loss: ', log_loss_tmp)

#calculate accuracy
accuracy_tmp = accuracy_score(y_test_val, y_bin)
print('Test accuracy: ', accuracy_tmp)


# Load model
model_fulldata = pickle.load(open('./model/xgb_model_fulldata.pkl', 'rb'))

# Set logger
logging.config.fileConfig('./config/logging.conf', disable_existing_loggers=False)
logger = logging.getLogger('root')

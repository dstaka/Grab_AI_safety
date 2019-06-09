import pandas as pd

import time
import logging
import logging.config


# Set logger
logging.config.fileConfig('./config/logging.conf', disable_existing_loggers=False)
logger = logging.getLogger('root')


df_agg_features=pd.read_csv('./data/agg_features_t5_190608.csv/part-00000-e288dcb3-d734-408a-9aca-1f3ebf01c79a-c000.csv')
df_agg_features.shape


# In[3]:


df_agg_features.head()


# In[4]:


list(df_agg_features)


# # Label data

# In[5]:


df_label=pd.read_csv('./labels/part-00000-e9445087-aa0a-433b-a7f6-7f4c19d78ad6-c000.csv')
df_label.shape


# In[6]:


# Check duplicated 
df_label[df_label.bookingID.isin(df_label[df_label.bookingID.duplicated()].bookingID)].sort_values(by=['bookingID', 'label'])


# In[7]:


# Exclude bookingIDs whose label is not determined
# remove_bookingID_list = df_label[df_label.bookingID.isin(df_label[df_label.bookingID.duplicated()].bookingID)].bookingID
# remove_bookingID_list = list(df_label[df_label.bookingID.duplicated()].bookingID)
df_label = df_label[~(df_label.bookingID.duplicated())]
df_label.shape


# In[8]:


# Check if duplicated bookingIDs are removed
df_label[df_label.bookingID.isin(df_label[df_label.bookingID.duplicated()].bookingID)].sort_values(by=['bookingID', 'label'])


# In[9]:


df=df_agg_features.merge(df_label, how='inner', on='bookingID')
df.shape


# In[10]:


df.bookingID.nunique()


# In[11]:


df.to_csv('./df_190603.csv', index=False)


# In[15]:


import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import sklearn

import xgboost as xgb

# from sklearn import cross_validation
# from sklearn.cross_validation import train_test_split
#from sklearn.grid_search import GridSearchCV #old version which will be removed in future
# from sklearn.model_selection import cross_validation
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, precision_recall_curve, auc, roc_curve
from sklearn.metrics import f1_score, accuracy_score, log_loss, precision_score, recall_score
from sklearn.metrics import make_scorer

from sklearn.model_selection import StratifiedShuffleSplit

#Random search
#from sklearn.grid_search import RandomizedSearchCV #old version which will be removed in future
# from sklearn.model_selection import RandomizedSearchCV
# from scipy.stats import randint as sp_randint
# import scipy as sp

#order is matter
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
# import seaborn as sns; sns.set(style='whitegrid')
# import plotly.graph_objs as go

import math
import time


from scipy import stats
import sklearn.preprocessing as sp
target='label'



# In[16]:


print(sklearn.__version__)


# In[17]:


df[target].value_counts()


# In[18]:


df_merge=df.drop('bookingID', axis=1)


# In[19]:


X = df_merge.iloc[:,0:len(df_merge.columns)-1]
y = df_merge[target]


# In[20]:


X.head()


# In[21]:


X.shape


# In[22]:


y.sum()/X.shape[0]


# In[23]:


X.columns


# In[24]:


sss = StratifiedShuffleSplit(n_splits=2, test_size=0.3, random_state=1021)
print(sss.get_n_splits(X, y))
print(sss)
for train_index, test_index in sss.split(X, y):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y[train_index], y[test_index]


# In[25]:


print(X_train.shape)
print(X_test.shape)
print()
print(y_train.shape)
print(y_test.shape)


# In[26]:


print(y_train.value_counts())
print()
print(y_test.value_counts())


# In[27]:


#check positive/negative ratio in Target
print(1.0*y_train.value_counts()[1]/y_train.value_counts()[0])
print(1.0*y_test.value_counts()[1]/y_test.value_counts()[0])


# In[28]:


# transform DMatrix type
dtrain = xgb.DMatrix(X_train,label=y_train, feature_names=X_train.columns)
dtest = xgb.DMatrix(X_test.as_matrix(),label=y_test.tolist(), feature_names=X_train.columns)


# In[ ]:


_scoring_metric_arg=['roc_auc']
_num_fold=5
_gscv_result_DF = pd.DataFrame() # initialize DataFrame for result saving
start_time_total = time.time()
print('Metric: ', _scoring_metric)

start_time_cv = time.time()

# transform DMatrix type
dtrain = xgb.DMatrix(X_train,label=y_train,feature_names=X_train.columns)
dtest = xgb.DMatrix(X_test.as_matrix(),label=y_test.tolist(), feature_names=X_train.columns)
xgb_model = xgb.XGBClassifier()
xgb_model.get_params().keys()
params_cv={'objective': ['binary:logistic'],
           'learning_rate': [0.001], #eta
           'reg_alpha': [0.1], # alpha(L1)
           'reg_lambda': [0.1], #lambda(L2)
            'gamma': [0], # min_split_loss
            'max_depth': [3], # max depth of a tree [default=6]
            'min_child_weight': [1], # [default=1]
            'max_delta_step' : [1],
            'subsample': [0.5, 1], # [default=1]
            'colsample_bytree': [0.5, 1], # [default=1]
            'colsample_bylevel': [0.5, 1], # [default=1]
            'nthread': [4],
            'scale_pos_weight': [1],
            'n_estimators': [100], # n_estimators = num_boost_round
            'seed': [1021]
    }

# Grid Search
#     stratified_gscv = cross_validation.StratifiedKFold(y_train, _num_fold, False, 1021) #y, n_folds, shuffle, and random_state
#     gscv = GridSearchCV(xgb_model, params_cv, scoring=_scoring_metric, cv=stratified_gscv, n_jobs=1, verbose=1)
#     gscv.fit(X_train, y_train)
stratified_gscv = StratifiedKFold(n_splits=_num_fold, random_state=1021, shuffle=False)
gscv = GridSearchCV(xgb_model, params_cv, scoring=_scoring_metric, cv=stratified_gscv.split(X_train, y_train))
gscv.fit(X_train, y_train)
##     kfolds = StratifiedKFold(5)
##     clf = GridSearchCV(estimator, parameters, scoring=qwk, cv=kfolds.split(X_train, y_train))
##     clf.fit(xtrain, ytrain)

print('CV', _scoring_metric,':', gscv.best_score_)
print(gscv.best_estimator_.get_params())


# In[ ]:


#train model by using tuned hyper-parameter
model_gscv=xgb.train(gscv.best_estimator_.get_params(), dtrain, num_boost_round=100) # n_estimators = num_boost_round
predicted_prob = model_gscv.predict(dtest)
fpr_test, tpr_test, thresholds_test= roc_curve(y_test, predicted_prob)
# plot ROC curve
plt.figure(figsize=(8,6))
plt.plot(fpr_test, tpr_test)
plt.title("ROC curve")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.show()

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


#calculate precision
#     precision_tmp = accuracy_score(y_test_val, y_bin)
#     print('Test accuracy: ', accuracy_tmp)
#calculate recall
#     precision_tmp = accuracy_score(y_test_val, y_bin)
#     print('Test accuracy: ', accuracy_tmp)


for _scoring_metric in _scoring_metric_arg:
print('Metric: ', _scoring_metric)
#display Feature Importance
print(model_gscv.get_fscore())
#         print
#         print('Feature Importance:')
#         for key, value in sorted(model_gscv.get_fscore().iteritems(), key=lambda (k,v): (v,k), reverse=True):
#             print("\t%s: %s" % (key, value))

duration = time.time() - start_time_cv
print('elapsed time: %.3f sec' % (duration))
print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
print()

#store results as DataFrame
_gscv_result_DF_part = pd.DataFrame(gscv.cv_results_) #gscv_result = gscv.cv_results
##add columns
#scores for "test" data
_gscv_result_DF_part['Test_roc_auc'] = roc_auc_tmp
_gscv_result_DF_part['Test_f1'] = f1_tmp
_gscv_result_DF_part['Test_log_loss'] = log_loss_tmp
_gscv_result_DF_part['Test_accuracy'] = accuracy_tmp
#metrics
_gscv_result_DF_part['metric'] = _scoring_metric
_gscv_result_DF = _gscv_result_DF.append(_gscv_result_DF_part)

# #save DataFrame
# _gscv_result_DF.to_csv(path_gscv_result, index=False)

print()
print()
duration = time.time() - start_time_total
print('Total elapsed time: %.3f sec' % (duration))
print("################################################################################################")
print("################################################################################################")


# In[29]:

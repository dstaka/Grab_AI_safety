import time
import logging
import logging.config

# Load model
model_fulldata = pickle.load(open('./model/xgb_model_fulldata.pkl', 'rb'))

# Set logger
logging.config.fileConfig('./config/logging.conf', disable_existing_loggers=False)
logger = logging.getLogger('root')

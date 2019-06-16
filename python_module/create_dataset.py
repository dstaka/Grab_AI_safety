import pandas as pd
import glob
import logging
import logging.config

# Set logger
logging.config.fileConfig('./config/logging.conf', disable_existing_loggers=False)
logger = logging.getLogger('root')


def merge_feature_and_label(_feature_file_path, _label_file_path, _output_file_path):
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
    df.to_csv(_output_file_path, index=False)
    return df

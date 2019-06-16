# Usage for creating training feature file
# nohup spark-submit --master local[*] --conf spark.pyspark.python=python --executor-cores 8 --executor-memory 40G --driver-memory 5G create_features.py train &
# Usage for creating testing feature file
# nohup spark-submit --master local[*] --conf spark.pyspark.python=python --executor-cores 8 --executor-memory 40G --driver-memory 5G create_features.py test &
from pyspark import SparkConf, SparkContext
from pyspark.sql import SQLContext, SparkSession
from pyspark.sql.types import *
from pyspark.sql import functions as func
import pandas as pd
import time
import logging
import logging.config
import sys
args = sys.argv
data_type = args[1] # Either "train" or "test"
# data_type = 'train'
_telematics_data_dir='./raw_data/' + data_type
_csv_filenames='/part-*.csv'
_feature_data_dir='./dataset/' + data_type

# Set logger
logging.config.fileConfig('./config/logging.conf', disable_existing_loggers=True)
logger = logging.getLogger('root')


# Set up Spark
spark = SparkSession\
    .builder\
    .appName("Create_features")\
    .getOrCreate()


def load_data_into_spark(_filepath):
    # Load data files
    struct = StructType([
            StructField('bookingID', StringType(), False),
            StructField('accuracy', DoubleType(), False),
            StructField('bearing', DoubleType(), False),
            StructField('accx', DoubleType(), False),
            StructField('accy', DoubleType(), False),
            StructField('accz', DoubleType(), False),
            StructField('gyrox', DoubleType(), False),
            StructField('gyroy', DoubleType(), False),
            StructField('gyroz', DoubleType(), False),
            StructField('second', DoubleType(), False),
            StructField('speed', DoubleType(), False)])
    try:
        sdf_raw = spark.read.csv(_filepath, schema=struct, header=True)
    except Exception as e:
        logger.error('Failed to load data from ' + str(_filepath))
        logger.error('Exception on load_data_into_spark(): '+str(e))
        raise
    return sdf_raw
    

def create_dataset(_dirpath, _sdf_raw):
    # Register table
    _sdf_raw.registerTempTable('sdf_raw')

    ## Calibrate acc values ##
    # If speed is negative value, replace with 0
    # If speed exceeds 300km (i.e. 83.34 m/s), replace with 83.34
    # Average records by second because there are records whose second values are same
    sdf_aggsec = spark.sql(
        "SELECT bookingID, second,\
            AVG(accuracy) AS accuracy,\
            AVG(bearing) AS bearing,\
            AVG(accx) AS accx, AVG(accy) AS accy, AVG(accz) AS accz,\
            AVG(gyrox) AS gyrox, AVG(gyroy) AS gyroy, AVG(gyroz) AS gyroz,\
            AVG(LEAST(GREATEST(speed, 0), 83.34)) AS speed\
        FROM sdf_raw\
        GROUP BY bookingID, second")
    sdf_aggsec.registerTempTable('sdf_aggsec')

    # Identify 3 percentile value of speed
    sdf_aggsec_speed3perc = spark.sql(
        "SELECT bookingID,\
            PERCENTILE(speed, 0.03) AS speed_3perc\
        FROM sdf_aggsec\
        GROUP BY bookingID")
    sdf_aggsec_speed3perc.registerTempTable('sdf_aggsec_speed3perc')

    # Extract records whose speed is within 3% percentile, then calculate avarage of acc in these records
    sdf_aggsec_speed3perc_lt = spark.sql(
        "SELECT aa.bookingID,\
            AVG(aa.accx) AS accx_offset,\
            AVG(aa.accy) AS accy_offset,\
            AVG(aa.accz) AS accz_offset\
        FROM sdf_aggsec aa LEFT JOIN sdf_aggsec_speed3perc bb ON aa.bookingID=bb.bookingID\
        WHERE aa.speed <= bb.speed_3perc\
        GROUP BY aa.bookingID")
    sdf_aggsec_speed3perc_lt.registerTempTable('sdf_aggsec_speed3perc_lt')

    # Caribrate acc by subtracting offset values
    sdf_aggsec_carib = spark.sql(
        "SELECT aa.bookingID, aa.second, aa.accuracy, aa.bearing,\
            aa.accx - accx_offset AS accx,\
            aa.accy - accy_offset AS accy,\
            aa.accz - accz_offset AS accz,\
            aa.gyrox, aa.gyroy, aa.gyroz,\
            aa.speed\
        FROM sdf_aggsec aa LEFT JOIN sdf_aggsec_speed3perc_lt bb ON aa.bookingID=bb.bookingID")
    sdf_aggsec_carib.registerTempTable('sdf_aggsec_carib')

    ## Feature Engineering 1 ##
    # 1st phase pre-processing
    # Calculate norm(strength) of acc3d and gyro3d
    sdf_1_1 = spark.sql(
        "SELECT *,\
            SQRT(accx*accx + accy*accy + accz*accz) AS acc3d,\
            SQRT(gyrox*gyrox + gyroy*gyroy + gyroz*gyroz) AS gyro3d,\
            SQRT(accx*accx + accy*accy + accz*accz)*SQRT(gyrox*gyrox + gyroy*gyroy + gyroz*gyroz)  AS acc3dgyro3d\
        FROM sdf_aggsec_carib")
    sdf_1_1.registerTempTable('sdf_1_1')

    # 2nd phase pre-processing
    # Extract records from t-5 point of time
    sdf_1_2 = spark.sql(
        "SELECT *,\
            LAG(bearing, 5) OVER(PARTITION BY bookingID ORDER BY second) AS bearing_t5,\
            LAG(accx, 5) OVER(PARTITION BY bookingID ORDER BY second) AS accx_t5,\
            LAG(accy, 5) OVER(PARTITION BY bookingID ORDER BY second) AS accy_t5,\
            LAG(accy, 5) OVER(PARTITION BY bookingID ORDER BY second) AS accz_t5,\
            LAG(gyrox, 5) OVER(PARTITION BY bookingID ORDER BY second) AS gyrox_t5,\
            LAG(gyroy, 5) OVER(PARTITION BY bookingID ORDER BY second) AS gyroy_t5,\
            LAG(gyroz, 5) OVER(PARTITION BY bookingID ORDER BY second) AS gyroz_t5,\
            LAG(second, 5) OVER(PARTITION BY bookingID ORDER BY second) AS second_t5,\
            LAG(speed, 5) OVER(PARTITION BY bookingID ORDER BY second) AS speed_t5,\
            LAG(acc3d, 5) OVER(PARTITION BY bookingID ORDER BY second) AS acc3d_t5,\
            LAG(gyro3d, 5) OVER(PARTITION BY bookingID ORDER BY second) AS gyro3d_t5,\
            LAG(acc3dgyro3d, 5) OVER(PARTITION BY bookingID ORDER BY second) AS acc3dgyro3d_t5\
        FROM sdf_1_1")
    sdf_1_2.registerTempTable('sdf_1_2')

    # 3rd phase
    # Compute record sub
    sdf_1_3 = spark.sql(
        "SELECT *,\
            COALESCE(ABS(bearing - bearing_t5), 0) AS bearing_sub_t5,\
            COALESCE(ABS(accx - accx_t5), 0) AS accx_sub_t5,\
            COALESCE(ABS(accy - accy_t5), 0) AS accy_sub_t5,\
            COALESCE(ABS(accz - accz_t5), 0) AS accz_sub_t5,\
            COALESCE(ABS(gyrox - gyrox_t5), 0) AS gyrox_sub_t5,\
            COALESCE(ABS(gyroy - gyroy_t5), 0) AS gyroy_sub_t5,\
            COALESCE(ABS(gyroz - gyroz_t5), 0) AS gyroz_sub_t5,\
            COALESCE(second - second_t5, 0) AS second_sub_t5,\
            COALESCE(ABS(speed - speed_t5), 0) AS speed_sub_t5,\
            COALESCE(ABS(acc3d - acc3d_t5), 0) AS acc3d_sub_t5,\
            COALESCE(ABS(gyro3d - gyro3d_t5), 0) AS gyro3d_sub_t5,\
            COALESCE(ABS(acc3dgyro3d - acc3dgyro3d_t5), 0) AS acc3dgyro3d_sub_t5\
        FROM sdf_1_2")
    sdf_1_3.registerTempTable('sdf_1_3')

    # 4th phase
    # Compute threshold
    sdf_1_4 = spark.sql(
        "SELECT bookingID,\
            PERCENTILE(bearing_sub_t5, 0.7) AS bearing_sub_t5_70perc,\
            PERCENTILE(accx_sub_t5, 0.7) AS accx_sub_t5_70perc,\
            PERCENTILE(accy_sub_t5, 0.7) AS accy_sub_t5_70perc,\
            PERCENTILE(accz_sub_t5, 0.7) AS accz_sub_t5_70perc,\
            PERCENTILE(gyrox_sub_t5, 0.7) AS gyrox_sub_t5_70perc,\
            PERCENTILE(gyroy_sub_t5, 0.7) AS gyroy_sub_t5_70perc,\
            PERCENTILE(gyroz_sub_t5, 0.7) AS gyroz_sub_t5_70perc,\
            PERCENTILE(second_sub_t5, 0.7) AS second_sub_t5_70perc,\
            PERCENTILE(acc3d_sub_t5, 0.7) AS acc3d_sub_t5_70perc,\
            PERCENTILE(gyro3d_sub_t5, 0.7) AS gyro3d_sub_t5_70perc,\
            PERCENTILE(acc3dgyro3d_sub_t5, 0.7) AS acc3dgyro3d_sub_t5_70perc\
        FROM sdf_1_3\
        GROUP BY bookingID")
    sdf_1_4.registerTempTable('sdf_1_4')

    # 5th phase
    # Aggregate features into bookingID level
    sdf_1_5 = spark.sql(
        "SELECT aa.bookingID,\
            COUNT(second) AS datapoint_num,\
            MAX(aa.second) AS travel_sec,\
            PERCENTILE(COALESCE(aa.speed_sub_t5/aa.second_sub_t5,0), 0.99)  AS speed_sub_t5_max,\
            STDDEV(COALESCE(aa.speed_sub_t5/aa.second_sub_t5,0)) AS speed_sub_t5_std,\
            AVG(aa.speed) AS speed_avg,\
            PERCENTILE(aa.speed, 0.5) AS speed_med,\
            PERCENTILE(aa.speed, 0.99) AS speed_max,\
            PERCENTILE(ABS(COALESCE(aa.bearing_sub_t5/aa.second_sub_t5, 0)), 0.9) AS disp_sub_t5_90perc,\
            COUNT(CASE WHEN aa.accx_sub_t5 > bb.accx_sub_t5_70perc THEN 1 END) accx_sub_t5_cnt70perc,\
            PERCENTILE(aa.speed, 0.8) AS speed_80perc,\
            PERCENTILE(aa.speed, 0.9) AS speed_90perc,\
            PERCENTILE(COALESCE(aa.speed_sub_t5/aa.second_sub_t5,0), 0.9) AS speed_sub_t5_90perc,\
            AVG(ABS(COALESCE(aa.bearing_sub_t5/aa.second_sub_t5, 0))) AS disp_sub_t5_avg,\
            COUNT(CASE WHEN aa.accy_sub_t5 > bb.accy_sub_t5_70perc THEN 1 END) accy_sub_t5_cnt70perc,\
            STDDEV(aa.speed) AS speed_std,\
            COUNT(CASE WHEN aa.accz_sub_t5 > bb.accz_sub_t5_70perc THEN 1 END) accz_sub_t5_cnt70perc,\
            COUNT(CASE WHEN aa.speed_sub_t5 > bb.accx_sub_t5_70perc THEN 1 END) speed_sub_t5_cnt70perc,\
            PERCENTILE(ABS(COALESCE(aa.bearing_sub_t5/aa.second_sub_t5, 0)), 0.99) AS disp_sub_t5_max,\
            PERCENTILE(COALESCE(aa.speed_sub_t5/aa.second_sub_t5,0), 0.5) AS speed_sub_t5_med,\
            PERCENTILE(aa.gyroy, 0.5) AS gyroy_med,\
            PERCENTILE(aa.gyroz, 0.5) AS gyroz_med,\
            PERCENTILE(aa.accx, 0.99) AS accx_max,\
            COUNT(CASE WHEN aa.gyrox_sub_t5 > bb.gyrox_sub_t5_70perc THEN 1 END) gyrox_sub_t5_cnt70perc,\
            AVG(COALESCE(aa.speed_sub_t5/aa.second_sub_t5,0)) AS speed_sub_t5_avg,\
            PERCENTILE(ABS(COALESCE(aa.bearing_sub_t5/aa.second_sub_t5, 0)), 0.8) AS disp_sub_t5_80perc,\
            STDDEV(ABS(COALESCE(aa.bearing_sub_t5/aa.second_sub_t5, 0))) AS disp_sub_t5_std,\
            STDDEV(aa.accz) AS accz_std,\
            PERCENTILE(aa.gyrox, 0.5) AS gyrox_med,\
            PERCENTILE(aa.accz, 0.99) AS accz_max,\
            PERCENTILE(aa.accy, 0.99) AS accy_max,\
            PERCENTILE(COALESCE(aa.accx_sub_t5/aa.second_sub_t5,0), 0.5) AS accx_sub_t5_med,\
            PERCENTILE(COALESCE(aa.speed_sub_t5/aa.second_sub_t5,0), 0.8) AS speed_sub_t5_80perc,\
            COUNT(CASE WHEN aa.gyroy_sub_t5 > bb.gyroy_sub_t5_70perc THEN 1 END) gyroy_sub_t5_cnt70perc\
        FROM sdf_1_3 aa LEFT JOIN sdf_1_4 bb ON aa.bookingID=bb.bookingID\
        GROUP BY 1")
    sdf_1_5.registerTempTable('sdf_1_5')


    ## Feature Engineering 2 ##
    # 1st phase pre-processing
    sdf_2_1 = spark.sql(
        "SELECT *,\
            ROW_NUMBER() OVER (PARTITION BY bookingID ORDER BY second) AS sequence\
        FROM sdf_1_1")
    sdf_2_1.registerTempTable('sdf_2_1')

    # 2nd phase pre-processing
    # Extract records from t-1 point of time
    sdf_2_2 = spark.sql(
        "SELECT *,\
            LAG(bearing, 1) OVER(PARTITION BY bookingID ORDER BY second) AS bearing_t1,\
            LAG(accx, 1) OVER(PARTITION BY bookingID ORDER BY second) AS accx_t1,\
            LAG(accy, 1) OVER(PARTITION BY bookingID ORDER BY second) AS accy_t1,\
            LAG(accy, 1) OVER(PARTITION BY bookingID ORDER BY second) AS accz_t1,\
            LAG(gyrox, 1) OVER(PARTITION BY bookingID ORDER BY second) AS gyrox_t1,\
            LAG(gyroy, 1) OVER(PARTITION BY bookingID ORDER BY second) AS gyroy_t1,\
            LAG(gyroz, 1) OVER(PARTITION BY bookingID ORDER BY second) AS gyroz_t1,\
            LAG(second, 1) OVER(PARTITION BY bookingID ORDER BY second) AS second_t1,\
            LAG(speed, 1) OVER(PARTITION BY bookingID ORDER BY second) AS speed_t1,\
            LAG(acc3d, 1) OVER(PARTITION BY bookingID ORDER BY second) AS acc3d_t1,\
            LAG(gyro3d, 1) OVER(PARTITION BY bookingID ORDER BY second) AS gyro3d_t1,\
            LAG(acc3dgyro3d, 1) OVER(PARTITION BY bookingID ORDER BY second) AS acc3dgyro3d_t1\
        FROM sdf_2_1")
    sdf_2_2.registerTempTable('sdf_2_2')

    # 3rd phase pre-processing
    # Calculate sign in between of each point of time
    sdf_2_3 = spark.sql(
        "SELECT bookingID, second, sequence,\
            SIGN(ABS(accx - COALESCE(accx_t1, 0))) AS sgn_accx,\
            SIGN(ABS(accy - COALESCE(accy_t1, 0))) AS sgn_accy,\
            SIGN(ABS(accz - COALESCE(accz_t1, 0))) AS sgn_accz,\
            SIGN(ABS(gyrox - COALESCE(gyrox_t1, 0))) AS sgn_gyrox,\
            SIGN(ABS(gyroy - COALESCE(gyroy_t1, 0))) AS sgn_gyroy,\
            SIGN(ABS(gyroz - COALESCE(gyroz_t1, 0))) AS sgn_gyroz,\
            SIGN(speed - COALESCE(speed_t1, 0)) AS sgn_speed,\
            SIGN(second - COALESCE(second_t1, 0)) AS sgn_second,\
            SIGN(acc3d - COALESCE(acc3d_t1, 0)) AS sgn_acc3d,\
            SIGN(gyro3d - COALESCE(gyro3d_t1, 0)) AS sgn_gyro3d,\
            SIGN(acc3dgyro3d - COALESCE(acc3dgyro3d_t1, 0)) AS sgn_acc3dgyro3d\
            FROM sdf_2_2")
    sdf_2_3.registerTempTable('sdf_2_3')

    # 4th phase pre-processing
    sdf_2_4 = spark.sql(
        "SELECT bookingID, sequence,\
            sgn_speed,\
            sequence - ROW_NUMBER() OVER (PARTITION BY bookingID, sgn_speed ORDER BY second) AS group_speed,\
            sgn_accx,\
            sequence - ROW_NUMBER() OVER (PARTITION BY bookingID, sgn_accx ORDER BY second) AS group_accx,\
            sgn_accy,\
            sequence - ROW_NUMBER() OVER (PARTITION BY bookingID, sgn_accy ORDER BY second) AS group_accy,\
            sgn_accz,\
            sequence - ROW_NUMBER() OVER (PARTITION BY bookingID, sgn_accy ORDER BY second) AS group_accz,\
            sgn_gyrox,\
            sequence - ROW_NUMBER() OVER (PARTITION BY bookingID, sgn_gyrox ORDER BY second) AS group_gyrox,\
            sgn_gyroy,\
            sequence - ROW_NUMBER() OVER (PARTITION BY bookingID, sgn_gyroy ORDER BY second) AS group_gyroy,\
            sgn_gyroz,\
            sequence - ROW_NUMBER() OVER (PARTITION BY bookingID, sgn_gyroz ORDER BY second) AS group_gyroz,\
            sgn_acc3d,\
            sequence - ROW_NUMBER() OVER (PARTITION BY bookingID, sgn_acc3d ORDER BY second) AS group_acc3d,\
            sgn_gyro3d,\
            sequence - ROW_NUMBER() OVER (PARTITION BY bookingID, sgn_gyro3d ORDER BY second) AS group_gyro3d,\
            sgn_acc3dgyro3d,\
            sequence - ROW_NUMBER() OVER (PARTITION BY bookingID, sgn_acc3dgyro3d ORDER BY second) AS group_acc3dgyro3d\
        FROM sdf_2_3")
    sdf_2_4.registerTempTable('sdf_2_4')

    # 5th phase pre-processing
    # Calculate the longest consecutive value increase in each feature
    sdf_2_5_speed = spark.sql(
        "SELECT DISTINCT bookingID,\
            MAX(COUNT(*)) OVER (PARTITION BY bookingID) AS speed_max_inc_num\
        FROM sdf_2_4\
        WHERE sgn_speed=1 GROUP BY bookingID, group_speed\
        ")
    sdf_2_5_speed.registerTempTable('sdf_2_5_speed')

    sdf_2_5_accx = spark.sql(
        "SELECT DISTINCT bookingID,\
            MAX(COUNT(*)) OVER (PARTITION BY bookingID) AS accx_max_inc_num\
        FROM sdf_2_4\
        WHERE sgn_accx=1 GROUP BY bookingID, group_accx\
        ")
    sdf_2_5_accx.registerTempTable('sdf_2_5_accx')

    sdf_2_5_accy = spark.sql(
        "SELECT DISTINCT bookingID,\
            MAX(COUNT(*)) OVER (PARTITION BY bookingID) AS accy_max_inc_num\
        FROM sdf_2_4\
        WHERE sgn_accy=1 GROUP BY bookingID, group_accy\
        ")
    sdf_2_5_accy.registerTempTable('sdf_2_5_accy')

    sdf_2_5_accz = spark.sql(
        "SELECT DISTINCT bookingID,\
            MAX(COUNT(*)) OVER (PARTITION BY bookingID) AS accz_max_inc_num\
        FROM sdf_2_4\
        WHERE sgn_accz=1 GROUP BY bookingID, group_accz\
        ")
    sdf_2_5_accz.registerTempTable('sdf_2_5_accz')

    sdf_2_5_gyrox = spark.sql(
        "SELECT DISTINCT bookingID,\
            MAX(COUNT(*)) OVER (PARTITION BY bookingID) AS gyrox_max_inc_num\
        FROM sdf_2_4\
        WHERE sgn_gyrox=1 GROUP BY bookingID, group_gyrox\
        ")
    sdf_2_5_gyrox.registerTempTable('sdf_2_5_gyrox')

    sdf_2_5_gyroy = spark.sql(
        "SELECT DISTINCT bookingID,\
            MAX(COUNT(*)) OVER (PARTITION BY bookingID) AS gyroy_max_inc_num\
        FROM sdf_2_4\
        WHERE sgn_gyroy=1 GROUP BY bookingID, group_gyroy\
        ")
    sdf_2_5_gyroy.registerTempTable('sdf_2_5_gyroy')

    sdf_2_5_gyroz = spark.sql(
        "SELECT DISTINCT bookingID,\
            MAX(COUNT(*)) OVER (PARTITION BY bookingID) AS gyroz_max_inc_num\
        FROM sdf_2_4\
        WHERE sgn_gyroz=1 GROUP BY bookingID, group_gyroz\
        ")
    sdf_2_5_gyroz.registerTempTable('sdf_2_5_gyroz')

    sdf_2_5_acc3d = spark.sql(
        "SELECT DISTINCT bookingID,\
            MAX(COUNT(*)) OVER (PARTITION BY bookingID) AS acc3d_max_inc_num\
        FROM sdf_2_4\
        WHERE sgn_acc3d=1 GROUP BY bookingID, group_acc3d\
        ")
    sdf_2_5_acc3d.registerTempTable('sdf_2_5_acc3d')

    sdf_2_5_gyro3d = spark.sql(
        "SELECT DISTINCT bookingID,\
            MAX(COUNT(*)) OVER (PARTITION BY bookingID) AS gyro3d_max_inc_num\
        FROM sdf_2_4\
        WHERE sgn_gyro3d=1 GROUP BY bookingID, group_gyro3d\
        ")
    sdf_2_5_gyro3d.registerTempTable('sdf_2_5_gyro3d')

    sdf_2_5_acc3dgyro3d = spark.sql(
        "SELECT DISTINCT bookingID,\
            MAX(COUNT(*)) OVER (PARTITION BY bookingID) AS acc3dgyro3d_max_inc_num\
        FROM sdf_2_4\
        WHERE sgn_acc3dgyro3d=1 GROUP BY bookingID, group_acc3dgyro3d\
        ")
    sdf_2_5_acc3dgyro3d.registerTempTable('sdf_2_5_acc3dgyro3d')

    # 6th phase pre-processing
    # Join tables into one
    sdf_2_6 = spark.sql(
        "SELECT sdf_2_5_speed.bookingID,\
            sdf_2_5_speed.speed_max_inc_num,\
            sdf_2_5_accx.accx_max_inc_num,\
            sdf_2_5_accy.accy_max_inc_num,\
            sdf_2_5_accz.accz_max_inc_num,\
            sdf_2_5_gyrox.gyrox_max_inc_num,\
            sdf_2_5_gyroy.gyroy_max_inc_num,\
            sdf_2_5_gyroz.gyroz_max_inc_num,\
            sdf_2_5_acc3d.acc3d_max_inc_num,\
            sdf_2_5_gyro3d.gyro3d_max_inc_num,\
            sdf_2_5_acc3dgyro3d.acc3dgyro3d_max_inc_num\
        FROM sdf_2_5_speed\
            LEFT JOIN sdf_2_5_accx ON sdf_2_5_speed.bookingID=sdf_2_5_accx.bookingID\
            LEFT JOIN sdf_2_5_accy ON sdf_2_5_speed.bookingID=sdf_2_5_accy.bookingID\
            LEFT JOIN sdf_2_5_accz ON sdf_2_5_speed.bookingID=sdf_2_5_accz.bookingID\
            LEFT JOIN sdf_2_5_gyrox ON sdf_2_5_speed.bookingID=sdf_2_5_gyrox.bookingID\
            LEFT JOIN sdf_2_5_gyroy ON sdf_2_5_speed.bookingID=sdf_2_5_gyroy.bookingID\
            LEFT JOIN sdf_2_5_gyroz ON sdf_2_5_speed.bookingID=sdf_2_5_gyroz.bookingID\
            LEFT JOIN sdf_2_5_acc3d ON sdf_2_5_speed.bookingID=sdf_2_5_acc3d.bookingID\
            LEFT JOIN sdf_2_5_gyro3d ON sdf_2_5_speed.bookingID=sdf_2_5_gyro3d.bookingID\
            LEFT JOIN sdf_2_5_acc3dgyro3d ON sdf_2_5_speed.bookingID=sdf_2_5_acc3dgyro3d.bookingID")
    sdf_2_6.registerTempTable('sdf_2_6')

    ## Feature Engineering final ##
    # Create final aggregated feature table
    sdf_final = spark.sql(
        "SELECT sdf_1_5.*,\
            sdf_2_6.speed_max_inc_num,\
            sdf_2_6.accx_max_inc_num,\
            sdf_2_6.accy_max_inc_num,\
            sdf_2_6.accz_max_inc_num,\
            sdf_2_6.gyrox_max_inc_num,\
            sdf_2_6.gyroy_max_inc_num,\
            sdf_2_6.gyroz_max_inc_num,\
            sdf_2_6.acc3d_max_inc_num,\
            sdf_2_6.gyro3d_max_inc_num,\
            sdf_2_6.acc3dgyro3d_max_inc_num\
            FROM sdf_1_5\
                LEFT JOIN sdf_2_6 ON sdf_1_5.bookingID=sdf_2_6.bookingID")
    sdf_final.registerTempTable('sdf_final')

    # Create feature file
    try:
        sdf_final.coalesce(1).write.mode('overwrite').csv(_dirpath, header=True)
    except Exception as e:
        logger.error('Failed to create features')
        logger.error('Exception on create_features(): '+str(e))
        raise

if __name__ == '__main__':
    logger.info('create_features.py start!')
    start = time.time()
    logger.info('Load data files from ' + _telematics_data_dir + _csv_filenames)
    sdf_raw = load_data_into_spark(_filepath=_telematics_data_dir+_csv_filenames)
    logger.info('create_dataset() start')
    create_dataset(_dirpath=_feature_data_dir, _sdf_raw=sdf_raw)
    process_time = round(time.time() - start, 2)
    logger.info('Elapsed time: ' + str(process_time) + 'sec')
    logger.info('create_features.py completed!')


# coding: utf-8

# In[1]:


from pyspark.sql.types import *
from pyspark.sql import functions as func
import pandas as pd
import time
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')


# In[3]:


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
#         StructField('second', IntegerType(), False),
        StructField('second', DoubleType(), False),
        StructField('speed', DoubleType(), False)])

# df = spark.read.csv("./features/part-00000-e6120af0-10c2-4248-97c4-81baf4304e5c-c000.csv", schema=struct, header=True)
sdf_raw = spark.read.csv("./features/part-*.csv", schema=struct, header=True)
sdf_raw.registerTempTable('sdf_raw')


# In[4]:


# Filter out records whose speed value is negative
# Average records in each second
sdf_aggsec = spark.sql(
    "SELECT bookingID, second,\
        AVG(bearing) AS bearing,\
        AVG(accx) AS accx, AVG(accy) AS accy, AVG(accz) AS accz,\
        AVG(gyrox) AS gyrox, AVG(gyroy) AS gyroy, AVG(gyroz) AS gyroz,\
        AVG(speed) AS speed\
    FROM sdf_raw\
    WHERE speed>=0\
    GROUP BY bookingID, second")
sdf_aggsec.registerTempTable('sdf_aggsec')


# In[6]:


# Identify 3 percentile value of speed
sdf_aggsec_speed3perc = spark.sql(
    "SELECT bookingID,\
        PERCENTILE(speed, 0.03) AS speed_3perc\
    FROM sdf_aggsec\
    GROUP BY bookingID")
sdf_aggsec_speed3perc.registerTempTable('sdf_aggsec_speed3perc')


# In[7]:


# Extract records whose speed is within 3% percentile, and calculate avarage of acc in these records
sdf_aggsec_speed3perc_lt = spark.sql(
    "SELECT aa.bookingID,\
        AVG(aa.accx) AS accx_offset,\
        AVG(aa.accy) AS accy_offset,\
        AVG(aa.accz) AS accz_offset\
    FROM sdf_aggsec aa LEFT JOIN sdf_aggsec_speed3perc bb ON aa.bookingID=bb.bookingID\
    WHERE aa.speed <= bb.speed_3perc\
    GROUP BY aa.bookingID")
sdf_aggsec_speed3perc_lt.registerTempTable('sdf_aggsec_speed3perc_lt')


# In[8]:


# Caribrate acc
sdf_aggsec_carib = spark.sql(
    "SELECT aa.bookingID, aa.second, aa.bearing,\
        aa.accx - accx_offset AS accx,\
        aa.accy - accy_offset AS accy,\
        aa.accz - accz_offset AS accz,\
        aa.gyrox, aa.gyroy, aa.gyroz,\
        aa.speed\
    FROM sdf_aggsec aa LEFT JOIN sdf_aggsec_speed3perc_lt bb ON aa.bookingID=bb.bookingID")
sdf_aggsec_carib.registerTempTable('sdf_aggsec_carib')


# In[9]:


# df=sdf_aggsec_carib.toPandas()
# df.to_csv('./data/eda_carib_10_190608.csv', index=False)


# # Feature Engineering

# In[10]:


# 1st phase pre-processing
# Calculate norm(strength) of acc3d and gyro3d
sdf_1 = spark.sql(
    "SELECT *,\
        SQRT(accx*accx + accy*accy + accz*accz) AS acc3d,\
        SQRT(gyrox*gyrox + gyroy*gyroy + gyroz*gyroz) AS gyro3d,\
        SQRT(accx*accx + accy*accy + accz*accz)*SQRT(gyrox*gyrox + gyroy*gyroy + gyroz*gyroz)  AS acc3dgyro3d\
    FROM sdf_aggsec_carib")
sdf_1.registerTempTable('sdf_1')


# In[11]:


# 2nd phase pre-processing
# Extract records from previous point of time
sdf_2 = spark.sql(
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
        LAG(acc3dgyro3d, 5) OVER(PARTITION BY bookingID ORDER BY second) AS acc3dgyro3d_t5,\
        LAG(bearing, 10) OVER(PARTITION BY bookingID ORDER BY second) AS bearing_t10,\
        LAG(accx, 10) OVER(PARTITION BY bookingID ORDER BY second) AS accx_t10,\
        LAG(accy, 10) OVER(PARTITION BY bookingID ORDER BY second) AS accy_t10,\
        LAG(accy, 10) OVER(PARTITION BY bookingID ORDER BY second) AS accz_t10,\
        LAG(gyrox, 10) OVER(PARTITION BY bookingID ORDER BY second) AS gyrox_t10,\
        LAG(gyroy, 10) OVER(PARTITION BY bookingID ORDER BY second) AS gyroy_t10,\
        LAG(gyroz, 10) OVER(PARTITION BY bookingID ORDER BY second) AS gyroz_t10,\
        LAG(second, 10) OVER(PARTITION BY bookingID ORDER BY second) AS second_t10,\
        LAG(speed, 10) OVER(PARTITION BY bookingID ORDER BY second) AS speed_t10,\
        LAG(acc3d, 10) OVER(PARTITION BY bookingID ORDER BY second) AS acc3d_t10,\
        LAG(gyro3d, 10) OVER(PARTITION BY bookingID ORDER BY second) AS gyro3d_t10,\
        LAG(acc3dgyro3d, 10) OVER(PARTITION BY bookingID ORDER BY second) AS acc3dgyro3d_t10\
    FROM sdf_1")
sdf_2.registerTempTable('sdf_2')


# In[12]:


# 3rd phase
# Compute record sub and div
sdf_3 = spark.sql(
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
        COALESCE(ABS(acc3dgyro3d - acc3dgyro3d_t5), 0) AS acc3dgyro3d_sub_t5,\
        COALESCE(ABS(bearing / bearing_t5), 0) AS bearing_div_t5,\
        COALESCE(ABS(accx / accx_t5), 0) AS accx_div_t5,\
        COALESCE(ABS(accy / accy_t5), 0) AS accy_div_t5,\
        COALESCE(ABS(accz / accz_t5), 0) AS accz_div_t5,\
        COALESCE(ABS(gyrox / gyrox_t5), 0) AS gyrox_div_t5,\
        COALESCE(ABS(gyroy / gyroy_t5), 0) AS gyroy_div_t5,\
        COALESCE(ABS(gyroz / gyroz_t5), 0) AS gyroz_div_t5,\
        COALESCE(second / second_t5, 0) AS second_div_t5,\
        COALESCE(ABS(speed / speed_t5), 0) AS speed_div_t5,\
        COALESCE(ABS(acc3d / acc3d_t5), 0) AS acc3d_div_t5,\
        COALESCE(ABS(gyro3d / gyro3d_t5), 0) AS gyro3d_div_t5,\
        COALESCE(ABS(acc3dgyro3d / acc3dgyro3d_t5), 0) AS acc3dgyro3d_div_t5,\
        COALESCE(ABS(bearing / bearing_t10), 0) AS bearing_div_t10,\
        COALESCE(ABS(bearing - bearing_t10), 0) AS bearing_sub_t10,\
        COALESCE(ABS(accx - accx_t10), 0) AS accx_sub_t10,\
        COALESCE(ABS(accy - accy_t10), 0) AS accy_sub_t10,\
        COALESCE(ABS(accz - accz_t10), 0) AS accz_sub_t10,\
        COALESCE(ABS(gyrox - gyrox_t10), 0) AS gyrox_sub_t10,\
        COALESCE(ABS(gyroy - gyroy_t10), 0) AS gyroy_sub_t10,\
        COALESCE(ABS(gyroz - gyroz_t10), 0) AS gyroz_sub_t10,\
        COALESCE(second - second_t10, 0) AS second_sub_t10,\
        COALESCE(ABS(speed - speed_t10), 0) AS speed_sub_t10,\
        COALESCE(ABS(acc3d - acc3d_t10), 0) AS acc3d_sub_t10,\
        COALESCE(ABS(gyro3d - gyro3d_t10), 0) AS gyro3d_sub_t10,\
        COALESCE(ABS(acc3dgyro3d - acc3dgyro3d_t10), 0) AS acc3dgyro3d_sub_t10,\
        COALESCE(ABS(accx / accx_t10), 0) AS accx_div_t10,\
        COALESCE(ABS(accy / accy_t10), 0) AS accy_div_t10,\
        COALESCE(ABS(accz / accz_t10), 0) AS accz_div_t10,\
        COALESCE(ABS(gyrox / gyrox_t10), 0) AS gyrox_div_t10,\
        COALESCE(ABS(gyroy / gyroy_t10), 0) AS gyroy_div_t10,\
        COALESCE(ABS(gyroz / gyroz_t10), 0) AS gyroz_div_t10,\
        COALESCE(second / second_t10, 0) AS second_div_t10,\
        COALESCE(ABS(speed / speed_t10), 0) AS speed_div_t10,\
        COALESCE(ABS(acc3d / acc3d_t10), 0) AS acc3d_div_t10,\
        COALESCE(ABS(gyro3d / gyro3d_t10), 0) AS gyro3d_div_t10,\
        COALESCE(ABS(acc3dgyro3d / acc3dgyro3d_t10), 0) AS acc3dgyro3d_div_t10\
    FROM sdf_2")
sdf_3.registerTempTable('sdf_3')


# In[13]:


# 4th phase
# Compute threshold
sdf_4 = spark.sql(
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
        PERCENTILE(acc3dgyro3d_sub_t5, 0.7) AS acc3dgyro3d_sub_t5_70perc,\
        PERCENTILE(bearing_div_t5, 0.7) AS bearing_div_t5_70perc,\
        PERCENTILE(accx_div_t5, 0.7) AS accx_div_t5_70perc,\
        PERCENTILE(accy_div_t5, 0.7) AS accy_div_t5_70perc,\
        PERCENTILE(accz_div_t5, 0.7) AS accz_div_t5_70perc,\
        PERCENTILE(gyrox_div_t5, 0.7) AS gyrox_div_t5_70perc,\
        PERCENTILE(gyroy_div_t5, 0.7) AS gyroy_div_t5_70perc,\
        PERCENTILE(gyroz_div_t5, 0.7) AS gyroz_div_t5_70perc,\
        PERCENTILE(second_div_t5, 0.7) AS second_div_t5_70perc,\
        PERCENTILE(acc3d_div_t5, 0.7) AS acc3d_div_t5_70perc,\
        PERCENTILE(gyro3d_div_t5, 0.7) AS gyro3d_div_t5_70perc,\
        PERCENTILE(acc3dgyro3d_div_t5, 0.7) AS acc3dgyro3d_div_t5_70perc,\
        PERCENTILE(bearing_sub_t10, 0.7) AS bearing_sub_t10_70perc,\
        PERCENTILE(accx_sub_t10, 0.7) AS accx_sub_t10_70perc,\
        PERCENTILE(accy_sub_t10, 0.7) AS accy_sub_t10_70perc,\
        PERCENTILE(accz_sub_t10, 0.7) AS accz_sub_t10_70perc,\
        PERCENTILE(gyrox_sub_t10, 0.7) AS gyrox_sub_t10_70perc,\
        PERCENTILE(gyroy_sub_t10, 0.7) AS gyroy_sub_t10_70perc,\
        PERCENTILE(gyroz_sub_t10, 0.7) AS gyroz_sub_t10_70perc,\
        PERCENTILE(second_sub_t10, 0.7) AS second_sub_t10_70perc,\
        PERCENTILE(acc3d_sub_t10, 0.7) AS acc3d_sub_t10_70perc,\
        PERCENTILE(gyro3d_sub_t10, 0.7) AS gyro3d_sub_t10_70perc,\
        PERCENTILE(acc3dgyro3d_sub_t10, 0.7) AS acc3dgyro3d_sub_t10_70perc,\
        PERCENTILE(bearing_div_t10, 0.7) AS bearing_div_t10_70perc,\
        PERCENTILE(accx_div_t10, 0.7) AS accx_div_t10_70perc,\
        PERCENTILE(accy_div_t10, 0.7) AS accy_div_t10_70perc,\
        PERCENTILE(accz_div_t10, 0.7) AS accz_div_t10_70perc,\
        PERCENTILE(gyrox_div_t10, 0.7) AS gyrox_div_t10_70perc,\
        PERCENTILE(gyroy_div_t10, 0.7) AS gyroy_div_t10_70perc,\
        PERCENTILE(gyroz_div_t10, 0.7) AS gyroz_div_t10_70perc,\
        PERCENTILE(second_div_t10, 0.7) AS second_div_t10_70perc,\
        PERCENTILE(acc3d_div_t10, 0.7) AS acc3d_div_t10_70perc,\
        PERCENTILE(gyro3d_div_t10, 0.7) AS gyro3d_div_t10_70perc,\
        PERCENTILE(acc3dgyro3d_div_t10, 0.7) AS acc3dgyro3d_div_t10_70perc\
    FROM sdf_3\
    GROUP BY bookingID")
sdf_4.registerTempTable('sdf_4')


# In[ ]:


# https://people.apache.org/~pwendell/spark-nightly/spark-master-docs/spark-2.3.0-SNAPSHOT-2017_12_08_04_01-26e6645-docs/api/sql/#percentile
# 5th phase
# Aggregate features into bookingID level
sdf_5 = spark.sql(
    "SELECT aa.bookingID,\
        MAX(aa.second) AS travel_sec,\
        PERCENTILE(aa.speed, 0.99) AS speed_max,\
        AVG(aa.speed) AS speed_avg,\
        STDDEV(aa.speed) AS speed_std,\
        PERCENTILE(aa.speed, 0.5) AS speed_med,\
        PERCENTILE(aa.speed, 0.8) AS speed_80perc,\
        PERCENTILE(aa.speed, 0.9) AS speed_90perc,\
        PERCENTILE(aa.accx, 0.99) AS accx_max,\
        AVG(aa.accx) AS accx_avg,\
        STDDEV(aa.accx) AS accx_std,\
        PERCENTILE(aa.accx, 0.5) AS accx_med,\
        PERCENTILE(aa.accx, 0.8) AS accx_80perc,\
        PERCENTILE(aa.accx, 0.9) AS accx_90perc,\
        PERCENTILE(aa.accy, 0.99) AS accy_max,\
        AVG(aa.accy) AS accy_avg,\
        STDDEV(aa.accy) AS accy_std,\
        PERCENTILE(aa.accy, 0.5) AS accy_med,\
        PERCENTILE(aa.accy, 0.8) AS accy_80perc,\
        PERCENTILE(aa.accy, 0.9) AS accy_90perc,\
        PERCENTILE(aa.accz, 0.99) AS accz_max,\
        AVG(aa.accz) AS accz_avg,\
        STDDEV(aa.accz) AS accz_std,\
        PERCENTILE(aa.accz, 0.5) AS accz_med,\
        PERCENTILE(aa.accz, 0.8) AS accz_80perc,\
        PERCENTILE(aa.accz, 0.9) AS accz_90perc,\
        PERCENTILE(aa.gyrox, 0.99) AS gyrox_max,\
        AVG(aa.gyrox) AS gyrox_avg,\
        STDDEV(aa.gyrox) AS gyrox_std,\
        PERCENTILE(aa.gyrox, 0.5) AS gyrox_med,\
        PERCENTILE(aa.gyrox, 0.8) AS gyrox_80perc,\
        PERCENTILE(aa.gyrox, 0.9) AS gyrox_90perc,\
        PERCENTILE(aa.gyroy, 0.99) AS gyroy_max,\
        AVG(aa.gyroy) AS gyroy_avg,\
        STDDEV(aa.gyroy) AS gyroy_std,\
        PERCENTILE(aa.gyroy, 0.5) AS gyroy_med,\
        PERCENTILE(aa.gyroy, 0.8) AS gyroy_80perc,\
        PERCENTILE(aa.gyroy, 0.9) AS gyroy_90perc,\
        PERCENTILE(aa.gyroz, 0.99) AS gyroz_max,\
        AVG(aa.gyroz) AS gyroz_avg,\
        STDDEV(aa.gyroz) AS gyroz_std,\
        PERCENTILE(aa.gyroz, 0.5) AS gyroz_med,\
        PERCENTILE(aa.gyroz, 0.8) AS gyroz_80perc,\
        PERCENTILE(aa.gyroz, 0.9) AS gyroz_90perc,\
        PERCENTILE(aa.acc3d, 0.99) AS acc3d_max,\
        AVG(aa.acc3d) AS acc3d_avg,\
        STDDEV(aa.acc3d) AS acc3d_std,\
        PERCENTILE(aa.acc3d, 0.5) AS acc3d_med,\
        PERCENTILE(aa.acc3d, 0.8) AS acc3d_80perc,\
        PERCENTILE(aa.acc3d, 0.9) AS acc3d_90perc,\
        PERCENTILE(aa.gyro3d, 0.99) AS gyro3d_max,\
        AVG(aa.gyro3d) AS gyro3d_avg,\
        STDDEV(aa.gyro3d) AS gyro3d_std,\
        PERCENTILE(aa.gyro3d, 0.5) AS gyro3d_med,\
        PERCENTILE(aa.gyro3d, 0.8) AS gyro3d_80perc,\
        PERCENTILE(aa.gyro3d, 0.9) AS gyro3d_90perc,\
        PERCENTILE(aa.acc3dgyro3d, 0.99) AS acc3dgyro3d_70perc_max,\
        AVG(aa.acc3dgyro3d) AS acc3dgyro3d_70perc_avg,\
        STDDEV(aa.acc3dgyro3d) AS acc3dgyro3d_70perc_std,\
        PERCENTILE(aa.acc3dgyro3d, 0.5) AS acc3dgyro3d_70perc_med,\
        PERCENTILE(aa.acc3dgyro3d, 0.8) AS acc3dgyro3d_70perc_80perc,\
        PERCENTILE(aa.acc3dgyro3d, 0.9) AS acc3dgyro3d_70perc_90perc,\
        PERCENTILE(COALESCE(aa.speed_sub_t5/aa.second_sub_t5,0), 0.99)  AS speed_sub_t5_max,\
        AVG(COALESCE(aa.speed_sub_t5/aa.second_sub_t5,0)) AS speed_sub_t5_avg,\
        STDDEV(COALESCE(aa.speed_sub_t5/aa.second_sub_t5,0)) AS speed_sub_t5_std,\
        PERCENTILE(COALESCE(aa.speed_sub_t5/aa.second_sub_t5,0), 0.5) AS speed_sub_t5_med,\
        PERCENTILE(COALESCE(aa.speed_sub_t5/aa.second_sub_t5,0), 0.8) AS speed_sub_t5_80perc,\
        PERCENTILE(COALESCE(aa.speed_sub_t5/aa.second_sub_t5,0), 0.9) AS speed_sub_t5_90perc,\
        PERCENTILE(COALESCE(aa.accx_sub_t5/aa.second_sub_t5,0), 0.99)  AS accx_sub_t5_max,\
        AVG(COALESCE(aa.accx_sub_t5/aa.second_sub_t5,0)) AS accx_sub_t5_avg,\
        STDDEV(COALESCE(aa.accx_sub_t5/aa.second_sub_t5,0)) AS accx_sub_t5_std,\
        PERCENTILE(COALESCE(aa.accx_sub_t5/aa.second_sub_t5,0), 0.5) AS accx_sub_t5_med,\
        PERCENTILE(COALESCE(aa.accx_sub_t5/aa.second_sub_t5,0), 0.8) AS accx_sub_t5_80perc,\
        PERCENTILE(COALESCE(aa.accx_sub_t5/aa.second_sub_t5,0), 0.9) AS accx_sub_t5_90perc,\
        PERCENTILE(COALESCE(aa.accy_sub_t5/aa.second_sub_t5,0), 0.99)  AS accy_sub_t5_max,\
        AVG(COALESCE(aa.accy_sub_t5/aa.second_sub_t5,0)) AS accy_sub_t5_avg,\
        STDDEV(COALESCE(aa.accy_sub_t5/aa.second_sub_t5,0)) AS accy_sub_t5_std,\
        PERCENTILE(COALESCE(aa.accy_sub_t5/aa.second_sub_t5,0), 0.5) AS accy_sub_t5_med,\
        PERCENTILE(COALESCE(aa.accy_sub_t5/aa.second_sub_t5,0), 0.8) AS accy_sub_t5_80perc,\
        PERCENTILE(COALESCE(aa.accy_sub_t5/aa.second_sub_t5,0), 0.9) AS accy_sub_t5_90perc,\
        PERCENTILE(COALESCE(aa.accz_sub_t5/aa.second_sub_t5,0), 0.99)  AS accz_sub_t5_max,\
        AVG(COALESCE(aa.accz_sub_t5/aa.second_sub_t5,0)) AS accz_sub_t5_avg,\
        STDDEV(COALESCE(aa.accz_sub_t5/aa.second_sub_t5,0)) AS accz_sub_t5_std,\
        PERCENTILE(COALESCE(aa.accz_sub_t5/aa.second_sub_t5,0), 0.5) AS accz_sub_t5_med,\
        PERCENTILE(COALESCE(aa.accz_sub_t5/aa.second_sub_t5,0), 0.8) AS accz_sub_t5_80perc,\
        PERCENTILE(COALESCE(aa.accz_sub_t5/aa.second_sub_t5,0), 0.9) AS accz_sub_t5_90perc,\
        PERCENTILE(COALESCE(aa.gyrox_sub_t5/aa.second_sub_t5,0), 0.99)  AS gyrox_sub_t5_max,\
        AVG(COALESCE(aa.gyrox_sub_t5/aa.second_sub_t5,0)) AS gyrox_sub_t5_avg,\
        STDDEV(COALESCE(aa.gyrox_sub_t5/aa.second_sub_t5,0)) AS gyrox_sub_t5_std,\
        PERCENTILE(COALESCE(aa.gyrox_sub_t5/aa.second_sub_t5,0), 0.5) AS gyrox_sub_t5_med,\
        PERCENTILE(COALESCE(aa.gyrox_sub_t5/aa.second_sub_t5,0), 0.8) AS gyrox_sub_t5_80perc,\
        PERCENTILE(COALESCE(aa.gyrox_sub_t5/aa.second_sub_t5,0), 0.9) AS gyrox_sub_t5_90perc,\
        PERCENTILE(COALESCE(aa.gyroy_sub_t5/aa.second_sub_t5,0), 0.99)  AS gyroy_sub_t5_max,\
        AVG(COALESCE(aa.gyroy_sub_t5/aa.second_sub_t5,0)) AS gyroy_sub_t5_avg,\
        STDDEV(COALESCE(aa.gyroy_sub_t5/aa.second_sub_t5,0)) AS gyroy_sub_t5_std,\
        PERCENTILE(COALESCE(aa.gyroy_sub_t5/aa.second_sub_t5,0), 0.5) AS gyroy_sub_t5_med,\
        PERCENTILE(COALESCE(aa.gyroy_sub_t5/aa.second_sub_t5,0), 0.8) AS gyroy_sub_t5_80perc,\
        PERCENTILE(COALESCE(aa.gyroy_sub_t5/aa.second_sub_t5,0), 0.9) AS gyroy_sub_t5_90perc,\
        PERCENTILE(COALESCE(aa.gyroz_sub_t5/aa.second_sub_t5,0), 0.99)  AS gyroz_sub_t5_max,\
        AVG(COALESCE(aa.gyroz_sub_t5/aa.second_sub_t5,0)) AS gyroz_sub_t5_avg,\
        STDDEV(COALESCE(aa.gyroz_sub_t5/aa.second_sub_t5,0)) AS gyroz_sub_t5_std,\
        PERCENTILE(COALESCE(aa.gyroz_sub_t5/aa.second_sub_t5,0), 0.5) AS gyroz_sub_t5_med,\
        PERCENTILE(COALESCE(aa.gyroz_sub_t5/aa.second_sub_t5,0), 0.8) AS gyroz_sub_t5_80perc,\
        PERCENTILE(COALESCE(aa.gyroz_sub_t5/aa.second_sub_t5,0), 0.9) AS gyroz_sub_t5_90perc,\
        PERCENTILE(COALESCE(aa.acc3d_sub_t5/aa.second_sub_t5,0), 0.99)  AS acc3d_sub_t5_max,\
        AVG(COALESCE(aa.acc3d_sub_t5/aa.second_sub_t5,0)) AS acc3d_sub_t5_avg,\
        STDDEV(COALESCE(aa.acc3d_sub_t5/aa.second_sub_t5,0)) AS acc3d_sub_t5_std,\
        PERCENTILE(COALESCE(aa.acc3d_sub_t5/aa.second_sub_t5,0), 0.5) AS acc3d_sub_t5_med,\
        PERCENTILE(COALESCE(aa.acc3d_sub_t5/aa.second_sub_t5,0), 0.8) AS acc3d_sub_t5_80perc,\
        PERCENTILE(COALESCE(aa.acc3d_sub_t5/aa.second_sub_t5,0), 0.9) AS acc3d_sub_t5_90perc,\
        PERCENTILE(COALESCE(aa.gyro3d_sub_t5/aa.second_sub_t5,0), 0.99)  AS gyro3d_sub_t5_max,\
        AVG(COALESCE(aa.gyro3d_sub_t5/aa.second_sub_t5,0)) AS gyro3d_sub_t5_avg,\
        STDDEV(COALESCE(aa.gyro3d_sub_t5/aa.second_sub_t5,0)) AS gyro3d_sub_t5_std,\
        PERCENTILE(COALESCE(aa.gyro3d_sub_t5/aa.second_sub_t5,0), 0.5) AS gyro3d_sub_t5_med,\
        PERCENTILE(COALESCE(aa.gyro3d_sub_t5/aa.second_sub_t5,0), 0.8) AS gyro3d_sub_t5_80perc,\
        PERCENTILE(COALESCE(aa.gyro3d_sub_t5/aa.second_sub_t5,0), 0.9) AS gyro3d_sub_t5_90perc,\
        PERCENTILE(COALESCE(aa.acc3dgyro3d_sub_t5/aa.second_sub_t5,0), 0.99)  AS acc3dgyro3d_sub_t5_max,\
        AVG(COALESCE(aa.acc3dgyro3d_sub_t5/aa.second_sub_t5,0)) AS acc3dgyro3d_sub_t5_avg,\
        STDDEV(COALESCE(aa.acc3dgyro3d_sub_t5/aa.second_sub_t5,0)) AS acc3dgyro3d_sub_t5_std,\
        PERCENTILE(COALESCE(aa.acc3dgyro3d_sub_t5/aa.second_sub_t5,0), 0.5) AS acc3dgyro3d_sub_t5_med,\
        PERCENTILE(COALESCE(aa.acc3dgyro3d_sub_t5/aa.second_sub_t5,0), 0.8) AS acc3dgyro3d_sub_t5_80perc,\
        PERCENTILE(COALESCE(aa.acc3dgyro3d_sub_t5/aa.second_sub_t5,0), 0.9) AS acc3dgyro3d_sub_t5_90perc,\
        PERCENTILE(COALESCE(SQRT(POW(aa.accx-aa.accx_t5, 2)+POW(aa.accy-aa.accy_t5, 2)+POW(aa.accz-aa.accz_t5, 2))/aa.speed_sub_t5, 0), 0.99) AS accx_vec_t5_max,\
        AVG(COALESCE(SQRT(POW(aa.accx-aa.accx_t5, 2)+POW(aa.accy-aa.accy_t5, 2)+POW(aa.accz-aa.accz_t5, 2))/aa.speed_sub_t5, 0)) AS accx_vec_t5_avg,\
        STDDEV(COALESCE(SQRT(POW(aa.accx-aa.accx_t5, 2)+POW(aa.accy-aa.accy_t5, 2)+POW(aa.accz-aa.accz_t5, 2))/aa.speed_sub_t5, 0)) AS accx_vec_t5_std,\
        PERCENTILE(COALESCE(SQRT(POW(aa.accx-aa.accx_t5, 2)+POW(aa.accy-aa.accy_t5, 2)+POW(aa.accz-aa.accz_t5, 2))/aa.speed_sub_t5, 0), 0.5) AS accx_vec_t5_med,\
        PERCENTILE(COALESCE(SQRT(POW(aa.accx-aa.accx_t5, 2)+POW(aa.accy-aa.accy_t5, 2)+POW(aa.accz-aa.accz_t5, 2))/aa.speed_sub_t5, 0), 0.8) AS accx_vec_t5_80perc,\
        PERCENTILE(COALESCE(SQRT(POW(aa.accx-aa.accx_t5, 2)+POW(aa.accy-aa.accy_t5, 2)+POW(aa.accz-aa.accz_t5, 2))/aa.speed_sub_t5, 0), 0.9) AS accx_vec_t5_90perc,\
        PERCENTILE(ABS(COALESCE(aa.bearing_sub_t5/aa.speed_sub_t5, 0)), 0.99) AS disp_sub_t5_max,\
        AVG(ABS(COALESCE(aa.bearing_sub_t5/aa.speed_sub_t5, 0))) AS disp_sub_t5_avg,\
        STDDEV(ABS(COALESCE(aa.bearing_sub_t5/aa.speed_sub_t5, 0))) AS disp_sub_t5_std,\
        PERCENTILE(ABS(COALESCE(aa.bearing_sub_t5/aa.speed_sub_t5, 0)), 0.5) AS disp_sub_t5_med,\
        PERCENTILE(ABS(COALESCE(aa.bearing_sub_t5/aa.speed_sub_t5, 0)), 0.8) AS disp_sub_t5_80perc,\
        PERCENTILE(ABS(COALESCE(aa.bearing_sub_t5/aa.speed_sub_t5, 0)), 0.9) AS disp_sub_t5_90perc\
    FROM sdf_3 aa\
    GROUP BY 1")
sdf_5.registerTempTable('sdf_5')


# In[ ]:


start = time.time()

sdf_5.coalesce(1).write.mode('overwrite').csv("./data/agg_features_t5_190608.csv", header=True)
process_time = round(time.time() - start, 2)
print(process_time)


# Grab_AI_safety
This repository is deliverables of Safety challenge in Grab AI for SEA.  
https://www.aiforsea.com/safety
## Problem statement
Given the telematics data for each trip and the label if the trip is tagged as dangerous driving, derive a model that can detect dangerous driving trips.
## Data
Data is available from this URL.  
https://s3-ap-southeast-1.amazonaws.com/grab-aiforsea-dataset/safety.zip
## Data dictionary
 - bookingID: trip id  
 - accuracy: accuracy inferred by GPS in meters  
 - bearing: GPS bearing in degree  
 - accx: accelerometer reading at x axis (m/s2)  
 - accy: accelerometer reading at y axis (m/s2)  
 - accz: accelerometer reading at z axis (m/s2)  
 - gyrox: gyroscope reading in x axis (rad/s)  
 - gyroy: gyroscope reading in y axis (rad/s)  
 - gyroz: gyroscope reading in z axis (rad/s)  
 - second: time of the record by number of seconds  
 - speed: speed measured by GPS in m/s  
## Environment:
 - OS: Ubuntu 18.04.2 LTS (Bionic Beaver)  
 - Python: Python 3.6.8  
 - As for Python packages, refer requirement.txt  
 - Spark: Spark 2.3.3  
## Directories
 - config: For common config file (for logger setting)  
 - raw_data: For telematics raw data files  
 - labels: For label files 
 - dataset: For modelling dataset created by create_features.py  
 - log: For job log file 
 - model: For predictive model pickle file  
 - python_module: For common Python module file  
 - jupyter_notebooks: For Jupyter Notebook files  
## Python codes
 - create_features.py: For creating features given telematics raw data files  
 - build_model.py: For building model to detect if a drive is danger  
 - detect_dangerdrive.py: For detecting danger drive and compute AUC score given dataset and label  
## Procedure to run the model for testing dataset
1. Set up Linux server  
2. Install Python 3.6.8 and Spark 2.3.3  
3. Install packages  
pip install -r requirements.txt  
4. Clone this Git repository on the server  
5. Change directory  
$ cd ./Grab_AI_safety  
6. Download testing data file on following directories  
 - Telematics data: ./raw_data/test  
 (e.g.) ./raw_data/test/part-00000-e6120af0-10c2-4248-97c4-81baf4304e5c-c000.csv  
 - Label data: ./label/test/  
 (e.g.) ./labels/test/part-00000-e9445087-aa0a-433b-a7f6-7f4c19d78ad6-c000.csv  
7. Create features for testing dataset by running Spark job  
$ nohup spark-submit --master local[*] --conf spark.pyspark.python=python --executor-cores 8 --executor-memory 40G --driver-memory 5G create_features.py test &  
 * executor-cores, executor-memory, and driver-memory options need to be set according to your environment  
 * Files will be created on ./dataset/test directory  
8. Detect dangerous drive by running Python program  
$ python detect_dangerdrive.py test  
### Note:
 - XGBoost model file has already been uploaded on this repository (See ./model/xgb_model_fulldata.pkl)  
 - detect_dangerdrive.py loads the model and make a prediction  
 - If you would like to build model by yourself, you need to follow these steps  
1. Download training data file on following directories  
 - Telematics data: ./raw_data/train  
  (e.g.) ./raw_data/train/part-00000-e6120af0-10c2-4248-97c4-81baf4304e5c-c000.csv  
 - Label data: ./label/train/  
  (e.g.) ./labels/train/part-00000-e9445087-aa0a-433b-a7f6-7f4c19d78ad6-c000.csv  
2. Create features for training dataset by running Spark job  
$ nohup spark-submit --master local[*] --conf spark.pyspark.python=python --executor-cores 8 --executor-memory 40G --driver-memory 5G create_features.py train &  
* Files will be created on ./dataset/train directory
3. Build XGBoost model by using training data  
$ python build_model.py  
4. Detect dangerous drive by running Python program  
$ python detect_dangerdrive.py train  
## Feature engineering
### Abnormal value handling
 - There are some abnormal values in data. For example, there are records whose speed=-1. If speed is negative value, the value is replaced with 0. Also, if speed exceeds 300km (i.e. 83.34 m/s), the value is replaced with 83.34.
 - There are records whose second values are same in each bookingID. (i.e. one travel records might have multiple records whose second values are same). In that case, values of a record is averaged by second.
### Accelerometer readings calibration
 - Accelerometer readings are affected by gravity. Effect of gravity in each axis depends on how a mobile device is inclined in a car. Hence, gravity effect is approximated by averaging accelerometer readings when speed is less than 3 percentile. Here, I assume that a car doesn't accelerate when speed is zero, or very slow. In these points of time, accelerometer measures only gravity effect. So I just calculate such a gravity effect by each drive, then subtract in each readings in order to calibration.  
### Features derived
 - Basic representative values such as max, median, 80 percentile, etc is computed in each drive.
 - Accelerometer and gyrometer readings provides values in each axis, so these values are aggregated by computing as following.  
  - acc3d: SQRT(accx * accx + accy * accy + accz * accz)  
  - gyro3d: SQRT(gyrox * gyrox + gyroy * gyroy + gyroz * gyroz)  
  - acc3dgyro3d: SQRT(accx * accx + accy * accy + accz * accz) * SQRT(gyrox * gyrox + gyroy * gyroy + gyroz * gyroz)  
 - Difference of readings between `t` and `t-5` are calculated in order to represent how much speed and direction differs within a certain period.  
 - The number of consecutive reading values increase is calculated.  
## Technologies employed
 - Spark is used for creating features by bookingID level given telematics data. Although Pandas could be enough to handle data provided for this challenge, I exploited Spark for the purpose of scalability because Grab has vast amounts of telematics data with millions of user base. By using Spark, the solution can be scaled easily.  
 - As for modelling framework, XGBoost with Scikit-learn is used for the challenge because aggregated data can be not so huge compared to raw telematics data. If Grab integrates the solution into Spark from end-to-end, Spark ML would be used for building model.  
 - Regarding modelling algorithm, XGBoost is employed because GBDT based model is strong in cross tabular data.  
## Findings & insights
 - A travel time is the most important feature according to XGBoost feature importance. It suggests that longer a drive, more a user may have more chance to feel danger. It makes sense but longer drive isn't necessary dangerous one.
 - In addition, there would be individual variation whether a user feels drive is dangerous or not. Also, a user might not report to Grab even if he had dangerous travel. Thus, I'm afraid that labelling could be biased.
 - Therefore, we may need to think of another way in order to annotate data for supervised learning, or to detect dangerous driver by using unsupervised learning. 
## Use cases of the model built in the project
 - Grab provides function that enables a user to report in case that a drive was dangerous. However, there would be individual variation whether a user feels drive is dangerous or not. Also, a user might not report to Grab even if he had dangerous travel. Hence, the detective model can be used to identify danger drive even though a user doesn't report so. By identifying a driver who is more likely to drive such a way, Grab can alert to the driver and educate him before causing serious accident.
 - Realize real-time dangerous drive detection by building a model which utilizes telematics data generated on halfway drive. Streaming data processing technologies such as Spark Streaming and Kafka could be helpful to ingest real-time data and make a prediction on the fly. By implementing such a real-time detection system, Grab can send push notification so that a driver is warned.
## Future work
 - Implement prototype application by using Spark Streaming and Kafka in order to realize real-time data ingestion and prediction system.
 - Collect image data during travel to build image recognition model to detect drowsy driving. The model will contribute to identify dangerous drive from another point of view. To collect such data, Grab may provide incentive to a driver who allows Grab to collect real-time image data via smartphone camera.  
 - Experiment on other modelling algorithms such as Deep Neural Network, unsupervised abnormal detection approach, etc  
 - Dockerize an application for productionalization

# Grab_AI_safety
This repository is deliverables of Safety challenge in Grab AI for SEA.  
https://www.aiforsea.com/safety
## Problem statement
Given the telematics data for each trip and the label if the trip is tagged as dangerous driving, derive a model that can detect dangerous driving trips.
## Data
Data is available from this URL.  
https://s3-ap-southeast-1.amazonaws.com/grab-aiforsea-dataset/safety.zip
## Environment:
 - OS: Ubuntu 18.04.2 LTS (Bionic Beaver)  
 - Python: Python 3.6.8  
  - As for Python packagesm, refer requirement.txt  
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
 

## Procedure to run the model for testing dataset
1. Set up Ubuntu server  
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
8. Run pre-built model by running Python program  
$ python detect_dangerdrive.py test  
## Note:
 - XGBoost model file is uploaded on this repository (./model/xgb_model_fulldata.pkl)  
 - detect_dangerdrive.py loads the model and make prediction  
 - If you would like to build model by yourself, you need to follow these steps  
1. Create features for testing dataset by running Spark job  
$ nohup spark-submit --master local[*] --conf spark.pyspark.python=python --executor-cores 8 --executor-memory 40G --driver-memory 5G create_features.py train & 
2. Build XGBoost model by using training data  
$ python build_model.py  
3. Run the model by running Python program  
$ python detect_dangerdrive.py train  


## Idea memo:
 - Label is given by customers, but some users may be more likely to label as "dangerous", vice versa.
 - This model helps Grab to identify characteristics & trend of dangerous driving even if a customer does not give feedback
 - By using Spark Streaming & Kafka, dangerous drive could be identified (by sudden acc, keep increasing speed, etc), then Grab can send push notification to a driver in order to alert.
 - Dockerize: https://qiita.com/kon_yu/items/7c40f4dfbd1cce006ce7

## 


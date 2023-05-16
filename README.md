# Disaster-Response-Classification

# Summary:
This project is following Udacity's data analysis project #2. In this project I built an ETL pipeline, a machine learning pipeline and then completeing web app portion of this project. 

# How to Run:
 * To run ETL pipeline enter: python3 data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db
 * To run ML pipeline enter: python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl
 * To run the web app enter: python3 run.py
# Files: 
process_data.py: Loads, cleans and stores data.
train_classifier.py: Creates, trains and evaluated a machine learning model.


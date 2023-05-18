# Disaster-Response-Classification

# Summary:
This project is following Udacity's data analysis project #2. In this project I built an ETL pipeline, a machine learning pipeline and then completeing web app portion of this project.

This project could prove to be very helpful to the disaster relief community. 
    It could be used to process incomming messages from areas experiencing a disaster and prioritize the messages that should be responded to. 
    It could also be used to identify locations that might be experiencing a disaster. It could accomplish this by being implemented in a large messaging app that continuously pulses recived messages against the model and sending an alert if a possible disaster message was encountered.

# How to Run:
 * To run ETL pipeline enter: python3 data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db
 * To run ML pipeline enter: python3 models/train_classifier.py data/DisasterResponse.db models/classifier.pkl
 * To run the web app enter: python3 run.py
# Files: 
* App
    - Templates
        * go.html : Outputs the reponse from the model after user clicks enter message.
        * master.html : Handles the layout of the web app. 
    - run.py : Handles the setup of the web app.
* data
    - disaster_categories.csv : Initial csv file holding disaster message categories.
    - disaster_messages.csv : Initial csv file holding disaster messgages.
    - DisasterResponse.db : SQLLite database holding out cleaned data.
    - process_data.py : Handles the ETL portion of the project.
* Models
    - classifier.pkl : The pickle file holding our trained classifier.
    - train_classifier.py : Handles the training of the classifier.
* gitignore : contains all files not wanted to be uploaded to GIT.
* README.md : The projects readme

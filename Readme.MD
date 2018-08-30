# Natural Language Classifier

This repository can be used as a reference to creating ML or Deep learning models using Tensorflow, Keras or any library of your choice.  This one has code for creating Natural Language classifier model that runs on IBM Watson Machine learning platform and can be configured to use runtime (CPU or GPU) as per your choice.  All the code to create and train the model is under the "build_code" folder.  Please note that whole of this code under "build_code" folder is zipped and deployed to IBM Watson Machine Learning platform and runs there.  

## Project Structure

  - build_code/IntentClassification.py: This is the main file that runs the code to create and train a model.

  - build_code/handlers: This has all the handlers like keras_model_handler.py, data_handler.py and cloud object storage handler (cos_handler.py)

  - utilities: This has one utility, but that's currently not been used.  You can remove it or can use it as per your requirement or preference.

  - Classify.py: You can run this python file to test your models either locally or model which is deployed on IBM Cloud:

  - Deployment.py: This file uses watson_machine_learning_client library to create definitions, train and store model, deploy model and use scoring endpoint.  This can be done by commenting or uncommenting a few lines of code for specific calls:

  - config.json: This file is not part of the git commit and you need to create this file which should have following format with details of wml_credentials and cos_credentials:

  ```sh
  {
    "cos_credentials":{
                        "apikey": "-----------------------------------",
                        "cos_hmac_keys": {
                          "access_key_id": "-----------------------------------",
                          "secret_access_key": "-----------------------------------"
                        },
                        "endpoints": "https://cos-service.bluemix.net/endpoints",
                        "iam_apikey_description": "-----------------------------------",
                        "iam_apikey_name": "auto-generated-apikey------------------------------------",
                        "iam_role_crn": "crn:v1:bluemix:public:iam::::serviceRole:Writer",
                        "iam_serviceid_crn": "crn:v1:bluemix:public:iam-identity::a/-----------------------------------::serviceid:ServiceId------------------------------------",
                        "resource_instance_id": "crn:v1:bluemix:public:cloud-object-storage:global:a/-----------------------------------:-----------------------------------::"
                      },
    "wml_credentials":{
                        "apikey": "------------------------------------vpfRo3SBZrOajK",
                        "iam_apikey_description": "Auto generated apikey during resource-key operation for Instance - crn:v1:bluemix:public:pm-20:us-south:a/-----------------------------------::",
                        "iam_apikey_name": "auto-generated-apikey------------------------------------",
                        "iam_role_crn": "crn:v1:bluemix:public:iam::::serviceRole:Writer",
                        "iam_serviceid_crn": "crn:v1:bluemix:public:iam-identity::a/-----------------------------------::-----------------------------------",
                        "instance_id": "-----------------------------------",
                        "password": "-----------------------------------",
                        "url": "https://ibm-watson-ml.mybluemix.net",
                        "username": "-----------------------------------"
                      },
    "deployment_id": "-----------------------------------",
    "service_endpoint": "https://s3-api.us-geo.objectstorage.softlayer.net",
    "auth_endpoint": "https://iam.bluemix.net/oidc/token"
  }
```

## Requirements to run locally
Step 1: Make sure you have following dependencies:
  - Python (I prefer to use Anaconda for installation)
  - Tensorflow >= 1.5.0
  - Keras
  - Scikit Learn

Step 2: Create "data" folder and put "data.csv" inside it.  This file should have following format:
```sh
utterances,intent
Tell me something about yourself,about_me
Tell me something about you ?,about_me
Tell me about yourself ?,about_me
Who are you ?,about_me
```
Note: More the examples of Utterances and their corresponding Intent, better the accuracy of your NLC model.

## Create Model locally
To create and train model locally, run the following command:
```sh
python build_code/IntentClassification.py --data_dir data --result_dir results --config_file model_config.json --data_file data.csv --framework keras
```
## Test your model locally
To test your model, run following command:
```sh
python Classify.py --data_dir data --result_dir results --config_file model_config.json --data_file data.csv --from_cloud False
```
## Deployment of Model to IBM Cloud
Once you are good with basic testing of your model locally, you may want to create and run training tasks that uses GPUs or you may want to monitor, do continuous learning, analyze your models and much more, then use "Deployment.py" file.  Make a few code changes and run each task separately like process_deployment, store_model and deploy_model.  Most of the required commands are mentioned in this file at the bottom that you can use by uncommenting the code.  

  ```sh
  python Deployment.py --data_dir data --result_dir results --config_file model_config.json --data_file data.csv
  ```

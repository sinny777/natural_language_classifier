#
#
# main() will be run when you invoke this action
#
# @param Cloud Functions actions accept a single parameter, which must be a JSON object.
#
# @return The output of this action, which must be a JSON object.
#
#
import sys
import urllib3, requests, json
import re
import json
import ibm_boto3
import pandas as pd
from ibm_botocore.client import Config
from io import StringIO

def getCOS(args):
  endpoint = args.get('endpoint','https://s3-api.us-geo.objectstorage.softlayer.net')
  api_key_id = args.get('apikey', args.get('apiKeyId', args.get('cos_credentials', {}).get('apikey', '')))
  service_instance_id = args.get('resource_instance_id', args.get('serviceInstanceId', args.get('cos_credentials', {}).get('resource_instance_id', '')))
  ibm_auth_endpoint = args.get('ibmAuthEndpoint', 'https://iam.ng.bluemix.net/oidc/token')
  cos = ibm_boto3.resource(service_name='s3',
    ibm_api_key_id=api_key_id,
    ibm_service_instance_id=service_instance_id,
    ibm_auth_endpoint=ibm_auth_endpoint,
    config=Config(signature_version='oauth'),
    endpoint_url=endpoint)
  return cos

def init_content(params):
    cos = getCOS(params)
    bytes_data = cos.Object(params["data_bucket"], "data.csv").get()["Body"].read()
    s1 = str(bytes_data,'utf-8')
    data_file = StringIO(s1)
    df = pd.read_csv(data_file)
    word_index_bytes = cos.Object(params["data_bucket"], "word_index.json").get()["Body"].read()
    word_index_file = str(word_index_bytes,'utf-8')
    intents = df["intent"].unique()
    intents = sorted(list(set(intents)))
    global scoring_params
    scoring_params = {
        "intents": intents,
        "word_index": json.loads(word_index_file)
    }
    return scoring_params


def get_scoring_payload(params):
    max_fatures = 500
    maxlen = 50

    preprocessed_records = []
    word_index = scoring_params['word_index']
    scoring_data = params["texts"]

    for data in scoring_data:
        comment = data
        cleanString = re.sub(r"[!\"#$%&()*+,-./:;<=>?@[\]^_`{|}~]", "", comment)
        splitted_comment = cleanString.split()[:maxlen]
        hashed_tokens = []

        for token in splitted_comment:
            index = word_index.get(token, 0)
            if index < 501 and index > 0:
                hashed_tokens.append(index)

        hashed_tokens_size = len(hashed_tokens)
        padded_tokens = [0]*(maxlen-hashed_tokens_size) + hashed_tokens
        preprocessed_records.append(padded_tokens)

    scoring_payload = {'values': preprocessed_records}
    return scoring_payload

def score(params):
    headers = urllib3.util.make_headers(basic_auth='{username}:{password}'.format(username=params["wml_credentials"]['username'], password=params["wml_credentials"]['password']))
    url = '{}/v3/identity/token'.format(params["wml_credentials"]['url'])
    response = requests.get(url, headers=headers)
    mltoken = json.loads(response.text).get('token')

    header = {'Content-Type': 'application/json', 'Authorization': 'Bearer ' + mltoken}
    ERROR_THRESHOLD = 0.15
    if "ERROR_THRESHOLD" in params:
        ERROR_THRESHOLD = params["ERROR_THRESHOLD"]

    payload_scoring = get_scoring_payload(params)
    print(str(payload_scoring))
    print(params["scoring_endpoint"])
    resp = requests.post(params["scoring_endpoint"], json=payload_scoring, headers=header)
    response_scoring = resp.json()

    result = response_scoring["values"][0][0]
    # filter out predictions below a threshold
    result = [[i,r] for i,r in enumerate(result) if r>ERROR_THRESHOLD]
    # sort by strength of probability
    result.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    resp_json = {"intents": []}
    for r in result:
        resp_json["intents"].append({"intent": scoring_params["intents"][r[0]], "score": r[1]})
    print(resp_json)
    return resp_json


def main(params):
    init_content(params)
    return score(params)

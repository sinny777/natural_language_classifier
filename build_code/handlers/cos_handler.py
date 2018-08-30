
import ibm_boto3
from ibm_botocore.client import Config, ClientError

class COSHandler(object):
    def __init__(self, SECRET_CONFIG):
        self.cos = ibm_boto3.resource(service_name='s3',
                ibm_api_key_id=SECRET_CONFIG["cos_credentials"]['apikey'],
                ibm_auth_endpoint=SECRET_CONFIG["auth_endpoint"],
                config=Config(signature_version='oauth'),
                endpoint_url=SECRET_CONFIG["service_endpoint"])

    def get_bucket_contents(self, bucket_name):
        print("Retrieving bucket contents from: {0}".format(bucket_name))
        try:
            files = self.cos.Bucket(bucket_name).objects.all()
            # for file in files:
            #     print("Item: {0} ({1} bytes).".format(file.key, file.size))
            return files
        except ClientError as be:
            print("CLIENT ERROR: {0}\n".format(be))
        except Exception as e:
            print("Unable to retrieve bucket contents: {0}".format(e))

    def upload_file(self, bucket_name, file_path, file_name):
        print("Uploading file: {0}".format(file_path))
        try:
            self.cos.Bucket(bucket_name).upload_file(file_path, file_name)
            print("File: {0} uploaded!".format(file_path))
        except ClientError as be:
            print("CLIENT ERROR: {0}\n".format(be))
        except Exception as e:
            print("Unable to upload file: {0}".format(e))

    def create_file(self, bucket_name, file_name, file_data):
        print("Creating new file: {0}".format(file_name))
        try:
            self.cos.Object(bucket_name, file_name).put(
                Body=file_data
            )
            print("File: {0} created!".format(file_name))
        except ClientError as be:
            print("CLIENT ERROR: {0}\n".format(be))
        except Exception as e:
            print("Unable to create file: {0}".format(e))

    def get_item(self, bucket_name, item_name):
        print("Retrieving item from bucket: {0}, key: {1}".format(bucket_name, item_name))
        try:
            file = self.cos.Object(bucket_name, item_name).get()
            # print("File Contents: {0}".format(file["Body"].read()))
            return file
        except ClientError as be:
            print("CLIENT ERROR: {0}\n".format(be))
        except Exception as e:
            print("Unable to retrieve file contents: {0}".format(e))

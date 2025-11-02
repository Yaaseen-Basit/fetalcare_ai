


import boto3 
bedrock = boto3.client(service_name='bedrock' , region_name="ap-south-1")

bedrock.list_foundation_models()
print(f"Model Name: ",bedrock.list_foundation_models())


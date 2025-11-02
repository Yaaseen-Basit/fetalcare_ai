import sagemaker
from sagemaker.sklearn.model import SKLearnModel
import boto3

# Initialize SageMaker session
session = sagemaker.Session()
SAGEMAKER_ROLE_ARN = 'arn:aws:iam::01234586:role/SageMaker-ml-role' 

# Set the role variable directly
role = SAGEMAKER_ROLE_ARN
print(f"  Using hardcoded role ARN: {role}")
#   S3 URIs
model_data_uri = 's3://model-bucket/ai-model/model.tar.gz'
source_dir_uri = 's3://model-bucket/ai-model/source.tar.gz'

#   Define model
model = SKLearnModel(
    model_data=model_data_uri,
    role=role,
    entry_point="inference.py",
    source_dir=source_dir_uri,
    framework_version="1.2-1",    # Any lightweight version works
    sagemaker_session=session
)

#   Deploy the model
predictor = model.deploy(
    initial_instance_count=1,
    instance_type="ml.t2.medium", 
    endpoint_name="your-ai-endpoint",
      wait=False
)

print("  Model deployed successfully!")
print("Endpoint name:", predictor.endpoint_name)

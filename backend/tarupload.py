import tarfile, shutil, boto3, os

# Step 1: Prepare folder
model_dir = "model_dir"
os.makedirs(model_dir, exist_ok=True)
shutil.copy("model.pkl", model_dir)
shutil.copy("inference.py", model_dir)

# Step 2: Create tar.gz
with tarfile.open("model.tar.gz", "w:gz") as tar:
    for f in os.listdir(model_dir):
        tar.add(os.path.join(model_dir, f), arcname=f)

# Step 3: Upload to S3
s3 = boto3.client("s3", region_name="ap-south-1")
s3.upload_file("model.tar.gz", "aicare-model-bucket", "model.tar.gz")

print("✅ Clean model.tar.gz uploaded to S3 successfully!")

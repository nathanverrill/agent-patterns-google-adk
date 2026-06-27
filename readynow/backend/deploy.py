import os
import vertexai
from google.cloud import storage
from vertexai import agent_engines
from app import app  # Imports your FastAPI app instance wrapper object

# Target project credentials parameters mapping to workshop parameters
gcp_project = os.getenv("GOOGLE_CLOUD_PROJECT", "qwiklabs-gcp-02-32eac43f4591")
gcp_location = "us-central1"
staging_bucket_name = f"{gcp_project}-agent-staging-bucket"

print(f"📦 Verifying Cloud Storage staging bucket: gs://{staging_bucket_name}...")
storage_client = storage.Client(project=gcp_project)
bucket = storage_client.bucket(staging_bucket_name)
if not bucket.exists():
    bucket.location = gcp_location
    storage_client.create_bucket(bucket, location=gcp_location)

# Initialize platform context configurations
vertexai.init(
    project=gcp_project,
    location=gcp_location,
    staging_bucket=f"gs://{staging_bucket_name}"
)

print("\n🚀 Pushing ReadyNow system configuration to upstream Agent Platform...")
try:
    remote_agent = agent_engines.create(
        app,
        requirements=["google-cloud-aiplatform[agent_engines,adk]", "fastapi", "uvicorn", "requests"]
    )
    print(f"\n✅ Challenge 6 Deployment Complete! Resource URI Name:\n{remote_agent.resource_name}")
except Exception as e:
    print(f"⚠️ Remote Platform Error: {e}")
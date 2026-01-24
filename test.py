from azure.ai.inference import EmbeddingsClient
from azure.core.credentials import AzureKeyCredential
import os
from dotenv import load_dotenv

load_dotenv()
endpoint = "https://models.github.ai/inference"
model_name = "openai/text-embedding-3-large"
token = os.environ["GITHUB_TOKEN"]

client = EmbeddingsClient(endpoint=endpoint, credential=AzureKeyCredential(token))

response = client.embed(input=["Hello world"], model=model_name)

print(response)

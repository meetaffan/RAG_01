from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
import os
from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage, UserMessage
from azure.core.credentials import AzureKeyCredential
from langchain_qdrant import QdrantVectorStore
from azure.ai.inference import EmbeddingsClient

from dotenv import load_dotenv


load_dotenv()

file_path = "./nodejs.pdf"

# Loading
loader = PyPDFLoader(file_path)
docs = loader.load()
# print(docs[5])

# Chunking
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap = 200
)

split_docs = text_splitter.split_documents(documents=docs)


# endpoint_embeddigs = "https://models.github.ai/inference"
endpoint = "https://models.github.ai/inference"
model_name_embeddings = "openai/text-embedding-3-large"
api_key_embeddings = os.environ["api_key"]

# Vector Embeddings
# embedding_model = OpenAIEmbeddings(
#     model=model_name_embeddings,
#     api_key=api_key_embeddings
#     # With the `text-embedding-3` class
#     # of models, you can specify the size
#     # of the embeddings you want returned.
#     # dimensions=1024
# )

client = EmbeddingsClient(
    endpoint=endpoint,
    credential=AzureKeyCredential(api_key_embeddings)
)

vector_store = QdrantVectorStore.from_documents(
    documents=split_docs,
    url = "http://localhost:6333",
    collection_name ="rag_01",
    embedding=client
)

print("Indexing of document done...")


# model = "openai/gpt-5-mini"
model = "gpt-5-mini"
token = os.environ["GITHUB_TOKEN"]

client = ChatCompletionsClient(
    endpoint=endpoint,
    credential=AzureKeyCredential(token),
)

response = client.complete(
    messages=[
        SystemMessage("You are a helpful assistant."),
        UserMessage("What is the capital of France?"),
    ],
    model=model
)

print(response.choices[0].message.content)



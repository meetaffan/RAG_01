from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_qdrant import QdrantVectorStore
from azure.ai.inference import ChatCompletionsClient, EmbeddingsClient
from azure.ai.inference.models import SystemMessage, UserMessage
from azure.core.credentials import AzureKeyCredential
import os
from dotenv import load_dotenv

load_dotenv()

# -------------------------
# 1️⃣ Load PDF and split into chunks
# -------------------------
file_path = "./nodejs.pdf"

loader = PyPDFLoader(file_path)
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

split_docs = text_splitter.split_documents(documents=docs)
print(f"Total chunks created: {len(split_docs)}")

# -------------------------
# 2️⃣ Wrap GitHub embeddings client
# -------------------------
class GitHubEmbeddingsWrapper:
    def __init__(self, client, model_name):
        self.client = client
        self.model_name = model_name

    def embed_documents(self, texts):
        response = self.client.embed(input=texts, model=self.model_name)
        return [item.embedding for item in response.data]

    def embed_query(self, text):
        response = self.client.embed(input=[text], model=self.model_name)
        return response.data[0].embedding

# GitHub embeddings client
endpoint = "https://models.github.ai/inference"
model_name_embeddings = "openai/text-embedding-3-large"
github_token = os.environ["GITHUB_TOKEN"]  # Your GitHub token

embeddings_client = EmbeddingsClient(
    endpoint=endpoint,
    credential=AzureKeyCredential(github_token)
)

embedding_model = GitHubEmbeddingsWrapper(
    client=embeddings_client,
    model_name=model_name_embeddings
)

# -------------------------
# 3️⃣ Index chunks in Qdrant
# -------------------------
vector_store = QdrantVectorStore.from_documents(
    documents=split_docs,
    url="http://localhost:6333",
    collection_name="rag_01",
    embedding=embedding_model
)

print("✅ Document indexing in Qdrant done!")

# -------------------------
# 4️⃣ Chat using GitHub GPT model
# -------------------------
chat_model_name = "gpt-5-mini"

chat_client = ChatCompletionsClient(
    endpoint=endpoint,
    credential=AzureKeyCredential(github_token)
)

response = chat_client.complete(
    messages=[
        SystemMessage("You are a helpful assistant."),
        UserMessage("What is the capital of France?")
    ],
    model=chat_model_name
)

print("💬 Chat Response:", response.choices[0].message.content)

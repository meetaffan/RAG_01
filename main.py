from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
from langchain_qdrant import QdrantVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from google import genai
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

# Vector Embeddings using HuggingFace (runs locally, no API key needed)
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vector_store = QdrantVectorStore.from_documents(
    documents=split_docs,
    url = "http://localhost:6333",
    collection_name ="rag_01",
    embedding=embedding_model
)

print("Indexing of document done...")

# Initialize Gemini client for chat
client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

# RAG Question-Answer Loop
print("\n" + "="*50)
print("RAG System Ready! Ask questions about the PDF.")
print("Type 'stop' to exit.")
print("="*50 + "\n")

while True:
    question = input("Your question: ").strip()
    
    if question.lower() == "stop":
        print("Exiting... Goodbye!")
        break
    
    if not question:
        print("Please enter a question.\n")
        continue
    
    # Retrieve relevant documents from vector store
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    relevant_docs = retriever.invoke(question)
    
    # Build context from retrieved documents
    context = "\n\n".join([doc.page_content for doc in relevant_docs])
    
    # Create prompt with context
    prompt = f"""Based on the following context, answer the question.

Context:
{context}

Question: {question}
Answer:"""
    
    # Get response from Gemini
    interaction = client.interactions.create(
        model="gemini-3-flash-preview",
        input=prompt
    )
    
    answer = interaction.outputs[-1].text
    
    print(f"\nAnswer: {answer}\n")
    print("Sources:")
    for idx, doc in enumerate(relevant_docs, 1):
        page_no = doc.metadata.get('page', 'N/A')
        print(f"  - Chunk {idx}, Page {page_no}")
    print("\n" + "-"*50 + "\n")



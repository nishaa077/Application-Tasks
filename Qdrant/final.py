from qdrant_client import QdrantClient, models
#from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import Qdrant
from qdrant_client.models import Distance, VectorParams
import os
from huggingface_hub import login

#load the pdf

loader = PyPDFLoader("Ikigai.pdf")
documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500,
                                                   chunk_overlap=50)
texts = text_splitter.split_documents(documents)

model = SentenceTransformer("all-MiniLM-L6-v2")
#embeddings = model.encode(texts)

client = QdrantClient(":memory:")
client.create_collection(
    collection_name="test_collection",
    vectors_config=VectorParams(size=model.get_sentence_embedding_dimension(), distance=Distance.COSINE)
)

client.upload_points(
    collection_name="test_collection",
    points=[
        models.Record(
            id= idx,
            vector=model.encode(doc.page_content).tolist(),
            payload=doc
        ) for idx, doc in enumerate(texts)
    ]

)

collection_stats = client.get_collection("test_collection")
num_vectors = collection_stats.vectors_count

if num_vectors > 0:
  print(f"Collection has {num_vectors} vectors.")
else:
  print(f"Collection is empty.")



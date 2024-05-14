import chromadb
client = chromadb.Client()

#print(client.heartbeat()) #checks if the service is running

collection = client.create_collection(name="my_collection")

#print(collection)
 
collection.modify("my_new_collection")


collection.add(
    documents=["This is a document", "This is another document"],
    metadatas=[{"source": "my_source"}, {"source": "my_source"}],
    ids=["id1", "id2"]
)

results = collection.query(
    query_texts=["This is a query document"],
    n_results=1
)

#print(collection.count())

newcollection = client.get_or_create_collection(name="hehe",metadata={"hnsw:space": "cosine"})
print(newcollection)

try: 
     client.delete_collection(name="hehe") 
     print("hehe collection deleted.") 
except ValueError as e: 
     print(f"Error: {e}") 
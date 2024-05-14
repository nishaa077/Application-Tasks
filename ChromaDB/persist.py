import chromadb

client = chromadb.PersistentClient(path="/home/nisha/chromadb/collpath")

print(client.heartbeat())

morpheus_collection = client.create_collection(name="morpheus_collection")


 # Adding embeddings and metadata 
morpheus_collection.add( 
     embeddings=[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], 
     metadatas=[ 
         {"location": "Zion", "description": "Last human city"}, 
         {"location": "Machine City", "description": "City inhabited by machines"}, 
     ], 
     ids=["loc_1", "loc_2"], 
 ) 

morpheus_collection.add(
    embeddings=[[1.2, 2.3, 4.5], [6.7, 8.2, 9.2]],
    documents=["This is a document", "This is another document"],
    metadatas=[{"source": "my_source"}, {"source": "my_source"}],
    ids=["id1", "id2"]
)



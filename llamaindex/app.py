from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.core import Settings
from llama_index.core import StorageContext
from llama_index.core import VectorStoreIndex
from llama_index.llms.huggingface import HuggingFaceInferenceAPI
from llama_index.embeddings.langchain import LangchainEmbedding
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from llama_index.core import load_index_from_storage
import os
from huggingface_hub import login

#huggingface.co -> settings -> access token -> create new 
Hftoken = "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"
login(token=Hftoken)

persist_Dir = "./db"

llm1 = HuggingFaceInferenceAPI(
    model_name = "HuggingFaceH4/zephyr-7b-alpha", #what are the other alternatives? HuggingFaceH4/zephyr-7b-alpha, meta-llama/Llama-2-7b(request is submitted), google/gemma-7b
    api_key = Hftoken
)

embedmodel = LangchainEmbedding(
    HuggingFaceInferenceAPIEmbeddings(
        api_key=Hftoken,
        model_name= "thenlper/gte-large"  #what are the other alternatives?
    )
)

Settings.llm = llm1
Settings.embed_model = embedmodel
Settings.num_output = 512

if not os.path.exists(persist_Dir):
    #create new index
    document = SimpleDirectoryReader("data").load_data()
    #parse document into nodes
    parser = SimpleNodeParser()
    nodes = parser.get_nodes_from_documents(document)
    #serviceContext = Settings(llm=llm1, embed_model=embedmodel, chunk_size=512)
    storageContext = StorageContext.from_defaults()
    index = VectorStoreIndex(nodes,
                             storage_context= storageContext)  #service_context = serviceContext,
    index.storage_context.persist(persist_dir= persist_Dir)

else:
    #load the existing index
    #serviceContext = Settings(llm=llm1, embed_model=embedmodel, chunk_size=512)
    storageContext = StorageContext.from_defaults(persist_dir= persist_Dir)
    index = load_index_from_storage(storage_context = storageContext)  #service_context = serviceContext,


#query 
    
user_promt = "what is ikigai?"
queryengine = index.as_query_engine()
response = queryengine.query(user_promt)
print(user_promt)
print(response) 

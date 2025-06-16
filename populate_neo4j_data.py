from langchain_text_splitters import TokenTextSplitter
import requests
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
#from text_splitter import split_text
from config_loader import load_config
config = load_config()

openai_cfg = config["openai"]
neo4j_cfg = config["neo4j"]
file_cfg = config["filestorage"]

text = requests.get(file_cfg["file_url"]).text

# Save the text to a file
with open(file_cfg["local_path"], "w", encoding="utf-8") as file:
    file.write(text)

def get_documents_doc(text, chunk_size=1000, chunk_overlap=200):
    text_splitter = TokenTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    documents = []
    for raw_text in text_splitter.split_text(text):
        print(f"Chunk length: {len(raw_text)} characters")
        documents.append(Document(page_content=raw_text))
    
    return documents

# read the text from the file
text=""
with open(file_cfg["local_path"], "r", encoding="utf-8") as file:
    text = file.read()

print(f"Original text length: {len(text)} characters")

# Write a code to count the number of lines in the cleaned text
line_count = len(text.splitlines())
print(f"\nNumber of lines in the cleaned text: {line_count}")

documents_2 = get_documents_doc(text)

print(f"Number of documents created: {len(documents_2)}")


# Loading LLM graph tranformer from langchain
from langchain_openai import AzureOpenAI,ChatOpenAI,AzureChatOpenAI

#documents_2 = documents_2[:1] 


llm = AzureChatOpenAI(
    model_name=openai_cfg["model_name"],
    temperature=openai_cfg["temperature"],
    max_tokens=openai_cfg["max_tokens"],
    request_timeout=openai_cfg["request_timeout"],
    api_version=openai_cfg["api_version"],
    api_key=openai_cfg["api_key"],
    azure_endpoint=openai_cfg["azure_endpoint"],
)

from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_core.prompts import ChatPromptTemplate

llm_transformer = LLMGraphTransformer(llm=llm,node_properties=True,relationship_properties=True,)
print(f"LLMGraphTransformer initialized with model: {llm.model_name}")
# Managing each document seperatly to control unhandled exception from LLM
def convert_with_exception_handling(llm_transformer, documents):
    graph_documents = []
    failed_documents = []
    
    for i, doc in enumerate(documents):
        try:
            graph_data = llm_transformer.convert_to_graph_documents([doc])
            graph_documents.extend(graph_data)
            print(f"Successfully processed document {i+1}")
        except Exception as e:
            print(f"Failed to process document {i+1}: {str(e)}")
            failed_documents.append((i, doc, str(e)))
            continue 
    
    return graph_documents, failed_documents

# Transform the text into a graph
print("\nTransforming documents into graph data...")
print(len(documents_2), "documents to process")
graph_data,invalid_data = convert_with_exception_handling(llm_transformer, documents_2)

# Adding name property in each node. looks like name is mostly used for cypher query searches

for graph_doc in graph_data:
    for node in graph_doc.nodes:
        node.properties["name"] = node.id
        
print(f"\nNumber of nodes in the first graph document: {len(graph_data[0].nodes)}")
print(f"\nNumber of relationships in the first graph document: {len(graph_data[0].relationships)}")

# Connect with neo4j with Langchain Neo4j integration
from langchain_neo4j import Neo4jGraph 

neo_4j_graph = Neo4jGraph(
    url=neo4j_cfg["uri"],
    username=neo4j_cfg["username"],
    password=neo4j_cfg["password"],
    refresh_schema=True
)

# adding all graph nodes to neo4j graph
neo_4j_graph.add_graph_documents(graph_data,include_source=True,baseEntityLabel=True)

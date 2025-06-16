import semantic_search
import streamlit as st
from langchain_openai import AzureChatOpenAI
from langchain_community.graphs.neo4j_graph import Neo4jGraph
from langchain.prompts.prompt import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from typing import Tuple, List, Optional
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores.neo4j_vector import remove_lucene_chars
from langchain_core.runnables import ConfigurableField, RunnableParallel, RunnablePassthrough
from langchain_core.runnables import (RunnableBranch,RunnableLambda,RunnableParallel,RunnablePassthrough)
from langchain_core.output_parsers import StrOutputParser

from config_loader import load_config
config = load_config()

openai_cfg = config["openai"]
neo4j_cfg = config["neo4j"]

st.title("Ask question on book Pride and Prejudice")


llm = AzureChatOpenAI(
    model_name=openai_cfg["model_name"],
    temperature=openai_cfg["temperature"],
    max_tokens=openai_cfg["max_tokens"],
    request_timeout=openai_cfg["request_timeout"],
    api_version=openai_cfg["api_version"],
    api_key=openai_cfg["api_key"],
    azure_endpoint=openai_cfg["azure_endpoint"],
)

neo_4j_graph_store = Neo4jGraph(
    url=neo4j_cfg["uri"],
    username=neo4j_cfg["username"],
    password=neo4j_cfg["password"],
    refresh_schema=True,
)

# ------------------------------------------------------------------------
# ------------------------------------------------------------------------
# ------------------------------------------------------------------------


# Better way ... will full text search and its output for anwser

# run this if you create a new index
#neo_4j_graph_store.query(
    #"CREATE FULLTEXT INDEX entity IF NOT EXISTS FOR (e:__Entity__) ON EACH [e.id]")

# Extract entities from text
class Entities(BaseModel):
    """Identifying information about entities."""

    names: List[str] = Field(
        ...,
        description="All the person, organization, book, place, or business entities that appear in the text",
    )

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are extracting organization, book, place,  and person entities from the text.",
        ),
        (
            "human",
            "Use the given format to extract information from the following "
            "input: {question}",
        ),
    ]
)

entity_chain = prompt | llm.with_structured_output(Entities)


def generate_full_text_query(input: str) -> str:
    """
    Generate a full-text search query for a given input string.

    This function constructs a query string suitable for a full-text search.
    It processes the input string by splitting it into words and appending a
    similarity threshold (~2 changed characters) to each word, then combines
    them using the AND operator. Useful for mapping entities from user questions
    to database values, and allows for some misspelings.
    """
    full_text_query = ""
    words = [el for el in remove_lucene_chars(input).split() if el]
    for word in words[:-1]:
        full_text_query += f" {word}~2 AND"
    full_text_query += f" {words[-1]}~2"
    return full_text_query.strip()

# Fulltext index query
def structured_retriever(question: str) -> str:
    """
    Collects the neighborhood of entities mentioned
    in the question
    """
    result = ""
    entities = entity_chain.invoke({"question": question})
    for entity in entities.names:
        response = neo_4j_graph_store.query(
            """CALL db.index.fulltext.queryNodes('entity', $query, {limit:2})
            YIELD node,score
            CALL {
              WITH node
              MATCH (node)-[r:!MENTIONS]->(neighbor)
              RETURN node.id + ' - ' + type(r) + ' -> ' + neighbor.id AS output
              UNION ALL
              WITH node
              MATCH (node)<-[r:!MENTIONS]-(neighbor)
              RETURN neighbor.id + ' - ' + type(r) + ' -> ' +  node.id AS output
            }
            RETURN output LIMIT 50
            """,
            {"query": generate_full_text_query(entity)},
        )
        result += "\n".join([el['output'] for el in response])
    return result


def get_unstructured_data(query):
    results = semantic_search.semantic_search(query, top_k=3)
    similar_data = []
    for i, (text, score) in enumerate(results):
        similar_data.append(text)
    return similar_data

def retriever(question: str):
    print(f"Search query: {question}")
    structured_data = structured_retriever(question)
    
    unstructured_data = get_unstructured_data(question)
    final_data = f"""Structured data:
{structured_data}
Unstructured data:
{"#Document ". join(unstructured_data)}
    """
    return final_data

def retriever_no_vector(question: str):
    print(f"Search query: {question}")
    structured_data = structured_retriever(question)
    final_data = f"""Structured data:
{structured_data}
    """
    return final_data


_search_query = RunnableLambda(lambda x: x["question"])

template = """Answer the question based only on the following context:
{context}

Question: {question}
Use natural language and be concise.
Answer:"""
prompt = ChatPromptTemplate.from_template(template)

chain = (
    RunnableParallel(
        {
            "context": _search_query | retriever,
            "question": RunnablePassthrough(),
        }
    )
    | prompt
    | llm
    | StrOutputParser()
)

query = st.text_input("Enter your question:")

output_found = chain.invoke({"question": query})

if query:
    st.subheader("Answer:")
    st.write(output_found)
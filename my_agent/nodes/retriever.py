import os
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore

from state import State

OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')

QDRANT_URL = os.environ.get('QDRANT_URL')
QDRANT_API_KEY = os.environ.get('QDRANT_API_KEY')

embeddings = OpenAIEmbeddings(
    model='text-embedding-3-small',
    openai_api_key=OPENAI_API_KEY
)

vectorstore = QdrantVectorStore.from_existing_collection(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
    embedding=embeddings,
    collection_name='fantix-llc'
)

db_retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

def retriever(state: State):
    """
    Retrieve documents based on the user's question

    Args:
        state (State): Current state of the conversation
    Returns:
        dict: State with updated question and documents
    """

    question = state['question']
    documents = db_retriever.invoke(question)

    return {"question": question, "documents": documents}

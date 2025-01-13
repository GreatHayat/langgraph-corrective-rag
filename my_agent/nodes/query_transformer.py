import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from state import State

OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')


llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    api_key=OPENAI_API_KEY,
)

system_prompt = """Transform the user’s question into a concise, clear, and search-engine-friendly query that focuses\
on retrieving relevant and accurate information. Ensure the phrasing avoids ambiguity and includes keywords\
likely to yield the best search results."""

prompt = ChatPromptTemplate.from_messages([
    ('system', system_prompt),
    ('user', 'User Question: {question}')
])

chain = prompt | llm | StrOutputParser()

def transform_query(state: State):
    """
    Transform a user's question into a search-engine-friendly query

    Args:
        query (str): User's question
    Returns:
        str: Search-engine-friendly query
    """

    question = state['question']
    documents = state['documents']

    response = chain.invoke({"question": question})
    
    return {"question": response, "documents": documents}
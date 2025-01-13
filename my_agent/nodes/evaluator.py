import os
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from state import State

OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')

class RetrievalEvaluator(BaseModel):
    """
    A class to evaluate the relevance of documents retrieved from the vector store.
    """

    relevant: str = Field(description="Whether the document related to the user's question? 'yes' or 'no'")

llm = ChatOpenAI(
    openai_api_key=OPENAI_API_KEY,
    temperature=0,
    model='gpt-4o-mini'
).with_structured_output(RetrievalEvaluator)

system_prompt = """Evaluate the document to determine its relevance to the user's question. 
If the document includes keywords or conveys semantic meaning related to the user's question, return `yes`. 
Otherwise, return `no`. Only respond with `yes` or `no` to indicate the document's relevance.
"""

prompt = ChatPromptTemplate.from_messages([
    ('system', system_prompt),
    ('user', 'User Question: \\n\\n {question} \\n\\n Retrieved Document: \\n\\n {document}')
])

chain = prompt | llm

def document_evaluator(state: State):
    """
    Evaluate the relevance of the retrieved document

    Args:
        state (State): Current state of the conversation
    Returns:
        dict: State with updated answer and relevance evaluation
    """

    question = state['question']
    documents = state['documents']

    web_search = 'no'
    filtered_docs = []

    for document in documents:
        response = chain.invoke({"question": question, "document": document.page_content})

        if response.relevant == 'yes':
            filtered_docs.append(document)

    if len(filtered_docs) / len(documents) <= 0.7:
        web_search = 'yes'
        
    return {"question": question, "documents": filtered_docs, "web_search": web_search}
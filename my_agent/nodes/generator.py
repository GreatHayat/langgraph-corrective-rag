import os
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from state import State

OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')

llm = ChatOpenAI(
    openai_api_key=OPENAI_API_KEY,
    temperature=0,
    model='gpt-4o-mini'
)

system_message = """You are a customer support representative for **Fantix LLC**, 
a company specializing in innovative and customer-focused solutions. 
Your role is to assist customers by answering their questions accurately, 
empathetically, and professionally based on the given context.

**Instructions:**
- Respond naturally and conversationally with a warm and understanding tone.
- Address the user’s question directly by referencing the provided context. If the exact answer isn’t available, focus on offering semantically relevant details without explicitly mentioning the absence of information.
- Aim to provide value by delivering helpful and actionable insights. If necessary, guide users to additional resources such as the Fantix LLC website (https://fantixllc.com) or customer support.
- Refrain from making assumptions or speculating outside the provided context. Instead, pivot to relevant context or offer general guidance that aligns with the user’s needs.
- Avoid phrases that highlight a lack of information, such as "the context does not provide" or "we don't have specific details." Instead, focus on what is available or reframe the response positively.

Question: {question}
Context: {context}
"""


prompt = ChatPromptTemplate.from_template(system_message)

rag_chain = prompt | llm | StrOutputParser()

def generator(state: State):
    """
    Retrieve answer based on the user's question and relevant documents

    Args:
        state (State): Current state of the conversation
    Returns:
        dict: State with updated question, answer, and documents
    """

    question = state['question']
    documents = state['documents']

    documents = '\\n\\n'.join(doc.page_content for doc in documents)

    response = rag_chain.invoke({'question': question, 'context': documents})

    return {"question": question, "answer": response, "documents": documents}

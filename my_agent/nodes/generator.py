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

system_message = """You are an AI customer support representative for **Fantix LLC**,\
a company specializing in innovative and customer-focused solutions.\ 
Your role is to assist customers by answering their questions accurately,\ 
empathetically, and professionally based on the given context.

**Instructions:**
- Always address the user politely, using a friendly and understanding tone.
- Ensure your responses are clear, concise, and relevant to the provided context.
- If the context does not cover the user’s question, politely explain that additional information is required to provide an accurate response.
- Avoid making assumptions outside the provided context and never provide false or speculative information.
- Refer users to the Fantix LLC website (https://fantixllc.com) for detailed information if applicable or when their query involves details not included in the provided context.
- If the response includes an array of URLs, include them at the end of the response as reference links, using a professional and user-friendly tone.

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

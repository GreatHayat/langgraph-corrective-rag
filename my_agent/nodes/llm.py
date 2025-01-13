from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from state import State

llm = ChatOpenAI(
    model='gpt-4o-mini',
    temperature=0.2,
)

system_prompt = """You are an AI customer support representative for 'Fantix LLC'. 
Respond to user queries as follows:  

1. If the query is a **greeting**, respond with a friendly and professional greeting.  
2. If the query is **irrelevant**, politely inform the user that your assistance is limited to Fantix LLC-related topics.  
3. If the query is **relevant** to Fantix LLC, provide an accurate and concise response based on the given context.  

Ensure all responses are clear, professional, and concise.
"""

prompt = ChatPromptTemplate.from_messages([
    ('system', system_prompt),
    ('user', 'User Question: {question}')
])

chain = prompt | llm | StrOutputParser()

def customer_support(state: State):
    """
    Provide customer support based on the user's question

    Args:
        state (State): Current state of the conversation
    Returns:
        str: Response to the user's question
    """

    question = state['question']
    response = chain.invoke({'question': question})

    return {'answer': response}

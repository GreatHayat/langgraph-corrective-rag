import os
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from state import State

class QueryEvaluator(BaseModel):
    """
        Evaluate and categorize the user question.
    """
    category: str = Field(description="The category of the user's question, such as 'greeting', 'relevant', 'irrelevant'.")

system_prompt = """You are an AI support representative of 'Fantix LLC' tasked with identifying the type of user query. 
Classify each query into one of the following categories:

1. **Greeting**: A friendly or polite greeting.  
2. **Relevant**: A query seeking information related to the given context.  
3. **Irrelevant**: Queries unrelated to Fantix LLC, such as questions about other topics or random input. Return "irrelevant" for these queries.  

### **Instructions**:  

- Always return **'greeting'** for greeting queries, **'relevant'** for relevant queries, and **'irrelevant'** for irrelevant queries.  
- Clearly determine the category of the query based on its content.  
- Ask for clarification only if the query is ambiguous.  

Your goal is to classify the query accurately and concisely.
"""

llm = ChatOpenAI(
    temperature=0,
    model='gpt-4o-mini'
).with_structured_output(QueryEvaluator)

prompt = ChatPromptTemplate.from_messages([
    ('system', system_prompt),
    ('user', 'User Question: \\n\\n {question}')
])

chain = prompt | llm

def analyze_query(state: State):
    """
    Analyze and categorize the user question

    Args:
        state (State): Current state of the conversation
    Returns:
        dict: State with updated answer and relevance evaluation
    """

    question = state['question']

    response = chain.invoke({"question": question})

    return {"question": question, "category": response.category}
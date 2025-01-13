from langgraph.graph import StateGraph, START, END

from state import State
from nodes.retriever import retriever
from nodes.generator import generator
from nodes.evaluator import document_evaluator
from nodes.decision_maker import decide_to_go_next, decide_response_generator
from nodes.query_transformer import transform_query
from nodes.web_search import tavily_web_search
from nodes.query_analyzer import analyze_query
from nodes.llm import customer_support


graph_builder = StateGraph(State)

# Nodes
graph_builder.add_node('query_analyzer', analyze_query)
graph_builder.add_node('chatbot', customer_support)
graph_builder.add_node('retriever', retriever)
graph_builder.add_node('generator', generator)
graph_builder.add_node('evaluator', document_evaluator)
graph_builder.add_node('query_transformer', transform_query)
graph_builder.add_node('search_web', tavily_web_search)

# Edges
graph_builder.add_edge(START, 'query_analyzer')
graph_builder.add_conditional_edges(
    'query_analyzer',
    decide_response_generator,
    {
        "llm": 'chatbot',
        "retriever": 'retriever',
    }
)
graph_builder.add_edge('chatbot', END)
# graph_builder.add_edge(START, 'retriever')
graph_builder.add_edge('retriever', 'evaluator')

graph_builder.add_conditional_edges(
    'evaluator',
    decide_to_go_next,
    {
        "transform_query": 'query_transformer',
        "generator": 'generator',
    }
)

graph_builder.add_edge('query_transformer', 'search_web')
graph_builder.add_edge('search_web', 'generator')
graph_builder.add_edge('generator', END)

graph = graph_builder.compile()
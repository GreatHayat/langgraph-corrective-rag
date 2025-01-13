from typing import TypedDict
from langchain_core.documents import Document

class State(TypedDict):
    question: str
    category: str
    documents: list[Document]
    web_search: str
    answer: str
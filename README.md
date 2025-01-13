# Corrective RAG (CRAG)

**Corrective RAG (CRAG)** is a strategy for Retrieval-Augmented Generation (RAG) that integrates self-reflection and self-grading mechanisms to enhance the accuracy of responses by evaluating the relevance of retrieved documents.

<img width="1440" alt="Screenshot 2025-01-13 at 7 38 14 AM" src="https://github.com/user-attachments/assets/7bfadf1f-9d05-4f2c-9380-8d3a64984f55" />

**Reference & Inspiration:**  
[Learn more about CRAG in LangGraph](https://langchain-ai.github.io/langgraph/tutorials/rag/langgraph_crag/)

## Prerequisites

To get started with this project, ensure you have access to the following:  

- **OpenAI API** – For language model operations.  
- **Qdrant Vector Database** – For vector storage and retrieval.  
- **Tavily API Key** – Required for specific integrations.  

---

## Getting Started

Follow these steps to set up and run the project:  

1. **Set up a virtual environment**  
   ```bash
   python -m venv env
   source env/bin/activate  # For Linux/macOS
   env\Scripts\activate     # For Windows
   ```

2. **Install dependencies**  
   Navigate to the project directory and run:  
   ```bash
   pip install -r my-agents/requirements.txt
   ```

3. **Open in LangGraph Studio**  
   - Launch the **LangGraph Studio** desktop application.  
   - Open the project folder within the application.  


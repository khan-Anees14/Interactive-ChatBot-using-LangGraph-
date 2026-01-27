from __future__ import annotations

import os
import tempfile
from typing import TypedDict, Annotated, Any, Dict, Optional

from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, SystemMessage

from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3

from langgraph.prebuilt import ToolNode, tools_condition    # used to add edges between tools and nodes
from langchain_community.tools import DuckDuckGoSearchRun   # this is a web search tools
from langchain_core.tools import tool   # to make custom tools
import requests

load_dotenv()

# ------------------------------------------
# 1. LLM + embeddings
# ------------------------------------------
llm = ChatOllama(model='llama3', temperature=0)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


# ------------------------------------------
# 2. PDF retriever store (per thread)
# ------------------------------------------
_THREAD_RETRIEVERS: Dict[str, Any] = {}
_THREAD_METADATA: Dict[str, Any] = {}


def _get_retriever(thread_id: Optional[str]):
    """Fetch the retriever for a thread if available."""
    if thread_id and thread_id in _THREAD_RETRIEVERS:
        return _THREAD_RETRIEVERS[thread_id]
    return None


def ingest_pdf(file_bytes: bytes, thread_id: str, filename: Optional[str] = None) -> dict:
    """
    Build a FAISS retriever for the uploaded PDF and store it for the thread.

    Returns a summary dict that can be surfaced in the UI.
    """
    if not file_bytes:
        raise ValueError("No bytes revieved for ingestion.")
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        tempfile.write(file_bytes)
        temp_path = temp_file.name

    try:
        loader = PyPDFLoader(temp_path)
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size = 1000, chunk_overlap=200, separators=["\n\n", " ", ""]
        )
        chunks = splitter.split_documents(docs)

        vecotr_store = FAISS.from_documents(chunks,embeddings )
        retriever = vecotr_store.as_retriever(search_type="similarity", search_kwargs={"k":4})

        _THREAD_RETRIEVERS[str(thread_id)] = retriever
        _THREAD_METADATA[str(thread_id)] = {
            "filename": filename or os.path.basename(temp_path),
            "documents": len(docs),
            "chunks": len(chunks),
        }

        return {
            "filename": filename or os.path.basename(temp_path),
            "documents": len(docs),
            "chunks": len(chunks),
        }
    finally:
        # the FAISS store keeps copies of the text, so the temp file is sage to remove.
        try:
            os.remove(temp_path)
        except OSError:
            pass

# -----------------------------------------------------
# 3. Tools conditions, making and edge connections
# ------------------------------------------------------

search_tool = DuckDuckGoSearchRun(region='us-en')

@tool
def calculator(firs_num: float, second_num: float, operation: str) -> dict:
    """
    Perform a basic arithemetic operation an two numbers.
    Suppoeted operations: add, sub, mul, div
    """
    try:
        if operation == 'add':
            result = firs_num + second_num
        elif operation == 'sub':
            result = firs_num - second_num
        elif operation == 'mul':
            result = firs_num * second_num
        elif operation == 'div':
            if second_num == 0:
                return {"error": "Division by zero is not allowed"}
            result = firs_num / second_num
        else:
            return {"error": f"Unsuupported operation '{operation}' "}
        return {"first_num": firs_num, "second_num": second_num, "operation": operation, "result": result}
    
    except Exception as e:
        return {"error": str(e)}
    

@tool
def get_stock_price(symbol: str) -> dict:
    """
    Fetch latest stock price for a given symbol (e.g. 'AAPL', 'TSLA')
    using Alpha Vantage with API key in the URL.
    """
    url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey=9YHECAKUSB2AJSC0"
    r = requests.get(url)
    return r.json()

@tool
def rag_tool(query: str, thread_id: Optional[str] = None) -> dict:
    """
    Retrieve relevant information from the uploaded PDF for this chat thread.
    Always include the thread_id when calling this tool.
    """ 
    retriever = _get_retriever(thread_id)
    if retriever is None:
        return {
            "error": "No document indexed for this chat. Upload a PDF first.",
            "query": query,
        }
    
    result = retriever.invoke(query)
    context = [doc.page_content for doc in result]
    metadata = [doc.metadata for doc in result]

    return {
        "query": query,
        "context": context,
        "metadata": metadata,
        "source_file": _THREAD_METADATA.get(str(thread_id), {}).get("filename"),
    }


# tool list
tools = [get_stock_price, search_tool, calculator, rag_tool]

# make the LLM tool-aware
#llm_with_tools = llm.bind_tools(tools)


# -----------------------------------------------------
# 4. State
# ------------------------------------------------------
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


# ------------------------------------------------------
# 5. Nodes
# ------------------------------------------------------
def chat_node(state: ChatState, config=None):
    """LLM node that may answer or request a tool call."""
    thread_id = None
    if config and isinstance(config, dict):
        thread_id = config.get("configurable", {}).get("thread_id")

    system_message = SystemMessage(
        content=(
            "You are a helpful assistant. For questions about the uploaded PDF, call "
            "the `rag_tool` and include the thread_id "
            f"`{thread_id}`. You can also use the web search, stock price, and "
            "calculator tools when helpful. If no document is available, ask the user "
            "to upload a PDF."
        )
    )

    messages = [system_message, *state["messages"]]
    response = llm.invoke(messages, config=config)
    return {"messages": [response]}


tool_node = ToolNode(tools)

# ------------------------------------------------------
# 6. Checkpointer
# ------------------------------------------------------
conn = sqlite3.connect(database="chatbot.db", check_same_thread=False)
checkpointer = SqliteSaver(conn=conn)

# ------------------------------------------------------
# 7. Graph
# ------------------------------------------------------
graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_node("tools", tool_node)

graph.add_edge(START, "chat_node")
graph.add_conditional_edges("chat_node", tools_condition)
graph.add_edge("tools", "chat_node")

chatbot = graph.compile(checkpointer=checkpointer)

# ------------------------------------------------------
# 8. Helpers
# ------------------------------------------------------
def retrieve_all_threads():
    all_threads = set()
    for checkpoint in checkpointer.list(None):
        all_threads.add(checkpoint.config["configurable"]["thread_id"])
    return list(all_threads)


def thread_has_document(thread_id: str) -> bool:
    return str(thread_id) in _THREAD_RETRIEVERS


def thread_document_metadata(thread_id: str) -> dict:
    return _THREAD_METADATA.get(str(thread_id), {})
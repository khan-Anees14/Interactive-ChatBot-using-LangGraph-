
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
# from langchain_community.chat_models import ChatOllama
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, BaseMessage
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3

from dotenv import load_dotenv
import requests

load_dotenv()

llm = ChatOllama(
    model='llama3',
    temperature=0
)

# ### Tools conditions, making and edge connections

from langgraph.prebuilt import ToolNode, tools_condition    # used to add edges between tools and nodes
from langchain_community.tools import DuckDuckGoSearchRun   # this is a web search tools
from langchain_core.tools import tool   # to make custom tools

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


# tool list
tools = [get_stock_price, search_tool, calculator]

# make the LLM tool-aware
#llm_with_tools = llm.bind_tools(tools)

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


# graph nodes
def chat_node(state: ChatState):
    """LLM node that may answer or request a tool call."""
    messages = state['messages']
   # response = llm_with_tools.invoke(messages)
    response = llm.invoke(messages)
    return {'messages': [response]}

tool_node = ToolNode(tools)     # Executes tools calls


# ********************* checkpointer ***************************
conn = sqlite3.connect(database='chatbot.db', check_same_thread=False)
checkpointer = SqliteSaver(conn= conn)


graph = StateGraph(ChatState)

graph.add_node("chat_node", chat_node)
graph.add_node("tools", tool_node)


graph.add_edge(START, "chat_node")

# route to tool node if tool requested
graph.add_conditional_edges("chat_node", tools_condition)

# AFTER tool execution → back to chat
graph.add_edge("tools", "chat_node")

#graph.add_edge("chat_node", END)

chatbot = graph.compile(checkpointer=checkpointer)



# ************************ Helper **********************************
def retrieve_all_thread():
    all_thread = set()
    for checkpointer in checkpointer.list(None):
        all_thread.add(checkpointer.config["configurable"]["thread_id"])
    return list(all_thread)




# from langchain_core.messages import HumanMessage

# chatbot.invoke(
#     {
#         "messages": [
#             HumanMessage(content="What is 25 * 4?")
#         ]
#     },
#     config={"configurable": {"thread_id": "t1"}}
# )

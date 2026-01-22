from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage
from langchain.chat_models import ChatOllama

# to store the memory
# from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver     # to store messages in sql database
import sqlite3        

## *************define llm*************************
llm  = ChatOllama(
    model = 'llama3',
    temperature = 0
)

# ************to add messages iteratively one after other***********************
from langgraph.graph.message import add_messages

class ChatState(TypedDict):

    messages: Annotated[list[BaseMessage], add_messages]      # base message add felxibilty as it conatins all HUman message, system message, llm message etc


# *****************define the chatting node ************************************
def chat_node(state: ChatState):

    # Take user query from state
    messages = state['messages']

    # send to llm
    response = llm.invoke(messages)

    # Response store state
    return {'messages': response}


conn = sqlite3.connect(database='chatbot.db', check_same_thread=False)
# ********************* Now define the Graph from here ***********************************
check_pointer = SqliteSaver(conn=conn)   # save the memory for checking, updating past things

graph = StateGraph(ChatState)

# add node
graph.add_node('chat_node', chat_node)

graph.add_edge(START, 'chat_node')
graph.add_edge('chat_node', END)

chatbot = graph.compile(checkpointer=check_pointer)     # add check pointer here while compiling

def retrieve_all_threads():
    all_threads = set()
    for checkpoint in check_pointer.list(None):
        all_threads.add(checkpoint.config['configurable']['thread_id'])

    return list(all_threads)

from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage
from langchain.chat_models import ChatOllama

# to store the memory
from langgraph.checkpoint.memory import MemorySaver

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


# ================================
# #Now define the Graph from here
# ================================
check_pointer = MemorySaver()   # save the memory for checking, updating past things

graph = StateGraph(ChatState)

# add node
graph.add_node('chat_node', chat_node)

graph.add_edge(START, 'chat_node')
graph.add_edge('chat_node', END)

chatbot = graph.compile(checkpointer=check_pointer)     # add check pointer here while compiling
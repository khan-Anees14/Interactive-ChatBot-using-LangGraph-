🤖 Streamlit LangGraph Chatbot

A simple yet powerful Streamlit-based chatbot built using LangGraph and LangChain. This project demonstrates how to maintain chat history using Streamlit session state and how to invoke a backend conversational graph for generating AI responses.

✨ Features

🧠 LangGraph-powered backend for structured conversational flows

💬 Persistent chat history using st.session_state

⚡ Real-time chat UI using Streamlit’s st.chat_message

🔄 Thread-based conversation handling using configurable IDs

🛠️ Clean and minimal codebase – easy to extend

🧩 How It Works

User enters a message in the Streamlit chat input.

The message is stored in st.session_state to persist chat history.

The input is sent to the LangGraph chatbot backend.

The backend processes the message and returns an AI response.

The response is displayed and stored for future context.


The langgraph_backend.py file defines:

Nodes (LLM calls, tools, logic)

State transitions

Message handling logic

This allows:

Deterministic conversation flows

Easy extension to tools, RAG, or agents

🌱 Future Enhancements

🔍 Retrieval-Augmented Generation (RAG)

🧾 Chat export (PDF / TXT)

🧑‍💼 Role-based agents

💾 Persistent database-backed memory

🔐 Authentication and user sessions

🙋‍♂️ Author

Mohmmad Anish
AI & ML Enthusiast | LangChain | LangGraph | Streamlit

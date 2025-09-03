from langchain_google_genai import GoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain.vectorstores.chroma import Chroma
from get_embedding import get_embedding_function
# import streamlit as st
from dotenv import load_dotenv
from langchain.memory import ConversationBufferWindowMemory
from typing import Optional


load_dotenv()

CHROMA_PATH = r"E:\Chatbot_GUB\chroma_db"

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Current conversation:
{history}

---

You are a chatbot. Answer the following question in a natural, human-like way, using only the information in the context.

Question: {question}
"""

##================================ FastAPI ================================#

# Global variables to store the database and LLM instance
db_instance = None
llm_instance = None
history_memory = {} # A dictionary to store memory for different sessions/users

def get_db():
    global db_instance
    if db_instance is None:
        embedding_function = get_embedding_function()
        db_instance = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
    return db_instance

def get_llm():
    global llm_instance
    if llm_instance is None:
        llm_instance = GoogleGenerativeAI(model="gemini-1.5-flash")
    return llm_instance

def get_memory_for_session(session_id: Optional[str] = "default"):
    if session_id not in history_memory:
        history_memory[session_id] = ConversationBufferWindowMemory(k=5)
    return history_memory[session_id]

def query_rag_with_history(query_text: str, session_id: Optional[str] = "default"):
    db = get_db()
    llm = get_llm()
    memory = get_memory_for_session(session_id)

    # Search the vector store for context
    results = db.similarity_search_with_score(query_text, k=5)
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _ in results])

    # Get the formatted history from memory
    history_str = memory.load_memory_variables({})['history']

    # Create the prompt with all variables
    prompt = PromptTemplate.from_template(PROMPT_TEMPLATE)
    formatted_prompt = prompt.format(
        context=context_text,
        history=history_str,
        question=query_text
    )

    # Now, invoke the LLM with this manually constructed prompt
    response = llm.invoke(formatted_prompt)

    # Update the memory with the new user input and AI response
    memory.save_context({"input": query_text}, {"output": response})

    # Sources
    sources = [doc.metadata.get("id", None) for doc, _ in results]

    return {"response": response, "sources": sources}




##================================ Streamlit Version ================================#

# def query_rag_with_history(query_text: str):
#     # Load DB and LLM once per session
#     if "db" not in st.session_state:
#         embedding_function = get_embedding_function()
#         st.session_state.db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
    
#     if "llm" not in st.session_state:
#         st.session_state.llm = GoogleGenerativeAI(model="gemini-1.5-flash")
    
#     # Initialize or retrieve conversation history
#     if "history_memory" not in st.session_state:
#         st.session_state.history_memory = ConversationBufferWindowMemory(k=5)

#     # Search the vector store for context
#     results = st.session_state.db.similarity_search_with_score(query_text, k=5)
#     context_text = "\n\n---\n\n".join([doc.page_content for doc, _ in results])

#     print("Retrieved Context:", context_text) 

#     # Get the formatted history from memory
#     history_str = st.session_state.history_memory.load_memory_variables({})['history']
    
#     # Create the prompt with all variables
#     prompt = PromptTemplate.from_template(PROMPT_TEMPLATE)
#     formatted_prompt = prompt.format(
#         context=context_text,
#         history=history_str,
#         question=query_text
#     )

#     # Now, invoke the LLM with this manually constructed prompt
#     response = st.session_state.llm.invoke(formatted_prompt)
    
#     # Update the memory with the new user input and AI response
#     st.session_state.history_memory.save_context({"input": query_text}, {"output": response})

#     # Sources
#     sources = [doc.metadata.get("id", None) for doc, _ in results]

#     return response, sources


# # ---------------- Streamlit UI ----------------
# def main():
#     st.set_page_config(page_title="RAG with Chroma + Gemini", layout="wide")
#     st.title("RAG with Chroma + Google Generative AI")

#     # Initialize chat history
#     if "messages" not in st.session_state:
#         st.session_state.messages = []

#     for message in st.session_state.messages:
#         with st.chat_message(message["role"]):
#             st.markdown(message["content"])

#     if query_text := st.chat_input("🔍 Enter your query:"):
#         with st.chat_message("user"):
#             st.markdown(query_text)
#         st.session_state.messages.append({"role": "user", "content": query_text})

#         with st.spinner("Thinking..."):
#             response, sources = query_rag_with_history(query_text)

#         with st.chat_message("assistant"):
#             st.markdown(response)
#             st.markdown(f"📖 Sources: {sources}")
        
#         st.session_state.messages.append({"role": "assistant", "content": response})

# if __name__ == "__main__":
#     main()
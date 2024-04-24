import streamlit as st
from langchain.prompts import PromptTemplate
from langchain_community.chat_models.openai import ChatOpenAI
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from utils import *
from streamlit_chat import message


# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Streamlit page configuration
st.title("Arabic Rag")

# Initialize or retrieve the vector database from session state
if 'db' not in st.session_state:
    st.session_state['db'] = []

# Session state to store responses
if 'response' not in st.session_state:
    st.session_state['response'] = []

# File upload section
uploaded_files = st.file_uploader("Upload your files", type=['pdf'], accept_multiple_files=True)  # Allow multiple PDF files
if uploaded_files:
    files_info = []
    for uploaded_file in uploaded_files:
        # Save each uploaded file to a temporary file (assuming server has necessary permissions)
        temp_file_path = f'temp_{uploaded_file.name}'
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        files_info.append({
            "file_path": temp_file_path,
            "file_name": uploaded_file.name
        })

    # Process the files and create/update vector database in session
    if not st.session_state['db']:
        create_vector_db_for_files(files_info)
        embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
        db = load_local_vectordb_using_qdrant("testing_arabic", embeddings)
        st.session_state['db'] = db
        st.success('Database updated with uploaded files!')

# User input for queries
user_prompt = st.text_input('Enter Your Query here..........')

# Button to trigger the response generation
if st.button("Press Enter"):
    if st.session_state['db']:
        st.session_state['response'] = arabic_qa(user_prompt, st.session_state['db'])
    else:
        st.error("No database found. Please upload a file first.")

# Display the response
if st.session_state['response']:
    message(st.session_state['response'])

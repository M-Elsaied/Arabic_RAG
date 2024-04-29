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
        # create_vector_db_for_files(files_info)
        embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
        db_files = load_local_vectordb_using_qdrant("testing_arabic", embeddings)
        db_tables = load_local_vectordb_using_qdrant("tables_arabic", embeddings)
        st.session_state['db'] = db_files
        st.session_state['db_tables'] = db_tables
        st.success('Database updated with uploaded files!')

# User input for queries
user_prompt = st.text_input('Enter Your Query here...')

# Button to trigger the response generation
if st.button("Press Enter"):
    if st.session_state['db']:
        st.info("""Processing your query, please be patient. This might take some time.

                إن الله مع الصابرين""")

        # st.session_state['response'] = arabic_qa(user_prompt, st.session_state['db'])
        response, reference_links = synthesize_responses(user_prompt, st.session_state['db'], st.session_state['db_tables'])

        # Save the response and links in the session state
        st.session_state['response'] = response
        st.session_state['reference_links'] = reference_links
    else:
        st.error("No database found. Please upload a file first.")

# Function to get the base64 encoding of the PDF file
def get_pdf_base64(file_path):
    with open(file_path, "rb") as pdf_file:
        encoded = base64.b64encode(pdf_file.read()).decode('utf-8')
    return encoded

# Display the response
if 'response' in st.session_state and st.session_state['response']:
    st.markdown(st.session_state['response'])

    if 'reference_links' in st.session_state and st.session_state['reference_links']:
        st.markdown("## Reference Document:")
        # Get the first link from the list
        link = st.session_state['reference_links'][0]

        # Extract the document name and page number from the link
        start = link.find('(') + 1
        end = link.find(')')
        url = link[start:end]
        doc_path, page = url.split('#page=')
        full_path = os.path.join('docs', doc_path.replace('./docs/', ''))  # Correct the path as needed

        # Check if the PDF file exists
        if os.path.exists(full_path):
            base64_pdf = get_pdf_base64(full_path)
            pdf_display = f'<embed src="data:application/pdf;base64,{base64_pdf}#page={page}" width="700" height="1000" type="application/pdf">'
            st.markdown(pdf_display, unsafe_allow_html=True)
        else:
            st.error(f"File not found: {full_path}")



# Button to trigger the response generation
# if st.button("Press Enter"):
#     if 'db' in st.session_state and 'db_tables' in st.session_state:
#         # Get response from a hard ensemble model
#         hybrid_response = arabic_qa_files_and_tables(user_prompt, st.session_state['db'], st.session_state['db_tables'])

#         # Get response from a hybrid model using both databases
#         hard_ensemble_response= synthesize_responses(user_prompt, st.session_state['db'], st.session_state['db_tables'])

#         # Save responses to session state
#         st.session_state['hard_ensemble_response'] = hard_ensemble_response
#         st.session_state['hybrid_response'] = hybrid_response
#     else:
#         st.error("No database found. Please upload a file first.")

# # Display the responses
# if 'hard_ensemble_response' in st.session_state:
#     st.subheader("Response from Hard Ensemble Model:")
#     st.write(st.session_state['hard_ensemble_response'])

# if 'hybrid_response' in st.session_state:
#     st.subheader("Response from Hybrid Model:")
#     st.write(st.session_state['hybrid_response'])

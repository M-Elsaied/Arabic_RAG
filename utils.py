import openai
import os
from langchain.prompts import ChatPromptTemplate
from langchain_community.chat_models import ChatOpenAI
from qdrant_client import QdrantClient
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import  RunnablePassthrough, RunnableParallel
import logging
from langchain_community.document_loaders import  Docx2txtLoader,PyMuPDFLoader
from langchain_community.vectorstores import Qdrant
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.docstore.document import Document
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()
openai.api_key  = os.getenv("OPENAI_API_KEY")
qdrant_url  = os.getenv('QDRANT_URI')
qdrant_api_key  = os.getenv('QDRANT_API_KEY')
from langchain_openai import OpenAIEmbeddings


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter(
    '%(asctime)s:%(name)s:%(levelname)s:%(message)s:%(funcName)s')
file_handler = logging.FileHandler('arabic_utils.log')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

class IncomingFileProcessor():
  def __init__(self, chunk_size=1000, chunk_overlap=200) -> None:
    self.chunk_size = chunk_size
    self.chunk_overlap = chunk_overlap

  def get_pdf_splits(self, pdf_file: str, filename:str):
    try:
      loader = PyMuPDFLoader(pdf_file)
      pages = loader.load()
      logger.info("Succesfully loaded the pdf file")
      textsplit = RecursiveCharacterTextSplitter(
        separators=["\n\n",".","\n"],
        chunk_size = self.chunk_size,
        chunk_overlap = self.chunk_overlap,
        length_function=len
      )
      doc_list = []
      for pg in pages:
        pg_splits = textsplit.split_text(pg.page_content)
        for pg_sub_split in pg_splits:
          pg_sub_split = clean_text(pg_sub_split)
          metadata = {'source': filename}
          doc_string = Document(page_content=pg_sub_split, metadata = metadata)
          doc_list.append(doc_string)
      logger.info("Successfully split the PDF file")
      return doc_list
    except:
      logger.critical(f"Error in loading pdf file:{str(e)}")
      raise Exception(str(e))
def create_vector_db_for_files(files_info):
  res = []
  try:
    for f_info in files_info:
      path = f_info["file_path"]
      filename = f_info["file_name"]
      file_extension = filename.split(".")[-1]
      logger.info(f"processing file: {filename} ")
      all_texts = load_data(path, file_extension, filename)
      if not all_texts:
        logger.warning(f"No text in the file: {filename}")
        continue

      embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
      collection_name = "testing_arabic"
      logger.info(f"Creating vectordb for file: {filename}")
      create_new_vectorstore_qdrant(all_texts, embeddings, collection_name, qdrant_url, qdrant_api_key)
      logger.info("Vectordb stored successfully on filesystem")
      res.append({
        "filename": filename,
        "collection_name": collection_name
      })
  except AssertionError as error:
    logger.critical(f"file doesn't contain any texts or maybe corrupted {error}")
  return res

def load_local_vectordb_using_qdrant(vectordb_folder_path, embed_fn):
  qdrant_client = QdrantClient(
    url = qdrant_url,
    api_key = qdrant_api_key
  )
  qdrant_store = Qdrant(qdrant_client, vectordb_folder_path, embed_fn)
  return qdrant_store



def load_data(file_path, file_extension, file_name):
  if file_extension.lower() == "pdf":
    logger.info("Enter in pdf file loader")
    file_processor = IncomingFileProcessor()
    texts = file_processor.get_pdf_splits(str(Path(__file__).parent.joinpath( file_path)), file_name)
    os.remove(file_path)
    logger.info("Successfully remove the pdf file")
    return texts

def create_new_vectorstore_qdrant(doc_list, embed_fn, COLLECTION_NAME, qdrant_url, qdrant_api_key):
  try:
    logger.info(f"Initializing Qdrant DB creation for collection: {COLLECTION_NAME}")
    qdrant = Qdrant.from_documents(
      documents=doc_list,
      embedding=embed_fn,
      url=qdrant_url,
      prefer_grpc=False,
      api_key=qdrant_api_key,
      collection_name=COLLECTION_NAME,
    )
    logger.info("Successfully created the vector DB")
    return qdrant
  except Exception as ex:
    logger.critical("Vectordb Failed: " + str(ex))
    raise Exception({"Error": str(ex)})


def arabic_qa(query, vectorstore):
  try:
    num_chunks = 3
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": num_chunks})
    template = """
      أنت مساعد مفيد وصادق ومتخصص في الاقتصاد والأوراق المالية، حاول على قدرالمستطاع أن تجاوب بصدق علما أن المعلومات التالية متوفرة لك:
        {context}

        السؤال : {question}
    """
    prompt = ChatPromptTemplate.from_template(template)
    setup_and_retrieval = RunnableParallel(
      {"context": retriever, "question": RunnablePassthrough()}
    )
    model = ChatOpenAI(model = "gpt-4", openai_api_key = os.getenv("OPENAI_API_KEY"), temperature=0.3)
    output_parser = StrOutputParser()
    #chain = setup_and_retrieval | prompt | model | output_parser
    context = setup_and_retrieval.invoke(query)
    prompt_answer = prompt.invoke(context)
    model_answer = model.invoke(prompt_answer)
    response = output_parser.invoke(model_answer)
    return response
  except Exception as e:
    raise Exception('open AI Key error')

#PSYNTHESIZED ANSWER
def synthesize_responses(query, db_files, db_tables):
    try:
        query = clean_text(query)
        # Get answers from files vector store
        response_files = arabic_qa(query, db_files)

        # Get answers from tables vector store
        response_tables = arabic_qa(query, db_tables)

        # Prepare the combined context for synthesis
        combined_answer = f"Response from files data: {response_files}\n\nResponse from tables data: {response_tables}"

        # Synthesis prompt
        synthesis_template = """
        Considering the information provided from different sources, synthesize a comprehensive response:
        {combined_answers}
        """

        # Setup model and generate synthesized response
        model = ChatOpenAI(model="gpt-4", openai_api_key=os.getenv("OPENAI_API_KEY"), temperature=0.5)
        synthesis_prompt = synthesis_template.format(combined_answers=combined_answer)
        synthesized_prompt_answer = model.invoke(synthesis_prompt)
        # Parse the synthesized response
        output_parser = StrOutputParser()
        synthesized_response = output_parser.invoke(synthesized_prompt_answer)

        return synthesized_response
    except Exception as e:
       raise Exception(f'An error occurred during the synthesis process: {e}')

#hybrid prompt

def arabic_qa_files_and_tables(query, db_files, db_tables):
    try:
        query = clean_text(query)
        num_chunks = 3
        # Create retrievers for both vector stores
        files_retriever = db_files.as_retriever(search_type="similarity", search_kwargs={"k": num_chunks})
        tables_retriever = db_tables.as_retriever(search_type="similarity", search_kwargs={"k": num_chunks})

        # Template for generating prompts
        template = """
            أنت مساعد مفيد وصادق ومتخصص في الاقتصاد والأوراق المالية، حاول على قدر المستطاع أن تجاوب بصدق علما أن المعلومات التالية متوفرة لك:
            {context}

            السؤال: {question}
        """
        prompt = ChatPromptTemplate.from_template(template)

        # Setup for parallel retrieval
        setup_and_retrieval = RunnableParallel(
            {"files_context": files_retriever, "tables_context": tables_retriever, "question": RunnablePassthrough()}
        )

        # Invoke the retrievers with the query and handle the result
        retrieval_results = setup_and_retrieval.invoke({"query": query, "question": query})

        # Combining contexts from both retrievals
        context_combiner = lambda results: f"{results['files_context']} \n\n {results['tables_context']}"
        combined_context = context_combiner(retrieval_results)

        # Generate the complete prompt and get the response
        prompt_answer = prompt.invoke({"context": combined_context, "question": query})
        model = ChatOpenAI(model="gpt-4", openai_api_key=os.getenv("OPENAI_API_KEY"), temperature=0.3)
        model_answer = model.invoke(prompt_answer)

        # Parse the output to get a clean response
        output_parser = StrOutputParser()
        response = output_parser.invoke(model_answer)

        return response
    except Exception as e:
        raise Exception(f'An error occurred during the Q&A process: {str(e)}')



##########################TEXT PREPROCESSING AND CLEANING##########################
import re
# import nltk
# nltk.download('punkt')
# nltk.download('wordnet')
# nltk.download('omw-1.4')
def normalize_whitespace(text):
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with a single space
    return text.strip()
def remove_headers_footers(text):
    # Example pattern: headers or footers to identify and remove
    header_footer_pattern = re.compile(r'Page \d+ of \d+|Confidential')
    return header_footer_pattern.sub('', text)
def remove_punctuation(text):
    # Remove all characters except words and space
    return re.sub(r'[^\w\s]', '', text)
def to_lowercase(text):
    return text.lower()
def remove_non_textual_elements(text):
    # Replace or remove specific non-text patterns
    return re.sub(r'[\u200b-\u200f\u202a-\u202e]', '', text)  # Removes zero-width spaces and similar characters
def remove_arabic_diacritics(text):
    # Arabic specific diacritics removal
    diacritics_pattern = re.compile(r'[\u064B-\u065F\u06D6-\u06DC\u06DF-\u06E8\u06EA-\u06ED]')
    return diacritics_pattern.sub('', text)
# from nltk.corpus import stopwords
# from nltk.tokenize import word_tokenize

def remove_stop_words(text, language='arabic'):
    words = word_tokenize(text)
    stop_words = set(stopwords.words(language))
    return ' '.join([word for word in words if word not in stop_words])
def clean_text(text):
    text = normalize_whitespace(text)
    text = remove_headers_footers(text)
    text = remove_punctuation(text)
    text = to_lowercase(text)
    text = remove_non_textual_elements(text)
    text = remove_arabic_diacritics(text)
    # text = remove_stop_words(text)  # Optional: Use if stop words are to be removed
    return text
################TABLES PROCESSED AS IMAGES#########################
import os
import base64
import requests
from utils import clean_text

def process_images(image_dir, api_key):
    # Function to encode the image
    def encode_image(image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    # Headers for the API request
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    # Data dictionary to hold the cleaned responses
    data = {}

    # Iterate over each image in the directory
    for filename in os.listdir(image_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):  # Filtering image files
            image_path = os.path.join(image_dir, filename)
            base64_image = encode_image(image_path)

            # Payload for each image
            payload = {
                "model": "gpt-4-turbo",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "افترض أنك ملخص مالي وقُدمت لك هذه الصورة، مهمتك هي تلخيص المعلومات الموجودة في هذا الجدول بطريقة موجزة بحيث يمكن استخدام هذه المعلومات في المستقبل. يجب أن يكون ملخصك دقيقًا ويشمل كل شيء."
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ],
                "max_tokens": 1000
            }

            # Send the request
            response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
            cleaned_content = clean_text(response.json()['choices'][0]['message']['content'])
            data[filename] = cleaned_content

    return data

#create db for image

def create_vector_db_for_table_data(table_data):
    """
    Create a vector database for provided text data.

    Args:
    text_data (dict): Dictionary with filenames as keys and text content as values.
    qdrant_url (str): URL to the Qdrant service.
    qdrant_api_key (str): API key for Qdrant.

    Returns:
    list: A list containing details about the processed text data collections.
    """
    results = []
    try:
        for filename, text in table_data.items():
            logger.info(f"Processing text for image: {filename}")
            if not text:
                logger.warning(f"No text available for image: {filename}")
                continue

            embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
            collection_name = "tables_arabic"
            logger.info(f"Creating vectordb for image: {filename}")
            create_new_vectorstore_qdrant([text], embeddings, collection_name, qdrant_url, qdrant_api_key)
            logger.info("Vectordb stored successfully on filesystem")
            results.append({
                "filename": filename,
                "collection_name": collection_name
            })
    except Exception as error:
        logger.critical(f"Error processing text data: {error}")
    return results
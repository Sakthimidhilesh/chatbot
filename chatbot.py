import streamlit as st
from PyPDF2 import PdfReader
from langchain.chains.question_answering.map_rerank_prompt import output_parser
from langchain.text_splitter import RecursiveCharacterTextSplitter
#from langchain.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

from langchain_community.chat_models import ChatOpenAI

#upload pdf files
st.header("document chatbot")

with st.sidebar:
    st.title("your document")
    file = st.file_uploader("upload the pdf file and ask questions", type="pdf")

#expract the text
if file is not None:
    pdf_reader = PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text+=page.extract_text()


#break into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        separators="\n",
        chunk_size=100,
        chunk_overlap=5,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    #generating embedding
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    #CREATING VECTOR STORE FAISS
    vector_store = FAISS.from_texts(chunks,embeddings)
    #user question
    user_question = st.text_input("ask your question here")
    #similarity search
    if user_question:
        match = vector_store.similarity_search(user_question)
        #define llm
        llm = ChatGroq (
            groq_api_key = "your api key",
            temperature = 0,
            max_tokens = 1000,
            model_name = "llama-3.1-8b-instant"
        )
        #output result
        chain = load_qa_chain(llm, chain_type = "stuff")
        response = chain.run(input_documents = match, question = user_question)
        st.write(response)

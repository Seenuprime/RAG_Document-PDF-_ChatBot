import os
import streamlit as st
from dotenv import load_dotenv
load_dotenv()

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_huggingface import HuggingFaceEmbeddings

# os.environ['GROQ_API_KEY'] = os.getenv("GROQ_API_KEY")
api = os.getenv('GROQ_API_KEY')
os.environ['HUGGING_FACE'] = os.getenv('HUGGING_FACE')
llm = ChatGroq(model='Llama3-8b-8192', groq_api_key=api)

prompt = ChatPromptTemplate.from_template(
    """
    Answer the question based on the given context.
    please provide the most accurate response based on the question.
    <context>{context}</context> <question>{input}</question>
    """
)

def context_vector_embedding():
    if "vectors" not in st.session_state:
        st.session_state.embedding = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')
        st.session_state.loader = PyPDFLoader('attention_all_youneed.pdf')
        st.session_state.docs = st.session_state.loader.load()
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        st.session_state.final_text = st.session_state.text_splitter.split_documents(st.session_state.docs[:50])
        st.session_state.vectors = FAISS.from_documents(st.session_state.final_text, st.session_state.embedding)


st.title("RAG Document ChatBot")
user_promt = st.text_input("Enter your query: ")
if st.button("ask"):
    context_vector_embedding()
    st.write("Your Database is ready")

import time

if user_promt:
    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = st.session_state.vectors.as_retriever()
    retriever_chain = create_retrieval_chain(retriever, document_chain)

    start_time = time.process_time()
    response = retriever_chain.invoke({"input": user_promt})
    print(f"Execution time: {time.process_time()-start_time}")
    
    st.write(response['answer'])

    with st.expander("Document similarity Search"):
        for i,doc in enumerate(response['context']):
            st.write(doc.page_content)
            st.write('------------------------')


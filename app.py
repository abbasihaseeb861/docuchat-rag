import streamlit as st
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

st.set_page_config(page_title="DocuChat-RAG", page_icon="🧬")
st.title("🧬 DocuChat-RAG")
st.write("Apni research paper upload karo aur us se sawal pucho!")

uploaded_file = st.file_uploader("PDF upload karo", type="pdf")

if uploaded_file is not None:
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.success("File upload ho gayi!")
    loader = PyPDFLoader("temp.pdf")
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(docs)
    st.write("Processing ho rahi hai...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = Chroma.from_documents(chunks, embeddings, persist_directory="./db")
    st.success("Ready! Ab sawal pucho")
    query = st.text_input("Tumhara sawal kya hai?")
    if query:
        results = db.similarity_search(query, k=3)
        st.write("### Relevant paragraphs:")
        for i, res in enumerate(results):
            st.write(f"**Source {i+1}:**")
            st.write(res.page_content)
            st.write("---")
else:
    st.info("Pehle upar se PDF upload karo")

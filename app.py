from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
import streamlit as st

loader = PyPDFLoader("Academic-Regulations.pdf")
text_documents = loader.load()

text_Splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 200)
documents = text_Splitter.split_documents(text_documents)

db = Chroma.from_documents(documents, OllamaEmbeddings(model = "all-minilm"))

llm = Ollama(model = "phi3")

prompt = ChatPromptTemplate.from_template("""
You are an academic assistant for technical programs. Answer questions regarding the university
using ONLY the following context. If the answer isn't in the context, say you don't know.

<context>
{context}
</context>

Question: {input}

Answer concisely and accurately in bullet points when appropriate:""")


document_chain =  create_stuff_documents_chain(llm, prompt)

retriever = db.as_retriever()

retrieval_chain = create_retrieval_chain(retriever, document_chain)

st.title("VITdocuMind: Academic Regulations Q&A")
st.write("Ask any question about the VIT Academic Regulations.")

user_input = st.text_input("Enter your question:")

if user_input:
    with st.spinner("Retrieving answer..."):
        response = retrieval_chain.invoke({"input": user_input})
        # Display the answer (handle dict or string)
        if isinstance(response, dict) and "answer" in response:
            st.markdown(response["answer"])
        else:
            st.markdown(str(response))
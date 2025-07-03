import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import HuggingFaceHub
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
import os
import pickle
from dotenv import load_dotenv


# Use secrets secrets.toml
if "HUGGINGFACEHUB_API_TOKEN" in st.secrets:
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = st.secrets["HUGGINGFACEHUB_API_TOKEN"]

@st.cache_resource
def get_vectorstore():
    index_path = "faiss_index"
    docstore_path = "docstore.pkl"

    embedding_function = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    if os.path.exists(index_path) and os.path.exists(docstore_path):
        with open(docstore_path, "rb") as f:
            documents = pickle.load(f)
        db = FAISS.load_local(index_path, embeddings=embedding_function, documents=documents)
    else:
        loader = PyPDFLoader("Academic-Regulations.pdf")
        text_documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", " ", ""]
        )
        documents = text_splitter.split_documents(text_documents)
        db = FAISS.from_documents(documents, embedding_function)
        db.save_local(index_path)

        with open(docstore_path, "wb") as f:
            pickle.dump(documents, f)

    return db

# Initialize with a spinner
with st.spinner("Initializing VITdocuMind..."):
    db = get_vectorstore()
    retriever = db.as_retriever(search_kwargs={"k": 4})

    # Hugging Face Hub LLM (Free open-access model)
    llm = HuggingFaceHub(
        repo_id="google/flan-t5-base",
        model_kwargs={"temperature": 0.3, "max_length": 512}
    )

    prompt = ChatPromptTemplate.from_template("""
    You are an expert assistant for VIT University's academic regulations. 
    Answer questions STRICTLY using only the provided context from the official document.
    
    Rules:
    1. Be precise and factual
    2. Cite section numbers when possible (e.g., "Section 4.3 states...")
    3. If unsure, say "This information is not specified in the academic regulations"
    
    Context:
    {context}
    
    Question: {input}
    
    Answer in this format:
    - [Concise point 1]
    - [Concise point 2]
    - [Relevant section reference if applicable]
    """)

    document_chain = create_stuff_documents_chain(llm, prompt)
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

# Streamlit UI
st.title("VITdocuMind: Academic Regulations Q&A")
st.caption("Official knowledge base for VIT University's FFCS Academic Regulations Version 4.0")

user_input = st.text_input("Ask about admission, courses, grading, or other regulations:")

if user_input:
    with st.spinner("Searching regulations..."):
        try:
            response = retrieval_chain.invoke({"input": user_input})
            answer = response.get("answer", "No answer found.")

            st.subheader("Answer:")
            st.markdown(answer)

            with st.expander("View relevant regulation excerpts"):
                for doc in response.get("context", []):
                    st.caption(f"From page {doc.metadata.get('page', 'N/A')}:")
                    st.text(doc.page_content[:300] + "...")
        except Exception as e:
            st.error(f"Error processing your question: {str(e)}")

st.divider()
st.markdown("""
**Sample questions to try:**
- What exams are required for B.Tech admission?
- How many credits are needed per semester?
- Explain the grading system
- What's the attendance policy?
""")


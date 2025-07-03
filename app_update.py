import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
import os

# Initialize with a spinner
with st.spinner("Initializing VITdocuMind..."):
    # Load and process PDF
    loader = PyPDFLoader("Academic-Regulations.pdf")
    text_documents = loader.load()
    
    # Split text
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""]
    )
    documents = text_splitter.split_documents(text_documents)
    
    # Create vector store
    persist_dir = "chroma_db"
    if os.path.exists(persist_dir):
        db = Chroma(persist_directory=persist_dir, 
                   embedding_function=OllamaEmbeddings(model="all-minilm"))
    else:
        db = Chroma.from_documents(
            documents=documents,
            embedding=OllamaEmbeddings(model="all-minilm"),
            persist_directory=persist_dir
        )
    
    # Initialize LLM
    llm = Ollama(model="phi3", temperature=0.3)
    
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
    
    # Create chains
    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = db.as_retriever(search_kwargs={"k": 4})   
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
            
            # Source chunks (for debugging)
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
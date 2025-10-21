import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import pipeline
import os
import tempfile
import pdfplumber
import re
from langchain_core.documents import Document
class PdfRagSystem:
    def __init__(self, pdf_path, chunk_size=1000, chunk_overlap=200, embedding_model="sentence-transformers/multi-qa-MiniLM-L6-cos-v1", qa_model="deepset/roberta-base-squad2"):
        self.pdf_path = pdf_path
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.embedding_model = embedding_model
        self.qa_model = qa_model
        self.documents = None
        self.vectorstore = None
        self.retriever = None
        self.qa_pipeline = None
        self._load_and_process()

    def _clean_text(self, text):
        """Clean extracted text to remove unwanted numerical fragments or artifacts."""
        # Remove standalone numbers or short numerical patterns (e.g., "11 8.1")
        text = re.sub(r'\b\d+(\s+\d+\.\d+)?\b', '', text)
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def _load_and_process(self):
        try:
            # Load PDF with pdfplumber
            documents = []
            with pdfplumber.open(self.pdf_path) as pdf:
                for page in pdf.pages:
                    text = page.extract_text()
                    if text:
                        cleaned_text = self._clean_text(text)
                        if cleaned_text:  # Only add non-empty cleaned text
                            documents.append({"page_content": cleaned_text, "metadata": {"page": page.page_number}})
            
            if not documents:
                raise ValueError("No valid text extracted from PDF.")

            # Convert to LangChain document format
            
            self.documents = [Document(page_content=doc["page_content"], metadata=doc["metadata"]) for doc in documents]

            # Split into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap
            )
            splits = text_splitter.split_documents(self.documents)

            # Create embeddings and vector store
            embeddings = HuggingFaceEmbeddings(model_name=self.embedding_model)
            self.vectorstore = Chroma.from_documents(
                documents=splits,
                embedding=embeddings,
                collection_name="pdf_rag"
            )
            self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 5})

            # Set up the QA pipeline
            self.qa_pipeline = pipeline("question-answering", model=self.qa_model, tokenizer=self.qa_model)

        except Exception as e:
            st.error(f"Error processing PDF: {str(e)}")

    def query(self, question):
        if not self.retriever or not self.qa_pipeline:
            return "System not initialized.", []
        if not question.strip() or question.lower() in ["give one question", "ask a question"]:
            return "Please ask a specific question about the PDF content.", []
        try:
            # Retrieve relevant chunks
            docs = self.retriever.invoke(question)
            context = "\n\n".join(self._clean_text(doc.page_content) for doc in docs)

            # Run question-answering
            result = self.qa_pipeline(question=question, context=context)
            # Check confidence score (threshold of 0.2 is empirical; adjust as needed)
            if result['score'] < 0.1:
                return "I couldn't find a confident answer. Try rephrasing your question or check the retrieved context.", [doc.page_content for doc in docs]
            return result['answer'], [doc.page_content for doc in docs]
        except Exception as e:
            return f"Error during query: {str(e)}", []

# Streamlit UI
st.title("Talk to Your PDF")
st.write("Upload a PDF and ask questions about its content (e.g., 'What is the main topic?').")

# Initialize session state
if "rag_system" not in st.session_state:
    st.session_state.rag_system = None
if "pdf_uploaded" not in st.session_state:
    st.session_state.pdf_uploaded = False

# Configuration options
st.sidebar.header("Settings")
chunk_size = st.sidebar.slider("Chunk Size", 500, 2000, 1000, 100)
chunk_overlap = st.sidebar.slider("Chunk Overlap", 50, 500, 200, 50)

# File uploader
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file is not None and not st.session_state.pdf_uploaded:
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_file_path = tmp_file.name

    # Initialize RAG system
    try:
        # Clear existing vector store if it exists
        if st.session_state.rag_system:
            st.session_state.rag_system.vectorstore.delete_collection()
        st.session_state.rag_system = PdfRagSystem(tmp_file_path, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        st.session_state.pdf_uploaded = True
        st.success("PDF loaded successfully!")
    except Exception as e:
        st.error(f"Failed to load PDF: {str(e)}")
    finally:
        # Clean up temporary file
        if os.path.exists(tmp_file_path):
            os.unlink(tmp_file_path)

# Question input
question = st.text_input("Ask a question about the PDF (e.g., 'What is the main conclusion?'):")

if question and st.session_state.rag_system:
    with st.spinner("Processing your question..."):
        answer, retrieved_contexts = st.session_state.rag_system.query(question)
        st.write("**Answer:**")
        st.write(answer)
        with st.expander("Retrieved Context (Debug)"):
            for i, context in enumerate(retrieved_contexts, 1):
                st.write(f"**Chunk {i}:**")
                st.write(context)
elif question and not st.session_state.rag_system:
    st.warning("Please upload a PDF first.")

# Clear session state when new file is uploaded
if uploaded_file and st.session_state.pdf_uploaded:
    st.session_state.pdf_uploaded = False
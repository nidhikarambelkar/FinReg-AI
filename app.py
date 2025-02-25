import streamlit as st
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import re
from datetime import datetime, timedelta
from PyPDF2.errors import PdfReadError

# Load environment variables
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=api_key)

def extract_metadata(text):
    match = re.search(r'RBI/\d{4}-\d{2}/\d+', text)
    circular_number = match.group(0) if match else "Unknown"
    
    date_match = re.search(r'\b(January|February|March|April|May|June|July|August|September|October|November|December) \d{1,2}, \d{4}\b', text)
    date = date_match.group(0) if date_match else "Unknown"
    
    return circular_number, date

def get_pdf_text(pdf_docs):
    extracted_data = []
    for pdf in pdf_docs:
        try:
            pdf_reader = PdfReader(pdf)
            text = "\n".join([page.extract_text() for page in pdf_reader.pages if page.extract_text()])
            circular_number, date = extract_metadata(text)
            extracted_data.append({"text": text, "circular": circular_number, "date": date})
        except PdfReadError:
            st.warning(f"Unable to read file: {pdf.name}")
        except Exception as e:
            st.warning(f"Error processing file: {pdf.name} - {str(e)}")
    return extracted_data

def get_text_chunks(data):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200, separators=["\n\n", "\n", " "])
    chunks = []
    for doc in data:
        for chunk in text_splitter.split_text(doc["text"]):
            chunks.append({"text": chunk, "circular": doc["circular"], "date": doc["date"]})
    return chunks

def get_vector_store(chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    text_data = [chunk["text"] for chunk in chunks]
    metadata = [{"circular": chunk["circular"], "date": chunk["date"]} for chunk in chunks]
    vector_store = FAISS.from_texts(text_data, embedding=embeddings, metadatas=metadata)
    vector_store.save_local("faiss_index")

def retrieve_relevant_docs():
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    return vector_store.as_retriever()

def get_conversational_chain():
    prompt_template ="""
    You are an *Advanced Financial Assistant* with deep expertise in RBI regulations, credit policies, and financial circulars.
    Your role is to provide *precise, regulation-backed answers* to finance professionals, compliance officers, and policymakers.  

    - *Answer with high financial accuracy* using retrieved RBI circulars.  
    - *If a direct answer is not found,* say:  
      "The answer is not available in the retrieved circulars. Please refer to the official RBI website for further details."  
    - *Always mention the exact RBI circular(s) and document references used* to support the answer.  

    *Context:*  
    {context}  

    *Question:*  
    {question}  

    *Answer:*  
    """
    model = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.2)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    retriever = retrieve_relevant_docs()
    return RetrievalQA.from_chain_type(llm=model, retriever=retriever, chain_type="stuff", chain_type_kwargs={"prompt": prompt})

def main():
    st.set_page_config("Regulatory Chatbot")
    st.subheader("FinReg AI - Your Advanced Financial Regulatory Assistant")
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    with st.sidebar:
        st.title("Upload Files")
        uploaded_files = st.file_uploader("Upload PDFs", accept_multiple_files=True, type=["pdf"])
        if uploaded_files:
            with st.spinner("Processing files..."):
                extracted_data = get_pdf_text(uploaded_files)
                if extracted_data:
                    text_chunks = get_text_chunks(extracted_data)
                    get_vector_store(text_chunks)
                    st.success("Processing complete.")
                else:
                    st.warning("No extractable text found in the uploaded files.")
    
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    user_query = st.chat_input("Ask a question")
    if user_query:
        st.session_state.chat_history.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)
        
        chain = get_conversational_chain()
        response = chain.run({"query": user_query})
        
        with st.chat_message("assistant"):
            st.markdown(response)
        st.session_state.chat_history.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()
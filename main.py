import streamlit as st
import os
import json
import requests
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
from typing import List, Dict
import pickle
from langchain_community.document_loaders import UnstructuredURLLoader

class SchemeResearchTool:
    def __init__(self):
        # Read API key from .config file
        try:
            with open('.config', 'r') as f:
                config = json.load(f)
                self.groq_api_key = config["GROQ_API_KEY"]
                
            if not self.groq_api_key:
                raise ValueError("API key is empty")
                
        except FileNotFoundError:
            st.error("'.config' file not found. Please create it with your GROQ API key.")
            st.stop()
        except json.JSONDecodeError:
            st.error("Invalid JSON format in .config file")
            st.stop()
        except Exception as e:
            st.error(f"Error reading API key: {str(e)}")
            st.stop()

        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
        self.llm = ChatGroq(groq_api_key=self.groq_api_key)
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

    def process_urls(self, urls: List[str]) -> List[Dict]:
        try:
            # Load content from URLs
            loader = UnstructuredURLLoader(urls=urls)
            documents = loader.load()
            
            # Split documents into chunks
            texts = self.text_splitter.split_documents(documents)
            return texts
        except Exception as e:
            st.error(f"Error processing URLs: {str(e)}")
            return []

    def process_url_file(self, file) -> List[Dict]:
        try:
            # Read URLs from uploaded file
            content = file.getvalue().decode("utf-8")
            urls = [url.strip() for url in content.split('\n') if url.strip()]
            return self.process_urls(urls)
        except Exception as e:
            st.error(f"Error processing URL file: {str(e)}")
            return []

    def create_and_save_vectorstore(self, texts) -> FAISS:
        vectorstore = FAISS.from_documents(texts, self.embeddings)
        # Save to pickle file
        with open('faiss_store_openai.pkl', 'wb') as f:
            pickle.dump(vectorstore, f)
        return vectorstore

    def load_vectorstore(self) -> FAISS:
        with open('faiss_store_openai.pkl', 'rb') as f:
            vectorstore = pickle.load(f)
        return vectorstore

    def answer_query(self, query: str, vectorstore: FAISS) -> Dict:
        qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=vectorstore.as_retriever()
        )

        enhanced_query = f"""Based on the provided documents, please provide a detailed response addressing:
        {query}
        Include relevant details about benefits, process, eligibility, and documentation if applicable to the query."""

        answer = qa_chain.run(enhanced_query)
        sources = [doc.metadata.get('source') for doc in vectorstore.similarity_search(query)]
        return {"answer": answer, "sources": sources}

def main():
    st.title("Scheme Research Application")
    st.write("Upload a file with URLs or enter URLs directly to analyze government schemes")

    try:
        tool = SchemeResearchTool()
        
        # Input method selection
        input_method = st.radio(
            "Choose input method:",
            ["Enter URL directly", "Upload file with URLs"]
        )
        
        texts = None
        
        if input_method == "Enter URL directly":
            url = st.text_input("Enter URL:")
            if url and st.button("Process URL"):
                with st.spinner("Processing URL..."):
                    texts = tool.process_urls([url])
        else:
            uploaded_file = st.file_uploader("Upload file with URLs (one URL per line)", type="txt")
            if uploaded_file and st.button("Process URLs"):
                with st.spinner("Processing URLs from file..."):
                    texts = tool.process_url_file(uploaded_file)
        
        if texts:
            vectorstore = tool.create_and_save_vectorstore(texts)
            st.success("Documents processed successfully!")
        
        # Query input
        query = st.text_input("Enter your question about the scheme:")
        
        if query and os.path.exists('faiss_store_openai.pkl'):
            with st.spinner("Searching for answer..."):
                vectorstore = tool.load_vectorstore()
                result = tool.answer_query(query, vectorstore)
                
                st.subheader("Answer:")
                st.write(result["answer"])
                
                st.subheader("Sources:")
                for source in result["sources"]:
                    st.write(source)
    
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()





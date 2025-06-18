
import os
import logging
import time
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from typing import List, Optional

logger = logging.getLogger(__name__)

def _get_embeddings_with_retries(texts, embedding_model:GoogleGenerativeAIEmbeddings, max_retries:int=5,initial_delay:int=5):
    """
    Generates embeddings for a list of texts with retry logic and exponential backoff.
    """
    all_embeddings =[]
    current_text_to_embed = list(texts)

    for attempt in range(max_retries):
        try:
            embeddings = embedding_model.embed_documents(current_text_to_embed)
            all_embeddings.extend(embeddings)

            logger.info(f"Successfully embedded {len(current_text_to_embed)} texts on attempt {attempt + 1}.")
            return all_embeddings
        except Exception as e:
            logger.warning(f"Embedding attempt {attempt + 1}/{max_retries} failed: {e}")
            if attempt < max_retries-1:
                wait_time = initial_delay*(2**attempt)
                logger.info(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                logger.error(f"Failed to embed texts after {max_retries} attempts.")
                raise

def create_vector_store(text_chunks,embedding_model:GoogleGenerativeAIEmbeddings,db_path:str):
    logger.info(f"Creating FAISS vector store at {db_path} with {len(text_chunks)} chunks...")
    try:
        texts_to_embeddings = [chunk.page_content for chunk in text_chunks]

        embeddings = _get_embeddings_with_retries(texts_to_embeddings,embedding_model)

        #create FAISS from document and embeddings

        vector_store = FAISS.from_embeddings(text_embeddings=list(zip(texts_to_embeddings, embeddings)),embedding=embedding_model)
        vector_store.add_documents(text_chunks)

        os.makedirs(db_path,exist_ok=True)
        logger.info(f'Creating the folder{db_path}')
        vector_store.save_local(db_path)
        logger.info(f"FAISS vector store created and saved successfully at {db_path}.")
        return vector_store
    except Exception as e:
        logger.error(f"Error creating FAISS vector store: {e}")
        raise

def load_vector_store(embedding_model:GoogleGenerativeAIEmbeddings, db_path:str):
    logger.info(f"Loading FAISS vector store from {db_path}...")
    try:
        vector_store = FAISS.load_local(db_path,embedding_model,allow_dangerous_deserialization=True)
        logger.info(f"FAISS vector store loaded successfully from {db_path}.")
        return vector_store
    except Exception as e:
         logger.error(f"Error loading FAISS vector store: {e}")
         raise
        
    

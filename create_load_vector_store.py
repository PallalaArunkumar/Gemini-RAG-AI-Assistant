import os
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from typing import List
from langchain_google_genai import GoogleGenerativeAIEmbeddings



def get_embedding_model(model_name : str = "models/embedding-001"):
  """
  Initializes and returns the embedding model."""
  return  GoogleGenerativeAIEmbeddings(model_name=model_name)
  

def create_vector_store(chunks:List[Document],embedding_model,db_path:str='faiss_index'):
  """
    Creates a FAISS vector store from document chunks and an embedding model.
    Saves the FAISS index to the specified path.
  """
  vector_store = FAISS.from_documents(chunks,embedding_model)

  os.makedirs(db_path,exist_ok=True)
  print(f'FAISS index saved locally at: {db_path}')
  vector_store.save_local(db_path)
  return vector_store 

def load_vector_store(embedding_model,db_path:str='faiss_index'):
  """
  Loads an existing FAISS vector store from the specified path
  """
  vector_store = FAISS.load_local(db_path,embedding_model,allow_dangerous_deserialization=True)
  return vector_store

if __name__ == "__main__":
  loaded_document = load_document('/kaggle/input/ai-wiki/Artificial intelligence - Wikipedia.pdf')
  text_chunks = split_documents(loaded_document)
  embedding_model = get_embedding_model("models/embedding-001")
  # Creating and Saving FAISS Vector Store
  vector_store = create_vector_store(text_chunks,embedding_model)

  #  Testing Loading FAISS Vector Store
  loaded_vector_store = load_vector_store(embedding_model)

  # Test a simple similarity search (optional, just to confirm it works)
  test_query = "What is the main contribution of this paper?"
  print(f"\nPerforming a test similarity search for: '{test_query}'")
  
  retrieved_docs = loaded_vector_store.similarity_search(test_query, k=2) # Retrieve top 2 similar documents
  for i, doc in enumerate(retrieved_docs):
      print(f"Retrieved Document {i+1} (Page: {doc.metadata.get('page')}):")
      print(doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content)
      print("-" * 30)

  print("\nFAISS Vector Store creation and testing complete!")

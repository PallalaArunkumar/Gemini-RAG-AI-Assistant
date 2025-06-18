
import os
import logging

from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

from langchain_google_genai import ChatGoogleGenerativeAI # For chat models like Gemini Pro
from langchain_google_genai import GoogleGenerativeAIEmbeddings # For Gemini Embeddings
# from create_load_vector_store import *
# from data_loader import *

 # --- Configure Logging ---
logging.basicConfig(
        level=logging.INFO, # Set the default logging level to INFO
        format='%(asctime)s - %(levelname)s - %(message)s',
        force=True
    )
logger = logging.getLogger(__name__) # Get a logger instance for this module

def setup_rag_components(document:str, faiss_db_path:str,llm_model_id:str):

        # --- Securely get the Google API Key from Kaggle Secrets ---
    # This is the correct way for Kaggle/Colab notebooks
    if "GOOGLE_API_KEY" not in os.environ:
        try:
            # For Kaggle Notebooks:
            from kaggle_secrets import UserSecretsClient
            user_secrets = UserSecretsClient()
            os.environ["GOOGLE_API_KEY"] = user_secrets.get_secret("GOOGLE_API_KEY")
            print("*"*50)
            print("Google API Key loaded from Kaggle Secrets.")
            print("*"*50)
            
            
        except ImportError:
            # Fallback for other environments if getpass also fails
            logger.error("Google API Key not found. Please ensure it is set as a Kaggle Secret (or Colab Secret) named 'GOOGLE_API_KEY'.")
            # print("Kaggle Secrets not available. Trying to load from environment or it will error.")
    else:
        loaded_key = os.environ.get("GOOGLE_API_KEY")
        logger.info(f"DEBUG: GOOGLE_API_KEY is loaded. First 5 chars: {loaded_key[:5]}*****")
        logger.info(f"DEBUG: Key length: {len(loaded_key)}") # A Gemini key is usually 39 characters
        

    if not os.environ.get("GOOGLE_API_KEY"):
        logger.error("Google API Key not found. Please ensure it is set as a Kaggle Secret")
        raise ValueError("Google API Key not found. Please ensure it is set as a Kaggle Secret (or Colab Secret) named 'GOOGLE_API_KEY'.")
        
    
    embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    if not os.path.exists(faiss_db_path):
        logger.info(f"FAISS index not found at {faiss_db_path}. Creating it now...")
        loaded_docs = load_document(document)
        text_chunks = split_documents(loaded_docs)
        vector_store = create_vector_store(text_chunks, embedding_model, db_path=faiss_db_path)
    else:
        vector_store = load_vector_store(embedding_model, db_path=faiss_db_path)
    
    retriever = vector_store.as_retriever(search_kwargs={"k": 2})
    
    logger.info(f"Initializing LLM: {llm_model_id} (Gemini)...")
    
    llm = ChatGoogleGenerativeAI(
            model=llm_model_id,
            temperature=0.7, 
        )
    logger.info(f"Successfully initialized Gemini model: {llm_model_id}")
    # --- Prompt Template ---
    # This template works well for both Flan-T5 and Gemini
    template = """Context: {context}

    Question: {question}

    Based on the provided context, please provide a detailed and comprehensive answer. If the answer is not present in the context, state that you don't know.
    Answer:"""
    
    prompt_template_obj = PromptTemplate(input_variables=["context", "question"], template=template)

     # --- Create RetrievalQA Chain ---
    # Use RetrievalQA.from_chain_type for simplicity in this structure
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff", # 'stuff' concatenates all retrieved docs into a single prompt
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt_template_obj}
    )

    return qa_chain

if __name__ == "__main__":
   

    
    GEMINI_LLM_MODEL = "models/gemini-2.0-flash"

    qa_chain = setup_rag_components(document="/kaggle/input/ai-wiki/Artificial intelligence - Wikipedia.pdf",
                                   faiss_db_path="/kaggle/working/faiss_index_2",
                                   llm_model_id=GEMINI_LLM_MODEL)
    logger.info("RAG setup complete. You can now ask questions.")

    while True:
        query = input("\nEnter your question (type 'exit' to quit): ")
        if query.lower() == 'exit':
            break
        
        try:
            # The qa_chain.invoke method uses the prompt template and LLM internally
            response = qa_chain.invoke({"query": query})
            
            print("\n--- Answer ---")
            print(response["result"])
            
            print("\n--- Source Documents ---")
            for i, doc in enumerate(response["source_documents"]):
                # Ensure metadata exists and page key is present
                page_info = doc.metadata.get('page', 'N/A')
                # st.markdown(f"**Document {i+1} (Page: {page_info}):**")
                print(f"Doc {i+1} (Page: {page_info}): {doc.page_content[:400]}...") # Print first 400 chars
                print("-" * 30)

        except Exception as e:
            logger.error(f"An error occurred: {e}")
            logger.info("Please ensure your Google API Key is correctly set as an environment variable.")
            import traceback
            traceback.print_exc()
    

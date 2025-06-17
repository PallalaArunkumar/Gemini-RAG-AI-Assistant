
import os
import torch
from langchain_community.llms import HuggingFacePipeline # Keep this for now, though we might not use it directly for generation
from langchain.prompts import PromptTemplate
# Removed: from langchain.chains import RetrievalQA
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

# For loading quantized models
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig

# # Import our custom functions from vector_store.py and data_loader.py
# from vector_store import load_vector_store, get_embedding_model
# from data_loader import split_documents, load_document

def setup_rag_components(
    document_path: str, 
    faiss_db_path: str, 
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    llm_model_id: str = "openlm-research/open_llama_3b"
):
    """
    Sets up the RAG components (embedding model, vector store, retriever, LLM pipeline).
    """
    embedding_model = get_embedding_model(embedding_model_name)

    if not os.path.exists(faiss_db_path):
        print(f"FAISS index not found at {faiss_db_path}. Creating it now...")
        loaded_docs = load_document(document_path)
        text_chunks = split_documents(loaded_docs)
        vector_store = create_vector_store(text_chunks, embedding_model, db_path=faiss_db_path)
    else:
        vector_store = load_vector_store(embedding_model, db_path=faiss_db_path)
    
    retriever = vector_store.as_retriever(search_kwargs={"k": 2})

    print(f"Initializing LLM from Hugging Face model: {llm_model_id} with 4-bit quantization...")
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(llm_model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        llm_model_id,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )
    
    # Create the Hugging Face pipeline directly
    llm_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=512,
        temperature=0.3,
        do_sample=True,
        top_p=0.9,
        repetition_penalty=1.1,
        return_full_text=False # Crucial for getting only generated text
    )
    print("LLM pipeline initialized.")
    
    return retriever, llm_pipeline

if __name__ == "__main__":
  document_path ="/content/various_TTS_paper.pdf" #os.path.join(project_root, "data", document_name)
  faiss_db_path = "/content/faiss_index"#os.path.join(project_root, "faiss_index") 

  llm_model_to_use = "openlm-research/open_llama_3b"

  print("Setting up RAG components...")
  retriever_obj, llm_pipeline_obj = setup_rag_components(
    document_path=document_path, 
    faiss_db_path=faiss_db_path, 
    llm_model_id=llm_model_to_use
  )
  print("RAG components setup complete.")

  # Define the prompt template for instruction-tuned models
  # This will be used to format the input for the LLM pipeline
  template = """<s>[INST] <<SYS>>
      You are a helpful, respectful and honest assistant. Answer the question based only on the provided context.
      If you don't know the answer, just say that you don't know. Do not try to make up an answer.
      Your answers should be concise and to the point.
      <</SYS>>
      
      Context: {context}
      
      Question: {question} [/INST]
      """
  # Helpful Answer:"""

  prompt_template_obj = PromptTemplate(input_variables=["context", "question"], template=template)

  while True:
  query = input("\nEnter your question (type 'exit' to quit): ")
  if query.lower() == 'exit':
      break
  if not query:
      print("Please enter a question.")
      continue

  print("Searching and generating answer...")
  try:
      # Step 1: Retrieve relevant documents
      retrieved_docs = retriever_obj.get_relevant_documents(query)
      
      # Step 2: Format context for the prompt
      context_text = "\n\n".join([doc.page_content for doc in retrieved_docs])
      
      # Step 3: Create the full prompt
      # The .format() method applies the values to the template
      formatted_prompt = prompt_template_obj.format(context=context_text, question=query)
      
      # Step 4: Generate answer using the Hugging Face pipeline directly
      # The pipeline expects a list of strings
      raw_response = llm_pipeline_obj(formatted_prompt)
      
      # The response from the pipeline is often a list of dicts.
      # We want the 'generated_text' from the first element.
      answer = raw_response[0]['generated_text'].strip()

      print("\nAnswer:")
      print(answer) 
      
      print("\n--- Source Documents ---")
      for i, doc in enumerate(retrieved_docs):
          print(f"Document {i+1} (Page: {doc.metadata.get('page')}):")
          print(doc.page_content + "...")
          print("-" * 30)

  except Exception as e:
      print(f"An error occurred: {e}")
      import traceback
      traceback.print_exc()
      print("Ensure you have a Colab GPU runtime (T4 recommended) and sufficient RAM.")
      print("Try restarting the Colab runtime if you encounter OOM errors.")

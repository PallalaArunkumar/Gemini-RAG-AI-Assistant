
import os
import torch
from langchain_community.llms import HuggingFacePipeline # Keep this for now, though we might not use it directly for generation
from langchain.prompts import PromptTemplate
# Removed: from langchain.chains import RetrievalQA
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

# For loading quantized models
from transformers import AutoTokenizer,AutoModelForSeq2SeqLM, AutoModelForCausalLM, pipeline, BitsAndBytesConfig


def setup_rag_components(
    document_path: str, 
    faiss_db_path: str, 
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    llm_model_id: str = "google/flan-t5-base"
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

    tokenizer = AutoTokenizer.from_pretrained(llm_model_id)

    # For T5 models, use AutoModelForSeq2SeqLM
    model = AutoModelForSeq2SeqLM.from_pretrained(
        llm_model_id,
        torch_dtype=torch.float32, # Use float32 or float16; float32 is default/safer for smaller models without explicit quantization
        device_map="auto" # Still useful for optimal device placement
    )

     # Create the Hugging Face pipeline for text2text-generation (for T5 models)
    llm_pipeline = pipeline(
        "text2text-generation", # <<< IMPORTANT: Changed task for T5
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=512, # Keep reasonable, T5 is efficient
        temperature=0.9, # Often a good default for T5, gives some creativity
        do_sample=True, # Allow sampling for more varied responses
        top_p=0.95,
        repetition_penalty=1.1,
        # T5 pipelines often don't need return_full_text=False as they inherently transform text
        # But for safety, we can keep it if issues arise. For now, remove as default behavior is better.
    )
    print("LLM pipeline initialized.")
    
    return retriever, llm_pipeline


if __name__ == "__main__":
    llm_model_to_use = "google/flan-t5-base" # <<< CHANGED TO FLAN-T5-BASE
    
print("Setting up RAG components...")
retriever_obj, llm_pipeline_obj = setup_rag_components(
    document_path="/kaggle/input/ai-wiki/Artificial intelligence - Wikipedia.pdf", 
    faiss_db_path="/kaggle/working/faiss_wiki_ai", 
    llm_model_id=llm_model_to_use
)
print("RAG components setup complete.")

# Define the prompt template for Flan-T5
# Flan-T5 often works well with simple Q&A formats
template = """Context: {context}

Question: {question}

Based on the provided context, please provide a detailed and comprehensive answer. If the answer is not present in the context, state that you don't know.
Answer:"""

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
        formatted_prompt = prompt_template_obj.format(context=context_text, question=query)
        
        # Step 4: Generate answer using the Hugging Face pipeline directly
        # The pipeline expects a list of strings
        raw_response = llm_pipeline_obj(formatted_prompt)
        
        # The response from the pipeline is often a list of dicts.
        # We want the 'generated_text' from the first element.
        answer = raw_response[0]['generated_text'].strip()

        print("\nAnswer:")
        print(answer) 
        
        # print("\n--- Source Documents ---")
        # for i, doc in enumerate(retrieved_docs):
        #     print(f"Document {i+1} (Page: {doc.metadata.get('page')}):")
        #     print(doc.page_content[:500] + "...")
        #     print("-" * 30)

    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()
        print("Ensure you have a Colab GPU runtime (T4 recommended) and sufficient RAM.")
        print("Try restarting the Colab runtime if you encounter OOM errors.")

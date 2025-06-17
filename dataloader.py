import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# data_loader.py

def load_document(filepath:str):
  loader = PyPDFLoader(filepath)
  documents = loader.load()
  print(f'Loaded {len(documents)} pages from the document')
  return documents

def split_documents(doc,chunk_size:int=1000, chunk_overlap:int=200):
  text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size,chunk_overlap=chunk_overlap,length_function=len)
  chunks = text_splitter.split_documents(doc)
  return chunks


if __name__ == "__main__":
  loaded_document = load_document('/kaggle/input/review-paper/various_TTS_paper.pdf')
  text_chunks = split_documents(loaded_document)

  for i,chunk in enumerate(text_chunks[:3]):
    print(f"Chunk {i+1} (Length: {len(chunk.page_content)} characters):")
    print(chunk.page_content[:500] + "..." if len(chunk.page_content) > 500 else chunk.page_content)
    print("-" * 30)

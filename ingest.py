"""
Knowledge ingestion pipeline for brand content.

Loads Markdown files from data/ directory (style guide, brand description,
example posts), splits into chunks, generates embeddings, and saves the
FAISS index to disk.

Usage: python ingest.py
"""

import os
import pickle
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import MarkdownTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

from config import settings


def ingest():
    data_dir = settings.data_dir
    print(f"Loading Markdown documents from {data_dir}/ ...")

    # Better way: Use DirectoryLoader + TextLoader directly for all .md files
    loader = DirectoryLoader(
        data_dir, 
        glob="**/*.md", 
        loader_cls=TextLoader, 
        loader_kwargs={"encoding": "utf-8"}
    )
    docs = loader.load()

    if not docs:
        print("No .md documents found to ingest!")
        return

    print(f"Loaded {len(docs)} documents.")

    print(f"Splitting documents into chunks of size {settings.chunk_size} "
          f"with overlap {settings.chunk_overlap}...")
    
    # Краще використовувати MarkdownTextSplitter замість звичайного RecursiveCharacterTextSplitter,
    # оскільки він розуміє специфіку Markdown (заголовки, списки, блоки коду)
    text_splitter = MarkdownTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    chunks = text_splitter.split_documents(docs)
    print(f"Generated {len(chunks)} chunks.")

    print(f"Generating embeddings using {settings.embedding_model} "
          f"and building FAISS index...")
    
    # Ініціалізуємо embeddings. OpenAI ключ береться з .env, 
    # бо в config.py вже є load_dotenv()
    embeddings = OpenAIEmbeddings(model=settings.embedding_model)
    vectorstore = FAISS.from_documents(chunks, embeddings)

    os.makedirs(settings.index_dir, exist_ok=True)

    print(f"Saving vector store to {settings.index_dir}/ ...")
    vectorstore.save_local(settings.index_dir)

    # Save chunks for BM25 retriever (гібридний пошук)
    chunks_path = os.path.join(settings.index_dir, "chunks.pkl")
    print(f"Saving chunks for BM25 to {chunks_path} ...")
    with open(chunks_path, "wb") as f:
        pickle.dump(chunks, f)

    print("✅ Ingestion complete!")


if __name__ == "__main__":
    ingest()

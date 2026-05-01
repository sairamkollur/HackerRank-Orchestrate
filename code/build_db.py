import os
import chromadb
from chromadb.utils import embedding_functions
from ingestion import load_and_chunk_corpus

def build_vector_database(data_dir="data", db_dir="chroma_db"):
    print("Step 1: Loading and chunking documents...")
    # This calls the function we built in the last step!
    chunks = load_and_chunk_corpus(data_dir)
    
    if not chunks:
        print("No chunks found. Exiting.")
        return

    print(f"\nStep 2: Initializing ChromaDB in ./{db_dir}...")
    # Create a persistent database that saves to a local folder
    client = chromadb.PersistentClient(path=db_dir)
    
    # We use a fast, lightweight, local embedding model from HuggingFace
    sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    
    # Create a "collection" (like a table in an SQL database)
    collection_name = "support_corpus"
    
    # Delete the collection if it already exists so we can start fresh
    try:
        client.delete_collection(name=collection_name)
        print("Overwriting existing collection...")
    except Exception:
        pass
        
    collection = client.create_collection(
        name=collection_name,
        embedding_function=sentence_transformer_ef
    )
    
    print("\nStep 3: Embedding and storing chunks (This might take a minute or two)...")
    
    # We need to separate the text, metadata, and give each chunk a unique ID
    documents = [chunk["text"] for chunk in chunks]
    metadatas = [chunk["metadata"] for chunk in chunks]
    ids = [f"chunk_{i}" for i in range(len(chunks))]
    
    # Add everything to the database
# Add everything to the database in smaller batches to avoid ChromaDB's size limits
    batch_size = 5000
    
    for i in range(0, len(documents), batch_size):
        # Calculate the end index for the current batch
        end_idx = min(i + batch_size, len(documents))
        print(f"  -> Processing batch from {i} to {end_idx}...")
        
        collection.add(
            documents=documents[i:end_idx],
            metadatas=metadatas[i:end_idx],
            ids=ids[i:end_idx]
        )
    
    print(f"\nSuccess! Stored {collection.count()} chunks in the database.")

if __name__ == "__main__":
    build_vector_database()
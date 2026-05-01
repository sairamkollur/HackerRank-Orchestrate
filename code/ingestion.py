import os
from pathlib import Path

# If you haven't already, run: pip install langchain-text-splitters
from langchain_text_splitters import RecursiveCharacterTextSplitter

def load_and_chunk_corpus(data_dir="data", chunk_size=1000, chunk_overlap=200):
    """
    Recursively loads all markdown files from the data directory, 
    extracts the company domain, and splits the text into chunks.
    """
    print(f"Scanning directory: {data_dir}...")
    
    # Initialize the text splitter
    # This specifically looks for Markdown headers and paragraphs to avoid cutting sentences in half
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )
    
    all_chunks = []
    base_path = Path(data_dir)
    
    # Grab every .md file in the directory and subdirectories
    for filepath in base_path.rglob('*.md'):
        # Extract company name from the folder structure (e.g., data/claude/...)
        # parts[0] is 'data', parts[1] is the company folder
        if len(filepath.parts) > 1:
            company_domain = filepath.parts[1].capitalize() 
        else:
            company_domain = "Unknown"
            
        try:
            with open(filepath, 'r', encoding='utf-8') as file:
                content = file.read()
                
            # Split the file's content into smaller chunks
            chunks = text_splitter.split_text(content)
            
            # Attach metadata to each chunk so the Search Engine knows exactly where it came from
            for chunk in chunks:
                all_chunks.append({
                    "text": chunk,
                    "metadata": {
                        "company": company_domain,
                        "source_file": filepath.name
                    }
                })
        except Exception as e:
            print(f"Error reading {filepath}: {e}")

    print(f"Successfully processed {len(all_chunks)} chunks from the corpus.")
    return all_chunks

# Quick test to make sure it works when you run this script directly
if __name__ == "__main__":
    # Make sure you run this from the root of your project where the 'data' folder is
    chunks = load_and_chunk_corpus("data")
    if chunks:
        print("\n--- Sample Chunk ---")
        print(f"Company: {chunks[0]['metadata']['company']}")
        print(f"Source: {chunks[0]['metadata']['source_file']}")
        print(f"Content: {chunks[0]['text'][:200]}...")
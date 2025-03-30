import os
import sys
from pathlib import Path

# Add the project root directory to Python path
project_root = str(Path(__file__).parent.parent)
sys.path.append(project_root)

import numpy as np
from src.vector_generator import generate_random_vectors, generate_query_vector
from src.index_operations import create_index, save_index, load_index
from src.search_operations import search_index

def main():
    # Parameters
    num_vectors = 10000
    dimension = 128
    k = 5
    
    print("1. Generating random vectors...")
    vectors = generate_random_vectors(num_vectors, dimension)
    
    print("2. Creating FAISS index...")
    index = create_index(vectors)
    
    print("3. Saving index to disk...")
    save_index(index, "vector_index.faiss")
    
    print("4. Loading index from disk...")
    loaded_index = load_index("vector_index.faiss")
    
    print("5. Generating query vector and performing search...")
    query = generate_query_vector(dimension)
    distances, indices = search_index(loaded_index, query, k)
    
    print("\nSearch Results:")
    print(f"Query vector shape: {query.shape}")
    print(f"\nTop {k} nearest neighbors:")
    for i in range(k):
        print(f"#{i+1}: Index: {indices[0][i]}, Distance: {distances[0][i]:.4f}")

if __name__ == "__main__":
    main() 
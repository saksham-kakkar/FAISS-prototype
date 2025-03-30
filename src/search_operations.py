import faiss
import numpy as np
from typing import Tuple

def search_index(index: faiss.Index, query_vector: np.ndarray, k: int = 5) -> Tuple[np.ndarray, np.ndarray]:
    """
    Search for k-nearest neighbors in the index.
    
    Args:
        index (faiss.Index): FAISS index to search in
        query_vector (np.ndarray): Query vector
        k (int): Number of nearest neighbors to return
    
    Returns:
        Tuple[np.ndarray, np.ndarray]: Distances and indices of nearest neighbors
    """
    distances, indices = index.search(query_vector, k)
    return distances, indices 

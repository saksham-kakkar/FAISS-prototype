import faiss
import numpy as np
from typing import Tuple

def create_index(vectors: np.ndarray, index_type: str = 'L2') -> faiss.Index:
    """
    Create a FAISS index for the input vectors.
    
    Args:
        vectors (np.ndarray): Input vectors to index
        index_type (str): Type of index to create ('L2', 'IP', 'IVF', 'HNSW')
    
    Returns:
        faiss.Index: Created FAISS index
    """
    dimension = vectors.shape[1]
    
    if index_type == 'L2':
        index = faiss.IndexFlatL2(dimension)
    elif index_type == 'IP':
        index = faiss.IndexFlatIP(dimension)
    elif index_type == 'IVF':
        # Create a quantizer
        quantizer = faiss.IndexFlatL2(dimension)
        # Create IVF index with 100 clusters
        index = faiss.IndexIVFFlat(quantizer, dimension, 100, faiss.METRIC_L2)
        # Train the index
        index.train(vectors)
    elif index_type == 'HNSW':
        # Create HNSW index
        index = faiss.IndexHNSWFlat(dimension, 32)
    else:
        raise ValueError("index_type must be one of: 'L2', 'IP', 'IVF', 'HNSW'")
    
    index.add(vectors)
    return index

def save_index(index: faiss.Index, filepath: str) -> None:
    """
    Save a FAISS index to disk.
    
    Args:
        index (faiss.Index): FAISS index to save
        filepath (str): Path where to save the index
    """
    faiss.write_index(index, filepath)

def load_index(filepath: str) -> faiss.Index:
    """
    Load a FAISS index from disk.
    
    Args:
        filepath (str): Path to the saved index
    
    Returns:
        faiss.Index: Loaded FAISS index
    """
    return faiss.read_index(filepath)
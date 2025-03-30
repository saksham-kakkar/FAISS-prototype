import numpy as np
import time
from typing import Tuple, List
import matplotlib.pyplot as plt
from src.vector_generator import generate_random_vectors, generate_query_vector
from src.index_operations import create_index
from src.search_operations import search_index

# Define available index types
INDEX_TYPES = {
    '1': 'L2',
    '2': 'IP',
    '3': 'IVF',
    '4': 'HNSW'
}

def get_index_description(index_type: str) -> str:
    """Get a description of the index type."""
    descriptions = {
        'L2': 'Flat L2 index - Simple but fast for small datasets',
        'IP': 'Flat Inner Product index - Good for cosine similarity',
        'IVF': 'Inverted File index - Good for large datasets',
        'HNSW': 'Hierarchical Navigable Small World - Good balance of speed and accuracy'
    }
    return descriptions.get(index_type, 'Unknown index type')

def brute_force_search(vectors: np.ndarray, query: np.ndarray, k: int = 5) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform brute force nearest neighbor search using NumPy.
    
    Args:
        vectors (np.ndarray): Database vectors
        query (np.ndarray): Query vector
        k (int): Number of nearest neighbors
    
    Returns:
        Tuple[np.ndarray, np.ndarray]: Distances and indices of nearest neighbors
    """
    # Compute L2 distances between query and all vectors
    distances = np.linalg.norm(vectors - query, axis=1)
    # Get indices of k smallest distances
    indices = np.argsort(distances)[:k]
    return distances[indices], indices

def run_benchmark(dimensions: List[int], num_vectors: int = 10000, k: int = 5, 
                 num_queries: int = 100, index_type: str = 'L2') -> Tuple[dict, dict]:
    """
    Run performance comparison between FAISS and brute force search.
    
    Args:
        dimensions (List[int]): List of dimensions to test
        num_vectors (int): Number of vectors in database
        k (int): Number of nearest neighbors
        num_queries (int): Number of queries to average over
        index_type (str): Type of FAISS index to use
    
    Returns:
        Tuple[dict, dict]: Dictionaries containing timing results
    """
    faiss_times = {}
    brute_times = {}
    
    print(f"\nRunning benchmark with {index_type} index")
    print(f"Index description: {get_index_description(index_type)}")
    
    for dim in dimensions:
        print(f"\nBenchmarking for dimension {dim}")
        
        # Generate database vectors
        vectors = generate_random_vectors(num_vectors, dim)
        queries = generate_random_vectors(num_queries, dim)
        
        # FAISS timing
        start_time = time.perf_counter()
        index = create_index(vectors, index_type)
        for query in queries:
            query = query.reshape(1, -1)
            distances, indices = search_index(index, query, k)
        faiss_time = (time.perf_counter() - start_time) / num_queries
        faiss_times[dim] = faiss_time
        
        # Brute force timing
        start_time = time.perf_counter()
        for query in queries:
            distances, indices = brute_force_search(vectors, query, k)
        brute_time = (time.perf_counter() - start_time) / num_queries
        brute_times[dim] = brute_time
        
        print(f"Average time per query:")
        print(f"FAISS: {faiss_time:.6f} seconds")
        print(f"Brute force: {brute_time:.6f} seconds")
        print(f"Speedup: {brute_time/faiss_time:.2f}x")
    
    return faiss_times, brute_times

def plot_comparison(dimensions: List[int], faiss_times: dict, brute_times: dict, 
                   index_type: str, save_path: str = "benchmark_results.png") -> None:
    """
    Plot performance comparison between FAISS and brute force search.
    
    Args:
        dimensions (List[int]): List of dimensions tested
        faiss_times (dict): FAISS timing results
        brute_times (dict): Brute force timing results
        index_type (str): Type of FAISS index used
        save_path (str): Path to save the plot
    """
    plt.figure(figsize=(10, 6))
    plt.plot(dimensions, [brute_times[d] for d in dimensions], 
             'ro-', label='Brute Force')
    plt.plot(dimensions, [faiss_times[d] for d in dimensions], 
             'bo-', label=f'FAISS ({index_type})')
    
    plt.xlabel('Dimension')
    plt.ylabel('Time per query (seconds)')
    plt.title(f'FAISS ({index_type}) vs Brute Force Search Performance')
    plt.legend()
    plt.grid(True)
    plt.yscale('log')
    
    plt.savefig(save_path)
    plt.close() 
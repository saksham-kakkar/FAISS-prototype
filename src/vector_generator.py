import numpy as np

def generate_random_vectors(num_vectors: int, dimension: int) -> np.ndarray:
    """
    Generate random vectors for demonstration purposes.
    
    Args:
        num_vectors (int): Number of vectors to generate
        dimension (int): Dimensionality of the vectors
    
    Returns:
        np.ndarray: Array of random vectors with shape (num_vectors, dimension)
    """
    return np.random.random((num_vectors, dimension)).astype('float32')

def generate_query_vector(dimension: int) -> np.ndarray:
    """
    Generate a single random query vector.
    
    Args:
        dimension (int): Dimensionality of the vector
    
    Returns:
        np.ndarray: A single random vector with shape (1, dimension)
    """
    return np.random.random((1, dimension)).astype('float32') 
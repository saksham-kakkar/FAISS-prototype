import os
import sys
from pathlib import Path

# Add the src directory to Python path
src_path = str(Path(__file__).parent.parent / "")
sys.path.append(src_path)

from src.benchmark import run_benchmark, plot_comparison, INDEX_TYPES, get_index_description

def print_index_options():
    """Print available index options."""
    print("\nAvailable FAISS index types:")
    for key, index_type in INDEX_TYPES.items():
        print(f"{key}. {index_type} - {get_index_description(index_type)}")

def get_user_choice() -> str:
    """Get user's choice of index type."""
    while True:
        print_index_options()
        choice = input("\nEnter the number of the index type you want to benchmark (or 'q' to quit): ")
        if choice.lower() == 'q':
            sys.exit(0)
        if choice in INDEX_TYPES:
            return INDEX_TYPES[choice]
        print("Invalid choice. Please try again.")

def main():
    # Test different dimensions
    dimensions = [8, 16, 32, 64, 128, 256, 512]
    num_vectors = 300000
    k = 5
    num_queries = 100
    
    # Get user's choice of index type
    index_type = get_user_choice()
    
    print("\nRunning benchmark...")
    print(f"Database size: {num_vectors} vectors")
    print(f"Number of queries: {num_queries}")
    print(f"k-NN: {k}")
    
    faiss_times, brute_times = run_benchmark(
        dimensions=dimensions,
        num_vectors=num_vectors,
        k=k,
        num_queries=num_queries,
        index_type=index_type
    )
    
    plot_comparison(dimensions, faiss_times, brute_times, index_type)
    print(f"\nBenchmark results have been saved to 'benchmark_results.png'")

if __name__ == "__main__":
    main() 
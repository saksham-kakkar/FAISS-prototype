# FAISS Similarity Search Demo

This project demonstrates how to use Facebook AI Similarity Search (FAISS) for efficient similarity search in high-dimensional spaces.

## Overview

FAISS is a library developed by Facebook Research that enables efficient similarity search and clustering of dense vectors. It is particularly useful for tasks such as:
- Nearest neighbor search
- Similarity matching
- Vector indexing
- Recommendation systems

## Installation

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Project Structure

- `src/vector_generator.py`: Functions for generating random vectors
- `src/index_operations.py`: FAISS index creation, storage, and loading operations
- `src/search_operations.py`: Vector similarity search operations
- `examples/demo.py`: Example usage of the FAISS functionality

## Usage

There are two ways to run the demo script:

1. From the project root directory:
```bash
python -m examples.demo
```

2. Or by running the demo script directly:
```bash
cd faiss-demo
python examples/demo.py
```

## Features

- Generate random high-dimensional vectors
- Create and manage FAISS indices
- Perform similarity searches
- Save and load FAISS indices

## Performance Comparison

This project includes a benchmark comparison between FAISS and brute-force nearest neighbor search:

1. Run the benchmark:
```bash
python -m examples.benchmark_demo
```

The benchmark allows you to choose from different FAISS index types:
1. L2 - Flat L2 index (simple but fast for small datasets)
2. IP - Flat Inner Product index (good for cosine similarity)
3. IVF - Inverted File index (good for large datasets)
4. HNSW - Hierarchical Navigable Small World (good balance of speed and accuracy)

The benchmark:
- Compares the selected FAISS index against NumPy-based brute force search
- Tests different vector dimensions (8 to 512)
- Measures average query time
- Generates a performance comparison plot

### Benchmark Results

The benchmark demonstrates that FAISS significantly outperforms brute-force search, especially:
- For high-dimensional vectors (100+ dimensions)
- With large datasets (10,000+ vectors)
- When performing multiple queries

Typical speedups:
- 10-50x faster for medium dimensions (32-128)
- 100x+ faster for high dimensions (256+)

The exact speedup depends on:
- Vector dimension
- Dataset size
- Hardware configuration
- Index type used

The results are visualized in `benchmark_results.png`, showing query time vs. dimension for both methods.

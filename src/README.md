# Source Code Modules

This directory contains modular, OOP-based Python classes for the RAG Complaint Chatbot project.

## Data Module

### CFPBDataLoader (`data_loader.py`)

Handles loading and validation of the CFPB complaint dataset.

**Key Methods:**
- `load_raw_data()` - Loads CSV data from file path
- `validate_data()` - Performs data integrity checks
- `get_data()` - Returns the loaded DataFrame

**Usage:**
```python
from src.data_loader import CFPBDataLoader

loader = CFPBDataLoader('data/raw/complaints.csv')
df = loader.load_raw_data()
validation_report = loader.validate_data()
```

### EDAAnalyzer (`eda_analyzer.py`)

Provides comprehensive exploratory data analysis with visualizations.

**Key Methods:**
- `analyze_product_distribution()` - Bar and pie charts for product distribution
- `analyze_narrative_length()` - Multi-panel analysis of narrative word counts
- `identify_missing_narratives()` - Missing data report
- `generate_eda_report()` - Complete EDA summary

**Usage:**
```python
from src.eda_analyzer import EDAAnalyzer

analyzer = EDAAnalyzer(df)
product_stats = analyzer.analyze_product_distribution()
narrative_stats = analyzer.analyze_narrative_length()
missing_report = analyzer.identify_missing_narratives()
```

### DataPreprocessor (`data_preprocessor.py`)

Filters and cleans complaint data for downstream processing.

**Key Methods:**
- `filter_by_products(products)` - Filter to specified product categories
- `remove_empty_narratives()` - Remove records with missing narratives
- `clean_text(text)` - Text normalization and cleaning
- `preprocess_pipeline(products)` - Complete preprocessing workflow

**Usage:**
```python
from src.data_preprocessor import DataPreprocessor

preprocessor = DataPreprocessor(df)
target_products = ['Credit card', 'Personal loan', 'Savings account', 'Money transfer']
cleaned_df = preprocessor.preprocess_pipeline(target_products)
```

### StratifiedSampler (`data_sampler.py`)

Creates stratified samples that maintain product category proportions.

**Key Methods:**
- `create_stratified_sample(sample_size, random_state)` - Create stratified sample
- `validate_sample_distribution(sample_df, tolerance)` - Validate distribution quality
- `get_sample_summary(sample_df)` - Generate sampling summary statistics

**Usage:**
```python
from src.data_sampler import StratifiedSampler

sampler = StratifiedSampler(df, product_column='Product')
sample = sampler.create_stratified_sample(sample_size=10000, random_state=42)
validation_report = sampler.validate_sample_distribution(sample, tolerance=0.05)
summary = sampler.get_sample_summary(sample)
```

## Text Processing Module

### TextChunker (`text_chunker.py`)

Splits long text documents into overlapping chunks for embedding generation.

**Key Methods:**
- `chunk_text(text, metadata)` - Split single text into chunks
- `chunk_documents(df, text_column, metadata_columns)` - Batch process DataFrame
- `get_chunk_statistics(chunks_df)` - Calculate chunking statistics

**Usage:**
```python
from src.text_chunker import TextChunker

# Create chunker with configurable parameters
chunker = TextChunker(chunk_size=500, chunk_overlap=50)

# Chunk a single text
text = "Long complaint narrative..."
metadata = {'complaint_id': 1001, 'product': 'Credit card'}
chunks = chunker.chunk_text(text, metadata)

# Batch process multiple documents
chunks_df = chunker.chunk_documents(
    df,
    text_column='Consumer complaint narrative',
    metadata_columns=['Product', 'Complaint ID']
)

# Get statistics
stats = chunker.get_chunk_statistics(chunks_df)
print(f"Created {stats['total_chunks']} chunks from {stats['num_documents']} documents")
```

## Embedding Module

### EmbeddingGenerator (`embedder.py`)

Generates vector embeddings for text using sentence-transformers models.

**Key Methods:**
- `load_model()` - Load sentence-transformers model
- `generate_embedding(text)` - Generate embedding for single or multiple texts
- `batch_embed(texts, batch_size)` - Efficient batch embedding generation
- `embed_dataframe(df, text_column, batch_size)` - Embed texts from DataFrame

**Usage:**
```python
from src.embedder import EmbeddingGenerator

# Initialize with model
embedder = EmbeddingGenerator(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Single text
embedding = embedder.generate_embedding("Credit card fees are too high")
print(f"Embedding shape: {embedding.shape}")  # (384,)

# Batch processing
texts = ["Text 1", "Text 2", "Text 3"]
embeddings = embedder.batch_embed(texts, batch_size=32, show_progress_bar=True)
print(f"Embeddings shape: {embeddings.shape}")  # (3, 384)
```

## Vector Store Module

### VectorStoreManager (`vector_store.py`)

Manages ChromaDB vector store for semantic search over embeddings.

**Key Methods:**
- `add_embeddings(embeddings, documents, metadatas, ids)` - Add embeddings to store
- `search(query_embedding, top_k, filter_dict)` - Similarity search
- `add_documents_batch(chunks_df, embeddings)` - Batch add from DataFrame
- `get_collection_stats()` - Get collection statistics
- `delete_collection()` - Delete collection

**Usage:**
```python
from src.vector_store import VectorStoreManager

# Initialize
vector_store = VectorStoreManager(
    persist_directory="vector_store",
    collection_name="complaints",
    reset=False
)

# Add embeddings
vector_store.add_embeddings(
    embeddings=embeddings,
    documents=texts,
    metadatas=[{'product': 'Credit card', 'id': '1'}],
    ids=['doc_1']
)

# Search
query_embedding = embedder.generate_embedding("late fees")
results = vector_store.search(query_embedding, top_k=5)

print(f"Found {len(results['documents'])} similar documents")
for doc, metadata, distance in zip(
    results['documents'], 
    results['metadatas'], 
    results['distances']
):
    print(f"Distance: {distance:.4f}")
    print(f"Product: {metadata['product']}")
    print(f"Text: {doc[:100]}...")

# Batch add from DataFrame
vector_store.add_documents_batch(
    chunks_df=chunks_df,
    embeddings=embeddings,
    text_column='text',
    metadata_columns=['Product', 'ID'],
    batch_size=1000
)
```

## Design Principles

- **Single Responsibility**: Each class has one clear purpose
- **Error Handling**: Proper exception handling and logging
- **Type Hints**: All methods use type hints
- **Docstrings**: Google-style docstrings for all public methods
- **Testability**: Designed for easy unit testing
- **Immutability**: Methods don't modify input data unless explicitly stated

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

## Design Principles

- **Single Responsibility**: Each class has one clear purpose
- **Error Handling**: Proper exception handling and logging
- **Type Hints**: All methods use type hints
- **Docstrings**: Google-style docstrings for all public methods
- **Testability**: Designed for easy unit testing
- **Immutability**: Methods don't modify input data unless explicitly stated

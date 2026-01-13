# RAG Complaint Chatbot

AI-powered chatbot for analyzing customer complaints in financial services using Retrieval-Augmented Generation (RAG).

## 🎯 Project Overview

**CrediTrust Financial** is building an intelligent complaint analysis system to transform customer feedback into actionable insights. This RAG-powered chatbot enables non-technical staff to query complaint data in natural language, dramatically reducing the time to identify trends from days to minutes.

### Key Features
- 🔍 Semantic search over 464K+ CFPB complaints
- 💬 Natural language query interface
- 📊 Multi-product analysis (Credit Cards, Personal Loans, Savings Accounts, Money Transfers)
- 🎯 Metadata filtering for targeted insights
- ⚡ Fast retrieval with vector similarity search

### Technology Stack
- **Embeddings**: sentence-transformers/all-MiniLM-L6-v2 (384 dimensions)
- **Vector Database**: ChromaDB (persistent storage)
- **LLM**: Open-source models (HuggingFace/Ollama)
- **UI Framework**: Gradio
- **Data Processing**: pandas, numpy, scikit-learn
- **Testing**: pytest with 100+ unit tests

## 📁 Project Structure

```
rag-complaint-chatbot/
├── src/                          # Modular Python modules (OOP)
│   ├── data_loader.py           # CFPB data loading
│   ├── eda_analyzer.py          # Exploratory data analysis
│   ├── data_preprocessor.py     # Text cleaning and filtering
│   ├── data_sampler.py          # Stratified sampling
│   ├── text_chunker.py          # Text chunking with overlap
│   ├── embedder.py              # Embedding generation
│   ├── vector_store.py          # ChromaDB management
│   └── README.md                # Module documentation
├── tests/                        # Unit tests (pytest)
│   ├── test_data_loader.py      # 11 tests
│   ├── test_eda_analyzer.py     # 16 tests
│   ├── test_data_preprocessor.py # 25 tests
│   ├── test_data_sampler.py     # 19 tests
│   ├── test_text_chunker.py     # 29 tests
│   ├── test_embedder.py         # Embedder tests
│   └── test_vector_store.py     # 25+ tests
├── notebooks/                    # Analysis notebooks
│   ├── 01_eda_preprocessing.ipynb
│   └── 02_chunking_embedding.ipynb
├── data/
│   ├── raw/                     # Original CFPB dataset
│   └── processed/               # Filtered/cleaned data
├── vector_store/                # Persisted ChromaDB index
├── docs/local/                  # Project documentation
│   ├── project-overview.md
│   ├── steps.md
│   ├── final-code-eval.md
│   └── interim-code-eval.md
├── app.py                       # Gradio UI (coming in Task 4)
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- 4GB+ RAM (for embedding generation)
- 2GB+ disk space (for models and vector store)

### Installation

```bash
# Clone repository
git clone <repository-url>
cd rag-complaint-chatbot

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download NLTK data (if needed)
python -c "import nltk; nltk.download('punkt')"
```

### Usage

#### 1. Data Preprocessing (Task 1)
```python
from src.data_loader import CFPBDataLoader
from src.data_preprocessor import DataPreprocessor

# Load data
loader = CFPBDataLoader('data/raw/complaints.csv')
df = loader.load_raw_data(nrows=1000000)  # Use 1M rows for performance

# Preprocess
preprocessor = DataPreprocessor(df)
cleaned_df = preprocessor.preprocess_pipeline(
    products=['Credit card', 'Personal loan', 'Savings account', 'Money transfer']
)
preprocessor.save_data(cleaned_df, 'data/processed/filtered_complaints.csv')
```

Or run the EDA notebook:
```bash
jupyter notebook notebooks/01_eda_preprocessing.ipynb
```

#### 2. Create Vector Store (Task 2)
```python
from src.data_sampler import StratifiedSampler
from src.text_chunker import TextChunker
from src.embedder import EmbeddingGenerator
from src.vector_store import VectorStoreManager

# Create stratified sample
sampler = StratifiedSampler(cleaned_df, product_column='Product')
sample = sampler.create_stratified_sample(sample_size=12000, random_state=42)

# Chunk documents
chunker = TextChunker(chunk_size=500, chunk_overlap=50)
chunks_df = chunker.chunk_documents(
    sample,
    text_column='Consumer complaint narrative',
    metadata_columns=['Product', 'Issue', 'Company']
)

# Generate embeddings
embedder = EmbeddingGenerator(model_name="sentence-transformers/all-MiniLM-L6-v2")
embeddings = embedder.batch_embed(chunks_df['text'].tolist(), batch_size=32)

# Build vector store
vector_store = VectorStoreManager(
    persist_directory="vector_store",
    collection_name="complaints"
)
vector_store.add_documents_batch(chunks_df, embeddings, text_column='text')
```

Or run the chunking/embedding notebook:
```bash
jupyter notebook notebooks/02_chunking_embedding.ipynb
```

#### 3. Query the System
```python
# Search similar complaints
query = "I was charged unexpected fees on my credit card"
query_embedding = embedder.generate_embedding(query)
results = vector_store.search(query_embedding, top_k=5)

for doc, metadata, distance in zip(
    results['documents'], 
    results['metadatas'], 
    results['distances']
):
    print(f"Distance: {distance:.4f}")
    print(f"Product: {metadata['Product']}")
    print(f"Text: {doc[:200]}...")
    print("-" * 80)
```

#### 4. Launch Interactive Chat Interface (Task 4)
```bash
# Run the Gradio application
python app.py
```

The app will start on `http://localhost:7860`. Features include:
- 💬 Natural language question input
- 🤖 AI-generated answers based on retrieved complaints
- 📄 Source document display with metadata
- 🗑️ Clear conversation button
- ✅ Real-time system status

**Example questions to try:**
- "What are the main complaints about credit cards?"
- "Why are customers unhappy with personal loans?"
- "What issues do people report with money transfers?"
- "Compare complaint trends across different products"

## 🧪 Testing

Run all tests:
```bash
pytest tests/ -v
```

Run specific test module:
```bash
pytest tests/test_data_loader.py -v
pytest tests/test_text_chunker.py -v
```

Run with coverage:
```bash
pytest tests/ --cov=src --cov-report=html
```

**Current Test Coverage**: 100+ unit tests across all modules

## 📊 Project Progress

### ✅ Completed (Tasks 1 & 2)

**Task 1: EDA and Data Preprocessing**
- [x] Data loading with validation
- [x] Product distribution analysis
- [x] Narrative length analysis
- [x] Missing data identification
- [x] Text cleaning (URLs, emails, boilerplate removal)
- [x] Filtered dataset saved to `data/processed/`

**Task 2: Text Chunking, Embedding, and Vector Store**
- [x] Stratified sampling (10K-15K complaints)
- [x] Text chunking (500 chars, 50 overlap)
- [x] Embedding generation (all-MiniLM-L6-v2)
- [x] ChromaDB vector store creation
- [x] Persistence and metadata support
- [x] Similarity search with filtering

### 🔄 In Progress

**Task 3: RAG Core Logic and Evaluation**
- [x] Retriever module
- [x] Prompt engineering
- [x] LLM integration
- [x] RAG pipeline
- [x] Evaluation framework

**Task 4: Interactive Chat Interface**
- [x] Gradio UI with chat interface
- [x] Source display with metadata
- [x] Chat history management
- [x] Clear functionality
- [x] System status display
- [x] Error handling

## 🏗️ Architecture

### Data Flow
```
CFPB Dataset (464K complaints)
    ↓
Preprocessing (filter products, clean text)
    ↓
Stratified Sampling (12K complaints)
    ↓
Text Chunking (500 chars, 50 overlap)
    ↓
Embedding Generation (all-MiniLM-L6-v2)
    ↓
Vector Store (ChromaDB)
    ↓
Semantic Search (top-k retrieval)
    ↓
LLM Generation (context + query → answer)
    ↓
User Interface (Gradio chatbot)
```

### Design Principles
- **Modularity**: Each class has a single responsibility
- **Testability**: Comprehensive unit test coverage
- **Documentation**: Google-style docstrings for all methods
- **Error Handling**: Robust validation and logging
- **Scalability**: Batch processing and efficient storage

## 📈 Key Metrics

| Metric              | Value      |
| ------------------- | ---------- |
| Total Complaints    | 464K+      |
| Sample Size         | 12,000     |
| Text Chunks         | ~30,000    |
| Embedding Dimension | 384        |
| Vector Store Size   | ~50MB      |
| Avg Chunks/Document | 2.5        |
| Test Coverage       | 100+ tests |

## 🎓 Technical Decisions

### Chunking Strategy
- **Size**: 500 characters (~100-150 words)
- **Overlap**: 50 characters (10%)
- **Rationale**: Balances context preservation with granularity

### Embedding Model
- **Model**: all-MiniLM-L6-v2
- **Size**: 80MB, 384 dimensions
- **Speed**: ~1000 sentences/sec on CPU
- **Quality**: Trained on 1B+ sentence pairs

### Vector Database
- **Choice**: ChromaDB
- **Persistence**: Disk-based storage
- **Features**: Metadata filtering, similarity search
- **Advantage**: Easy setup, no server required

## 📚 Documentation

- [Project Overview](docs/local/project-overview.md) - Business context and objectives
- [Implementation Steps](docs/local/steps.md) - Detailed task breakdown
- [Module Documentation](src/README.md) - API reference
- [Evaluation Criteria](docs/local/final-code-eval.md) - Grading rubric

## 🔧 Development

### Adding a New Module

1. Create module in `src/`
```python
# src/my_module.py
class MyModule:
    """Module description."""
    
    def __init__(self):
        """Initialize module."""
        pass
```

2. Create tests in `tests/`
```python
# tests/test_my_module.py
def test_my_module():
    module = MyModule()
    assert module is not None
```

3. Update documentation in `src/README.md`

4. Run tests
```bash
pytest tests/test_my_module.py -v
```

### Git Workflow

```bash
# Create feature branch
git checkout -b task-3

# Make changes and commit
git add .
git commit -m "feat: implement retriever module"

# Push and create PR
git push origin task-3
```

## 🤝 Contributing

1. Follow PEP 8 style guide
2. Write docstrings for all classes and methods
3. Add unit tests for new functionality
4. Update documentation as needed
5. Use meaningful commit messages

## 📝 License

This project is for educational purposes as part of the 10Academy AI Mastery program.

## 👥 Team

- Data & AI Engineer: Building RAG system for complaint analysis
- Product Manager: Asha (internal stakeholder)
- Target Users: Product, Support, and Compliance teams

## 🔗 Resources

- [Gradio Documentation](https://www.gradio.app/docs)
- [ChromaDB Docs](https://docs.trychroma.com)
- [Sentence Transformers](https://www.sbert.net)
- [CFPB Complaint Database](https://www.consumerfinance.gov/data-research/consumer-complaints/)

---

**Last Updated**: January 13, 2026  
**Project Status**: Task 4 Complete (Interactive Chat Interface Ready)  
**Next Milestone**: Final Submission & Demo
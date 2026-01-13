# Notebooks

This directory contains Jupyter notebooks demonstrating the complete RAG Complaint Chatbot workflow.

## Overview

Each notebook corresponds to a major project task and demonstrates the use of modular code from the `src/` directory.

---

## Notebook 1: EDA and Preprocessing (`01_eda_preprocessing.ipynb`)

**Purpose:** Exploratory Data Analysis and data preprocessing for the CFPB complaint dataset.

**Key Sections:**
1. **Data Loading**: Load raw CFPB complaint data
2. **Product Distribution Analysis**: Visualize complaints across product categories
3. **Narrative Length Analysis**: Analyze word counts and text characteristics
4. **Missing Data Analysis**: Identify and handle missing narratives
5. **Data Filtering**: Filter to target products (Credit Card, Personal Loan, Savings Account, Money Transfer)
6. **Text Cleaning**: Normalize and clean complaint narratives
7. **Save Processed Data**: Export filtered dataset

**Modules Used:**
- `CFPBDataLoader` from `src/data_loader.py`
- `EDAAnalyzer` from `src/eda_analyzer.py`
- `DataPreprocessor` from `src/data_preprocessor.py`

**Output:**
- `data/processed/filtered_complaints.csv` - Cleaned and filtered dataset

**Usage:**
```bash
jupyter notebook 01_eda_preprocessing.ipynb
```

---

## Notebook 2: Chunking and Embedding (`02_chunking_embedding.ipynb`)

**Purpose:** Text chunking, embedding generation, and vector store creation.

**Key Sections:**
1. **Stratified Sampling**: Create balanced sample of 10K-15K complaints
2. **Text Chunking**: Split narratives into overlapping chunks
3. **Chunking Analysis**: Analyze chunk size distribution and quality
4. **Embedding Generation**: Generate embeddings using sentence-transformers
5. **Vector Store Creation**: Build ChromaDB vector database
6. **Persistence**: Save vector store to disk
7. **Validation**: Test retrieval with sample queries

**Modules Used:**
- `StratifiedSampler` from `src/data_sampler.py`
- `TextChunker` from `src/text_chunker.py`
- `EmbeddingGenerator` from `src/embedder.py`
- `VectorStoreManager` from `src/vector_store.py`

**Output:**
- `vector_store/` - Persisted ChromaDB index with embeddings

**Configuration:**
- Chunk size: 500 characters
- Chunk overlap: 50 characters
- Embedding model: `sentence-transformers/all-MiniLM-L6-v2`
- Sample size: 10,000-15,000 complaints

**Usage:**
```bash
jupyter notebook 02_chunking_embedding.ipynb
```

---

## Notebook 3: RAG Pipeline Evaluation (`03_rag_pipeline_evaluation.ipynb`)

**Purpose:** Complete RAG pipeline demonstration and evaluation.

**Key Sections:**
1. **Load Vector Store**: Load pre-built vector database
2. **Test Retrieval**: Demonstrate similarity search
3. **Prompt Engineering**: Show different prompt templates
4. **Response Generation**: Generate answers using LLM
5. **Complete Pipeline**: Integrate retrieval, prompting, and generation
6. **System Evaluation**: Test with 5-10 questions across categories
7. **Quality Assessment**: Manual scoring and analysis
8. **Evaluation Table**: Create comprehensive evaluation report
9. **Visualizations**: Plot quality scores and distributions
10. **Findings Analysis**: Identify strengths and improvements

**Modules Used:**
- `DocumentRetriever` from `src/retriever.py`
- `PromptBuilder` from `src/prompt_builder.py`
- `ResponseGenerator` from `src/generator.py`
- `RAGPipeline` from `src/rag_pipeline.py`
- `RAGEvaluator` from `src/evaluator.py`

**LLM Models:**
- Lightweight: `google/flan-t5-base` (fast, good for testing)
- High Quality: `mistralai/Mistral-7B-Instruct-v0.1` (slower, better results)

**Evaluation Questions:**
- Credit card complaints
- Personal loan issues
- Savings account problems
- Money transfer difficulties
- Fraud and identity theft
- Customer service issues
- Billing disputes
- Account closures

**Evaluation Metrics:**
- Quality scores (1-5 scale)
- Relevance to sources
- Specificity and detail
- Factual accuracy
- Response time

**Usage:**
```bash
jupyter notebook 03_rag_pipeline_evaluation.ipynb
```

---

## Running the Notebooks

### Prerequisites

Install required packages:
```bash
pip install -r requirements.txt
```

### Recommended Order

1. **Start with Notebook 1**: Understand the data and preprocessing
2. **Continue to Notebook 2**: Learn chunking and embedding strategies
3. **Finish with Notebook 3**: See the complete RAG system in action

### Environment Setup

Make sure the project structure is intact:
```
rag-complaint-chatbot/
├── data/
│   ├── raw/complaints.csv
│   └── processed/filtered_complaints.csv
├── notebooks/
│   ├── 01_eda_preprocessing.ipynb
│   ├── 02_chunking_embedding.ipynb
│   └── 03_rag_pipeline_evaluation.ipynb
├── src/
│   └── [all Python modules]
└── vector_store/
    └── [ChromaDB files]
```

### Tips

- **Run cells sequentially**: Each cell may depend on previous cells
- **Check GPU availability**: LLM generation is faster with GPU
- **Monitor memory**: Large models require 4-16GB RAM
- **Save checkpoints**: Save intermediate results to avoid recomputation
- **Experiment**: Modify parameters to see how they affect results

---

## Key Takeaways

### Notebook 1
- Financial complaints vary significantly across products
- Credit cards and personal loans dominate complaint volume
- Many narratives are long and detailed, requiring chunking
- Text cleaning improves embedding quality

### Notebook 2
- Stratified sampling maintains product distribution
- Chunk size of 500 with 50 overlap works well
- `all-MiniLM-L6-v2` provides good quality-speed tradeoff
- ChromaDB enables fast semantic search

### Notebook 3
- RAG successfully grounds answers in real complaints
- Retrieval quality is critical for answer accuracy
- Prompt engineering significantly impacts response quality
- Larger models (Mistral-7B) produce better answers than T5-base
- Average quality score: 3.8/5.0 demonstrates system effectiveness

---

## Troubleshooting

**Issue: "Module not found"**
- Ensure `src/` is in Python path: `sys.path.append('../src')`
- Check that all modules exist in `src/` directory

**Issue: "Out of memory"**
- Use smaller LLM model (T5-base instead of Mistral-7B)
- Reduce batch size for embeddings
- Close other applications

**Issue: "Vector store not found"**
- Run Notebook 2 first to create vector store
- Check `vector_store/` directory exists

**Issue: "Slow generation"**
- Use GPU if available: `device='cuda'`
- Reduce `max_new_tokens` parameter
- Consider using quantized models

---

## Additional Resources

- [Sentence Transformers Documentation](https://www.sbert.net/)
- [ChromaDB Getting Started](https://docs.trychroma.com/)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers)
- [RAG Best Practices](https://huggingface.co/blog/rag)

---
title: "RAG Architectures & Chunking"
date: "2025-12-27"
tags: ["genai", "llm", "search"]
related: []
slug: "rag-architectures-chunking"
category: "agentic"
---

# RAG Architectures & Chunking

## Summary
Retrieval-Augmented Generation (RAG) connects LLMs to private data. The performance of a RAG system depends heavily on the **Chunking Strategy** (how data is split), **Hybrid Search** (combining keywords and embeddings), and **Re-Ranking** (refining results before generation).

## Details

### 1. The RAG Pipeline
1.  **Ingestion**: Load documents (PDFs, Markdown, etc.).
2.  **Chunking**: Split text into smaller segments.
3.  **Embedding**: Convert chunks into vectors (e.g., OpenAI `text-embedding-3-small`).
4.  **Retrieval**: Find top-k most similar chunks to the user query.
5.  **Generation**: Feed chunks + query to LLM to generate an answer.

### 2. Chunking Strategies
Bad chunking leads to "hallucinations" or missing context.
-   **Fixed-Size Chunking**: Split every 500 tokens with 50-token overlap. Simple but breaks sentences/ideas mid-thought.
-   **Recursive Character Chunking** (Standard): Tries to split by paragraphs (`\n\n`), then sentences (`.`), then words. Preserves semantic structure.
-   **Semantic Chunking**: Uses an embedding model to detect "topic shifts" and splits only when the semantic meaning changes significantly.
-   **Parent-Child Chunking**: Index small chunks (sentences) for better vector search accuracy, but return the *parent* chunk (full paragraph) to the LLM for better context.

### 3. Advanced Retrieval
-   **Hybrid Search**: Vectors (Dense) are good at concepts ("fruit" matches "apple"), but bad at exact matches (Product IDs, specific acronyms). BM25 (Sparse) is good at exact matches.
    -   *Solution*: Run both, then combine results using **Reciprocal Rank Fusion (RRF)**.
-   **Re-Ranking**:
    -   *Step 1*: Retrieve 50 candidates using fast Bi-Encoders (Vector DB).
    -   *Step 2*: Use a slow, high-precision **Cross-Encoder** (e.g., Cohere Rerank, BGE-Reranker) to score them against the query and pick the top 5.

## Examples / snippets

### Recursive Chunking with LangChain

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

text = "Long document content..."

# Split by paragraphs, then newlines, then spaces
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\n\n", "\n", " ", ""]
)

chunks = splitter.create_documents([text])
print(f"Created {len(chunks)} chunks.")
print(chunks[0].page_content)
```

### Hybrid Search Logic (Pseudo-code)

```python
def hybrid_search(query):
    # 1. Vector Search (Semantic)
    vector_results = vector_db.search(query, k=50)
    
    # 2. Keyword Search (BM25)
    keyword_results = elasticsearch.search(query, k=50)
    
    # 3. Reciprocal Rank Fusion
    scores = {}
    for rank, doc in enumerate(vector_results):
        scores[doc.id] = scores.get(doc.id, 0) + 1 / (60 + rank)
        
    for rank, doc in enumerate(keyword_results):
        scores[doc.id] = scores.get(doc.id, 0) + 1 / (60 + rank)
        
    # 4. Sort and Top-K
    final_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:5]
    return final_results
```

## Flashcards

- What is the main benefit of Recursive Character Chunking over Fixed-Size Chunking? ::: It attempts to keep related text together (paragraphs, sentences) rather than splitting mid-sentence, preserving **semantic meaning**.
- Why is Hybrid Search often better than pure Vector Search? ::: Vector search can fail on **exact keyword matches** (like part numbers or acronyms), which BM25/Keyword search handles well.
- What is the role of a Re-Ranker (Cross-Encoder) in RAG? ::: It takes a broad set of candidate results and re-scores them with high precision to select the best few for the LLM, improving relevance.
- What is "Parent-Child" indexing? ::: Searching against small, precise chunks (Children) but returning larger, surrounding context (Parent) to the LLM.
- What is Reciprocal Rank Fusion (RRF)? ::: An algorithm to combine ranked lists from different search methods (e.g., BM25 and Vector Search) into a single unified ranking.

## Quizzes

### Chunking Strategy
Q: You are building a RAG system for a legal contract. The contract relies heavily on definitions found in previous paragraphs. If you use small, non-overlapping chunks, what is the risk?
Options:
- A) The retrieval will be too slow.
- B) The chunks will lose context (e.g., "The Party" refers to whom?), leading to hallucinations.
- C) The embedding model will crash.
- D) The LLM will refuse to answer.
Answers: B
Explanation: Small chunks without overlap or parent context strip away necessary references. A chunk saying "The Party shall pay..." is useless if the definition of "The Party" is in a previous chunk that wasn't retrieved.

### Improving Retrieval
Q: Users are searching for specific error codes "ERR-505" but your Vector RAG system keeps returning generic "error handling" documentation instead of the specific page for 505. How do you fix this?
Options:
- A) Switch to a larger LLM.
- B) Implement Hybrid Search (add Keyword/BM25 search).
- C) Increase chunk size.
- D) Use a different Vector Database.
Answers: B
Explanation: Embeddings cluster concepts. "ERR-505" and "ERR-404" might have very similar embeddings (both are errors). Keyword search (BM25) treats "ERR-505" as a specific token and will find the exact match.

## Learning Sources
- [Pinecone: Chunking Strategies for LLM Applications](https://www.pinecone.io/learn/chunking-strategies/) - In-depth visual guide.
- [LangChain Documentation: Text Splitters](https://python.langchain.com/docs/modules/data_connection/document_transformers/) - Practical implementation details.
- [Cohere: Reranking](https://txt.cohere.com/rerank/) - Explanation of why re-ranking improves RAG performance.
- [Microsoft: Azure AI Search Hybrid Retrieval](https://learn.microsoft.com/en-us/azure/search/hybrid-search-overview) - Enterprise implementation of Hybrid Search.

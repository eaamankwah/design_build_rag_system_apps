# Mastering RAG: Build Smart, Data-Driven Applications

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![LangChain](https://img.shields.io/badge/LangChain-0.1.16-green.svg)](https://langchain.com/)
[![IBM watsonx.ai](https://img.shields.io/badge/IBM-watsonx.ai-blue.svg)](https://www.ibm.com/products/watsonx-ai)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🚀 Overview

This repository contains a comprehensive collection of Retrieval-Augmented Generation (RAG) projects that demonstrate advanced AI application development capabilities. The projects showcase expertise in building intelligent, data-driven systems that combine the power of Large Language Models (LLMs) with external knowledge sources.

## 🎯 What is RAG?

Retrieval-Augmented Generation (RAG) is a cutting-edge technique that enhances Large Language Models by integrating external data sources into their reasoning process. While LLMs excel at broad reasoning, their knowledge is limited to training data up to a specific cutoff date. RAG addresses this limitation by:

* ***Dynamic Knowledge Integration**: Retrieving relevant information and inserting it into model prompts
* **Private Data Processing**: Enabling AI to reason about proprietary or newly introduced data
* **Contextual Accuracy**: Ensuring responses are accurate, timely, and contextually relevant

## 🏗️ Architecture Overview

### Core Components

1. **Indexing Pipeline** (Offline)
   * Document loading and preprocessing
   * Text splitting and chunking
   * Vector embedding generation
   * Storage in vector databases

2. **Retrieval & Generation** (Runtime)
   * Query processing and similarity search
   * Relevant context retrieval
   * LLM prompt augmentation
   * Response generation

```mermaid
graph LR
    A[Documents] --> B[Text Splitter]
    B --> C[Embeddings]
    C --> D[Vector Store]
    E[User Query] --> F[Retriever]
    D --> F
    F --> G[LLM]
    G --> H[Response]
```

## 📚 Project Portfolio

### 1. 🔒 Private Document Summarization
**Status:** ✅ Complete | **Tech Stack:** RAG, LangChain, IBM watsonx.ai

A sophisticated document processing system that enables secure summarization and Q&A capabilities for private documents without external data exposure.

#### Key Features
* **Multi-Model Support**: Integration with FLAN-T5-XL and Llama 3.3-70B models
* **Intelligent Chunking**: Optimized text splitting for efficient processing
* **Conversational Memory**: Context-aware dialogue capabilities
* **Source Attribution**: Traceable responses with document references

#### Technical Implementation
```python
# Core architecture components
* Document Loader: TextLoader for file ingestion
* Text Splitter: CharacterTextSplitter with 1000-character chunks
* Embeddings: HuggingFace sentence transformers
* Vector Store: ChromaDB for efficient retrieval
* LLM Integration: IBM watsonx.ai models via LangChain
```

### 2. 🔍 Granite 3 Retrieval Agent
**Status:** 🚧 In Development | **Tech Stack:** LlamaIndex, Granite 3.0

Advanced retrieval system supporting multiple data formats (PDFs, HTML, text files) with precision-focused insights.

### 3. 🌐 Web Data RAG System
**Status:** 📋 Planned | **Tech Stack:** LangChain, Llama 3.1

Real-time web data processing and analysis system for dynamic, context-aware interactions.

### 4. 📺 YouTube Content Processor
**Status:** 📋 Planned | **Tech Stack:** FAISS, RAG, NLP

Automated video transcript extraction, summarization, and interactive Q&A system development.

### 5. 🤝 AI Icebreaker Bot
**Status:** 📋 Planned | **Tech Stack:** Granite 3.0, LlamaIndex, ProxyCurl API

LinkedIn profile analysis and personalized conversation starter generation for professional networking.

## 🛠️ Technical Stack

### Core Technologies
* **Language Models**: IBM watsonx.ai (Granite, Llama, FLAN-T5)
* **Frameworks**: LangChain, LlamaIndex
* **Vector Databases**: ChromaDB, FAISS
* **Embeddings**: HuggingFace Transformers, Sentence-BERT
* **APIs**: ProxyCurl (LinkedIn), IBM watsonx.ai

---

### Project 2. Granite 3 Retrieval Agent with LlamaIndex

An advanced RAG application using LlamaIndex framework with IBM Granite 3.0 model for precise document retrieval and expert-level query responses on scientific and technical documents.

#### Key Features
* **LlamaIndex Framework**: Complete RAG pipeline implementation using modern indexing techniques
* **Granite 3.8B Instruct Model**: IBM's latest instruction-tuned dense decoder model trained on 12 trillion tokens
* **Advanced Document Processing**: PDF processing with intelligent chunking (500-character nodes)
* **Vector Embeddings**: IBM Slate-125M English retriever for semantic understanding  
* **Precision Retrieval**: Configurable similarity search with top-k results
* **Scientific Focus**: Optimized for research papers and technical documentation
* **Interactive Query Engine**: Real-time question-answering with contextual responses
* **Multi-Language Support**: 12 natural languages and 116 programming languages

#### Technical Architecture
```mermaid
graph TB
    subgraph "Document Loading & Processing"
        A[PDF Documents] --> B[SimpleDirectoryReader]
        B --> C[Document Objects]
        C --> D[SentenceSplitter<br/>500 char chunks]
        D --> E[Node Objects]
    end
    
    subgraph "Embedding & Indexing"
        E --> F[IBM Slate-125M<br/>Embedding Model]
        F --> G[VectorStoreIndex]
        G --> H[Vector Embeddings]
    end
    
    subgraph "Query Processing"
        I[User Query] --> J[Retriever<br/>Top-K Similarity]
        H --> J
        J --> K[Retrieved Context]
    end
    
    subgraph "Response Generation"
        K --> L[Granite 3.8B<br/>Instruct Model]
        I --> L
        L --> M[Contextual Response]
    end
    
    style A fill:#ffebee
    style G fill:#f3e5f5
    style L fill:#e3f2fd
    style M fill:#e8f5e8
```

```python
# LlamaIndex RAG Pipeline
PDF Loading → Document Parsing → Node Splitting → Vector Embedding → Index Creation

# Core Components:
├── SimpleDirectoryReader (PDF ingestion)
├── SentenceSplitter (500-char chunking)  
├── WatsonxEmbeddings (IBM Slate-125M)
├── VectorStoreIndex (Efficient storage)
├── Query Engine (Retrieval + Generation)
└── WatsonxLLM (Granite 3.8B Instruct)
```

#### Technology Stack
* **LLM Model**: IBM Granite 3.8B Instruct (ibm/granite-3-8b-instruct)
* **Framework**: LlamaIndex 0.10.65
* **Embedding Model**: IBM Slate-125M English Retriever
* **Document Processing**: SimpleDirectoryReader, SentenceSplitter
* **Vector Store**: VectorStoreIndex with similarity search
* **Platform**: IBM watsonx.ai
* **API Integration**: llama-index-llms-ibm, llama-index-embeddings-ibm
* **Configuration**: Temperature 0.1, max_new_tokens 75, top-k retrieval

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](https://creativecommons.org/licenses/by/4.0/) file for details.

## 🔗 Resources

* [IBM watsonx.ai Documentation](https://docs.anthropic.com)
* [LangChain Documentation](https://langchain.com/docs)
* [RAG Best Practices Guide](https://example.com/rag-guide)


# BeagleMind Updates

## A. Milestone 2

### Core RAG Functionality

In this milestone, I focused on finalizing the core RAG functionality by implementing the previously discussed features.

I developed a **Gradio interface** to interact with the chatbot and visually inspect the results, including the sources used to answer each query. These sources are passed through a **reranking model** to ensure that only the most relevant ones are selected before being sent to the LLM.

---

### Chatbot Interface

* Developed using **Gradio** for interactive exploration of BeagleBoard documentation.
* **Markdown-rendered answer area**, supporting:

  * Clean formatting (bold text, bullet points, inline code)
  * Clickable links to documentation files
  * Embedded images and references

The user can ask questions like *"How to blink an LED using BeagleBoard?"*, and the system responds with relevant reformulated documentation snippets.

---

### Sources Section

* Right panel labeled **Sources & References** displays:

  * File name, path, and clickable GitHub link
  * Scoring metrics: **Composite, Rerank, and Content Quality**
  * Formatted content preview for readability

**Note:** Reranking model implementation details are in `retrieval.py`.

---

### Repository Tree Overview

The `src` directory contains the core logic for the QA system:

* `main.py`: Entry point
* `gradio_app.py`: Gradio UI
* `qa_system.py`: Manages end-to-end QA pipeline
* `retrieval.py`: Document retrieval with scoring
* `search_vectorstore.py`: Searches Milvus vector store
* `github_direct_ingester.py`: Pulls data from GitHub repos
* `graph_qa.py`: Prototype for graph-based QA
* `router.py`: Routes requests
* `config.py`: Stores configs and model params

---

### CI Integration with GitLab

To streamline integration with **OpenBeagle**:

* Integrated with **GitLab CI** for continuous deployment.
* Pipeline (WIP) to include **automated testing and linting** to:

  * Maintain code quality
  * Catch errors early
  * Ensure deployment consistency

## B. Milestone 1

### Vector Database Implementation

* Chose **Milvus** for the vector store, based on benchmarking with [VectorDBBench](https://github.com/zilliztech/VectorDBBench)
* Prioritized scalability, performance, and ecosystem support

---

### Chunking Strategy

* Used [Chonkie](https://github.com/chonkie-inc/chonkie), which uses the **CHOMP pipeline** for modular and semantic chunking
* Improves granularity and chunk retrieval accuracy

---

### Retrieval Strategy

* Proposing a **Graph-RAG** approach:

  * Uses structured relationships (e.g., diagrams ↔ code ↔ documentation)
  * Inspired by [GRAG Paper](https://arxiv.org/pdf/2408.08921)
  * Scripts in development to benchmark impact

---

### Embedding Models

* **Primary**: [`BAAI/bge-large-en`](https://huggingface.co/BAAI/bge-large-zh)
* **Reranker Candidate**: [`all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)

---

### Data Source

* Sourced from BeagleBoard GitHub repos (code, diagrams, documentation)
* Indexing metadata to enable semantically rich queries

## C. Milestone 0

### Model Selection

#### a. Mistral-Small-3.1-24B-Instruct-2503-GGUF

* 24B params, instruction-tuned, quantized
* Handles 128K tokens
* Ideal for local deployment (e.g., 4090 GPU)
* Strong multilingual and code performance

#### b. Phi-4 (14B)

* Developed by Microsoft
* Excels in reasoning and competitive programming
* Compact with performance matching larger models

#### c. Qwen2.5-Coder-7B

* Optimized for code gen and repair
* Supports 131K context length
* Memory-efficient with 4-bit quantization

#### d. Qwen3 30B-A3B

* Dual-mode reasoning
* Agentic task support
* Ideal for long-form processing and tool-based interaction

---

### Selection Criteria

* Model size
* MMLU, MBPP, EvalPlus, MATH
* Inference speed (tokens/s)

### Fine-Tuning Environment

* **Google Colab Pro**: Good for prototyping and tuning
* **HF Inference Endpoints**: Considered for hosting (higher cost)
* **Unsloth**: For efficient 4-bit fine-tuning

---

### Data Collection Pipeline

* GitHub Scraper + API
* Markdown, code, PDF processing
* External sources:

  * [eLinux.org](https://elinux.org/BeagleBoard_Community)
  * Datasheets
  * Community forums (if allowed)

### Fine-Tuning Dataset Prep

* QA pair generation via:

  * Manual annotation
  * LLM-based synthetic prompts (verified)

---

## Fine-Tuned Model Evaluation

### Agentic Evaluation

* Evaluates reasoning, planning, tool usage

Tools considered:

* **DeepEval**
* **Opik**
* **JudgeLM**
* **AgentBench**

### Metric-Based Evaluation

* **Perplexity**
* **BLEU / ROUGE**
* **F1 Score**
* **BERTScore**
* **Exact Match (EM)**
* **Latency / Throughput**

---

## Demo Preparation

Working on improving the RAG architecture using advanced retrieval methods:

### Techniques

1. **Graph RAG (GRAG)**: Leverages structured entity relationships
2. **Contextual Semantic Search**: Uses semantic embeddings + cross-encoders
3. **Dense Passage Retrieval (DPR)**: Efficient dual-encoder retrieval

---

These approaches aim to outperform traditional RAG by improving the **contextual relevance** and **accuracy** of responses, especially on the BeagleBoard dataset.

➡️ **Demo will be recorded in an introductory video.**

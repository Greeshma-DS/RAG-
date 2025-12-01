# Medically Grounded RAG System for Diabetes Education using Open-Source LLMs

**Final Project – MS Data Science, University of New Haven**  
**Authors:** Greeshma Chanduri, Kamineni Pravalika  
**Date:** December 2025  

**Presentation Slides:** [slides/Diabetes_RAG_Presentation](https://youtu.be/jxGDuHZlGH0?si=g0ekO1GufbB5AwwX)

## Overview
A fully reproducible, end-to-end Retrieval-Augmented Generation (RAG) system for patient-facing diabetes education using **only open-source components** deployable on a laptop.

Key features:
- 4 trusted diabetes PDFs as the sole knowledge source
- ChromaDB + nomic-embed-text embeddings
- Three LLMs via Ollama: Llama-3.1-8B, Mistral-7B-Instruct-v0.3, Phi-3-Medium
- Automatic embedding-based grounding/hallucination metric
- Zero detected hallucinations with Mistral-7B under strict retrieval constraints

## Quick Start (one-command setup)

```bash
# 1. Clone the repo
git clone https://github.com/yourusername/diabetes-rag-open-source-llms.git
cd diabetes-rag-open-source-llms

# 2. Install dependencies
pip install -r code/requirements.txt

# 3. Start Ollama (in another terminal)
ollama serve

# 4. Pull models (first time only)
ollama pull nomic-embed-text
ollama pull llama3.1:8b
ollama pull mistral:7b-instruct
ollama pull phi3:medium

# 5. Build the vector index (only once)
python code/rag.py --build

# 6. Run interactive demo
python code/rag.py --demo

# 7. Reproduce full benchmark (30 answers + grounding scores)
python code/rag.py --evaluate

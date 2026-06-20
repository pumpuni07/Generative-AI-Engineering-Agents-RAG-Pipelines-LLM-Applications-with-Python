# Building AI Agents & RAG Systems from Scratch with Python
## IBM AI Engineering Professional Certificate — Labs

**Author:** Jack Pumpuni Frimpong-Manso  
**GitHub:** [@pumpuni07](https://github.com/pumpuni07)  


## Repository Overview

Seven fully implemented, production-quality Python labs covering the complete
IBM AI Engineering track — from multi-agent chatbots to RAG pipelines with
Gradio front-ends and IBM watsonx.ai integration.

| File | Lab | Key Concepts |
|------|-----|--------------|
| `lab1_ai_agents_chatbot.py` | Building AI Agents from Scratch | Query routing, agent design, keyword matching, interactive loops |
| `lab2_rag_huggingface_gpt2.py` | RAG with Hugging Face | DPR retrieval, GPT-2 generation, beam search parameter tuning |
| `lab3_rag_pytorch_song_safety.py` | RAG with PyTorch | Cosine vs dot-product similarity, embedding-based content classification |
| `lab4_langchain_prompt_engineering.py` | In-Context Engineering | Prompt templates, LangChain chains, few-shot prompting, role/tone control |
| `lab5_gradio_interface.py` | Gradio Interface Design | All Gradio input types, LLM chatbot front-end, Gradio + watsonx.ai |
| `lab6_watsonx_llm_quickstart.py` | IBM watsonx.ai Quickstart | Llama, Mixtral & Granite models, parameter tuning, terminal Q&A |
| `lab7_qabot_pdf_rag.py` | Full RAG QA Bot | PDF loading, chunking, embeddings, ChromaDB, RetrievalQA, Gradio UI |

---

## Lab Descriptions

### Lab 1 — Building AI Agents from Scratch (`lab1_ai_agents_chatbot.py`)

A restaurant chatbot for **The Daily Dish** that routes user queries to
specialised agents:

- **QueryProcessor**: Lowercase → clean punctuation → expand synonyms
- **Router**: Classifies each query as `weather` or `restaurant`
- **WeatherAgent**: Live OpenWeatherMap API + simulation fallback
- **DailyDishAgent**: Menu + FAQ knowledge base with keyword matching
- **Interactive loop** with built-in unit tests (`--test` flag)

```bash
python lab1_ai_agents_chatbot.py          # Chat mode
python lab1_ai_agents_chatbot.py --test   # Unit tests
```

---

### Lab 2 — RAG with Hugging Face + GPT-2 (`lab2_rag_huggingface_gpt2.py`)

Full DPR + GPT-2 RAG pipeline:

- Facebook DPR context/question encoders for dense passage retrieval
- Dot product and cosine similarity retrieval (exercise solution)
- GPT-2 generation with and without retrieved context
- Parameter tuning exercise: `max_length`, `min_length`, `length_penalty`, `num_beams`

```bash
pip install transformers torch faiss-cpu
python lab2_rag_huggingface_gpt2.py
```

---

### Lab 3 — RAG with PyTorch: Song Safety (`lab3_rag_pytorch_song_safety.py`)

Content moderation classifier for a social media platform:

- SentenceTransformer embeddings (MiniLM-L6-v2)
- Pre-answered Q&A knowledge base for child-safety classification
- `RAG_QA()` with **cosine similarity** (exercise solution, full derivation)
- Evaluation comparing dot product vs cosine accuracy

```bash
pip install torch sentence-transformers
python lab3_rag_pytorch_song_safety.py
```

---

### Lab 4 — In-Context Engineering with LangChain (`lab4_langchain_prompt_engineering.py`)

LLM behaviour control through prompt structure alone:

- Role + tone `PromptTemplate` ("You are a game master...")
- `FewShotPromptTemplate` for consistent output format
- LangChain `PromptTemplate | LLM | StrOutputParser` chains
- Exercise 1: 4-configuration LLM parameter comparison
- Multi-turn `ConversationManager` with message history
- Interactive Game Master chat loop

```bash
pip install langchain langchain-openai openai python-dotenv
export OPENAI_API_KEY=your_key
python lab4_langchain_prompt_engineering.py --chat
```

---

### Lab 5 — Gradio Interface Design (`lab5_gradio_interface.py`)

Four Gradio demos in one file:

- **Demo 1** (`--demo sum`): Numeric calculator — intro to `gr.Number`
- **Demo 2** (`--demo sentences`): Sentence combiner — Exercise solution
- **Demo 3** (`--demo inputs`): All common input types — `gr.Slider`, `gr.Dropdown`, `gr.CheckboxGroup`, `gr.Radio`, `gr.Checkbox`
- **Demo 4** (`--demo chat`): Full LLM chatbot via IBM watsonx.ai

```bash
pip install gradio ibm-watsonx-ai langchain-ibm
python lab5_gradio_interface.py --demo inputs
python lab5_gradio_interface.py --demo chat
```

---

### Lab 6 — IBM watsonx.ai Quickstart (`lab6_watsonx_llm_quickstart.py`)

Terminal Q&A bot with IBM watsonx.ai foundation models:

- **Llama** (`meta-llama/llama-3-2-11b-vision-instruct`) — default
- **Mixtral** (`mistralai/mistral-small-3-1-24b-instruct-2503`)
- **Granite** (`ibm/granite-4-h-small`)
- Three run modes: `single` query, `chat` loop, `compare` models side-by-side
- Parameter guide: `MAX_NEW_TOKENS`, `TEMPERATURE` effects documented

```bash
pip install ibm-watsonx-ai langchain-ibm
python lab6_watsonx_llm_quickstart.py --mode single --model llama
python lab6_watsonx_llm_quickstart.py --mode compare
```

---

### Lab 7 — Full RAG QA Bot (`lab7_qabot_pdf_rag.py`)

Production-quality PDF Q&A bot — all 6 lab tasks fully implemented:

| Task | Function | Component |
|------|----------|-----------|
| 1 | `document_loader()` | PyPDFLoader |
| 2 | `text_splitter()` | RecursiveCharacterTextSplitter (chunk=1000, overlap=200) |
| 3 | `watsonx_embedding()` | IBM Slate 125M embeddings |
| 4 | `vector_database()` | ChromaDB vector store |
| 5 | `retriever()` | Similarity search retriever |
| 6 | `retriever_qa()` | RetrievalQA chain (`stuff` type) + Gradio UI |

```bash
pip install gradio ibm-watsonx-ai langchain langchain-community \
            langchain-ibm chromadb pypdf huggingface_hub
python lab7_qabot_pdf_rag.py
# Open http://127.0.0.1:7860 → upload a PDF → ask questions
```

---

## Setup

```bash
git clone https://github.com/pumpuni07/ai-agents-labs.git
cd ai-agents-labs
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

---

## Concepts Covered

| Concept | Labs |
|---------|------|
| Agent architecture & query routing | 1 |
| Text preprocessing & synonym expansion | 1 |
| Dense Passage Retrieval (DPR) | 2 |
| GPT-2 beam search & generation parameters | 2 |
| Cosine similarity vs dot product | 2, 3 |
| Embedding-based semantic search | 2, 3, 7 |
| Content classification with RAG | 3 |
| Prompt templates & in-context engineering | 4 |
| Few-shot prompting | 4 |
| LangChain chains & output parsers | 4, 7 |
| LLM parameter tuning | 2, 4, 6 |
| Multi-turn conversation management | 4 |
| Gradio UI design (all input/output types) | 5 |
| IBM watsonx.ai (Llama, Mixtral, Granite) | 5, 6, 7 |
| PDF document loading (PyPDFLoader) | 7 |
| Recursive character text splitting | 7 |
| Vector database (ChromaDB) | 7 |
| RetrievalQA chain types (stuff, map_reduce) | 7 |
| End-to-end RAG pipeline | 7 |

---

## References

- IBM Skills Network — Generative AI Engineering Professional Certificate
- Hugging Face Transformers: https://huggingface.co/docs/transformers
- LangChain Documentation: https://python.langchain.com/docs
- IBM watsonx.ai SDK: https://ibm.github.io/watsonx-ai-python-sdk/
- Gradio Documentation: https://www.gradio.app/docs
- Sentence Transformers: https://www.sbert.net
- ChromaDB: https://www.trychroma.com/

---

*© 2024 Jack Pumpuni Frimpong-Manso. Code extended from IBM Skills Network lab materials (Apache 2.0).*

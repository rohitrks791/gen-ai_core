# GenAI Learning Project

This repository is a hands-on learning project for Generative AI, built using Python, LangChain, Hugging Face, NVIDIA endpoints, embeddings, and vector search workflows.

The goal of this project is to learn how modern AI applications work in real life: chatbots, prompt engineering, RAG pipelines, vector databases, and model integration.

---

## What I learned in this project

This repo covers the main building blocks of modern GenAI:

- LLM basics and how models generate text
- Prompt engineering and structured prompts
- System, Human, and AI messages in chat workflows
- LangChain integrations with different providers
- Embedding generation and semantic similarity
- Document loading and text chunking concepts
- Local and cloud-hosted model usage
- API-based AI app patterns with Python
- Streamlit and web app integration concepts
- Vector database and similarity-search concepts

Note: RAG is explained as a concept in the project notes, but this repository does not yet contain a full end-to-end RAG implementation with actual document retrieval and vector search in production code.

---

## Tech stack

- Python 3.11+
- LangChain
- LangGraph
- Hugging Face
- NVIDIA AI Endpoints
- OpenAI / Groq / Mistral / Google GenAI ecosystem libraries
- FAISS
- Sentence Transformers
- FastAPI
- Streamlit
- Python dotenv

---

## Project structure

```text
gen-ai/
├── README.md
├── pyproject.toml
├── requirements.txt
├── src/
│   └── gen_ai/
│       ├── __init__.py
│       ├── test.py
│       ├── readme.md
│       ├── chatmodels/
│       │   ├── chatbot.py
│       │   ├── chatbot copy.py
│       │   ├── huggingface.py
│       │   ├── localmodel.py
│       │   ├── testchat.py
│       │   └── UIchatbot.py
│       ├── embeddingmodels/
│       │   ├── embeddings.py
│       │   └── huggingface_embeddings.py
│       └── StreamSworn/
│           ├── core.py
│           ├── corewith_pydanticschema.py
│           ├── UIcore.py
│           └── UIcore2.py
```

---

## Core concepts practiced

### 1. LLM chat models

The project experiments with chat-based models through LangChain wrappers.

Examples include:

- NVIDIA Nemotron
- Hugging Face models
- chat model wrappers using `ChatNVIDIA`, `ChatHuggingFace`, and `HuggingFaceEndpoint`

This teaches how to:

- initialize a model
- create messages
- send prompts
- receive structured responses

### 2. Prompt roles

In chat flows, a message is not just plain text. It can have a role.

- `SystemMessage`: sets the rules and behavior for the assistant
- `HumanMessage`: contains the user input
- `AIMessage`: stores the model-generated output

This is a key concept in conversational AI.

### 3. Embeddings

The embeddings examples convert text into vector representations.

This helps for:

- semantic matching
- similarity search
- ranking relevant information
- building RAG pipelines

Example idea:

```python
vector = embedder.embed_query("You are a GenAI expert.")
```

The output is a numerical vector representing the text meaning.

### 4. RAG concepts learned, but not fully built yet

The repository includes notes and learning material about Retrieval-Augmented Generation, but the actual codebase does not yet implement a complete RAG app.

Typical RAG flow that is being studied conceptually:

1. Load documents
2. Clean and preprocess text
3. Split into chunks
4. Generate embeddings
5. Store them in a vector database
6. Convert user query into embedding
7. Retrieve top relevant chunks
8. Pass them to the LLM as context
9. Generate a grounded answer

This project is currently focused more on understanding the pieces than on building the full pipeline end-to-end.

### 5. Vector database and similarity search

The idea of embeddings and semantic search is learned here, even though the repository does not yet use a full FAISS/vector DB implementation in the active app code.

Examples of applications:

- document search
- chatbot memory with context retrieval
- enterprise Q&A systems
- custom knowledge assistants

### 6. UI and app integration

There are also Streamlit and UI-related files, which show that GenAI projects are not only backend logic—they can also be packaged as small interactive apps.

This teaches:

- building user interfaces for AI tools
- making LLM apps interactive
- creating demos and prototypes quickly

---

## Setup

### 1. Create environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

Or if you use uv:

```bash
uv sync
```

### 3. Add environment variables

Create a `.env` file and add required API keys if using cloud models.

Example:

```bash
OPENAI_API_KEY=your_key_here
HUGGINGFACE_API_KEY=your_key_here
```

---

## Run examples

### Check LangChain installation

```bash
uv run python -m gen_ai.test
```

### Run chat examples

There are multiple examples in the `chatmodels` folder:

```bash
python src/gen_ai/chatmodels/chatbot.py
```

### Run embedding examples

```bash
python src/gen_ai/embeddingmodels/embeddings.py
```

---

## Important learning notes

- GenAI is not only about prompting; it also involves data, embeddings, model integration, and retrieval concepts.
- Model selection depends on the use case: speed, quality, cost, local vs cloud, and context length.
- RAG is an important GenAI pattern, but this repo has not yet implemented the full retrieval pipeline in code.
- Prompt design strongly affects output quality.
- Chat applications should clearly define system instructions.
- Embeddings are the bridge between raw text and semantic search.

---

## Real-world use cases explored here

- AI assistant / chatbot
- Prompt-based conversational workflows
- Embedding and similarity-search learning
- Document understanding concepts
- AI-driven customer support prototypes
- small GenAI app interfaces

This repository is a learning ground for the building blocks of AI apps, not yet a complete production RAG application.

---

## Final takeaway

This project is a practical learning sandbox for understanding how modern Generative AI systems are built. It focuses on the core workflow behind real AI products:

Model + Prompt + Context + Retrieval + Response

This repo shows the stepping stones for moving from beginner AI experiments to actual AI application development.

---

## License

This project is for learning and experimentation purposes.

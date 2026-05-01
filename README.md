# 🤖 Support Triage AI Agent (v2.0 Turbo)

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![ChromaDB](https://img.shields.io/badge/ChromaDB-RAG-orange.svg)
![OpenRouter](https://img.shields.io/badge/OpenRouter-LLM-green.svg)
![Pandas](https://img.shields.io/badge/Pandas-Data-blueviolet.svg)

An enterprise-grade, terminal-based AI Agent designed to autonomously triage, classify, and respond to customer support tickets for HackerRank, Claude, and Visa. 

Built for the **HackerRank Orchestrate** hackathon, this agent goes beyond simple LLM wrappers by implementing semantic caching, zero-downtime model fallbacks, and strict deterministic security gates.

---

## 🌟 Key Features

*   **Zero-Downtime Fallback Chain:** Free-tier APIs often fail or rate-limit. This agent implements an automatic retry and fallback chain (`gpt-oss-120b` → `llama-3.3-70b` → `nemotron-3-nano`), ensuring 100% uptime and execution resilience.
*   **Semantic Caching Engine:** Calculates an MD5 hash of incoming tickets. If a similar issue was recently solved, it serves the cached response instantly—saving API costs and reducing latency to 0ms.
*   **Pre-Flight Security Gates:** Uses deterministic keyword screening to instantly escalate High-Risk issues (fraud, prompt injection, live outages) *before* hitting the database or LLM, ensuring absolute safety and saving tokens.
*   **Contextual RAG (Retrieval-Augmented Generation):** Uses `ChromaDB` and `all-MiniLM-L6-v2` embeddings to pull the exact Support Corpus policy needed from the provided Markdown knowledge base. 
*   **Resilient JSON Output:** Wraps LLM outputs in a robust parsing layer. If the AI hallucinates bad JSON, the system auto-corrects or safe-escalates without crashing the batch loop.
*   **Interactive CLI:** Features a color-coded terminal UI (`colorama`) with a live interactive chat mode for real-time testing.

---

## 🏗️ Architecture Pipeline

The agent processes tickets through a rigorous 6-stage pipeline to ensure speed and accuracy:

1.  **Stage 0: Cache Check:** Checks if the ticket's MD5 hash exists in the local Semantic Cache.
2.  **Stage 1: Pre-Flight Gates:** Deterministically scans for security threats or critical bugs.
3.  **Stage 2: Parallel RAG:** Queries the local ChromaDB collection in a dedicated thread with an 8-second hard timeout.
4.  **Stage 3: Corpus Coverage Gate:** Filters out irrelevant documents based on a strict L2 Distance Threshold (0.85). 
5.  **Stage 4: Prompt Construction:** Trims the context to max 400 characters per document and injects it into a highly optimized, hyper-concise system prompt.
6.  **Stage 5: LLM Inference & Parse:** Calls the OpenRouter API, parses the JSON, validates enums, and returns the formatted triage data.

---

## 📁 Project Structure
```text
HackerRank-Orchestrate/
├── chroma_db/               # Generated ChromaDB vector store
├── code/                    # Application source code
│   ├── build_db.py          # Script to initialize the vector DB
│   ├── ingestion.py         # Script to ingest markdown corpus
│   ├── main.py              # Main agent logic and CLI
│   └── test_api.py          # API testing utility
├── data/                    # Markdown knowledge base (Claude, HackerRank, Visa)
├── support_tickets/         # Input CSVs and agent outputs
├── .gitignore               # Git ignore rules
└── requirements.txt         # Python dependencies
```
## ⚙️ Setup & Installation

* **1. Clone the Repository:**
```git clone [https://github.com/sairamkollur/HackerRank-Orchestrate.git](https://github.com/sairamkollur/HackerRank-Orchestrate.git)
cd HackerRank-Orchestrate
```
* **2. Install Dependencies:**
```
pip install -r requirements.txt
```

* **3. Environment Variables**
Create a `.env` file in the root directory of the project and add your OpenRouter API key:

* **Code snippet:**
```
OPENROUTER_API_KEY="your_openrouter_api_key_here"
```
(Optional) To rebuild the vector database from scratch using the markdown files in the `data/` folder, run:
```
python code/build_db.py
```

## 🚀 Usage
Run the main application file from your terminal:

```
python code/main.py
```
Upon launching, you will be greeted by the Triage OS Menu with three options:

* **[1] Process CSV (Batch Mode)**
Reads `support_tickets/support_tickets.csv.`

Processes all tickets sequentially, utilizing the semantic cache for duplicate issues.

Saves the cleanly formatted results to support_tickets/output.csv.

* **[2] Interactive Chat (Live Mode)**
An interactive terminal loop where you can manually type a Company, Subject, and Issue.

Watch the RAG retrieval and LLM triage happen in real-time.

* **[3] Exit**
Closes the application safely.

Built for the HackerRank Orchestrate Hackathon
Developed by Kollur Sai Ram
# NeoStats ShipAssist – Streamlit Chatbot (RAG + Web Search + Modes)

An extensible blueprint chatbot that **understands**, **retrieves**, and **searches**.

## Why this use-case?
We target **Universal Shipment Tracking & Warranty Support**. Users paste tracking numbers (e.g., `STB100349244098109`) or ask policy questions. The bot:
- Pulls relevant context from your uploaded policy manuals using **RAG**.
- Performs **live web search** for the latest courier updates when needed.
- Lets users toggle **Concise** vs **Detailed** answers.

## Features
- 🔍 RAG over your own documents (txt/pdf) with chunking + FAISS (fallback to cosine).
- 🌐 Live Web Search via SerpAPI/Tavily; DuckDuckGo fallback if no API key.
- 🔁 Response Modes: Concise vs Detailed.
- 🧩 Pluggable LLM providers: **OpenAI**, **Groq**, **Gemini**.
- 🛡️ Robust try/except and simple logging via Streamlit messages.

## Project Structure
```
project/
├── config/
│   └── config.py
├── models/
│   ├── llm.py
│   └── embeddings.py
├── utils/
│   ├── rag.py
│   ├── web_search.py
│   └── text.py
├── app.py
├── requirements.txt
└── README.md
```

## Local Setup
```bash
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Set keys
export OPENAI_API_KEY="..."
export GROQ_API_KEY="..."
export GEMINI_API_KEY="..."
export SERPAPI_API_KEY="..."   # optional
export TAVILY_API_KEY="..."    # optional

streamlit run app.py
```

## Streamlit Cloud Deployment
1. Push this repo to GitHub.
2. On Streamlit Cloud, create a new app pointing to `app.py`.
3. In **Secrets**, add any keys you will use (OpenAI/Groq/Gemini + SerpAPI or Tavily).
4. Deploy. Done!

## Notes
- If FAISS is unavailable on your platform, the app falls back to NumPy cosine search.
- PDF extraction uses `pdfminer.six` for a no-Java dependency flow.
- `sentence-transformers/all-MiniLM-L6-v2` for fast, capable embeddings by default.

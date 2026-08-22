# 🇸🇬 "What Would Lee Kuan Yew Do?" (WwLKYD) Chatbot

An AI Chatbot trained on founding Prime Minister Lee Kuan Yew's speeches, memoirs, interviews, and public articles. Built with **RAG (Retrieval-Augmented Generation)**, **Persona Prompt Engineering**, **Real-Time Evaluation Metrics (Eval Bonus)**, and a **Gradio Web Interface**.

---

## 🌟 Key Features

1. **Lee Kuan Yew Persona**: Responds in LKY's direct, pragmatic, sharp, highly disciplined, and authoritative tone across life, geopolitics, history, and governance.
2. **RAG Vector Knowledge Base**: Retrieves exact excerpts from LKY's speeches (*From Third World to First*, *One Man's View of the World*, National Day Rallies).
3. **RAG Evaluation Suite (Bonus)**: Evaluates RAG performance in real-time across:
   - **Faithfulness** (Context Groundedness)
   - **Context Relevance**
   - **Answer Relevance & Persona Alignment**
4. **Interactive Gradio UI**: Includes Source Citation Inspector, Live Evaluation Score Dashboard, and Preset Sample Questions.

---

## 📁 Repository Structure

```
├── Lee_Kuan_Yew_Chatbot.ipynb  # Primary Python Notebook for submission/demo
├── app.py                       # Standalone Python script for Gradio app execution
├── requirements.txt             # Python dependencies
├── data/
│   └── lky_knowledge_base.json  # Curated corpus of LKY speeches & memoirs
└── README.md                    # Project documentation
```

---

## 🚀 Quick Start Guide

### Option A: Run via Jupyter Notebook / Google Colab
Open [`Lee_Kuan_Yew_Chatbot.ipynb`](Lee_Kuan_Yew_Chatbot.ipynb) in Jupyter Notebook or Google Colab and run all cells sequentially.

### Option B: Run Standalone Gradio App
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the application:
   ```bash
   python app.py
   ```
3. Open the local link (e.g. `http://127.0.0.1:7860`) in your browser.

---

## 💡 Example Questions

- *"How should small nations navigate big power rivalry between the US and China?"*
- *"What is your secret to eradicating corruption in Singapore?"*
- *"What advice do you have for young people facing hardship?"*
- *"Why is meritocracy so critical for Singapore's survival?"*
"""
Modern NLP Generation Pipeline (April 2026)
Model: Qwen3-Instruct via Ollama (local GPU inference)
Data: Auto-downloaded at runtime
"""
import os, json, warnings
import pandas as pd
warnings.filterwarnings("ignore")

TASK = "summarization"
OLLAMA_MODEL = "qwen3:8b"
OLLAMA_URL = "http://localhost:11434/api/generate"


def query_ollama(prompt, temperature=0.7, max_tokens=512):
    import requests
    try:
        r = requests.post(OLLAMA_URL, json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False,
                          "options": {"temperature": temperature, "num_predict": max_tokens}}, timeout=120)
        r.raise_for_status()
        return r.json().get("response", "")
    except Exception as e:
        print(f"Ollama error: {e}")
        return None


def load_data():
    from datasets import load_dataset as _hf_load
    df = _hf_load("stanfordnlp/imdb", split="train").to_pandas()
    return df


def run_summarization(df):
    text_col = next((c for c in df.columns if df[c].dtype == "object" and df[c].str.len().mean() > 50), df.select_dtypes("object").columns[0])
    results = []
    for i, text in enumerate(df[text_col].dropna().head(10)):
        summary = query_ollama(f"Summarize concisely:\n\n{text[:2000]}\n\nSummary:")
        if summary:
            results.append({"original": text[:200], "summary": summary})
            print(f"  [{i+1}] {summary[:100]}...")
    return results


def run_translation(df):
    text_col = df.select_dtypes("object").columns[0]
    results = []
    for i, text in enumerate(df[text_col].dropna().head(10)):
        translated = query_ollama(f"Translate to English:\n\n{text[:1000]}\n\nTranslation:")
        if translated:
            results.append({"original": text[:100], "translated": translated})
            print(f"  [{i+1}] {translated[:100]}...")
    return results


def run_chatbot():
    print("\n💬 Chatbot (type 'quit' to exit)")
    history = []
    while True:
        user = input("\nYou: ").strip()
        if user.lower() in ("quit", "exit", "q"): break
        history.append(f"User: {user}")
        resp = query_ollama(f"Continue:\n\n{'\n'.join(history[-6:])}\n\nAssistant:", temperature=0.8)
        if resp:
            history.append(f"Assistant: {resp}")
            print(f"Bot: {resp}")


def main():
    print("=" * 60)
    print(f"NLP GENERATION — {OLLAMA_MODEL} — Task: {TASK}")
    print("=" * 60)
    test = query_ollama("Say hello.", max_tokens=10)
    if not test:
        print("⚠ Ollama not reachable. Run: ollama serve && ollama pull " + OLLAMA_MODEL)
        return
    df = load_data()
    if TASK == "summarization" and df is not None: run_summarization(df)
    elif TASK == "translation" and df is not None: run_translation(df)
    elif TASK == "chatbot": run_chatbot()
    elif TASK == "generation": query_ollama("Write a creative story about AI:", temperature=0.9, max_tokens=1024)
    else:
        if df is not None: run_summarization(df)
        else: run_chatbot()


if __name__ == "__main__":
    main()

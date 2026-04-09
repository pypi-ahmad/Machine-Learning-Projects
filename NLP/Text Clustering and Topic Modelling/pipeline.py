"""
Modern NLP Generation Pipeline (April 2026)
Models: Qwen3-Instruct (chat/generation/summarization) + NLLB-200 (translation) + BART (summarization baseline)
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
    df = _hf_load("SetFit/20_newsgroups", split="train").to_pandas()
    return df


def run_summarization(df):
    """Qwen3-Instruct for general summarization + BART as classic baseline."""
    text_col = next((c for c in df.columns if df[c].dtype == "object" and df[c].str.len().mean() > 50), df.select_dtypes("object").columns[0])
    texts = df[text_col].dropna().head(10).tolist()

    # ═══ PRIMARY: Qwen3-Instruct via Ollama ═══
    qwen_results = []
    for i, text in enumerate(texts):
        summary = query_ollama(f"Summarize concisely:\n\n{text[:2000]}\n\nSummary:")
        if summary:
            qwen_results.append({"original": text[:200], "summary": summary})
            print(f"  Qwen3 [{i+1}] {summary[:100]}...")
    if qwen_results:
        print(f"✓ Qwen3-Instruct summarized {len(qwen_results)} texts")

    # ═══ BASELINE: BART (facebook/bart-large-cnn) ═══
    try:
        import torch
        from transformers import BartForConditionalGeneration, BartTokenizer
        device = "cuda" if torch.cuda.is_available() else "cpu"
        tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
        model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn").to(device)
        bart_results = []
        for i, text in enumerate(texts[:5]):
            inputs = tokenizer(text[:1024], return_tensors="pt", truncation=True, max_length=1024).to(device)
            summary_ids = model.generate(**inputs, max_length=150, min_length=30, num_beams=4, length_penalty=2.0)
            summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            bart_results.append(summary)
            print(f"  BART [{i+1}] {summary[:100]}...")
        print(f"✓ BART summarized {len(bart_results)} texts")
    except Exception as e:
        print(f"✗ BART baseline: {e}")


def run_translation(df):
    """NLLB-200 (Meta) — 200+ language pairs, offline, multilingual."""
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_id = "facebook/nllb-200-distilled-600M"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id).to(device)
    text_col = df.select_dtypes("object").columns[0]

    # Translate to multiple target languages
    targets = [("fra_Latn", "French"), ("deu_Latn", "German"), ("spa_Latn", "Spanish"), ("zho_Hans", "Chinese")]
    for tgt_code, tgt_name in targets:
        results = []
        for i, text in enumerate(df[text_col].dropna().head(3)):
            inputs = tokenizer(text[:512], return_tensors="pt", truncation=True).to(device)
            translated_ids = model.generate(**inputs, forced_bos_token_id=tokenizer.convert_tokens_to_ids(tgt_code), max_new_tokens=256)
            translated = tokenizer.decode(translated_ids[0], skip_special_tokens=True)
            results.append({"original": text[:100], "translated": translated})
            print(f"  → {tgt_name} [{i+1}] {translated[:100]}...")
        print(f"✓ NLLB-200 → {tgt_name}: {len(results)} texts")


def run_generation(df):
    """Qwen3-Instruct for text generation / next-word prediction."""
    prompts = [
        "Write a creative short story about artificial intelligence discovering emotions:",
        "Complete this sentence: The future of machine learning is",
        "Explain quantum computing to a 10-year-old:",
    ]
    # Use data context if available
    if df is not None:
        text_col = next((c for c in df.columns if df[c].dtype == "object"), None)
        if text_col:
            samples = df[text_col].dropna().head(3).tolist()
            prompts = [f"Continue this text creatively:\n\n{t[:300]}\n\nContinuation:" for t in samples]

    for i, prompt in enumerate(prompts):
        response = query_ollama(prompt, temperature=0.9, max_tokens=256)
        if response:
            print(f"  [{i+1}] {response[:200]}...")
    print(f"✓ Qwen3-Instruct generated {len(prompts)} texts")


def run_chatbot():
    """Qwen3-Instruct interactive chatbot."""
    print("\n💬 Chatbot (type 'quit' to exit)")
    history = []
    while True:
        user = input("\nYou: ").strip()
        if user.lower() in ("quit", "exit", "q"): break
        history.append(f"User: {user}")
        resp = query_ollama(f"You are a helpful assistant. Continue this conversation:\n\n{'\n'.join(history[-6:])}\n\nAssistant:", temperature=0.8)
        if resp:
            history.append(f"Assistant: {resp}")
            print(f"Bot: {resp}")


def main():
    print("=" * 60)
    print(f"NLP GENERATION — Qwen3-Instruct + NLLB-200 + BART | Task: {TASK}")
    print("=" * 60)
    if TASK != "translation":
        test = query_ollama("Say hello.", max_tokens=10)
        if not test:
            print("⚠ Ollama not reachable. Run: ollama serve && ollama pull " + OLLAMA_MODEL)
            if TASK != "translation":
                return
    df = load_data()
    if TASK == "summarization" and df is not None: run_summarization(df)
    elif TASK == "translation" and df is not None: run_translation(df)
    elif TASK == "chatbot": run_chatbot()
    elif TASK == "generation": run_generation(df)
    else:
        if df is not None: run_summarization(df)
        else: run_chatbot()


if __name__ == "__main__":
    main()

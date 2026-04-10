"""
Modern NLP Generation Pipeline (April 2026)

Task      : summarization
Primary   : Qwen3-Instruct (8B) via Ollama — chat, generation, summarisation.
Translation: NLLB-200-distilled-600M (Meta) — 200+ language pairs, offline.
Baseline  : BART-large-CNN (summarisation only).

Chatbot mode runs a scripted demo conversation from the dataset first,
then offers an interactive session. All modes report wall-clock timing
and export results to metrics.json.

Compute: Ollama manages GPU for Qwen3; NLLB/BART use torch + CUDA if available.
Data   : Auto-downloaded at runtime.
"""
import os, json, time, warnings
import pandas as pd
warnings.filterwarnings("ignore")

TASK = "summarization"
OLLAMA_MODEL = "qwen3:8b"
OLLAMA_URL = "http://localhost:11434/api/generate"
SAVE_DIR = os.path.dirname(os.path.abspath(__file__))


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
    df = _hf_load("EdinburghNLP/xsum", split="train").to_pandas()
    return df


# ═══════════════════════════════════════════════════════════════
# SUMMARISATION — Qwen3 + BART baseline
# ═══════════════════════════════════════════════════════════════
def run_summarization(df):
    text_col = next((c for c in df.columns if df[c].dtype == "object" and df[c].str.len().mean() > 50), df.select_dtypes("object").columns[0])
    texts = df[text_col].dropna().head(10).tolist()
    metrics = {}

    # Detect gold reference summaries (xsum: 'summary'; cnn: 'highlights')
    ref_col = None
    for c in ("summary", "highlights", "abstract", "target"):
        if c in df.columns and c != text_col:
            ref_col = c
            break
    refs = df[ref_col].dropna().head(10).tolist() if ref_col else None
    if refs:
        print(f"  Gold references found in column '{ref_col}'")

    # PRIMARY: Qwen3-Instruct via Ollama
    t0 = time.perf_counter()
    qwen_summaries = []
    for i, text in enumerate(texts):
        summary = query_ollama(f"Summarize concisely:\n\n{text[:2000]}\n\nSummary:")
        if summary:
            qwen_summaries.append(summary.strip())
            print(f"  Qwen3 [{i+1}] {summary[:100]}...")
    elapsed = time.perf_counter() - t0
    if qwen_summaries:
        print(f"  Qwen3-Instruct: {len(qwen_summaries)} summaries in {elapsed:.1f}s")
    m = {"count": len(qwen_summaries), "time_s": round(elapsed, 1)}
    if refs and qwen_summaries:
        m.update(_compute_rouge(qwen_summaries, refs[:len(qwen_summaries)], "Qwen3"))
    metrics["Qwen3-Instruct"] = m

    # BASELINE: BART
    try:
        import torch
        from transformers import BartForConditionalGeneration, BartTokenizer
        device = "cuda" if torch.cuda.is_available() else "cpu"
        tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
        model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn").to(device)
        t0 = time.perf_counter()
        bart_summaries = []
        for i, text in enumerate(texts[:5]):
            inputs = tokenizer(text[:1024], return_tensors="pt", truncation=True, max_length=1024).to(device)
            summary_ids = model.generate(**inputs, max_length=150, min_length=30, num_beams=4, length_penalty=2.0)
            summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            bart_summaries.append(summary)
            print(f"  BART [{i+1}] {summary[:100]}...")
        elapsed = time.perf_counter() - t0
        print(f"  BART: {len(bart_summaries)} summaries in {elapsed:.1f}s")
        m = {"count": len(bart_summaries), "time_s": round(elapsed, 1)}
        if refs and bart_summaries:
            m.update(_compute_rouge(bart_summaries, refs[:len(bart_summaries)], "BART"))
        metrics["BART"] = m
    except Exception as e:
        print(f"  BART baseline failed: {e}")
    return metrics


def _compute_rouge(hypotheses, references, model_name):
    """Compute ROUGE-1/2/L F1 scores. Uses rouge-scorer if available, else
    falls back to a simple n-gram overlap implementation."""
    try:
        from rouge_score import rouge_scorer as rs
        scorer = rs.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
        r1, r2, rl = [], [], []
        for hyp, ref in zip(hypotheses, references):
            s = scorer.score(str(ref), str(hyp))
            r1.append(s["rouge1"].fmeasure)
            r2.append(s["rouge2"].fmeasure)
            rl.append(s["rougeL"].fmeasure)
        avg = lambda xs: round(sum(xs) / max(len(xs), 1) * 100, 2)
        out = {"rouge1": avg(r1), "rouge2": avg(r2), "rougeL": avg(rl)}
        print(f"  [{model_name}] ROUGE-1: {out['rouge1']}  ROUGE-2: {out['rouge2']}  ROUGE-L: {out['rougeL']}")
        return out
    except ImportError:
        pass
    # Fallback: simple unigram overlap (ROUGE-1 approximation)
    try:
        scores = []
        for hyp, ref in zip(hypotheses, references):
            h_toks = set(str(hyp).lower().split())
            r_toks = set(str(ref).lower().split())
            if not r_toks:
                continue
            overlap = len(h_toks & r_toks)
            p = overlap / max(len(h_toks), 1)
            r = overlap / max(len(r_toks), 1)
            f1 = 2 * p * r / max(p + r, 1e-9)
            scores.append(f1)
        avg_f1 = round(sum(scores) / max(len(scores), 1) * 100, 2) if scores else 0.0
        print(f"  [{model_name}] ROUGE-1 (approx): {avg_f1}")
        return {"rouge1_approx": avg_f1}
    except Exception:
        return {}


# ═══════════════════════════════════════════════════════════════
# TRANSLATION — NLLB-200
# ═══════════════════════════════════════════════════════════════
def _extract_texts(df, n=20):
    """Extract source texts from the dataset.  Handles WMT-style nested
    'translation' dicts ({"de": ..., "en": ...}) as well as plain
    string columns.  Returns (texts, references_or_None).
    References are available when the dataset contains parallel pairs."""
    # WMT-style: column named 'translation' containing dicts
    if "translation" in df.columns:
        sample = df["translation"].dropna().head(n).tolist()
        if sample and isinstance(sample[0], dict):
            # Prefer English source if available, else first key
            src_key = "en" if "en" in sample[0] else list(sample[0].keys())[0]
            ref_key = [k for k in sample[0].keys() if k != src_key]
            texts = [str(row.get(src_key, "")) for row in sample]
            refs = {k: [str(row.get(k, "")) for row in sample] for k in ref_key} if ref_key else None
            return texts, refs
    # Plain string column
    for c in df.columns:
        if df[c].dtype == "object" and df[c].str.len().mean() > 10:
            return df[c].dropna().head(n).astype(str).tolist(), None
    text_cols = df.select_dtypes("object").columns
    if len(text_cols):
        return df[text_cols[0]].dropna().head(n).astype(str).tolist(), None
    return df.iloc[:, 0].astype(str).head(n).tolist(), None


def run_translation(df):
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_id = "facebook/nllb-200-distilled-600M"
    print(f"  Loading {model_id} on {device} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id).to(device)
    metrics = {}

    texts, refs = _extract_texts(df, n=10)
    print(f"  Source texts: {len(texts)}")

    targets = [("fra_Latn", "French"), ("deu_Latn", "German"),
               ("spa_Latn", "Spanish"), ("zho_Hans", "Chinese")]
    total_t0 = time.perf_counter()

    for tgt_code, tgt_name in targets:
        t0 = time.perf_counter()
        translations = []
        for i, text in enumerate(texts):
            inputs = tokenizer(str(text)[:512], return_tensors="pt", truncation=True).to(device)
            out_ids = model.generate(**inputs,
                                     forced_bos_token_id=tokenizer.convert_tokens_to_ids(tgt_code),
                                     max_new_tokens=256)
            translated = tokenizer.decode(out_ids[0], skip_special_tokens=True)
            translations.append(translated)
            if i < 3:
                print(f"    -> {tgt_name} [{i+1}] {translated[:100]}...")
        elapsed = time.perf_counter() - t0
        lang_metrics = {"count": len(translations), "time_s": round(elapsed, 1)}

        # BLEU if we have reference translations for this language
        lang_code_short = tgt_code.split("_")[0][:2]   # fra_Latn -> fr
        iso_map = {"fr": "fra_Latn", "de": "deu_Latn", "es": "spa_Latn", "zh": "zho_Hans"}
        if refs:
            for rk in refs:
                if rk == lang_code_short or iso_map.get(rk) == tgt_code:
                    try:
                        from sacrebleu.metrics import BLEU
                        bleu = BLEU()
                        score = bleu.corpus_score(translations, [refs[rk][:len(translations)]])
                        lang_metrics["bleu"] = round(score.score, 2)
                        print(f"    BLEU ({tgt_name}): {score.score:.2f}")
                    except ImportError:
                        try:
                            from nltk.translate.bleu_score import corpus_bleu
                            ref_tok = [[r.split()] for r in refs[rk][:len(translations)]]
                            hyp_tok = [t.split() for t in translations]
                            b = corpus_bleu(ref_tok, hyp_tok)
                            lang_metrics["bleu"] = round(b * 100, 2)
                            print(f"    BLEU ({tgt_name}): {b*100:.2f}")
                        except Exception:
                            pass
                    break

        print(f"  NLLB-200 -> {tgt_name}: {len(translations)} texts in {elapsed:.1f}s")
        metrics[f"NLLB-200_{tgt_name}"] = lang_metrics

    total_elapsed = time.perf_counter() - total_t0
    metrics["NLLB-200_total"] = {"time_s": round(total_elapsed, 1), "languages": len(targets),
                                   "model": model_id, "device": str(device)}
    return metrics


# ═══════════════════════════════════════════════════════════════
# GENERATION — Qwen3-Instruct
# ═══════════════════════════════════════════════════════════════
def run_generation(df):
    prompts = [
        "Write a creative short story about artificial intelligence discovering emotions:",
        "Complete this sentence: The future of machine learning is",
        "Explain quantum computing to a 10-year-old:",
    ]
    if df is not None:
        text_col = next((c for c in df.columns if df[c].dtype == "object"), None)
        if text_col:
            samples = df[text_col].dropna().head(3).tolist()
            prompts = [f"Continue this text creatively:\n\n{t[:300]}\n\nContinuation:" for t in samples]

    t0 = time.perf_counter()
    responses = []
    for i, prompt in enumerate(prompts):
        response = query_ollama(prompt, temperature=0.9, max_tokens=256)
        if response:
            responses.append(response)
            print(f"  [{i+1}] {response[:200]}...")
    elapsed = time.perf_counter() - t0
    print(f"  Qwen3-Instruct: {len(responses)} texts in {elapsed:.1f}s")
    return {"Qwen3-Instruct": {"count": len(responses), "time_s": round(elapsed, 1)}}


# ═══════════════════════════════════════════════════════════════
# CHATBOT — Qwen3-Instruct | demo + interactive
# ═══════════════════════════════════════════════════════════════
def extract_demo_turns(df, n=5):
    """Pull sample user utterances from the loaded dialogue dataset for a
    scripted demo conversation (avoids relying solely on interactive input)."""
    samples = []
    # daily-dialogs: column named 'dialog' (list of turns) or first text col
    for col in ("dialog", "utterance", "text", "question", "input"):
        if col in df.columns:
            vals = df[col].dropna().head(n * 3)
            for v in vals:
                if isinstance(v, list):
                    samples.extend([str(t) for t in v[:2]])
                elif isinstance(v, str) and len(v.strip()) > 5:
                    samples.append(v.strip()[:200])
                if len(samples) >= n:
                    break
            break
    if not samples:
        text_cols = df.select_dtypes("object").columns
        if len(text_cols):
            samples = df[text_cols[0]].dropna().astype(str).head(n).tolist()
    return samples[:n]


def run_chatbot(df=None):
    """Qwen3-Instruct chatbot: scripted demo + optional interactive session."""
    system = "You are a helpful, concise assistant."
    metrics = {"model": OLLAMA_MODEL, "turns": []}

    # ── Scripted demo from dataset ──
    demo_prompts = []
    if df is not None:
        demo_prompts = extract_demo_turns(df, n=5)
    if not demo_prompts:
        demo_prompts = ["Hello!", "What can you help me with?",
                        "Explain machine learning in one sentence.",
                        "What is the capital of France?", "Thanks, goodbye!"]

    print()
    print("--- Demo Conversation (scripted) ---")
    history = []
    for user_msg in demo_prompts:
        history.append(f"User: {user_msg}")
        ctx = "\n".join(history[-6:])
        prompt = f"{system}\n\n{ctx}\nAssistant:"
        t0 = time.perf_counter()
        resp = query_ollama(prompt, temperature=0.7, max_tokens=256)
        latency = time.perf_counter() - t0
        if resp:
            resp = resp.strip()
            history.append(f"Assistant: {resp}")
            print(f"  User : {user_msg}")
            print(f"  Bot  : {resp[:200]}")
            print(f"  ({latency:.1f}s)")
            metrics["turns"].append({"user": user_msg, "bot": resp[:300],
                                      "latency_s": round(latency, 1)})
        else:
            print(f"  User : {user_msg}")
            print(f"  Bot  : [no response]")

    avg_lat = 0
    if metrics["turns"]:
        avg_lat = sum(t["latency_s"] for t in metrics["turns"]) / len(metrics["turns"])
        print()
        print(f"  Demo: {len(metrics['turns'])} turns, avg latency {avg_lat:.1f}s")
    metrics["demo_avg_latency_s"] = round(avg_lat, 1)

    # ── Interactive session (skipped in non-interactive environments) ──
    import sys
    if sys.stdin.isatty():
        print()
        print("--- Interactive Chat (type 'quit' to exit) ---")
        while True:
            try:
                user = input("You: ").strip()
            except EOFError:
                break
            if user.lower() in ("quit", "exit", "q", ""):
                break
            history.append(f"User: {user}")
            ctx = "\n".join(history[-6:])
            prompt = f"{system}\n\n{ctx}\nAssistant:"
            t0 = time.perf_counter()
            resp = query_ollama(prompt, temperature=0.8, max_tokens=512)
            latency = time.perf_counter() - t0
            if resp:
                resp = resp.strip()
                history.append(f"Assistant: {resp}")
                print(f"Bot: {resp}")
                print(f"  ({latency:.1f}s)")
    else:
        print("  (non-interactive environment — skipping live chat)")
    return metrics


def run_eda(df, save_dir):
    """Input data statistics for generation tasks."""
    print("\n" + "=" * 60)
    print("EXPLORATORY DATA ANALYSIS")
    print("=" * 60)
    if df is not None:
        print(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns")
        desc = df.describe(include="all").T
        desc.to_csv(os.path.join(save_dir, "eda_summary.csv"))
        text_cols = df.select_dtypes(include=["object"]).columns.tolist()
        for col in text_cols[:3]:
            lengths = df[col].astype(str).str.len()
            print(f"  {col}: mean_len={lengths.mean():.0f}, median={lengths.median():.0f}")
        print("Summary statistics saved to eda_summary.csv")
    else:
        print("  No structured dataset (chatbot/generation mode)")
    print("EDA complete.")


def main():
    print("=" * 60)
    print(f"NLP GENERATION | Task: {TASK} | Model: {OLLAMA_MODEL}")
    print("=" * 60)

    # Ollama connectivity check (not needed for translation)
    if TASK != "translation":
        test = query_ollama("Say hello.", max_tokens=10)
        if not test:
            print("Ollama not reachable. Run: ollama serve && ollama pull " + OLLAMA_MODEL)
            if TASK != "translation":
                return

    df = load_data()
    run_eda(df, SAVE_DIR)
    metrics = {}

    if TASK == "summarization" and df is not None:
        metrics = run_summarization(df)
    elif TASK == "translation" and df is not None:
        metrics = run_translation(df)
    elif TASK == "chatbot":
        metrics = run_chatbot(df)
    elif TASK == "generation":
        metrics = run_generation(df)
    else:
        if df is not None:
            metrics = run_summarization(df)
        else:
            metrics = run_chatbot()

    # Export metrics
    out_path = os.path.join(SAVE_DIR, "metrics.json")
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    print()
    print(f"Metrics saved to {out_path}")


if __name__ == "__main__":
    main()

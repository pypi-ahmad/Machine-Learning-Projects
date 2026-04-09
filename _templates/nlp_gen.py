"""NLP Generation/Translation/Chatbot template: Qwen3-Instruct (Ollama) — April 2026"""
import textwrap


def generate(project_path, config):
    task_type = config.get("task", "summarization")  # summarization, translation, chatbot, generation

    return textwrap.dedent(f'''\
        """
        Modern NLP Generation Pipeline (April 2026)
        Model: Qwen3-Instruct via Ollama (local GPU inference)
        Tasks: summarization, translation, chatbot, text generation
        """
        import os, json, warnings
        import pandas as pd
        from pathlib import Path

        warnings.filterwarnings("ignore")

        TASK = "{task_type}"
        OLLAMA_MODEL = "qwen3:8b"
        OLLAMA_URL = "http://localhost:11434/api/generate"


        def query_ollama(prompt, model=OLLAMA_MODEL, temperature=0.7, max_tokens=512):
            """Send a prompt to Ollama and return the response."""
            import requests
            payload = {{
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {{
                    "temperature": temperature,
                    "num_predict": max_tokens,
                }}
            }}
            try:
                resp = requests.post(OLLAMA_URL, json=payload, timeout=120)
                resp.raise_for_status()
                return resp.json().get("response", "")
            except Exception as e:
                print(f"Ollama error: {{e}}")
                return None


        def load_data():
            data_dir = Path(os.path.dirname(__file__))
            csv_files = list(data_dir.glob("*.csv")) + list(data_dir.glob("*.tsv")) + list(data_dir.glob("*.txt"))
            if csv_files:
                if csv_files[0].suffix == ".txt":
                    text = csv_files[0].read_text(encoding="utf-8", errors="ignore")
                    return pd.DataFrame({{"text": text.split("\\n\\n")[:50]}})
                sep = "\\t" if csv_files[0].suffix == ".tsv" else ","
                df = pd.read_csv(csv_files[0], sep=sep, nrows=100)
                return df
            print("No data files found — running in interactive mode.")
            return None


        def run_summarization(df):
            """Summarize texts using Qwen3."""
            text_col = None
            for c in df.columns:
                if df[c].dtype == "object" and df[c].str.len().mean() > 50:
                    text_col = c
                    break
            if not text_col:
                text_col = df.select_dtypes("object").columns[0]

            results = []
            samples = df[text_col].dropna().head(10)
            for i, text in enumerate(samples):
                prompt = f"Summarize the following text concisely:\\n\\n{{text[:2000]}}\\n\\nSummary:"
                summary = query_ollama(prompt)
                if summary:
                    results.append({{"original": text[:200] + "...", "summary": summary}})
                    print(f"  [{{i+1}}/{{len(samples)}}] {{summary[:100]}}...")

            return results


        def run_translation(df):
            """Translate texts using Qwen3."""
            text_col = df.select_dtypes("object").columns[0]
            results = []
            for i, text in enumerate(df[text_col].dropna().head(10)):
                prompt = f"Translate the following text to English. Only output the translation, nothing else:\\n\\n{{text[:1000]}}\\n\\nTranslation:"
                translated = query_ollama(prompt)
                if translated:
                    results.append({{"original": text[:100], "translated": translated}})
                    print(f"  [{{i+1}}] {{translated[:100]}}...")
            return results


        def run_chatbot():
            """Interactive chatbot using Qwen3."""
            print("\\n💬 Chatbot Mode (type 'quit' to exit)")
            history = []
            while True:
                user_input = input("\\nYou: ").strip()
                if user_input.lower() in ("quit", "exit", "q"):
                    break
                history.append(f"User: {{user_input}}")
                context = "\\n".join(history[-6:])
                prompt = f"You are a helpful assistant. Continue the conversation:\\n\\n{{context}}\\n\\nAssistant:"
                response = query_ollama(prompt, temperature=0.8)
                if response:
                    history.append(f"Assistant: {{response}}")
                    print(f"Bot: {{response}}")
                else:
                    print("Bot: [No response from Ollama]")


        def run_generation(df):
            """Text generation using Qwen3."""
            if df is not None:
                text_col = df.select_dtypes("object").columns[0]
                seed = df[text_col].dropna().iloc[0][:200]
            else:
                seed = "Once upon a time"

            prompt = f"Continue writing from the following text. Be creative and detailed:\\n\\n{{seed}}\\n\\nContinuation:"
            generated = query_ollama(prompt, temperature=0.9, max_tokens=1024)
            if generated:
                print(f"\\nGenerated text:\\n{{generated}}")
            return generated


        def main():
            print("=" * 60)
            print("MODERN NLP GENERATION PIPELINE")
            print(f"Model: {{OLLAMA_MODEL}} via Ollama | Task: {{TASK}}")
            print("=" * 60)

            # Check Ollama connectivity
            test = query_ollama("Say hello in one word.", max_tokens=10)
            if test is None:
                print("\\n⚠ Ollama not reachable. Ensure Ollama is running:")
                print("  ollama serve")
                print(f"  ollama pull {{OLLAMA_MODEL}}")
                return

            print(f"✓ Ollama connected (model: {{OLLAMA_MODEL}})")

            df = load_data()

            if TASK == "summarization" and df is not None:
                results = run_summarization(df)
            elif TASK == "translation" and df is not None:
                results = run_translation(df)
            elif TASK == "chatbot":
                run_chatbot()
                return
            elif TASK == "generation":
                run_generation(df)
                return
            else:
                print(f"Task '{{TASK}}' with available data — running summarization by default.")
                if df is not None:
                    results = run_summarization(df)
                else:
                    run_chatbot()
                    return

            # Save results
            if results:
                out_path = os.path.join(os.path.dirname(__file__), "generation_results.json")
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(results, f, indent=2, ensure_ascii=False)
                print(f"\\n✓ Results saved to {{out_path}}")


        if __name__ == "__main__":
            main()
    ''')

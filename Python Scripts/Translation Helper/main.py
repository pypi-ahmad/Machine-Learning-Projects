"""Translation Helper — CLI tool.

Translate text between languages using LibreTranslate (self-hosted)
or the MyMemory free API (no key required for basic use).

Usage:
    python main.py
    python main.py --text "Hello, world!" --to es
    python main.py --text "Bonjour" --from fr --to en
    python main.py --list-languages
"""

import argparse
import json
import sys
import urllib.request
import urllib.parse


# MyMemory: free, no key, 500 chars/day per IP
MYMEMORY_URL = "https://api.mymemory.translated.net/get"

LANGUAGES = {
    "af":"Afrikaans","sq":"Albanian","am":"Amharic","ar":"Arabic",
    "hy":"Armenian","az":"Azerbaijani","eu":"Basque","be":"Belarusian",
    "bn":"Bengali","bs":"Bosnian","bg":"Bulgarian","ca":"Catalan",
    "zh":"Chinese","hr":"Croatian","cs":"Czech","da":"Danish",
    "nl":"Dutch","en":"English","eo":"Esperanto","et":"Estonian",
    "fi":"Finnish","fr":"French","gl":"Galician","ka":"Georgian",
    "de":"German","el":"Greek","gu":"Gujarati","ht":"Haitian Creole",
    "ha":"Hausa","he":"Hebrew","hi":"Hindi","hu":"Hungarian",
    "is":"Icelandic","id":"Indonesian","ga":"Irish","it":"Italian",
    "ja":"Japanese","kn":"Kannada","kk":"Kazakh","km":"Khmer",
    "ko":"Korean","ku":"Kurdish","lo":"Lao","lv":"Latvian",
    "lt":"Lithuanian","mk":"Macedonian","ms":"Malay","ml":"Malayalam",
    "mt":"Maltese","mr":"Marathi","mn":"Mongolian","my":"Myanmar",
    "ne":"Nepali","no":"Norwegian","ps":"Pashto","fa":"Persian",
    "pl":"Polish","pt":"Portuguese","pa":"Punjabi","ro":"Romanian",
    "ru":"Russian","sr":"Serbian","si":"Sinhala","sk":"Slovak",
    "sl":"Slovenian","so":"Somali","es":"Spanish","sw":"Swahili",
    "sv":"Swedish","tl":"Filipino","tg":"Tajik","ta":"Tamil",
    "te":"Telugu","th":"Thai","tr":"Turkish","uk":"Ukrainian",
    "ur":"Urdu","uz":"Uzbek","vi":"Vietnamese","cy":"Welsh",
    "xh":"Xhosa","yi":"Yiddish","yo":"Yoruba","zu":"Zulu",
}


def translate_mymemory(text: str, from_lang: str, to_lang: str) -> dict:
    langpair = f"{from_lang}|{to_lang}"
    params   = urllib.parse.urlencode({"q": text, "langpair": langpair})
    url      = f"{MYMEMORY_URL}?{params}"
    try:
        with urllib.request.urlopen(url, timeout=8) as resp:
            data = json.loads(resp.read())
    except Exception as e:
        raise ValueError(f"Network error: {e}")

    if data.get("responseStatus") != 200:
        raise ValueError(f"API error: {data.get('responseDetails','unknown')}")

    match = data.get("responseData", {})
    return {
        "translation": match.get("translationText",""),
        "confidence":  match.get("match", 0),
        "from":        from_lang,
        "to":          to_lang,
        "original":    text,
    }


def display_translation(result: dict) -> None:
    from_name = LANGUAGES.get(result["from"], result["from"])
    to_name   = LANGUAGES.get(result["to"],   result["to"])
    conf      = result["confidence"]
    print(f"\n  Original ({from_name}):")
    print(f"  {result['original']}")
    print(f"\n  Translation ({to_name}):")
    print(f"  {result['translation']}")
    if conf:
        print(f"\n  Confidence: {conf:.0%}")
    print()


def list_languages() -> None:
    print(f"\n  Supported languages ({len(LANGUAGES)}):")
    items = sorted(LANGUAGES.items(), key=lambda x: x[1])
    for i in range(0, len(items), 3):
        row = items[i:i+3]
        print("  " + "  ".join(f"{code:<5} {name:<18}" for code, name in row))
    print()


def interactive():
    print("=== Translation Helper ===")
    print("Commands: translate | list | quit\n")
    while True:
        cmd = input("> ").strip().lower()
        if cmd in ("quit", "q", "exit"): break
        elif cmd == "list":
            list_languages()
        elif cmd == "translate":
            text      = input("  Text to translate: ").strip()
            from_lang = input("  From language code [auto]: ").strip() or "auto"
            to_lang   = input("  To language code [en]: ").strip() or "en"
            if not text: continue
            try:
                result = translate_mymemory(text, from_lang, to_lang)
                display_translation(result)
            except ValueError as e: print(f"  Error: {e}")
        else:
            # Treat input as text to translate to English
            try:
                result = translate_mymemory(cmd, "auto", "en")
                display_translation(result)
            except ValueError as e: print(f"  Error: {e}")
        print()


def main():
    parser = argparse.ArgumentParser(description="Translation Helper")
    parser.add_argument("--text",           metavar="TEXT", help="Text to translate")
    parser.add_argument("--from",           dest="from_lang", default="auto", metavar="LANG")
    parser.add_argument("--to",             dest="to_lang",   default="en",   metavar="LANG")
    parser.add_argument("--list-languages", action="store_true")
    args = parser.parse_args()

    try:
        if args.list_languages:
            list_languages()
        elif args.text:
            result = translate_mymemory(args.text, args.from_lang, args.to_lang)
            display_translation(result)
        else:
            interactive()
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr); sys.exit(1)


if __name__ == "__main__":
    main()

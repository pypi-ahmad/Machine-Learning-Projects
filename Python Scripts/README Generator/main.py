"""README Generator — CLI developer tool.

Generate a professional README.md for a project by answering
a series of prompts. Supports templates and auto-detection of
language/framework from the project directory.

Usage:
    python main.py
    python main.py --project /path/to/project
    python main.py --template minimal
"""

import argparse
import os
import re
import sys
from datetime import date


ANSI = {"bold": "\033[1m", "cyan": "\033[96m", "green": "\033[92m",
        "yellow": "\033[93m", "red": "\033[91m", "reset": "\033[0m"}


def c(text, color):
    return f"{ANSI.get(color,'')}{text}{ANSI['reset']}"


# ── Project auto-detection ─────────────────────────────────────────────────

LANG_INDICATORS = {
    "Python":     ["*.py", "requirements.txt", "setup.py", "pyproject.toml", "Pipfile"],
    "JavaScript": ["package.json", "*.js", "*.mjs"],
    "TypeScript": ["tsconfig.json", "*.ts"],
    "Rust":       ["Cargo.toml", "*.rs"],
    "Go":         ["go.mod", "*.go"],
    "Java":       ["pom.xml", "build.gradle", "*.java"],
    "C++":        ["CMakeLists.txt", "*.cpp", "*.cc"],
    "Ruby":       ["Gemfile", "*.rb"],
    "PHP":        ["composer.json", "*.php"],
}

FRAMEWORK_INDICATORS = {
    "Django":   ["manage.py", "django"],
    "Flask":    ["flask", "app.py"],
    "FastAPI":  ["fastapi", "uvicorn"],
    "React":    ["react", "jsx"],
    "Vue":      ["vue.config.js", "nuxt.config.js"],
    "Next.js":  ["next.config.js"],
    "Express":  ["express"],
    "Spring":   ["spring"],
    "Rails":    ["rails"],
    "Streamlit":["streamlit"],
}


def detect_language(project_dir: str) -> str:
    if not os.path.isdir(project_dir):
        return "Unknown"
    files = os.listdir(project_dir)
    for lang, indicators in LANG_INDICATORS.items():
        for ind in indicators:
            if "*" in ind:
                ext = ind.replace("*", "")
                if any(f.endswith(ext) for f in files):
                    return lang
            elif ind in files:
                return lang
    return "Unknown"


def detect_framework(project_dir: str) -> str:
    if not os.path.isdir(project_dir):
        return ""
    all_text = ""
    for fn in os.listdir(project_dir):
        if fn in ("requirements.txt", "package.json", "Gemfile"):
            try:
                with open(os.path.join(project_dir, fn)) as f:
                    all_text += f.read().lower()
            except Exception:
                pass
    for fw, indicators in FRAMEWORK_INDICATORS.items():
        if any(ind.lower() in all_text for ind in indicators):
            return fw
    return ""


def detect_license(project_dir: str) -> str:
    for fn in os.listdir(project_dir) if os.path.isdir(project_dir) else []:
        if fn.upper().startswith("LICENSE"):
            try:
                with open(os.path.join(project_dir, fn)) as f:
                    first = f.read(200).upper()
                if "MIT"       in first: return "MIT"
                if "APACHE"    in first: return "Apache 2.0"
                if "GPL"       in first: return "GPL v3"
                if "BSD"       in first: return "BSD"
                if "UNLICENSE" in first: return "Unlicense"
                return "See LICENSE file"
            except Exception:
                pass
    return "MIT"


# ── Templates ─────────────────────────────────────────────────────────────────

def build_readme(data: dict) -> str:
    name        = data.get("name", "My Project")
    desc        = data.get("description", "A great project.")
    lang        = data.get("language", "Python")
    framework   = data.get("framework", "")
    install_cmd = data.get("install", "pip install -r requirements.txt")
    usage_ex    = data.get("usage", f"python main.py")
    features    = data.get("features", [])
    contrib     = data.get("contributing", True)
    license_    = data.get("license", "MIT")
    author      = data.get("author", "")
    github_url  = data.get("github", "")
    badges      = data.get("badges", True)
    year        = date.today().year

    lines = []

    # Title and badges
    lines.append(f"# {name}\n")

    if badges:
        if github_url:
            slug = github_url.rstrip("/").split("github.com/")[-1]
            lines.append(f"![GitHub stars](https://img.shields.io/github/stars/{slug}?style=flat-square)")
        if license_:
            lines.append(f"![License](https://img.shields.io/badge/license-{license_.replace(' ','%20')}-blue.svg?style=flat-square)")
        lang_badge = f"![{lang}](https://img.shields.io/badge/{lang}-{'3776AB' if lang=='Python' else '4FC08D'}.svg?style=flat-square&logo={lang.lower()})"
        lines.append(lang_badge)
        lines.append("")

    lines.append(f"> {desc}\n")

    if framework:
        lines.append(f"**Built with:** {lang} · {framework}\n")
    else:
        lines.append(f"**Built with:** {lang}\n")

    # Table of contents
    lines += [
        "## Table of Contents",
        "- [Features](#features)",
        "- [Installation](#installation)",
        "- [Usage](#usage)",
    ]
    if contrib:
        lines.append("- [Contributing](#contributing)")
    lines += ["- [License](#license)", ""]

    # Features
    lines.append("## Features\n")
    if features:
        for f in features:
            lines.append(f"- {f}")
    else:
        lines += ["- Feature 1", "- Feature 2", "- Feature 3"]
    lines.append("")

    # Installation
    lines += [
        "## Installation",
        "",
        "```bash",
    ]
    if github_url:
        lines.append(f"git clone {github_url}")
    else:
        lines.append(f"git clone https://github.com/yourusername/{name.lower().replace(' ', '-')}.git")
    lines += [
        f"cd {name.lower().replace(' ', '-')}",
        install_cmd,
        "```",
        "",
    ]

    # Usage
    lines += [
        "## Usage",
        "",
        "```bash",
        usage_ex,
        "```",
        "",
    ]

    # Contributing
    if contrib:
        lines += [
            "## Contributing",
            "",
            "Contributions are welcome! Please feel free to submit a Pull Request.",
            "",
            "1. Fork the project",
            "2. Create your feature branch (`git checkout -b feature/AmazingFeature`)",
            "3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)",
            "4. Push to the branch (`git push origin feature/AmazingFeature`)",
            "5. Open a Pull Request",
            "",
        ]

    # License
    lines += [
        "## License",
        "",
        f"Distributed under the {license_} License."
        + (f" See `LICENSE` for more information." if license_ != "See LICENSE file" else ""),
        "",
    ]

    if author:
        lines += [f"---\n\nMade with ❤️ by **{author}** · {year}"]

    return "\n".join(lines) + "\n"


def build_minimal(data: dict) -> str:
    name = data.get("name", "My Project")
    desc = data.get("description", "A great project.")
    return f"# {name}\n\n{desc}\n\n## Installation\n\n```bash\n# install instructions here\n```\n\n## Usage\n\n```bash\n# usage here\n```\n\n## License\n\n{data.get('license', 'MIT')}\n"


# ── Interactive wizard ─────────────────────────────────────────────────────────

def ask(prompt: str, default: str = "") -> str:
    suffix = f" [{default}]" if default else ""
    try:
        val = input(c(f"  {prompt}{suffix}: ", "cyan")).strip()
    except (EOFError, KeyboardInterrupt):
        print()
        sys.exit(0)
    return val if val else default


def run_wizard(project_dir: str = ".") -> dict:
    print(c("\n  README Generator Wizard\n", "bold"))

    lang    = detect_language(project_dir)
    fw      = detect_framework(project_dir)
    lic     = detect_license(project_dir)

    name   = ask("Project name", os.path.basename(os.path.abspath(project_dir)))
    desc   = ask("Short description", "A useful Python project.")
    author = ask("Author name", "")
    github = ask("GitHub URL (optional)", "")
    lang   = ask("Primary language", lang)
    fw     = ask("Framework (optional)", fw)
    lic    = ask("License", lic)
    install= ask("Install command", "pip install -r requirements.txt")
    usage  = ask("Usage example", "python main.py")

    print(c("\n  Enter features (one per line, empty line to stop):", "cyan"))
    features = []
    while True:
        try:
            f = input("    - ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not f:
            break
        features.append(f)

    return {
        "name": name, "description": desc, "author": author, "github": github,
        "language": lang, "framework": fw, "license": lic,
        "install": install, "usage": usage, "features": features,
        "contributing": True, "badges": True,
    }


def main():
    parser = argparse.ArgumentParser(description="Generate a project README.md")
    parser.add_argument("--project",  metavar="DIR",      help="Project directory", default=".")
    parser.add_argument("--template", metavar="TEMPLATE", help="Template: full|minimal", default="full")
    parser.add_argument("--output",   metavar="FILE",     help="Output file", default="README.md")
    parser.add_argument("--preview",  action="store_true", help="Print to stdout without saving")
    args = parser.parse_args()

    data = run_wizard(args.project)

    if args.template == "minimal":
        content = build_minimal(data)
    else:
        content = build_readme(data)

    if args.preview:
        print(c("\n── Preview ──────────────────────────────\n", "bold"))
        print(content)
    else:
        out = os.path.join(args.project, args.output)
        if os.path.exists(out):
            confirm = ask(f"\n  {args.output} already exists. Overwrite?", "n").lower()
            if confirm not in ("y", "yes"):
                print(c("  Aborted.", "yellow"))
                return

        with open(out, "w", encoding="utf-8") as f:
            f.write(content)
        print(c(f"\n  ✓ Saved to {out}", "green"))
        print(c(f"  Lines: {len(content.splitlines())}  |  Chars: {len(content)}", "reset"))


if __name__ == "__main__":
    main()

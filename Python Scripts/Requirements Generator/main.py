"""Requirements Generator — CLI tool.

Scan Python files in a directory to detect third-party imports
and generate a requirements.txt with optional version pinning.

Usage:
    python main.py
    python main.py /path/to/project
"""

import ast
import importlib.metadata
import json
import re
import subprocess
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Standard library module list (rough set — avoid false positives)
# ---------------------------------------------------------------------------

STDLIB_TOP: set[str] = set(
    "abc,aifc,argparse,ast,asynchat,asyncio,asyncore,atexit,audioop,"
    "base64,bdb,binascii,binhex,bisect,builtins,bz2,calendar,cgi,"
    "cgitb,chunk,cmath,cmd,code,codecs,codeop,colorsys,compileall,"
    "concurrent,configparser,contextlib,contextvars,copy,copyreg,"
    "cProfile,csv,ctypes,curses,dataclasses,datetime,dbm,decimal,"
    "difflib,dis,distutils,doctest,email,encodings,enum,errno,"
    "faulthandler,fcntl,filecmp,fileinput,fnmatch,fractions,ftplib,"
    "functools,gc,getopt,getpass,gettext,glob,grp,gzip,hashlib,"
    "heapq,hmac,html,http,idlelib,imaplib,imghdr,imp,importlib,"
    "inspect,io,ipaddress,itertools,json,keyword,lib2to3,linecache,"
    "locale,logging,lzma,mailbox,mailcap,marshal,math,mimetypes,"
    "mmap,modulefinder,multiprocessing,netrc,nis,nntplib,numbers,"
    "operator,optparse,os,ossaudiodev,pathlib,pdb,pickle,pickletools,"
    "pipes,pkgutil,platform,plistlib,poplib,posix,posixpath,pprint,"
    "profile,pstats,pty,pwd,py_compile,pyclbr,pydoc,queue,quopri,"
    "random,re,readline,reprlib,rlcompleter,runpy,sched,secrets,"
    "select,selectors,shelve,shlex,shutil,signal,site,smtpd,smtplib,"
    "sndhdr,socket,socketserver,spwd,sqlite3,sre_compile,sre_constants,"
    "sre_parse,ssl,stat,statistics,string,stringprep,struct,subprocess,"
    "sunau,symtable,sys,sysconfig,syslog,tabnanny,tarfile,telnetlib,"
    "tempfile,termios,test,textwrap,threading,time,timeit,tkinter,"
    "token,tokenize,tomllib,trace,traceback,tracemalloc,tty,turtle,"
    "turtledemo,types,typing,unicodedata,unittest,urllib,uu,uuid,"
    "venv,warnings,wave,weakref,webbrowser,winreg,winsound,wsgiref,"
    "xdrlib,xml,xmlrpc,zipapp,zipfile,zipimport,zlib,zoneinfo,"
    # common sub-packages accessed as top-level
    "collections,contextlib,pathlib,dataclasses,_thread,abc,io".split(",")
)

# Flatten (some entries may have commas inside the set init above)
STDLIB_TOP = {m.strip() for m in ",".join(STDLIB_TOP).split(",") if m.strip()}


# ---------------------------------------------------------------------------
# Import extraction
# ---------------------------------------------------------------------------

def extract_imports_from_file(path: Path) -> set[str]:
    try:
        tree = ast.parse(path.read_text(errors="replace"))
    except SyntaxError:
        return set()
    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module and node.level == 0:
                imports.add(node.module.split(".")[0])
    return imports


def scan_directory(root: Path) -> dict[str, list[str]]:
    """Return {module: [files_that_import_it]}."""
    results: dict[str, list[str]] = {}
    for py_file in root.rglob("*.py"):
        if any(part.startswith(".") for part in py_file.parts):
            continue  # skip hidden dirs
        if "venv" in py_file.parts or "env" in py_file.parts:
            continue
        for mod in extract_imports_from_file(py_file):
            results.setdefault(mod, []).append(str(py_file.relative_to(root)))
    return results


def filter_third_party(all_imports: dict[str, list[str]]) -> dict[str, list[str]]:
    return {m: files for m, files in all_imports.items() if m not in STDLIB_TOP}


# ---------------------------------------------------------------------------
# Version pinning
# ---------------------------------------------------------------------------

KNOWN_PACKAGE_MAP: dict[str, str] = {
    # import name → PyPI name
    "cv2":          "opencv-python",
    "sklearn":      "scikit-learn",
    "PIL":          "Pillow",
    "bs4":          "beautifulsoup4",
    "yaml":         "PyYAML",
    "dotenv":       "python-dotenv",
    "wx":           "wxPython",
    "gi":           "PyGObject",
    "usb":          "pyusb",
    "serial":       "pyserial",
    "jwt":          "PyJWT",
    "dateutil":     "python-dateutil",
    "attr":         "attrs",
    "google.cloud": "google-cloud",
    "sklearn":      "scikit-learn",
    "tensorflow":   "tensorflow",
    "torch":        "torch",
    "transformers": "transformers",
    "streamlit":    "streamlit",
    "gradio":       "gradio",
    "flask":        "Flask",
    "django":       "Django",
    "fastapi":      "fastapi",
    "pydantic":     "pydantic",
    "sqlalchemy":   "SQLAlchemy",
    "aiohttp":      "aiohttp",
    "httpx":        "httpx",
    "requests":     "requests",
    "numpy":        "numpy",
    "pandas":       "pandas",
    "matplotlib":   "matplotlib",
    "seaborn":      "seaborn",
    "plotly":       "plotly",
    "scipy":        "scipy",
}


def get_installed_version(package_name: str) -> str | None:
    try:
        return importlib.metadata.version(package_name)
    except importlib.metadata.PackageNotFoundError:
        return None


def module_to_package(module_name: str) -> str:
    return KNOWN_PACKAGE_MAP.get(module_name, module_name)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

MENU = """
Requirements Generator
----------------------
1. Scan directory for imports
2. Generate requirements.txt (no versions)
3. Generate requirements.txt (pinned to installed)
4. Show third-party imports + source files
0. Quit
"""


def main():
    root_dir: Path | None = None
    third_party: dict[str, list[str]] = {}

    if len(sys.argv) > 1:
        root_dir = Path(sys.argv[1])
        if not root_dir.is_dir():
            print(f"Not a directory: {root_dir}")
            sys.exit(1)

    while True:
        print(MENU)
        choice = input("Choice: ").strip()

        if choice == "0":
            print("Bye!")
            break

        elif choice == "1":
            if not root_dir:
                path_str = input("  Project directory: ").strip() or "."
                root_dir = Path(path_str)
            if not root_dir.is_dir():
                print(f"  Not a directory: {root_dir}")
                root_dir = None
                continue
            print(f"\n  Scanning {root_dir}…")
            all_imports = scan_directory(root_dir)
            third_party = filter_third_party(all_imports)
            print(f"  Found {len(all_imports)} unique imports, {len(third_party)} third-party.")
            packages = sorted(module_to_package(m) for m in third_party)
            print("  Third-party: " + ", ".join(packages) if packages else "  None detected.")

        elif choice in ("2", "3"):
            if not third_party:
                print("  Run option 1 first to scan imports.")
                continue
            lines = []
            for module in sorted(third_party):
                pkg = module_to_package(module)
                if choice == "3":
                    ver = get_installed_version(pkg)
                    lines.append(f"{pkg}=={ver}" if ver else pkg)
                else:
                    lines.append(pkg)
            output = "\n".join(lines)
            out_path = (root_dir / "requirements.txt") if root_dir else Path("requirements.txt")
            out_path.write_text(output + "\n")
            print(f"\n  Written to: {out_path}")
            print(output)

        elif choice == "4":
            if not third_party:
                print("  Run option 1 first.")
                continue
            print(f"\n  {'Module':<25} {'Package':<25} Files")
            print("  " + "─" * 70)
            for module, files in sorted(third_party.items()):
                pkg = module_to_package(module)
                print(f"  {module:<25} {pkg:<25} {', '.join(files[:3])}"
                      + ("…" if len(files) > 3 else ""))

        else:
            print("  Invalid choice.")


if __name__ == "__main__":
    main()

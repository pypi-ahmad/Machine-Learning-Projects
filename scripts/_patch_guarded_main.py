#!/usr/bin/env python3
"""One-time script — replace  `main()`  with  `guarded_main(main, OUTPUT_DIR)`
in every project's run.py file.

Run once:   python scripts/_patch_guarded_main.py
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

OLD = 'if __name__ == "__main__":\n    main()\n'
NEW = 'if __name__ == "__main__":\n    from shared.utils import guarded_main\n    guarded_main(main, OUTPUT_DIR)\n'

patched = 0
for run_py in sorted(ROOT.glob("Deep Learning Projects*/run.py")):
    text = run_py.read_text(encoding="utf-8")
    if OLD in text and "guarded_main" not in text:
        text = text.replace(OLD, NEW)
        run_py.write_text(text, encoding="utf-8")
        patched += 1
        print(f"  [PATCHED] {run_py.parent.name}")
    else:
        print(f"  [SKIP]    {run_py.parent.name}")

print(f"\nDone. Patched {patched} files.")

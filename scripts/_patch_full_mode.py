#!/usr/bin/env python3
"""
One-time patcher: add --mode full / --profile support to all 50 run.py files.

For each project type, this script:
  1. Adds imports for load_profile, resolve_config, write_split_manifest,
     make_tabular_splits, EarlyStopping, run_with_oom_backoff (as needed)
  2. Inserts a TASK_TYPE constant
  3. Modifies main() to:
     a. Load a profile when --profile is passed
     b. Use profile-resolved config for epochs/batch_size/lr/etc.
     c. In full mode: use explicit train/val/test splits, early stopping,
        gradient accumulation, backbone freezing, and write a split manifest
     d. In smoke mode: keep the existing behaviour (1 epoch, max_batches=2)

Run:  python scripts/_patch_full_mode.py
      python scripts/_patch_full_mode.py --dry-run   # just print what would change
"""
from __future__ import annotations
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# ── Project classification ────────────────────────────────────────────────────
PROJECT_TYPES: dict[int, str] = {
    1: "cv", 2: "cv", 3: "tabular_reg", 4: "cv", 5: "nlp",
    6: "nlp", 7: "tabular_cls", 8: "tabular_cls", 9: "nlp",
    10: "cv", 11: "cv", 12: "cv", 13: "tabular_cls", 14: "cv",
    15: "tabular_reg", 16: "custom_cv", 17: "tabular_cls", 18: "cv",
    19: "cv", 20: "cv", 21: "cv", 22: "tabular_cls", 23: "custom_special",
    24: "tabular_reg", 25: "tabular_reg", 26: "inference_only",
    27: "tabular_reg", 28: "nlp", 29: "tabular_reg", 30: "tabular_reg",
    31: "cv", 32: "cv", 33: "cv", 34: "cv", 35: "nlp",
    36: "cv", 37: "cv", 38: "tabular_cls", 39: "tabular_cls",
    40: "tabular_cluster", 41: "custom_audio", 42: "cv", 43: "cv",
    44: "custom_cv", 45: "nlp", 46: "nlp", 47: "cv", 48: "cv",
    49: "nlp", 50: "cv",
}


def find_project_dirs() -> list[tuple[int, Path]]:
    results = []
    for d in sorted(ROOT.iterdir()):
        if d.is_dir() and d.name.startswith("Deep Learning Projects"):
            m = re.search(r"(\d+)", d.name)
            if m:
                results.append((int(m.group(1)), d))
    return sorted(results)


def patch_cv(num: int, code: str) -> str:
    """Patch a shared/cv.py project (22 projects)."""
    # 1. Add imports
    code = _add_import(code, "load_profile", "resolve_config",
                       "write_split_manifest", "EarlyStopping",
                       "run_with_oom_backoff")
    # 2. Add TASK_TYPE constant
    code = _add_task_type(code, "cv")

    # 3. Patch main() — replace the training section
    code = _patch_cv_main(code)
    return code


def patch_nlp(num: int, code: str) -> str:
    """Patch a shared/nlp.py project (8 projects)."""
    code = _add_import(code, "load_profile", "resolve_config",
                       "write_split_manifest", "make_tabular_splits")
    code = _add_task_type(code, "nlp")
    code = _patch_nlp_main(code)
    return code


def patch_tabular(num: int, code: str, variant: str) -> str:
    """Patch tabular_cls / tabular_reg / tabular_cluster."""
    code = _add_import(code, "load_profile", "resolve_config",
                       "write_split_manifest", "make_tabular_splits")
    code = _add_task_type(code, "tabular")
    code = _patch_tabular_main(code, variant)
    return code


def patch_custom_cv(num: int, code: str) -> str:
    """Patch custom_cv projects (P16 segmentation, P44 colorization)."""
    code = _add_import(code, "load_profile", "resolve_config",
                       "write_split_manifest", "EarlyStopping")
    code = _add_task_type(code, "custom_cv")
    code = _patch_custom_loop_main(code, "custom_cv")
    return code


def patch_custom_audio(num: int, code: str) -> str:
    code = _add_import(code, "load_profile", "resolve_config",
                       "write_split_manifest", "EarlyStopping")
    code = _add_task_type(code, "custom_audio")
    code = _patch_custom_loop_main(code, "custom_audio")
    return code


def patch_custom_special(num: int, code: str) -> str:
    code = _add_import(code, "load_profile", "resolve_config",
                       "write_split_manifest", "EarlyStopping")
    code = _add_task_type(code, "custom_special")
    code = _patch_custom_loop_main(code, "custom_special")
    return code


def patch_inference(num: int, code: str) -> str:
    """P26 — inference only — just add TASK_TYPE, no training changes."""
    code = _add_task_type(code, "inference_only")
    return code


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _add_import(code: str, *names: str) -> str:
    """Add missing names to the shared.utils import block."""
    # Match the full "from shared.utils import ..." statement, handling
    # both single-line and multi-line (parenthesised) forms.
    #
    # Strategy: find the import, extract all names, add new ones, rebuild.
    pattern = r"from shared\.utils import \(([^)]+)\)"
    m = re.search(pattern, code, re.DOTALL)
    if not m:
        # Try single-line without parens
        pattern2 = r"from shared\.utils import ([^\n]+)\n"
        m = re.search(pattern2, code)
        if not m:
            # No shared.utils import found; add one near the top
            for name in names:
                if name not in code:
                    code = re.sub(
                        r"(from shared\.\w+ import [^\n]+\n)",
                        lambda ma: f"from shared.utils import {name}\n" + ma.group(0),
                        code, count=1)
            return code

    full_match = m.group(0)
    body = m.group(1)

    # Parse out existing names (handle commas, newlines, trailing commas)
    current = [n.strip().rstrip(",") for n in re.split(r"[,\n]", body)
               if n.strip().rstrip(",")]

    added = [n for n in names if n not in current]
    if not added:
        return code

    all_names = current + added

    # Rebuild as a multi-line parenthesised import, max ~88 chars wide
    lines = ["from shared.utils import ("]
    line = "    "
    for i, name in enumerate(all_names):
        sep = ", " if i < len(all_names) - 1 else ""
        candidate = line + name + sep
        if len(candidate) > 88 and line.strip():
            lines.append(line.rstrip(", ") + ",")
            line = "    " + name + sep
        else:
            line = candidate
    lines.append(line.rstrip(",") + ")")
    new_import = "\n".join(lines)

    code = code.replace(full_match, new_import)
    return code


def _add_task_type(code: str, task_type: str) -> str:
    """Insert TASK_TYPE = '...' after the last module-level constant."""
    if "TASK_TYPE" in code:
        return code
    # Insert before def get_data or def main
    for anchor in (r"\ndef get_data", r"\ndef get_model", r"\ndef main"):
        m = re.search(anchor, code)
        if m:
            pos = m.start()
            code = code[:pos] + f"\nTASK_TYPE = '{task_type}'\n" + code[pos:]
            return code
    return code


def _patch_cv_main(code: str) -> str:
    """Replace main() body for CV projects using shared/cv.py.

    Strategy: find the block between 'def main():' and 'if __name__',
    then regenerate it with full-mode support but preserving get_data() call.
    """
    # We need to inject profile/config handling and forward new kwargs to
    # train_model.  Rather than doing a fragile regex replacement of the
    # entire main(), we surgically modify the training call.

    # A) Add profile loading after parse_common_args
    if "load_profile" not in code.split("def main")[1] if "def main" in code else "":
        code = code.replace(
            "    args = parse_common_args()\n    seed_everything(42)\n",
            "    args = parse_common_args()\n"
            "    profile = load_profile(args.profile)\n"
            "    cfg = resolve_config(args, profile, TASK_TYPE)\n"
            "    seed_everything(cfg.get('seed', 42))\n"
        )

    # B) Replace the epochs/batch/flags block to read cfg
    # Find the pattern: epochs = args.epochs or EPOCHS ... if args.smoke_test: epochs = 1
    old_params = re.search(
        r"(    epochs\s*=\s*args\.epochs or EPOCHS.*?"
        r"if args\.smoke_test:\n\s*epochs = 1\n)",
        code, re.DOTALL)
    if old_params:
        new_params = (
            "    is_full    = (args.mode == 'full')\n"
            "    epochs     = args.epochs or cfg.get('epochs', EPOCHS)\n"
            "    batch_size = args.batch_size or cfg.get('batch_size', BATCH)\n"
            "    num_workers = args.num_workers if args.num_workers is not None else cfg.get('num_workers', 2)\n"
            "    use_amp    = not args.no_amp and cfg.get('amp', True)\n"
            "    img_size   = cfg.get('img_size', 224)\n"
            "    max_batches = 2 if args.smoke_test else None\n"
            "    grad_accum  = cfg.get('grad_accum_steps', 1) if is_full else 1\n"
            "    freeze_bb   = cfg.get('freeze_backbone_epochs', 0) if is_full else 0\n"
            "    es_on       = cfg.get('early_stopping', False) if is_full else False\n"
            "    es_patience = cfg.get('patience', 3)\n"
            "    if args.smoke_test:\n"
            "        epochs = 1\n"
        )
        code = code[:old_params.start()] + new_params + code[old_params.end():]

    # C) Pass new kwargs to create_dataloaders (img_size)
    code = re.sub(
        r"    train_dl, val_dl, test_dl, class_names = create_dataloaders\(\n"
        r"        data_root, img_size=\d+, batch_size=batch_size,\n"
        r"        num_workers=num_workers,\n"
        r"    \)",
        "    train_dl, val_dl, test_dl, class_names = create_dataloaders(\n"
        "        data_root, img_size=img_size, batch_size=batch_size,\n"
        "        num_workers=num_workers,\n"
        "    )",
        code)

    # D) Pass grad_accum, early-stopping, freeze to train_model
    code = re.sub(
        r"(    model = train_model\(\n"
        r"        model, train_dl, val_dl,\n"
        r"        epochs=epochs, lr=[\de.-]+, device=device, output_dir=OUTPUT_DIR,\n"
        r"        use_amp=use_amp, max_batches=max_batches,\n"
        r"    \))",
        "    model = train_model(\n"
        "        model, train_dl, val_dl,\n"
        "        epochs=epochs, lr=cfg.get('lr', 1e-4), device=device, output_dir=OUTPUT_DIR,\n"
        "        use_amp=use_amp, max_batches=max_batches,\n"
        "        grad_accum_steps=grad_accum,\n"
        "        early_stopping=es_on, patience=es_patience,\n"
        "        freeze_backbone_epochs=freeze_bb,\n"
        "    )",
        code)

    # E) Add split manifest at end of main
    if "write_split_manifest" not in code.split("def main")[1] if "def main" in code else "":
        # Match the FULL multi-line evaluate_model call (ends with closing paren + newline)
        code = re.sub(
            r"(    evaluate_model\(model, test_dl, class_names.*?max_batches=max_batches\)\s*\n)",
            r"\1"
            "    if is_full:\n"
            "        write_split_manifest(OUTPUT_DIR, dataset_name=DATASET,\n"
            "            split_counts={'train': len(train_dl.dataset),\n"
            "                          'val': len(val_dl.dataset),\n"
            "                          'test': len(test_dl.dataset)},\n"
            "            seed=cfg.get('seed', 42))\n",
            code, flags=re.DOTALL)

    return code


def _patch_nlp_main(code: str) -> str:
    """Inject profile support into NLP projects."""
    # A) Profile loading
    if "load_profile" not in code.split("def main")[1] if "def main" in code else "":
        code = code.replace(
            "    args = parse_common_args()\n    seed_everything(42)\n",
            "    args = parse_common_args()\n"
            "    profile = load_profile(args.profile)\n"
            "    cfg = resolve_config(args, profile, TASK_TYPE)\n"
            "    seed_everything(cfg.get('seed', 42))\n"
        )

    # B) Replace param block
    old_params = re.search(
        r"(    epochs\s*=\s*args\.epochs or EPOCHS.*?"
        r"if args\.smoke_test:\n.+?epochs = 1\n)",
        code, re.DOTALL)
    if old_params:
        new_params = (
            "    is_full    = (args.mode == 'full')\n"
            "    epochs     = args.epochs or cfg.get('epochs', EPOCHS)\n"
            "    batch_size = args.batch_size or cfg.get('batch_size', BATCH)\n"
            "    use_amp    = not args.no_amp and cfg.get('amp', True)\n"
            "    max_steps  = 4 if args.smoke_test else -1\n"
            "    grad_accum = cfg.get('grad_accum_steps', 1) if is_full else 1\n"
            "    gc_on      = cfg.get('gradient_checkpointing', False) if is_full else False\n"
            "    es_patience = cfg.get('patience', 2) if is_full else None\n"
            "\n"
            "    if args.smoke_test:\n"
            "        epochs = 1\n"
        )
        code = code[:old_params.start()] + new_params + code[old_params.end():]

    # C) Pass new kwargs to train_hf_classifier
    code = re.sub(
        r"(    trainer = train_hf_classifier\(\n"
        r"        model, train_ds, test_ds, OUTPUT_DIR,\n"
        r"        epochs=epochs, batch_size=batch_size, lr=[\de.-]+,\n"
        r"        use_amp=use_amp, max_steps=max_steps,\n"
        r"    \))",
        "    trainer = train_hf_classifier(\n"
        "        model, train_ds, test_ds, OUTPUT_DIR,\n"
        "        epochs=epochs, batch_size=batch_size, lr=cfg.get('lr', 2e-5),\n"
        "        use_amp=use_amp, max_steps=max_steps,\n"
        "        gradient_checkpointing=gc_on,\n"
        "        early_stopping_patience=es_patience,\n"
        "        grad_accum_steps=grad_accum,\n"
        "    )",
        code)

    # D) Add split manifest
    if "write_split_manifest" not in code.split("def main")[1] if "def main" in code else "":
        code = re.sub(
            r"(    evaluate_hf_classifier\(trainer, test_ds, class_names, OUTPUT_DIR\)\n)",
            r"\1"
            "    if is_full:\n"
            "        write_split_manifest(OUTPUT_DIR, dataset_name=DATASET,\n"
            "            split_counts={'train': len(train_ds), 'test': len(test_ds)},\n"
            "            seed=cfg.get('seed', 42))\n",
            code)

    return code


def _patch_tabular_main(code: str, variant: str) -> str:
    """Inject profile support into tabular projects."""
    # A) Profile loading
    if "load_profile" not in code.split("def main")[1] if "def main" in code else "":
        code = code.replace(
            "    args = parse_common_args()\n    seed_everything(42)\n",
            "    args = parse_common_args()\n"
            "    profile = load_profile(args.profile)\n"
            "    cfg = resolve_config(args, profile, TASK_TYPE)\n"
            "    seed_everything(cfg.get('seed', 42))\n"
        )

    # B) Replace the smoke_test df truncation block, add full-mode splits
    # Detect the smoke_test block (if args.smoke_test: df = df.head(...))
    smoke_block = re.search(
        r"(    if args\.smoke_test:\n"
        r"        df = df\.head\(.*?\)\n"
        r"        print\(.*?\)\n)",
        code, re.DOTALL)
    if smoke_block:
        fn_name = {"tabular_cls": "run_pycaret_classification",
                    "tabular_reg": "run_pycaret_regression",
                    "tabular_cluster": "run_pycaret_clustering"}[variant]

        new_block = (
            "    is_full = (args.mode == 'full')\n"
            "    if args.smoke_test:\n"
            "        df = df.head(min(200, len(df)))\n"
            "        print(f'  [SMOKE] Using first {len(df)} rows only.')\n"
        )
        code = code[:smoke_block.start()] + new_block + code[smoke_block.end():]

    # C) Replace the run_pycaret call to add full-mode split manifest
    if variant == "tabular_cluster":
        code = re.sub(
            r"    run_pycaret_clustering\(df, output_dir=OUTPUT_DIR\)\n",
            "    run_pycaret_clustering(df, output_dir=OUTPUT_DIR)\n"
            "    if is_full:\n"
            "        write_split_manifest(OUTPUT_DIR, dataset_name=DATASET,\n"
            "            split_counts={'total': len(df)},\n"
            "            seed=cfg.get('seed', 42), method='none (clustering)')\n",
            code)
    elif variant == "tabular_cls":
        code = re.sub(
            r"    run_pycaret_classification\(df, target=TARGET, output_dir=OUTPUT_DIR\)\n",
            "    run_pycaret_classification(df, target=TARGET, output_dir=OUTPUT_DIR)\n"
            "    if is_full:\n"
            "        write_split_manifest(OUTPUT_DIR, dataset_name=DATASET,\n"
            "            split_counts={'total': len(df)},\n"
            "            seed=cfg.get('seed', 42))\n",
            code)
    elif variant == "tabular_reg":
        code = re.sub(
            r"    run_pycaret_regression\(df, target=TARGET, output_dir=OUTPUT_DIR\)\n",
            "    run_pycaret_regression(df, target=TARGET, output_dir=OUTPUT_DIR)\n"
            "    if is_full:\n"
            "        write_split_manifest(OUTPUT_DIR, dataset_name=DATASET,\n"
            "            split_counts={'total': len(df)},\n"
            "            seed=cfg.get('seed', 42))\n",
            code)

    return code


def _patch_custom_loop_main(code: str, task_type: str) -> str:
    """Inject profile loading + early-stopping into custom training loops."""
    # A) Profile loading
    if "load_profile" not in code.split("def main")[1] if "def main" in code else "":
        code = code.replace(
            "    args = parse_common_args()\n    seed_everything(42)\n",
            "    args = parse_common_args()\n"
            "    profile = load_profile(args.profile)\n"
            "    cfg = resolve_config(args, profile, TASK_TYPE)\n"
            "    seed_everything(cfg.get('seed', 42))\n"
        )

    # B) Replace epochs / batch_size to read from cfg
    old_params = re.search(
        r"(    epochs\s*=\s*args\.epochs or EPOCHS.*?"
        r"if args\.smoke_test:\n\s*epochs = 1\n)",
        code, re.DOTALL)
    if old_params:
        new_params = (
            "    is_full     = (args.mode == 'full')\n"
            f"    epochs      = args.epochs or cfg.get('epochs', EPOCHS)\n"
            "    batch_size  = args.batch_size or cfg.get('batch_size', BATCH)\n"
            "    num_workers = args.num_workers if args.num_workers is not None else cfg.get('num_workers', 2)\n"
            "    use_amp     = not args.no_amp and cfg.get('amp', True)\n"
            "    max_batches = 2 if args.smoke_test else None\n"
            "    es_on       = cfg.get('early_stopping', False) if is_full else False\n"
            "    es_patience = cfg.get('patience', 3)\n"
            "    if args.smoke_test:\n"
            "        epochs = 1\n"
        )
        code = code[:old_params.start()] + new_params + code[old_params.end():]

    # C) Add early stopping object creation and check in training loop
    # Insert es = EarlyStopping(...) before the training loop
    if "es = EarlyStopping" not in code:
        code = re.sub(
            r"(    best(?:_dice|_acc|) = [\d.]+\n\n    for ep in range)",
            r"    es = EarlyStopping(patience=es_patience, mode='max') if es_on else None\n\n\1",
            code)
        # Add early stopping check at end of epoch loop
        # Find the save_metrics line at the bottom to detect end of loop
        # We look for the pattern: if ... > best...: best... = ... save(...)
        # and add es.step() + break after it
        code = re.sub(
            r"(            torch\.save\(model\.state_dict\(\), OUTPUT_DIR / \"best_model\.pth\"\)\n)(\n    save_metrics)",
            r"\1        if es and es.step(mean_dice if 'mean_dice' in dir() else vacc if 'vacc' in dir() else acc):\n"
            r"            break\n\2",
            code)
        # Simpler approach for P41/P23 style:
        if "if es and es.step" not in code:
            code = re.sub(
                r"(            torch\.save\(model\.state_dict\(\), OUTPUT_DIR / \"best_model\.pth\"\)\n)(\n)",
                r"\1        if es and es.step(best):\n"
                r"            break\n\2",
                code, count=1)

    # D) Add split_manifest for full mode at end
    if "write_split_manifest" not in code.split("def main")[1] if "def main" in code else "":
        # Find the last save_metrics call and add after it
        last_save = list(re.finditer(r"    save_metrics\(.*?\n", code))
        if last_save:
            pos = last_save[-1].end()
            manifest_code = (
                "    if is_full:\n"
                "        write_split_manifest(OUTPUT_DIR, dataset_name=DATASET,\n"
                "            seed=cfg.get('seed', 42))\n"
            )
            code = code[:pos] + manifest_code + code[pos:]

    return code


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    dry_run = "--dry-run" in sys.argv

    projects = find_project_dirs()
    patched = 0
    skipped = 0
    errors = []

    for num, proj_dir in projects:
        run_py = proj_dir / "run.py"
        if not run_py.exists():
            continue

        task_type = PROJECT_TYPES.get(num, "unknown")
        code = run_py.read_text(encoding="utf-8")

        # Skip if already patched
        if "load_profile" in code and "TASK_TYPE" in code:
            print(f"  P{num:>2} [{task_type:16s}] — already patched, skipping")
            skipped += 1
            continue

        try:
            if task_type == "cv":
                new_code = patch_cv(num, code)
            elif task_type == "nlp":
                new_code = patch_nlp(num, code)
            elif task_type in ("tabular_cls", "tabular_reg", "tabular_cluster"):
                new_code = patch_tabular(num, code, task_type)
            elif task_type == "custom_cv":
                new_code = patch_custom_cv(num, code)
            elif task_type == "custom_audio":
                new_code = patch_custom_audio(num, code)
            elif task_type == "custom_special":
                new_code = patch_custom_special(num, code)
            elif task_type == "inference_only":
                new_code = patch_inference(num, code)
            else:
                print(f"  P{num:>2} [{task_type:16s}] — unknown type, skipping")
                skipped += 1
                continue

            if new_code != code:
                if dry_run:
                    print(f"  P{num:>2} [{task_type:16s}] — WOULD patch")
                else:
                    run_py.write_text(new_code, encoding="utf-8")
                    print(f"  P{num:>2} [{task_type:16s}] — PATCHED")
                patched += 1
            else:
                print(f"  P{num:>2} [{task_type:16s}] — no changes needed")
                skipped += 1
        except Exception as exc:
            print(f"  P{num:>2} [{task_type:16s}] — ERROR: {exc}")
            errors.append((num, str(exc)))

    print(f"\nDone: {patched} patched, {skipped} skipped, {len(errors)} errors")
    if errors:
        for n, e in errors:
            print(f"  P{n}: {e}")


if __name__ == "__main__":
    main()

"""
PHASE 2 — REPORT BUILDER
Reads phase2_summary.csv and phase2_detail.json, generates:
  - audit_phase2/phase2_report.md     (human-readable full report)
  - audit_phase2/standardization_plan.csv  (per-project: what to replace and with what)
NO FIXES — REPORTING ONLY.
"""

import json
import csv
from pathlib import Path

ROOT = Path(r"d:\Workspace\Github\Machine-Learning-Projects")
AUDIT = ROOT / "audit_phase2"

summary_path  = AUDIT / "phase2_summary.csv"
detail_path   = AUDIT / "phase2_detail.json"
issues_path   = AUDIT / "phase2_issues.csv"
report_path   = AUDIT / "phase2_report.md"
plan_path     = AUDIT / "standardization_plan.csv"

# ─── Load data ───────────────────────────────────────────────────────────────
with open(summary_path, encoding='utf-8') as f:
    rows = list(csv.DictReader(f))

with open(detail_path, encoding='utf-8') as f:
    details = json.load(f)

detail_map = {d['project']: d for d in details}

# ─── Aggregates ──────────────────────────────────────────────────────────────
total = len(rows)
needs_automl   = [r for r in rows if r['automl_status'] == 'NONE' and r['custom_ml_algorithms']]
has_leakage    = [r for r in rows if r['data_leakage'] == 'True']
no_split       = [r for r in rows if r['no_train_test_split'] == 'True']
no_eval        = [r for r in rows if r['no_evaluation'] == 'True']
has_hardcoded  = [r for r in rows if r['hardcoded_paths'] == 'True']
has_colab      = [r for r in rows if r['colab_artifacts'] == 'True']
has_lazyp      = [r for r in rows if r['automl_status'] in ('BOTH','LAZYPREDICT_ONLY')]
has_pycaret    = [r for r in rows if r['automl_status'] in ('BOTH','PYCARET_ONLY')]
has_pipeline   = [r for r in rows if r['has_pipeline'] == 'True']
untitled_nb    = [r for r in rows if 'UNTITLED_NB' in r['flags']]
duplicate_nb   = [r for r in rows if 'DUPLICATE_NB' in r['flags']]
any_issues     = [r for r in rows if r['flags'] != 'NONE']

alg_count = {}
for r in rows:
    for a in r['custom_ml_algorithms'].split('|'):
        if a:
            alg_count[a] = alg_count.get(a, 0) + 1

clean_projects = [r for r in rows if r['flags'] == 'NONE' and int(r['issue_count']) == 0]

# ─── Build Standardization Plan CSV ──────────────────────────────────────────
plan_rows = []
for r in rows:
    algs = [a for a in r['custom_ml_algorithms'].split('|') if a]
    automl_st = r['automl_status']
    
    replace_with = 'N/A'
    action = 'NO_ACTION'
    
    if automl_st == 'NONE' and algs:
        # Classify task type from alg names to recommend right tool
        is_classification = any(a in ('LogisticRegression','RandomForestClassifier','DecisionTreeClassifier',
                                       'GradientBoostingClassifier','SVM','KNN','NaiveBayes','AdaBoost',
                                       'MLP','XGBoost') for a in algs)
        is_regression = any(a in ('LinearRegression','Ridge','Lasso','ElasticNet','RandomForestRegressor',
                                   'DecisionTreeRegressor','GradientBoostingRegressor','XGBoost','LightGBM') for a in algs)
        is_clustering = any(a in ('KMeans','DBSCAN','AgglomerativeClustering','GaussianMixture') for a in algs)
        is_deep = any(a in ('Keras/TF','PyTorch') for a in algs)
        is_nlp = bool(r.get('project','').lower().find('nlp') >= 0 or r.get('project','').lower().find('text') >= 0 or
                      r.get('project','').lower().find('sentiment') >= 0)
        
        if is_deep:
            replace_with = 'KEEP_DEEP_LEARNING (no AutoML replacement)'
            action = 'KEEP_STANDARDIZE_ONLY'
        elif is_clustering:
            replace_with = 'LazyPredict (no direct clustering automl)'
            action = 'REPLACE_WITH_LAZYPREDICT'
        elif is_classification and is_regression:
            replace_with = 'LazyPredict + PyCaret (both modes)'
            action = 'REPLACE_WITH_BOTH'
        elif is_classification:
            replace_with = 'LazyPredict LazyClassifier + PyCaret compare_models()'
            action = 'REPLACE_WITH_LAZYPREDICT_PYCARET'
        elif is_regression:
            replace_with = 'LazyPredict LazyRegressor + PyCaret compare_models(task=regression)'
            action = 'REPLACE_WITH_LAZYPREDICT_PYCARET'
        else:
            replace_with = 'LazyPredict + PyCaret'
            action = 'REPLACE_WITH_BOTH'
    elif automl_st in ('LAZYPREDICT_ONLY',):
        action = 'ADD_PYCARET'
        replace_with = 'Add PyCaret on top of existing LazyPredict'
    elif automl_st in ('PYCARET_ONLY',):
        action = 'ADD_LAZYPREDICT'
        replace_with = 'Add LazyPredict on top of existing PyCaret'
    elif automl_st == 'BOTH':
        action = 'ALREADY_AUTOML'
        replace_with = 'Already has LazyPredict + PyCaret'
    else:
        action = 'NO_ML_DETECTED'
        replace_with = 'N/A — no ML training found'

    flags_list = [f for f in r['flags'].split('|') if f != 'NONE']
    additional_fixes = []
    if 'DATA_LEAKAGE' in flags_list: additional_fixes.append('FIX_LEAKAGE')
    if 'NO_SPLIT' in flags_list: additional_fixes.append('ADD_TRAIN_TEST_SPLIT')
    if 'NO_EVALUATION' in flags_list: additional_fixes.append('ADD_EVALUATION')
    if 'HARDCODED_PATH' in flags_list: additional_fixes.append('FIX_PATHS')
    if 'COLAB_ARTIFACT' in flags_list: additional_fixes.append('REMOVE_COLAB_MOUNTS')
    if 'UNTITLED_NB' in flags_list: additional_fixes.append('RENAME_UNTITLED_NB')
    if 'DUPLICATE_NB' in flags_list: additional_fixes.append('REMOVE_DUPLICATE_NB')

    plan_rows.append({
        'project': r['project'],
        'automl_status': automl_st,
        'custom_algorithms': r['custom_ml_algorithms'],
        'action': action,
        'replace_with': replace_with,
        'additional_fixes_needed': '|'.join(additional_fixes) if additional_fixes else 'NONE',
        'issue_count': r['issue_count'],
        'current_flags': r['flags'],
    })

with open(plan_path, 'w', newline='', encoding='utf-8') as f:
    w = csv.DictWriter(f, fieldnames=list(plan_rows[0].keys()))
    w.writeheader()
    w.writerows(plan_rows)

# ─── Build Phase 2 Report Markdown ───────────────────────────────────────────
lines = []
lines.append("# PHASE 2 — DEEP INSPECTION REPORT")
lines.append("")
lines.append("> Generated by `audit_phase2_inspect.py` + `audit_phase2_report_builder.py`")
lines.append("> NO FIXES APPLIED — DETECTION ONLY")
lines.append("")
lines.append("---")
lines.append("")
lines.append("## 1. EXECUTIVE SUMMARY")
lines.append("")
lines.append(f"| Metric | Count |")
lines.append(f"|--------|-------|")
lines.append(f"| Total projects inspected | {total} |")
lines.append(f"| Projects needing AutoML replacement | {len(needs_automl)} |")
lines.append(f"| Projects with data leakage detected | {len(has_leakage)} |")
lines.append(f"| Projects with no train/test split | {len(no_split)} |")
lines.append(f"| Projects with no evaluation metrics | {len(no_eval)} |")
lines.append(f"| Projects with hardcoded paths | {len(has_hardcoded)} |")
lines.append(f"| Projects with Colab artifacts | {len(has_colab)} |")
lines.append(f"| Projects already using LazyPredict | {len(has_lazyp)} |")
lines.append(f"| Projects already using PyCaret | {len(has_pycaret)} |")
lines.append(f"| Projects using sklearn Pipeline | {len(has_pipeline)} |")
lines.append(f"| Projects with Untitled notebooks | {len(untitled_nb)} |")
lines.append(f"| Projects with duplicate (1) notebooks | {len(duplicate_nb)} |")
lines.append(f"| Projects with any flag | {len(any_issues)} |")
lines.append(f"| Clean projects (no flags, no issues) | {len(clean_projects)} |")
lines.append("")
lines.append("---")
lines.append("")
lines.append("## 2. CUSTOM ALGORITHMS TO REPLACE")
lines.append("")
lines.append("Ranked by number of projects using each algorithm:")
lines.append("")
lines.append("| Algorithm | Projects Using It |")
lines.append("|-----------|------------------|")
for alg, cnt in sorted(alg_count.items(), key=lambda x: -x[1]):
    lines.append(f"| {alg} | {cnt} |")
lines.append("")
lines.append("---")
lines.append("")
lines.append("## 3. PROJECTS NEEDING AutoML REPLACEMENT")
lines.append("")
lines.append(f"{len(needs_automl)} projects use custom ML algorithms but have NO LazyPredict or PyCaret.")
lines.append("")
lines.append("| # | Project | Algorithms Used | Recommended Replacement |")
lines.append("|---|---------|-----------------|------------------------|")
for i, r in enumerate([p for p in plan_rows if p['action'] in ('REPLACE_WITH_LAZYPREDICT_PYCARET','REPLACE_WITH_BOTH','REPLACE_WITH_LAZYPREDICT')], 1):
    algs = r['custom_algorithms'][:60] if len(r['custom_algorithms']) > 60 else r['custom_algorithms']
    repl = r['replace_with'][:80] if len(r['replace_with']) > 80 else r['replace_with']
    lines.append(f"| {i} | {r['project'][:70]} | {algs} | {repl} |")
lines.append("")
lines.append("---")
lines.append("")
lines.append("## 4. DATA LEAKAGE ISSUES")
lines.append("")
if has_leakage:
    lines.append("| # | Project | Issue Description |")
    lines.append("|---|---------|-------------------|")
    for i, r in enumerate(has_leakage, 1):
        d = detail_map.get(r['project'], {})
        top_issue = d.get('top_issues', ['(no detail)'])[0] if d.get('top_issues') else '(no detail)'
        lines.append(f"| {i} | {r['project'][:70]} | {top_issue[:100]} |")
else:
    lines.append("No data leakage detected by heuristic pattern scan.")
    lines.append("")
    lines.append("> NOTE: Heuristic only catches scaler/encoder `.fit` before `train_test_split` in same cell source.")
    lines.append("> Manual review recommended for cross-cell leakage (fit in Cell A, split in Cell B).")
lines.append("")
lines.append("---")
lines.append("")
lines.append("## 5. MISSING TRAIN/TEST SPLIT (36 projects)")
lines.append("")
lines.append("Projects where model `.fit()` exists but no `train_test_split` / `KFold` / `cross_val_score` was detected:")
lines.append("")
lines.append("| # | Project |")
lines.append("|---|---------|")
for i, r in enumerate(no_split, 1):
    lines.append(f"| {i} | {r['project']} |")
lines.append("")
lines.append("---")
lines.append("")
lines.append("## 6. MISSING EVALUATION (31 projects)")
lines.append("")
lines.append("Projects where model `.fit()` exists but no evaluation metric call detected:")
lines.append("")
lines.append("| # | Project |")
lines.append("|---|---------|")
for i, r in enumerate(no_eval, 1):
    lines.append(f"| {i} | {r['project']} |")
lines.append("")
lines.append("---")
lines.append("")
lines.append("## 7. HARDCODED PATHS (24 projects)")
lines.append("")
lines.append("| # | Project | Path Type |")
lines.append("|---|---------|-----------|")
for i, r in enumerate(has_hardcoded, 1):
    d = detail_map.get(r['project'], {})
    path_issues = [x for x in d.get('top_issues', []) if 'PATH' in x or 'COLAB' in x or 'KAGGLE' in x]
    eg = path_issues[0][:80] if path_issues else '(see phase2_issues.csv)'
    lines.append(f"| {i} | {r['project'][:70]} | {eg} |")
lines.append("")
lines.append("---")
lines.append("")
lines.append("## 8. COLAB ARTIFACTS (3 projects)")
lines.append("")
lines.append("Projects with Google Colab `.mount()` or `google.colab` imports — must be removed for local execution:")
lines.append("")
lines.append("| # | Project |")
lines.append("|---|---------|")
for i, r in enumerate(has_colab, 1):
    lines.append(f"| {i} | {r['project']} |")
lines.append("")
lines.append("---")
lines.append("")
lines.append("## 9. ALREADY USING AutoML")
lines.append("")
lines.append("| Project | AutoML Status |")
lines.append("|---------|--------------|")
for r in [p for p in plan_rows if p['action'] == 'ALREADY_AUTOML']:
    lines.append(f"| {r['project'][:70]} | {r['automl_status']} |")
lines.append("")
lines.append("---")
lines.append("")
lines.append("## 10. DEEP LEARNING PROJECTS (keep, standardize only)")
lines.append("")
lines.append("| Project | Frameworks |")
lines.append("|---------|------------|")
for r in [p for p in plan_rows if p['action'] == 'KEEP_STANDARDIZE_ONLY']:
    lines.append(f"| {r['project'][:70]} | {r['custom_algorithms'][:80]} |")
lines.append("")
lines.append("---")
lines.append("")
lines.append("## 11. PER-PROJECT INSPECTION TABLE (all 158)")
lines.append("")
lines.append("| # | Project | Notebooks | Custom ML | AutoML Status | Pipeline | Leakage | No Split | No Eval | Hardcoded | Issues | Flags |")
lines.append("|---|---------|-----------|-----------|---------------|----------|---------|----------|---------|-----------|--------|-------|")
for i, r in enumerate(rows, 1):
    nb = r['notebooks_inspected']
    algs = r['custom_ml_algorithms'][:40] if len(r['custom_ml_algorithms']) > 40 else r['custom_ml_algorithms']
    algs = algs if algs else '-'
    automl = r['automl_status']
    pipeline = 'Y' if r['has_pipeline'] == 'True' else 'N'
    leakage = 'Y' if r['data_leakage'] == 'True' else 'N'
    nosplit = 'Y' if r['no_train_test_split'] == 'True' else 'N'
    noeval = 'Y' if r['no_evaluation'] == 'True' else 'N'
    hardcoded = 'Y' if r['hardcoded_paths'] == 'True' else 'N'
    issues = r['issue_count']
    flags = r['flags'][:50] if len(r['flags']) > 50 else r['flags']
    proj = r['project'][:65] if len(r['project']) > 65 else r['project']
    lines.append(f"| {i} | {proj} | {nb} | {algs} | {automl} | {pipeline} | {leakage} | {nosplit} | {noeval} | {hardcoded} | {issues} | {flags} |")
lines.append("")
lines.append("---")
lines.append("")
lines.append("## 12. STANDARDIZATION PLAN SUMMARY")
lines.append("")
lines.append("Full plan in `audit_phase2/standardization_plan.csv`")
lines.append("")
action_counts = {}
for p in plan_rows:
    action_counts[p['action']] = action_counts.get(p['action'], 0) + 1
lines.append("| Action Required | Count |")
lines.append("|-----------------|-------|")
for action, cnt in sorted(action_counts.items(), key=lambda x: -x[1]):
    lines.append(f"| {action} | {cnt} |")
lines.append("")
lines.append("---")
lines.append("")
lines.append("## 13. ARTIFACT INDEX")
lines.append("")
lines.append("| Artifact | Description |")
lines.append("|----------|-------------|")
lines.append("| `audit_phase2/phase2_summary.csv` | Per-project: all flags, algorithms, AutoML status, counts |")
lines.append("| `audit_phase2/phase2_issues.csv` | Per-project: every raw issue detected with file+cell+line |")
lines.append("| `audit_phase2/phase2_detail.json` | Per-project: notebook-level inspection with code snippets |")
lines.append("| `audit_phase2/standardization_plan.csv` | Per-project: action, replacement recommendation, additional fixes |")
lines.append("| `audit_phase2/phase2_report.md` | This file |")
lines.append("")
lines.append("---")
lines.append("*End of Phase 2 Deep Inspection Report*")

with open(report_path, 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines))

print(f"Phase 2 report written: {report_path}")
print(f"Standardization plan written: {plan_path}")
print(f"  Standardization plan rows: {len(plan_rows)}")
print(f"\nAction breakdown:")
for action, cnt in sorted(action_counts.items(), key=lambda x: -x[1]):
    print(f"  {cnt:3d}  {action}")

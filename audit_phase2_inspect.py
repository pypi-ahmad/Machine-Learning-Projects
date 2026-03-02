"""
PHASE 2 — DEEP INSPECTION SCRIPT
Parses every .ipynb notebook line-by-line across all 158 projects.
Extracts: data loading, preprocessing, model training, evaluation.
Detects: broken code, hardcoded paths, data leakage, wrong ML practices,
         redundant steps, inconsistent pipelines, custom ML needing replacement.
NO FIXES — DETECTION ONLY.
"""

import os
import json
import re
import csv
from pathlib import Path

ROOT = Path(r"d:\Workspace\Github\Machine-Learning-Projects")
AUDIT_DIR = ROOT / "audit_phase2"
AUDIT_DIR.mkdir(exist_ok=True)

# ─── PATTERN BANKS ──────────────────────────────────────────────────────────

DATA_LOAD_PATTERNS = [
    (r'pd\.read_csv\s*\(', 'read_csv'),
    (r'pd\.read_excel\s*\(', 'read_excel'),
    (r'pd\.read_json\s*\(', 'read_json'),
    (r'pd\.read_table\s*\(', 'read_table'),
    (r'pd\.read_parquet\s*\(', 'read_parquet'),
    (r'pd\.read_hdf\s*\(', 'read_hdf'),
    (r'np\.load\s*\(', 'np_load'),
    (r'np\.loadtxt\s*\(', 'np_loadtxt'),
    (r'load_dataset\s*\(', 'load_dataset'),
    (r'datasets\.load_', 'sklearn_load_dataset'),
    (r'fetch_\w+\s*\(', 'sklearn_fetch'),
    (r'ImageDataGenerator\s*\(', 'keras_ImageDataGenerator'),
    (r'flow_from_directory\s*\(', 'flow_from_directory'),
    (r'tf\.data', 'tf_data'),
    (r'torch\.utils\.data', 'torch_dataloader'),
    (r'cv2\.imread\s*\(', 'cv2_imread'),
    (r'open\s*\([^)]*["\']r["\']', 'file_open_read'),
    (r'requests\.get\s*\(', 'requests_get'),
    (r'urllib', 'urllib_download'),
    (r'kaggle', 'kaggle_api'),
]

PREPROCESS_PATTERNS = [
    (r'train_test_split\s*\(', 'train_test_split'),
    (r'StandardScaler\s*\(', 'StandardScaler'),
    (r'MinMaxScaler\s*\(', 'MinMaxScaler'),
    (r'RobustScaler\s*\(', 'RobustScaler'),
    (r'Normalizer\s*\(', 'Normalizer'),
    (r'LabelEncoder\s*\(', 'LabelEncoder'),
    (r'OneHotEncoder\s*\(', 'OneHotEncoder'),
    (r'get_dummies\s*\(', 'get_dummies'),
    (r'LabelBinarizer\s*\(', 'LabelBinarizer'),
    (r'OrdinalEncoder\s*\(', 'OrdinalEncoder'),
    (r'SimpleImputer\s*\(', 'SimpleImputer'),
    (r'KNNImputer\s*\(', 'KNNImputer'),
    (r'fillna\s*\(', 'fillna'),
    (r'dropna\s*\(', 'dropna'),
    (r'drop_duplicates\s*\(', 'drop_duplicates'),
    (r'\.fit_transform\s*\(', 'fit_transform'),
    (r'\.transform\s*\(', 'transform'),
    (r'PCA\s*\(', 'PCA'),
    (r'Pipeline\s*\(', 'Pipeline'),
    (r'ColumnTransformer\s*\(', 'ColumnTransformer'),
    (r'TfidfVectorizer\s*\(', 'TfidfVectorizer'),
    (r'CountVectorizer\s*\(', 'CountVectorizer'),
    (r'tokenize|Tokenizer', 'Tokenizer'),
    (r'ImageDataGenerator', 'ImageDataGenerator'),
    (r'resize\s*\(|reshape\s*\(', 'reshape'),
    (r'feature_importances_', 'feature_importances'),
]

TRAIN_PATTERNS = [
    # Custom algorithms to flag for replacement
    (r'LogisticRegression\s*\(', 'LogisticRegression', True),
    (r'LinearRegression\s*\(', 'LinearRegression', True),
    (r'Ridge\s*\(', 'Ridge', True),
    (r'Lasso\s*\(', 'Lasso', True),
    (r'ElasticNet\s*\(', 'ElasticNet', True),
    (r'DecisionTreeClassifier\s*\(', 'DecisionTreeClassifier', True),
    (r'DecisionTreeRegressor\s*\(', 'DecisionTreeRegressor', True),
    (r'RandomForestClassifier\s*\(', 'RandomForestClassifier', True),
    (r'RandomForestRegressor\s*\(', 'RandomForestRegressor', True),
    (r'GradientBoostingClassifier\s*\(', 'GradientBoostingClassifier', True),
    (r'GradientBoostingRegressor\s*\(', 'GradientBoostingRegressor', True),
    (r'XGBClassifier\s*\(|XGBRegressor\s*\(|xgb\.train\s*\(', 'XGBoost', True),
    (r'LGBMClassifier\s*\(|LGBMRegressor\s*\(|lgb\.train\s*\(', 'LightGBM', True),
    (r'CatBoostClassifier\s*\(|CatBoostRegressor\s*\(', 'CatBoost', True),
    (r'SVC\s*\(|SVR\s*\(|SVM\s*\(', 'SVM', True),
    (r'KNeighborsClassifier\s*\(|KNeighborsRegressor\s*\(', 'KNN', True),
    (r'GaussianNB\s*\(|MultinomialNB\s*\(|BernoulliNB\s*\(', 'NaiveBayes', True),
    (r'AdaBoostClassifier\s*\(|AdaBoostRegressor\s*\(', 'AdaBoost', True),
    (r'BaggingClassifier\s*\(|ExtraTreesClassifier\s*\(', 'Ensemble', True),
    (r'MLPClassifier\s*\(|MLPRegressor\s*\(', 'MLP', True),
    (r'KMeans\s*\(', 'KMeans', True),
    (r'DBSCAN\s*\(', 'DBSCAN', True),
    (r'AgglomerativeClustering\s*\(', 'AgglomerativeClustering', True),
    (r'GaussianMixture\s*\(', 'GaussianMixture', True),
    (r'ARIMA\s*\(|auto_arima\s*\(', 'ARIMA', True),
    (r'tf\.keras|keras\.Sequential|keras\.Model', 'Keras/TF', True),
    (r'torch\.nn|nn\.Module', 'PyTorch', True),
    (r'model\.fit\s*\(', 'model_fit_call', False),
    (r'clf\.fit\s*\(|reg\.fit\s*\(|estimator\.fit\s*\(', 'estimator_fit', False),
    (r'LazyClassifier\s*\(|LazyRegressor\s*\(', 'LazyPredict', False),
    (r'setup\s*\(.*pycaret|from pycaret', 'PyCaret', False),
    (r'GridSearchCV\s*\(', 'GridSearchCV', False),
    (r'RandomizedSearchCV\s*\(', 'RandomizedSearchCV', False),
    (r'cross_val_score\s*\(', 'cross_val_score', False),
]

EVAL_PATTERNS = [
    (r'accuracy_score\s*\(', 'accuracy_score'),
    (r'classification_report\s*\(', 'classification_report'),
    (r'confusion_matrix\s*\(', 'confusion_matrix'),
    (r'roc_auc_score\s*\(', 'roc_auc_score'),
    (r'roc_curve\s*\(', 'roc_curve'),
    (r'f1_score\s*\(', 'f1_score'),
    (r'precision_score\s*\(|recall_score\s*\(', 'precision_recall'),
    (r'mean_squared_error\s*\(', 'MSE'),
    (r'mean_absolute_error\s*\(', 'MAE'),
    (r'r2_score\s*\(', 'R2'),
    (r'silhouette_score\s*\(', 'silhouette_score'),
    (r'cross_val_score\s*\(', 'cross_val_score'),
    (r'model\.evaluate\s*\(', 'keras_evaluate'),
    (r'model\.score\s*\(|\.score\s*\(', 'score_method'),
    (r'model\.predict\s*\(|\.predict\s*\(', 'predict_call'),
]

ISSUE_PATTERNS = [
    # Hardcoded absolute paths
    (r'["\'][A-Za-z]:\\\\', 'HARDCODED_WIN_PATH'),
    (r'["\'][A-Za-z]:[/\\]', 'HARDCODED_WIN_PATH'),
    (r'"\/(?:home|root|Users|var|tmp|data)\/[^"]+?"', 'HARDCODED_UNIX_PATH'),
    (r"'\/(?:home|root|Users|var|tmp|data)\/[^']+?'", 'HARDCODED_UNIX_PATH'),
    (r'["\']\/kaggle\/input', 'HARDCODED_KAGGLE_PATH'),
    (r'["\']\/content\/', 'HARDCODED_COLAB_PATH'),
    # Data leakage signals
    (r'(StandardScaler|MinMaxScaler|RobustScaler|Normalizer).*\.fit\b(?!.*train)', 'POTENTIAL_LEAKAGE_SCALER_FULL_DATA'),
    (r'(LabelEncoder|OrdinalEncoder).*\.fit\b(?!.*train)', 'POTENTIAL_LEAKAGE_ENCODER_FULL_DATA'),
    # Fit before split
    (r'\.fit_transform\s*\((?:X|df|data)\b', 'POTENTIAL_LEAKAGE_FIT_BEFORE_SPLIT'),
    # Train_test_split missing or after model fit
    # Model saved outputs (pickle, h5)
    (r'joblib\.dump\s*\(|pickle\.dump\s*\(', 'SAVED_MODEL_PICKLE'),
    (r'model\.save\s*\(', 'SAVED_MODEL_H5'),
    # Missing evaluation (has fit but no score/predict)
    # Magic numbers
    (r'test_size\s*=\s*0\.[0-9]+', 'TEST_SIZE_DEFINED'),
    (r'random_state\s*=\s*\d+', 'RANDOM_STATE_DEFINED'),
    # Import without use signals
    (r'^import warnings|warnings\.filterwarnings', 'WARNINGS_SUPPRESSED'),
    # Raw experimental leftover
    (r'print\s*\(\s*["\']TODO|#\s*TODO|#\s*FIXME', 'TODO_MARKER'),
    (r'#.*hack|#.*workaround|#.*temp\b', 'HACK_COMMENT'),
    # Drive mount (colab artifacts)
    (r'drive\.mount\s*\(|from google\.colab', 'COLAB_ARTIFACT'),
    # GPU-only keras calls
    (r'\.to\s*\(\s*["\']cuda', 'GPU_ONLY_CUDA'),
    # Missing train_test_split entirely but has model fit
]

REDUNDANT_PATTERNS = [
    (r'\.head\s*\(\s*\)', 'head_call'),
    (r'\.describe\s*\(\s*\)', 'describe_call'),
    (r'\.info\s*\(\s*\)', 'info_call'),
    (r'\.isnull\s*\(\s*\)\.sum\s*\(\s*\)', 'null_check'),
    (r'sns\.|seaborn\.', 'seaborn_plot'),
    (r'plt\.|matplotlib\.', 'matplotlib_plot'),
    (r'\.value_counts\s*\(\s*\)', 'value_counts'),
]

# ─── HELPERS ─────────────────────────────────────────────────────────────────

def extract_cells(nb_path):
    """Return list of (cell_type, source_lines) from an ipynb file."""
    try:
        with open(nb_path, 'r', encoding='utf-8', errors='replace') as f:
            nb = json.load(f)
        cells = []
        for cell in nb.get('cells', []):
            ctype = cell.get('cell_type', 'code')
            src = cell.get('source', [])
            if isinstance(src, list):
                source = ''.join(src)
            else:
                source = src
            cells.append((ctype, source))
        return cells
    except Exception as e:
        return []

def scan_lines(source, patterns):
    """Return list of (line_number, matched_label) for each hit."""
    hits = []
    for i, line in enumerate(source.splitlines(), 1):
        for pat_tuple in patterns:
            pattern = pat_tuple[0]
            label = pat_tuple[1]
            if re.search(pattern, line, re.IGNORECASE):
                hits.append((i, label, line.strip()[:120]))
    return hits

def detect_leakage(source):
    """Heuristic: scaler/encoder .fit before train_test_split appears."""
    issues = []
    lines = source.splitlines()
    split_line = None
    fit_lines = []
    for i, line in enumerate(lines, 1):
        if re.search(r'train_test_split\s*\(', line):
            split_line = i
        if re.search(r'\.(fit|fit_transform)\s*\(', line):
            fit_lines.append(i)
    if split_line:
        for fl in fit_lines:
            if fl < split_line:
                ctx = lines[fl-1].strip()[:120]
                if re.search(r'Scaler|Encoder|Imputer|Vectorizer', ctx, re.IGNORECASE):
                    issues.append((fl, 'DATA_LEAKAGE_FIT_BEFORE_SPLIT', ctx))
    return issues

def detect_no_evaluation(source):
    """Flag if model is fitted but never evaluated."""
    has_fit = bool(re.search(r'\.fit\s*\(', source))
    has_eval = bool(re.search(
        r'accuracy_score|classification_report|confusion_matrix|r2_score|'
        r'mean_squared_error|mean_absolute_error|roc_auc_score|\.score\s*\(|'
        r'model\.evaluate|silhouette_score', source))
    if has_fit and not has_eval:
        return True
    return False

def detect_no_split(source):
    """Flag if model is fitted but train_test_split is absent."""
    has_fit = bool(re.search(r'\.fit\s*\(', source))
    has_split = bool(re.search(r'train_test_split\s*\(|KFold\s*\(|StratifiedKFold\s*\(|cross_val', source))
    if has_fit and not has_split:
        return True
    return False

# ─── MAIN INSPECTION LOOP ────────────────────────────────────────────────────

def inspect_project(proj_dir):
    proj_name = proj_dir.name
    result = {
        'project': proj_name,
        'notebooks_inspected': 0,
        'scripts_inspected': 0,
        'data_loading': [],
        'preprocessing': [],
        'model_training': [],
        'evaluation': [],
        'custom_ml_to_replace': [],
        'issues': [],
        'has_lazypredict': False,
        'has_pycaret': False,
        'has_pipeline': False,
        'no_train_test_split': False,
        'no_evaluation': False,
        'data_leakage': False,
        'hardcoded_paths': False,
        'colab_artifacts': False,
        'has_untitled_notebook': False,
        'has_duplicate_numbered_nb': False,
        'redundant_eda_count': 0,
        'notebooks': [],
    }

    nb_files = list(proj_dir.rglob('*.ipynb'))
    py_files = list(proj_dir.rglob('*.py'))
    result['notebooks_inspected'] = len(nb_files)
    result['scripts_inspected'] = len(py_files)

    all_source = ''

    for nb_path in nb_files:
        nb_rel = str(nb_path.relative_to(ROOT))
        if 'Untitled' in nb_path.name:
            result['has_untitled_notebook'] = True
        if re.search(r'\(1\)\.ipynb$', nb_path.name):
            result['has_duplicate_numbered_nb'] = True

        cells = extract_cells(nb_path)
        code_cells = [(i+1, src) for i,(ct,src) in enumerate(cells) if ct=='code']
        full_nb_source = '\n'.join(src for _,src in code_cells)
        all_source += '\n' + full_nb_source

        nb_info = {
            'file': nb_rel,
            'code_cells': len(code_cells),
            'data_loading': [],
            'preprocessing': [],
            'model_training': [],
            'evaluation': [],
            'issues': [],
        }

        for cell_num, src in code_cells:
            dl = scan_lines(src, DATA_LOAD_PATTERNS)
            if dl:
                for ln, lbl, ctx in dl:
                    nb_info['data_loading'].append(f'cell{cell_num}:L{ln} [{lbl}] {ctx}')
                    result['data_loading'].append(f'{nb_rel} cell{cell_num}:L{ln} [{lbl}]')

            pp = scan_lines(src, PREPROCESS_PATTERNS)
            if pp:
                for ln, lbl, ctx in pp:
                    nb_info['preprocessing'].append(f'cell{cell_num}:L{ln} [{lbl}] {ctx}')

            tr = scan_lines(src, [(p[0], p[1]) for p in TRAIN_PATTERNS])
            if tr:
                for ln, lbl, ctx in tr:
                    nb_info['model_training'].append(f'cell{cell_num}:L{ln} [{lbl}] {ctx}')
                    # Flag custom ML for replacement
                    for tp in TRAIN_PATTERNS:
                        if tp[1] == lbl and len(tp) > 2 and tp[2]:
                            if lbl not in result['custom_ml_to_replace']:
                                result['custom_ml_to_replace'].append(lbl)
                            break

            ev = scan_lines(src, EVAL_PATTERNS)
            if ev:
                for ln, lbl, ctx in ev:
                    nb_info['evaluation'].append(f'cell{cell_num}:L{ln} [{lbl}] {ctx}')

            iss = scan_lines(src, ISSUE_PATTERNS)
            if iss:
                for ln, lbl, ctx in iss:
                    nb_info['issues'].append(f'cell{cell_num}:L{ln} [{lbl}] {ctx}')
                    result['issues'].append(f'{nb_rel} cell{cell_num}:L{ln} [{lbl}]')

            # Leakage detection per cell
            leakage = detect_leakage(src)
            for ln, lbl, ctx in leakage:
                nb_info['issues'].append(f'cell{cell_num}:L{ln} [{lbl}] {ctx}')
                result['issues'].append(f'{nb_rel} cell{cell_num}:L{ln} [{lbl}]')
                result['data_leakage'] = True

        result['notebooks'].append(nb_info)
        result['preprocessing'] += [x for x in nb_info['preprocessing']]
        result['model_training'] += [x for x in nb_info['model_training']]
        result['evaluation'] += [x for x in nb_info['evaluation']]

        # Redundant EDA count
        redundant_hits = scan_lines(full_nb_source, REDUNDANT_PATTERNS)
        result['redundant_eda_count'] += len(redundant_hits)

    # Python scripts
    for py_path in py_files:
        try:
            src = py_path.read_text(encoding='utf-8', errors='replace')
            all_source += '\n' + src
        except:
            pass

    # Whole-project checks
    if all_source:
        result['has_lazypredict'] = bool(re.search(r'LazyClassifier|LazyRegressor|lazypredict', all_source, re.I))
        result['has_pycaret'] = bool(re.search(r'pycaret|from pycaret', all_source, re.I))
        result['has_pipeline'] = bool(re.search(r'Pipeline\s*\(', all_source))
        result['no_train_test_split'] = detect_no_split(all_source) if all_source.strip() else False
        result['no_evaluation'] = detect_no_evaluation(all_source) if all_source.strip() else False

        # Hardcoded paths
        if re.search(r'["\'][A-Za-z]:\\\\|["\'][A-Za-z]:[/\\]|["\']\/kaggle\/input|["\']\/content\/', all_source):
            result['hardcoded_paths'] = True
        # Colab artifacts
        if re.search(r'drive\.mount\s*\(|from google\.colab|google\.colab', all_source):
            result['colab_artifacts'] = True

    return result

# ─── RUN ALL PROJECTS ─────────────────────────────────────────────────────────

def main():
    proj_dirs = sorted([
        d for d in ROOT.iterdir()
        if d.is_dir() and d.name not in ('.git', 'audit_phase1', 'audit_phase2', 'venv', '__pycache__')
    ])

    print(f"Inspecting {len(proj_dirs)} projects...")

    all_results = []
    summary_rows = []
    issue_rows = []

    for i, proj_dir in enumerate(proj_dirs, 1):
        print(f"  [{i:03d}/{len(proj_dirs)}] {proj_dir.name[:80]}")
        r = inspect_project(proj_dir)
        all_results.append(r)

        custom_ml_str = '|'.join(r['custom_ml_to_replace']) if r['custom_ml_to_replace'] else ''
        automl_status = 'NONE'
        if r['has_pycaret'] and r['has_lazypredict']:
            automl_status = 'BOTH'
        elif r['has_pycaret']:
            automl_status = 'PYCARET_ONLY'
        elif r['has_lazypredict']:
            automl_status = 'LAZYPREDICT_ONLY'

        flags = []
        if r['hardcoded_paths']:     flags.append('HARDCODED_PATH')
        if r['data_leakage']:        flags.append('DATA_LEAKAGE')
        if r['no_train_test_split']: flags.append('NO_SPLIT')
        if r['no_evaluation']:       flags.append('NO_EVALUATION')
        if r['colab_artifacts']:     flags.append('COLAB_ARTIFACT')
        if r['has_untitled_notebook']: flags.append('UNTITLED_NB')
        if r['has_duplicate_numbered_nb']: flags.append('DUPLICATE_NB')

        summary_rows.append({
            'project': r['project'],
            'notebooks_inspected': r['notebooks_inspected'],
            'scripts_inspected': r['scripts_inspected'],
            'custom_ml_algorithms': custom_ml_str,
            'automl_status': automl_status,
            'has_pipeline': r['has_pipeline'],
            'no_train_test_split': r['no_train_test_split'],
            'no_evaluation': r['no_evaluation'],
            'data_leakage': r['data_leakage'],
            'hardcoded_paths': r['hardcoded_paths'],
            'colab_artifacts': r['colab_artifacts'],
            'redundant_eda_steps': r['redundant_eda_count'],
            'data_loading_count': len(r['data_loading']),
            'preprocessing_count': len(r['preprocessing']),
            'model_training_count': len(r['model_training']),
            'evaluation_count': len(r['evaluation']),
            'issue_count': len(r['issues']),
            'flags': '|'.join(flags) if flags else 'NONE',
        })

        for iss in r['issues']:
            issue_rows.append({'project': r['project'], 'issue': iss})

    # Write summary CSV
    summary_path = AUDIT_DIR / 'phase2_summary.csv'
    with open(summary_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        w.writeheader()
        w.writerows(summary_rows)

    # Write issues CSV
    issues_path = AUDIT_DIR / 'phase2_issues.csv'
    with open(issues_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=['project', 'issue'])
        w.writeheader()
        w.writerows(issue_rows)

    # Write per-project detailed JSON
    detail_path = AUDIT_DIR / 'phase2_detail.json'
    # Slim down for JSON output (drop raw line matches to keep size manageable)
    slim = []
    for r in all_results:
        slim.append({
            'project': r['project'],
            'notebooks_inspected': r['notebooks_inspected'],
            'custom_ml_to_replace': r['custom_ml_to_replace'],
            'has_lazypredict': r['has_lazypredict'],
            'has_pycaret': r['has_pycaret'],
            'has_pipeline': r['has_pipeline'],
            'no_train_test_split': r['no_train_test_split'],
            'no_evaluation': r['no_evaluation'],
            'data_leakage': r['data_leakage'],
            'hardcoded_paths': r['hardcoded_paths'],
            'colab_artifacts': r['colab_artifacts'],
            'redundant_eda_count': r['redundant_eda_count'],
            'issue_count': len(r['issues']),
            'data_loading_snippets': r['data_loading'][:10],
            'model_training_snippets': r['model_training'][:15],
            'evaluation_snippets': r['evaluation'][:10],
            'top_issues': r['issues'][:20],
            'notebooks': [
                {
                    'file': nb['file'],
                    'code_cells': nb['code_cells'],
                    'data_loading': nb['data_loading'][:5],
                    'preprocessing': nb['preprocessing'][:5],
                    'model_training': nb['model_training'][:10],
                    'evaluation': nb['evaluation'][:5],
                    'issues': nb['issues'][:15],
                }
                for nb in r['notebooks']
            ]
        })
    with open(detail_path, 'w', encoding='utf-8') as f:
        json.dump(slim, f, indent=2)

    # Aggregate statistics
    total_custom_ml = {}
    total_flags = {}
    projects_needing_automl = 0
    for r in summary_rows:
        for alg in r['custom_ml_algorithms'].split('|'):
            if alg:
                total_custom_ml[alg] = total_custom_ml.get(alg, 0) + 1
        for fl in r['flags'].split('|'):
            if fl != 'NONE':
                total_flags[fl] = total_flags.get(fl, 0) + 1
        if r['automl_status'] == 'NONE' and r['custom_ml_algorithms']:
            projects_needing_automl += 1

    print(f"\n{'='*60}")
    print(f"Phase 2 Complete")
    print(f"  Projects inspected : {len(all_results)}")
    print(f"  Projects needing AutoML replacement: {projects_needing_automl}")
    print(f"  Projects with data leakage: {total_flags.get('DATA_LEAKAGE',0)}")
    print(f"  Projects with no train/test split: {total_flags.get('NO_SPLIT',0)}")
    print(f"  Projects with no evaluation: {total_flags.get('NO_EVALUATION',0)}")
    print(f"  Projects with hardcoded paths: {total_flags.get('HARDCODED_PATH',0)}")
    print(f"  Projects with Colab artifacts: {total_flags.get('COLAB_ARTIFACT',0)}")
    print(f"  Top custom algorithms to replace:")
    for alg, cnt in sorted(total_custom_ml.items(), key=lambda x: -x[1])[:15]:
        print(f"    {alg}: {cnt} projects")
    print(f"\nArtifacts:")
    print(f"  {summary_path}")
    print(f"  {issues_path}")
    print(f"  {detail_path}")

if __name__ == '__main__':
    main()

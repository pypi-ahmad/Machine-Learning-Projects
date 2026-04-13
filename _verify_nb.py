"""Temporary verification script — delete after use."""
import json

path = r'E:\Github\Machine-Learning-Projects\Classification\Medical Appointment No-Show Prediction\Medical Appointment No-Show Prediction.ipynb'
with open(path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

full_src = ''
for c in nb['cells']:
    if c['cell_type'] == 'code':
        full_src += ''.join(c['source']) + '\n'

checks = {
    'matplotlib.use Agg': "matplotlib.use('Agg')" in full_src,
    'lead_time_days feature': 'lead_time_days' in full_src,
    'list(X_tr.columns) safe iteration': 'list(X_tr.columns)' in full_src,
    'LabelEncoder fit on combined vals': 'all_vals = pd.concat' in full_src,
    'No old LabelEncoder().fit_transform': 'LabelEncoder().fit_transform' not in full_src,
    'No old or/and precedence bug': "'patient' in c.lower() or" not in full_src,
    'clean drop_cols by dtype': 'select_dtypes' in full_src,
    'PR-AUC in baseline': 'average_precision_score(y_test, y_prob_b)' in full_src,
    'ROC curve plot': 'roc_curve(y_test, y_prob)' in full_src,
    'negative age handling': "df['Age'] < 0" in full_src,
}
print('Bug fix verification:')
for name, passed in checks.items():
    status = 'PASS' if passed else 'FAIL'
    print(f'  [{status}] {name}')
print()
all_ok = all(checks.values())
print('All checks passed!' if all_ok else 'SOME CHECKS FAILED!')


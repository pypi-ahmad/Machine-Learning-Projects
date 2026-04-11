"""Check remaining DA notebooks for data file issues."""
import json, re, os
from pathlib import Path

processed = {
    '2016 General Election Poll Analysis', '911 Calls Exploratory Analysis', 'Airbnb Data Analysis',
    'Bank Payment Fraud Detection', 'CLV Non-Contractual', 'CLV Online Retail', 'Coffee Quality Analysis',
    'COVID-19 Global Data Analysis', 'COVID-19 Tracking', 'Customer Lifetime Value Prediction',
    'Cybersecurity Anomaly Detection', 'Data Science Salaries Analysis', 'Drive Data Analysis',
    'FIFA 21 Data Cleaning', 'FIFA Data Analysis', 'Food Delivery Analysis', 'Heart Failure Prediction',
    'Indians Diabetes Prediction', 'Medical Insurance Cost Analysis', 'Melbourne Housing Price Analysis',
    'Mobile Price Prediction'
}

da_dir = Path('Data Analysis')
for nb_dir in sorted(da_dir.iterdir()):
    if not nb_dir.is_dir(): continue
    if nb_dir.name in processed: continue
    nb_files = list(nb_dir.glob('*.ipynb'))
    if not nb_files: continue
    for nb_file in nb_files:
        try:
            nb = json.loads(nb_file.read_text('utf-8'))
            code = '\n'.join(''.join(c.get('source', '')) for c in nb['cells'] if c['cell_type'] == 'code')
            pattern = r"pd\.read_(?:csv|excel)\([\"']([^\"']+)[\"']"
            for m in re.finditer(pattern, code):
                fname = m.group(1).split('/')[-1].split(os.sep)[-1]
                if fname.startswith('http') or fname.startswith('..'):
                    continue
                if not (nb_dir / fname).exists():
                    candidates = list(nb_dir.glob('*'))
                    data_files = [c.name for c in candidates if c.is_file() and c.suffix in ('.csv', '.xlsx', '.json')]
                    print(f'{nb_dir.name}: MISSING {fname} -> dir contains: {data_files[:3]}')
        except Exception as e:
            print(f'{nb_dir.name}: error: {e}')
print('Scan complete')

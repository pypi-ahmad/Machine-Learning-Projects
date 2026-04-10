"""Verify critical OpenML datasets and build replacement table."""
from sklearn.datasets import fetch_openml

# Key OpenML datasets to verify
tests = {
    42178: "telecom churn",
    1461: "bank marketing",
    53: "heart statlog",
    37: "diabetes pima",
    1597: "credit card fraud",
    1510: "breast cancer",
    31: "credit german",
    40: "sonar",
    41: "glass",
    287: "wine quality",
    242: "energy efficiency",
    1462: "banknote",
    1511: "wholesale",
    1523: "turkiye student",
    1590: "adult",
    44126: "mobile price",
    42352: "student performance",
    44129: "higgs",
    42570: "mercedes",
    42712: "bike sharing",
    4353: "concrete",
    43463: "insurance",
}

for data_id, desc in tests.items():
    try:
        d = fetch_openml(data_id=data_id, as_frame=True, parser="auto")
        cols = list(d.frame.columns)
        target_name = d.target.name if hasattr(d.target, 'name') else 'N/A'
        print(f"OK  {data_id:>6} ({desc:25s}): target='{target_name}' cols={cols[:5]}...{cols[-3:]}")
    except Exception as e:
        print(f"ERR {data_id:>6} ({desc:25s}): {e}")

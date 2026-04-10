"""Search OpenML for correct dataset IDs to use as replacements."""
from sklearn.datasets import fetch_openml
import warnings
warnings.filterwarnings("ignore")

searches = [
    ("churn", [42178, 40588]),
    ("bank marketing", [1461, 1558]),
    ("heart disease", [53, 188]),
    ("water quality", [1448, 40685]),
    ("weather rain", [44064, 43566]),
    ("hr analytics", [43603, 1030]),
    ("credit card fraud", [1597, 44307]),
    ("house prices", [537, 531]),
    ("loan", [31, 1040]),
]
for name, ids in searches:
    print(f"\n--- {name} ---")
    for did in ids:
        try:
            d = fetch_openml(data_id=did, as_frame=True, parser="auto")
            dname = d.details.get("name", "?")
            ncols = len(d.frame.columns)
            cols = list(d.frame.columns)
            tgt = cols[-1]
            print(f"  {did}: {dname} ({len(d.frame)} rows, {ncols} cols, target='{tgt}')")
            print(f"       first cols: {cols[:6]}")
        except Exception as e:
            print(f"  {did}: FAIL - {type(e).__name__}")

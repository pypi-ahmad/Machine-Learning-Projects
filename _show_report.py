import json

r = json.load(open('audit_phase6/phase6_stress_report.json'))
for proj in r['projects']:
    print(f"P{proj['project']:03d}:")
    for run in proj['runs']:
        print(f"  Run: {run['run_type']}  status={run['status']}  time={run['total_time_s']}s")
        print(f"    cells: ok={run['cells_ok']} err={run['cells_error']} skip={run['cells_skipped']}")
        for f in run['failures']:
            tag = ""
            if f['is_pycaret']:
                tag = " [PYCARET]"
            elif f['is_lazypredict']:
                tag = " [LAZY]"
            elif f['is_standardized']:
                tag = " [STD]"
            print(f"    cell[{f['index']:2d}]{tag}: {f['error_message'][:120]}")
    for fc in proj.get('failures', []):
        print(f"  RCA: cell[{fc['cell_index']}] category={fc['category']}")

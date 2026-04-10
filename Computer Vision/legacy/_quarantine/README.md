# Legacy Quarantine

Files moved here during the **WS1 Pre-Push Hardening** pass (2026-03-03).
They are no longer referenced by active code but kept for historical reference.

## Quarantined Scripts (`scripts/`)

| File | Reason |
|------|--------|
| `_gen_project_readmes.py` | Superseded by `scripts/_gen_final_docs.py` (generates root + per-project + overview) |
| `smoke_test.py` | Phase 1B smoke test, superseded by `scripts/smoke_3b3.py` (9 checks) |
| `_audit.py` | One-off Task 0 inventory script — no longer needed |
| `_extend_dataset_configs.py` | One-off scaffolding — dataset configs are now stable |
| `bootstrap_dataset_configs.py` | One-off scaffolding — dataset configs are now stable |

## Quarantined Phase Reports (`phase_reports/`)

| File | Reason |
|------|--------|
| `PHASE0_ANALYSIS.md` | Internal migration analysis — not intended for public repo |
| `PHASE2_REPORT.md` | Internal Phase 2 report |
| `PHASE3_MODERNIZATION.md` | Internal Phase 3 modernization plan |
| `PHASE3B_3_REPORT.md` | Internal report (documents yolo11→yolo26 migration) |
| `PHASE3B_4_REPORT.md` | Internal report (dataset config + downloader) |
| `PHASE3B_5_REPORT.md` | Internal report (docs, gitignore, CI hooks) |

> **Note:** Phase reports contain historical references to `yolo11` / `yolov8`
> in their migration narratives. This is intentional — they document the
> migration itself, not current code.

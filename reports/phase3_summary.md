# Phase 3.0 — Quality & Stability Upgrades

Generated: 2026-03-04 03:56 UTC

## Changes Applied

1. **Class weight clamping** — weights clamped to [0.5, 5.0], normalised mean=1
2. **Gradient clipping** — `max_grad_norm=1.0`
3. **Label smoothing** — `label_smoothing=0.05`
4. **Cosine LR schedule** — `lr_scheduler_type=cosine`
5. **Focal loss** — `(1-p_t)^gamma` for many-class problems (gamma=1.5-2.0)
6. **Multilabel fix** — BCEWithLogitsLoss + clamped pos_weight + per-label threshold tuning
7. **Per-project YAML configs** — `config/projects/<slug>.yaml`

## Overview

| Metric | Value |
|--------|-------|
| Re-trained projects | 10 |
| Succeeded | 0 |
| Failed | 10 |
| Total time | 164s (2.7 min) |

## Before / After Comparison

| # | Project | Action | Phase 2 | Phase 3 | Delta | Status |
|---|---------|--------|---------|---------|-------|--------|
| 1 | trip-advisor-hotel-reviews | classify | 0.2689 | **FAILED** | -0.2689 | regressed |
| 2 | cyberbullying-classification | classify | 0.0498 | **FAILED** | -0.0498 | regressed |
| 3 | e-commerce-product-classification | classify | 0.2131 | **FAILED** | -0.2131 | regressed |
| 4 | economic-news-articles | classify | 0.0392 | **FAILED** | -0.0392 | regressed |
| 5 | fake-news-detection | classify | 0.3588 | **FAILED** | -0.3588 | regressed |
| 6 | news-headline-classification | classify | 0.0476 | **FAILED** | -0.0476 | regressed |
| 7 | paper-subject-prediction | classify | 0.3826 | **FAILED** | -0.3826 | regressed |
| 8 | review-classification | classify | 0.3333 | **FAILED** | -0.3333 | regressed |
| 9 | twitter-sentiment-analysis | classify | 0.1230 | **FAILED** | -0.1230 | regressed |
| 10 | toxic-comment-classification | classify_multilabel | 0.0000 | **FAILED** | +0.0000 | same |

**0/10 improved**

## Per-Project Details

### trip-advisor-hotel-reviews

- **Status:** FAILED
- **Action:** classify
- **Model:** `?`
- **Time:** 22.6s
- **Error:** `Attempting to unscale FP16 gradients.`

### cyberbullying-classification

- **Status:** FAILED
- **Action:** classify
- **Model:** `?`
- **Time:** 10.6s
- **Error:** `Attempting to unscale FP16 gradients.`

### e-commerce-product-classification

- **Status:** FAILED
- **Action:** classify
- **Model:** `?`
- **Time:** 14.7s
- **Error:** `Attempting to unscale FP16 gradients.`

### economic-news-articles

- **Status:** FAILED
- **Action:** classify
- **Model:** `?`
- **Time:** 8.2s
- **Error:** `Attempting to unscale FP16 gradients.`

### fake-news-detection

- **Status:** FAILED
- **Action:** classify
- **Model:** `?`
- **Time:** 25.7s
- **Error:** `Attempting to unscale FP16 gradients.`

### news-headline-classification

- **Status:** FAILED
- **Action:** classify
- **Model:** `?`
- **Time:** 22.7s
- **Error:** `Attempting to unscale FP16 gradients.`

### paper-subject-prediction

- **Status:** FAILED
- **Action:** classify
- **Model:** `?`
- **Time:** 16.7s
- **Error:** `Attempting to unscale FP16 gradients.`

### review-classification

- **Status:** FAILED
- **Action:** classify
- **Model:** `?`
- **Time:** 7.2s
- **Error:** `Attempting to unscale FP16 gradients.`

### twitter-sentiment-analysis

- **Status:** FAILED
- **Action:** classify
- **Model:** `?`
- **Time:** 11.2s
- **Error:** `Attempting to unscale FP16 gradients.`

### toxic-comment-classification

- **Status:** FAILED
- **Action:** classify_multilabel
- **Model:** `?`
- **Time:** 24.1s
- **Error:** `Attempting to unscale FP16 gradients.`


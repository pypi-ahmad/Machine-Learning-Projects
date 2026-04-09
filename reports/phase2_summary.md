# Phase 2.1 -- Training Results Summary

Generated: 2026-03-04 00:29 UTC

## Overview

| Metric | Value |
|--------|-------|
| Total projects | 21 |
| Succeeded | 21 |
| Failed | 0 |
| Total time | 48165s (802.7 min) |

## Results Table

| # | Project | Action | Model | Mode | Size | Main Metric | Time |
|---|---------|--------|-------|------|------|-------------|------|
| 1 | e-commerce-clothing-reviews | classify | deberta-v3-base | full | 7225 | F1w=0.7343 | 1477s |
| 2 | trip-advisor-hotel-reviews | classify | deberta-v3-base | full | 7225 | F1w=0.2689 | 1866s |
| 3 | cyberbullying-classification | classify | deberta-v3-base | full | 7225 | F1w=0.0498 | 3058s |
| 4 | e-commerce-product-classification | classify | deberta-v3-base | full | 7225 | F1w=0.2131 | 1537s |
| 5 | economic-news-articles | classify | deberta-v3-base | full | 1025 | F1w=0.0392 | 424s |
| 6 | fake-news-detection | classify | deberta-v3-base | full | 7225 | F1w=0.3588 | 2046s |
| 7 | news-headline-classification | classify | deberta-v3-base | full | 7225 | F1w=0.0476 | 1015s |
| 8 | paper-subject-prediction | classify | deberta-v3-base | full | 7225 | F1w=0.3826 | 945s |
| 9 | review-classification | classify | deberta-v3-base | full | 2167 | F1w=0.3333 | 470s |
| 10 | spam-message-detection | classify | deberta-v3-base | full | 4025 | F1w=0.8039 | 940s |
| 11 | twitter-sentiment-analysis | classify | deberta-v3-base | full | 7225 | F1w=0.1230 | 1695s |
| 12 | toxic-comment-classification | classify_multilabel | deberta-v3-base | full | 7225 | F1mi=0.0000 | 2003s |
| 13 | world-war-i-letters | embed_cluster | all-MiniLM-L6-v2 | inference | 60 | Sil=0.9694 | 0s |
| 14 | kaggle-survey-questions-clustering | embed_cluster | all-MiniLM-L6-v2 | inference | 35 | Sil=0.1243 | 0s |
| 15 | medium-articles-clustering | embed_cluster | all-MiniLM-L6-v2 | inference | 337 | Sil=0.0703 | 0s |
| 16 | newsgroups-posts-clustering | embed_cluster | all-MiniLM-L6-v2 | inference | 628 | Sil=0.0299 | 0s |
| 17 | stories-clustering | embed_cluster | all-MiniLM-L6-v2 | inference | 1000 | Sil=0.0693 | 0s |
| 18 | bbc-articles-summarization | summarize | bart-large-cnn | LoRA | 1801 | R-L=0.3283 | 1566s |
| 19 | english-to-french-translation | translate | nllb-200-distilled-600M | LoRA | 27075 | BLEU=36.7 | 29122s |
| 20 | automated-image-captioning | caption | blip-image-captioning-large | inference | 200 | R-L=0.4755 | 0s |
| 21 | name-generate-from-languages | char_rnn | CharRNN (LSTM) | full | 20074 | VL=2.3473 | 0s |

**21 succeeded, 0 failed, total 48165s**

## Per-Project Details

### e-commerce-clothing-reviews

- **Status:** OK
- **Action:** classify
- **Model:** `microsoft/deberta-v3-base`
- **Training mode:** full
- **Dataset size:** 7225
- **Time:** 1476.9s
- **Metrics:** `{"accuracy": 0.8166666666666667, "f1_weighted": 0.7342507645259939, "f1_macro": 0.44954128440366975, "f1_micro": 0.8166666666666667}`
- **Val metrics:** `{"loss": 0.7102319002151489, "accuracy": 0.8164705882352942, "f1": 0.7339774459006401, "f1_macro": 0.4494818652849741, "runtime": 24.7392, "samples_per_second": 51.538, "steps_per_second": 12.895, "ep`
- **Outputs:** `outputs/e-commerce-clothing-reviews/`

### trip-advisor-hotel-reviews

- **Status:** OK
- **Action:** classify
- **Model:** `microsoft/deberta-v3-base`
- **Training mode:** full
- **Dataset size:** 7225
- **Time:** 1865.8s
- **Metrics:** `{"accuracy": 0.44, "f1_weighted": 0.2688888888888889, "f1_macro": 0.12222222222222223, "f1_micro": 0.44}`
- **Val metrics:** `{"loss": 1.6051108837127686, "accuracy": 0.44, "f1": 0.2688888888888889, "f1_macro": 0.12222222222222223, "runtime": 21.6753, "samples_per_second": 58.823, "steps_per_second": 14.717, "epoch": 2.0}`
- **Outputs:** `outputs/trip-advisor-hotel-reviews/`

### cyberbullying-classification

- **Status:** OK
- **Action:** classify
- **Model:** `microsoft/deberta-v3-base`
- **Training mode:** full
- **Dataset size:** 7225
- **Time:** 3058.1s
- **Metrics:** `{"accuracy": 0.17066666666666666, "f1_weighted": 0.049761579347000755, "f1_macro": 0.048595292331055424, "f1_micro": 0.17066666666666666}`
- **Val metrics:** `{"loss": 1.791668176651001, "accuracy": 0.17098039215686275, "f1": 0.049931313449693336, "f1_macro": 0.04867157847733869, "runtime": 14.9332, "samples_per_second": 85.38, "steps_per_second": 21.362, "`
- **Outputs:** `outputs/cyberbullying-classification/`

### e-commerce-product-classification

- **Status:** OK
- **Action:** classify
- **Model:** `microsoft/deberta-v3-base`
- **Training mode:** full
- **Dataset size:** 7225
- **Time:** 1537.0s
- **Metrics:** `{"accuracy": 0.384, "f1_weighted": 0.2130867052023121, "f1_macro": 0.13872832369942195, "f1_micro": 0.384}`
- **Val metrics:** `{"loss": 1.3492696285247803, "accuracy": 0.3835294117647059, "f1": 0.21263705482192874, "f1_macro": 0.13860544217687074, "runtime": 26.9349, "samples_per_second": 47.336, "steps_per_second": 11.843, "`
- **Outputs:** `outputs/e-commerce-product-classification/`

### economic-news-articles

- **Status:** OK
- **Action:** classify
- **Model:** `microsoft/deberta-v3-base`
- **Training mode:** full
- **Dataset size:** 1025
- **Time:** 424.0s
- **Metrics:** `{"accuracy": 0.15023474178403756, "f1_weighted": 0.039244993772156754, "f1_macro": 0.03731778425655977, "f1_micro": 0.15023474178403756}`
- **Val metrics:** `{"loss": 443.23583984375, "accuracy": 0.14835164835164835, "f1": 0.03833009096166991, "f1_macro": 0.036910457963089546, "runtime": 2.8464, "samples_per_second": 63.939, "steps_per_second": 16.161, "ep`
- **Outputs:** `outputs/economic-news-articles/`

### fake-news-detection

- **Status:** OK
- **Action:** classify
- **Model:** `microsoft/deberta-v3-base`
- **Training mode:** full
- **Dataset size:** 7225
- **Time:** 2046.4s
- **Metrics:** `{"accuracy": 0.5226666666666666, "f1_weighted": 0.35881844716870986, "f1_macro": 0.3432574430823117, "f1_micro": 0.5226666666666666}`
- **Val metrics:** `{"loss": 0.6921529173851013, "accuracy": 0.5223529411764706, "f1": 0.358461678334394, "f1_macro": 0.34312210200927357, "runtime": 25.1273, "samples_per_second": 50.742, "steps_per_second": 12.695, "ep`
- **Outputs:** `outputs/fake-news-detection/`

### news-headline-classification

- **Status:** OK
- **Action:** classify
- **Model:** `microsoft/deberta-v3-base`
- **Training mode:** full
- **Dataset size:** 7225
- **Time:** 1015.5s
- **Metrics:** `{"accuracy": 0.16666666666666666, "f1_weighted": 0.04761904761904762, "f1_macro": 0.006802721088435374, "f1_micro": 0.16666666666666666}`
- **Val metrics:** `{"loss": 29.079240798950195, "accuracy": 0.16705882352941176, "f1": 0.04782732447817837, "f1_macro": 0.006816436251920123, "runtime": 15.5989, "samples_per_second": 81.737, "steps_per_second": 20.45, `
- **Outputs:** `outputs/news-headline-classification/`

### paper-subject-prediction

- **Status:** OK
- **Action:** classify
- **Model:** `microsoft/deberta-v3-base`
- **Training mode:** full
- **Dataset size:** 7225
- **Time:** 945.1s
- **Metrics:** `{"accuracy": 0.5433333333333333, "f1_weighted": 0.3825629949604032, "f1_macro": 0.23470122390208784, "f1_micro": 0.5433333333333333}`
- **Val metrics:** `{"loss": 1.0349103212356567, "accuracy": 0.5435294117647059, "f1": 0.3827905308464849, "f1_macro": 0.2347560975609756, "runtime": 16.2556, "samples_per_second": 78.434, "steps_per_second": 19.624, "ep`
- **Outputs:** `outputs/paper-subject-prediction/`

### review-classification

- **Status:** OK
- **Action:** classify
- **Model:** `microsoft/deberta-v3-base`
- **Training mode:** full
- **Dataset size:** 2167
- **Time:** 469.5s
- **Metrics:** `{"accuracy": 0.5, "f1_weighted": 0.3333333333333333, "f1_macro": 0.3333333333333333, "f1_micro": 0.5}`
- **Val metrics:** `{"loss": 0.694200336933136, "accuracy": 0.49869451697127937, "f1": 0.33188380746172247, "f1_macro": 0.3327526132404181, "runtime": 4.7399, "samples_per_second": 80.803, "steps_per_second": 20.254, "ep`
- **Outputs:** `outputs/review-classification/`

### spam-message-detection

- **Status:** OK
- **Action:** classify
- **Model:** `microsoft/deberta-v3-base`
- **Training mode:** full
- **Dataset size:** 4025
- **Time:** 940.0s
- **Metrics:** `{"accuracy": 0.8660287081339713, "f1_weighted": 0.8038522880628144, "f1_macro": 0.4641025641025641, "f1_micro": 0.8660287081339713}`
- **Val metrics:** `{"loss": 0.6650784611701965, "accuracy": 0.8663853727144867, "f1": 0.8043607981795385, "f1_macro": 0.4642049736247174, "runtime": 14.6201, "samples_per_second": 48.632, "steps_per_second": 12.175, "ep`
- **Outputs:** `outputs/spam-message-detection/`

### twitter-sentiment-analysis

- **Status:** OK
- **Action:** classify
- **Model:** `microsoft/deberta-v3-base`
- **Training mode:** full
- **Dataset size:** 7225
- **Time:** 1694.8s
- **Metrics:** `{"accuracy": 0.2806666666666667, "f1_weighted": 0.1230199548846087, "f1_macro": 0.08766267568974492, "f1_micro": 0.2806666666666667}`
- **Val metrics:** `{"loss": 1.572443962097168, "accuracy": 0.2807843137254902, "f1": 0.12311179952691426, "f1_macro": 0.08769136558481323, "runtime": 26.9039, "samples_per_second": 47.391, "steps_per_second": 11.857, "e`
- **Outputs:** `outputs/twitter-sentiment-analysis/`

### toxic-comment-classification

- **Status:** OK
- **Action:** classify_multilabel
- **Model:** `microsoft/deberta-v3-base`
- **Training mode:** full
- **Dataset size:** 7225
- **Time:** 2002.9s
- **Metrics:** `{"f1_micro": 0.0, "f1_macro": 0.0, "f1_weighted": 0.0}`
- **Val metrics:** `{"loss": 31.213857650756836, "f1_micro": 0.0, "f1_macro": 0.0, "f1_weighted": 0.0, "f1_samples": 0.0, "runtime": 25.4272, "samples_per_second": 50.143, "steps_per_second": 12.546, "epoch": 2.0}`
- **Outputs:** `outputs/toxic-comment-classification/`

### world-war-i-letters

- **Status:** OK
- **Action:** embed_cluster
- **Model:** `all-MiniLM-L6-v2`
- **Training mode:** inference
- **Dataset size:** 60
- **Time:** 0.0s
- **Metrics:** `{"method": "hdbscan", "n_clusters": 2, "n_noise": 0, "silhouette": 0.9693729281425476, "calinski_harabasz": 918.1208973020053}`
- **Outputs:** `outputs/world-war-i-letters/`

### kaggle-survey-questions-clustering

- **Status:** OK
- **Action:** embed_cluster
- **Model:** `all-MiniLM-L6-v2`
- **Training mode:** inference
- **Dataset size:** 35
- **Time:** 0.0s
- **Metrics:** `{"method": "kmeans", "n_clusters": 5, "n_noise": 0, "silhouette": 0.12426846474409103, "calinski_harabasz": 3.9084130102577084}`
- **Outputs:** `outputs/kaggle-survey-questions-clustering/`

### medium-articles-clustering

- **Status:** OK
- **Action:** embed_cluster
- **Model:** `all-MiniLM-L6-v2`
- **Training mode:** inference
- **Dataset size:** 337
- **Time:** 0.0s
- **Metrics:** `{"method": "kmeans", "n_clusters": 5, "n_noise": 0, "silhouette": 0.07026674598455429, "calinski_harabasz": 16.88853355658109}`
- **Outputs:** `outputs/medium-articles-clustering/`

### newsgroups-posts-clustering

- **Status:** OK
- **Action:** embed_cluster
- **Model:** `all-MiniLM-L6-v2`
- **Training mode:** inference
- **Dataset size:** 628
- **Time:** 0.0s
- **Metrics:** `{"method": "hdbscan", "n_clusters": 12, "n_noise": 90, "silhouette": 0.029858820140361786, "calinski_harabasz": 130371725135458.4}`
- **Outputs:** `outputs/newsgroups-posts-clustering/`

### stories-clustering

- **Status:** OK
- **Action:** embed_cluster
- **Model:** `all-MiniLM-L6-v2`
- **Training mode:** inference
- **Dataset size:** 1000
- **Time:** 0.0s
- **Metrics:** `{"method": "hdbscan", "n_clusters": 2, "n_noise": 299, "silhouette": 0.06927366554737091, "calinski_harabasz": 11.502719239283097}`
- **Outputs:** `outputs/stories-clustering/`

### bbc-articles-summarization

- **Status:** OK
- **Action:** summarize
- **Model:** `facebook/bart-large-cnn`
- **Training mode:** LoRA
- **Dataset size:** 1801
- **Time:** 1566.4s
- **Metrics:** `{"rouge1": 0.4643006485926036, "rouge2": 0.3061719096625183, "rougeL": 0.32831576191042966}`
- **Val metrics:** `{"loss": 0.5176393985748291, "rouge1": 0.4634527599320004, "rouge2": 0.2962159455591071, "rougeL": 0.3230504237289995, "runtime": 484.2783, "samples_per_second": 0.415, "steps_per_second": 0.209, "epo`
- **Outputs:** `outputs/bbc-articles-summarization/`

### english-to-french-translation

- **Status:** OK
- **Action:** translate
- **Model:** `facebook/nllb-200-distilled-600M`
- **Training mode:** LoRA
- **Dataset size:** 27075
- **Time:** 29122.4s
- **Metrics:** `{"bleu": 36.73834080059138, "chrf": 58.262638201737246}`
- **Val metrics:** `{"loss": 0.7164146304130554, "bleu": 36.51581023422148, "chrf": 58.178639612079174, "runtime": 489.1142, "samples_per_second": 2.913, "steps_per_second": 1.458, "epoch": 2.0}`
- **Outputs:** `outputs/english-to-french-translation/`

### automated-image-captioning

- **Status:** OK
- **Action:** caption
- **Model:** `Salesforce/blip-image-captioning-large`
- **Training mode:** inference
- **Dataset size:** 200
- **Time:** 0.0s
- **Metrics:** `{"rougeL": 0.4754979656409096, "bleu": 0.4967896197584665}`
- **Outputs:** `outputs/automated-image-captioning/`

### name-generate-from-languages

- **Status:** OK
- **Action:** char_rnn
- **Model:** `CharRNN (LSTM)`
- **Training mode:** full
- **Dataset size:** 20074
- **Time:** 0.0s
- **Metrics:** `{"best_val_loss": 2.3472728679180146}`
- **Outputs:** `outputs/name-generate-from-languages/`


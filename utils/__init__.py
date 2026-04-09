"""Shared utilities for NLP Projects workspace.

Available modules (Phase 0-1):
    utils.logger              - Structured logging
    utils.seed                - Reproducibility seed helper
    utils.data_io             - CSV / JSON / TXT loaders + savers
    utils.dataset_downloader  - Kaggle dataset downloader
    utils.dataset_finder      - Project registry, data discovery, auto-download
    utils.nlp_preprocess      - Text normalization, TF-IDF, splitting
    utils.metrics             - Universal metrics (classification, regression, etc.)
    utils.baselines           - Baseline models per project type

Phase 2 modules:
    utils.training_common         - Device, seed, output dirs, class weights, CharRNN
    utils.train_text_classifier   - DeBERTa single-label / multi-label classification
    utils.train_summarizer        - BART-large-CNN + LoRA summarization
    utils.train_translator        - NLLB-200 + LoRA EN->FR translation
    utils.embeddings_and_topics   - Sentence-transformer embeddings, clustering, topics
    utils.captioning              - BLIP image captioning (inference + evaluation)
"""

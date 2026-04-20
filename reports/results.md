# Results: Machine-Generated Text Detection

## Validation results

| Rank | Experiment | Encoder | Training style | Train size | Accuracy | F1 | Machine precision | Machine recall |
|---:|---|---|---|---:|---:|---:|---:|---:|
| 1 | `mdeberta_processed_article` | mDeBERTa-v3 base | article-style | 204,274 | 0.7627 | 0.7757 | 0.7354 | 0.8207 |
| 2 | `xlm_roberta_processed_article` | XLM-RoBERTa base | article-style | 204,274 | 0.6987 | 0.7355 | 0.6554 | 0.8380 |
| 3 | `xlm_roberta_original_default` | XLM-RoBERTa base | default | 12,000 | 0.6290 | 0.7038 | 0.5857 | 0.8813 |
| 4 | `minilm_original_default` | MiniLM multilingual | default | 12,000 | 0.6283 | 0.6853 | 0.5942 | 0.8093 |
| 5 | `mdeberta_original_default` | mDeBERTa-v3 base | default | 12,000 | 0.6073 | 0.6749 | 0.5758 | 0.8153 |
| 6 | `minilm_processed_article` | MiniLM multilingual | article-style | 204,274 | 0.5320 | 0.3377 | 0.5774 | 0.2387 |

## Conclusion

The best configuration was `mdeberta_processed_article`.

Key observations:

- Long-text splitting and preprocessing improved the larger transformer encoders.
- mDeBERTa-v3 produced the best F1 and the most balanced classification quality.
- XLM-RoBERTa achieved high machine recall but lower precision.
- MiniLM was computationally cheaper but less robust under the processed training setup.

# Seed Stability Summary

Method summarized: `ensemble_lgbm_w0p500_per_category`.

## Mean ± Std

- `val_f1`: 0.3781 ± 0.0010 (range 0.3772-0.3792)
- `val_pr_auc`: 0.3189 ± 0.0015 (range 0.3177-0.3205)
- `test_f1`: 0.3817 ± 0.0021 (range 0.3798-0.3840)
- `test_precision`: 0.2985 ± 0.0024 (range 0.2958-0.3003)
- `test_recall`: 0.5296 ± 0.0151 (range 0.5194-0.5470)
- `test_pr_auc`: 0.3313 ± 0.0025 (range 0.3285-0.3328)
- `test_roc_auc`: 0.7678 ± 0.0007 (range 0.7670-0.7682)
- `test_balanced_accuracy`: 0.6738 ± 0.0040 (range 0.6709-0.6783)

## By Seed

| seed | method_id | model_label | threshold_mode | val_f1 | val_pr_auc | test_f1 | test_precision | test_recall | test_pr_auc | test_roc_auc | test_balanced_accuracy | source_dir |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 7 | ensemble_lgbm_w0p500_per_category | Ensemble LGBM(0.50) + XGB(0.50) | per_category | 0.3772 | 0.3177 | 0.3814 | 0.3003 | 0.5225 | 0.3328 | 0.7682 | 0.6723 | data\processed\model_results\seed_stability_onehot\seed_007 |
| 42 | ensemble_lgbm_w0p500_per_category | Ensemble LGBM(0.50) + XGB(0.50) | per_category | 0.3792 | 0.3205 | 0.3840 | 0.2958 | 0.5470 | 0.3326 | 0.7682 | 0.6783 | data\processed\model_results\seed_stability_onehot\seed_042 |
| 123 | ensemble_lgbm_w0p500_per_category | Ensemble LGBM(0.50) + XGB(0.50) | per_category | 0.3778 | 0.3183 | 0.3798 | 0.2993 | 0.5194 | 0.3285 | 0.7670 | 0.6709 | data\processed\model_results\seed_stability_onehot\seed_123 |

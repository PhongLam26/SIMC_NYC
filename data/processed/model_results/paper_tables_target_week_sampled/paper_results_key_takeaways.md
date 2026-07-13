# Key takeaways for paper writing

## Most important claims to use

1. **Final model:** Ensemble LGBM(0.50) + XGB(0.50) with per_category thresholding is the selected final method because it ranks first by validation F1.
2. **Future-test performance:** test F1 = 0.3840, precision = 0.2958, recall = 0.5470, PR-AUC = 0.3326.
3. **Semantics-aware decision layer:** category-specific thresholds improve the validation-ranked decision strategy and fit the paper's semantic service-demand framing.
4. **Modeling protocol:** all thresholds are selected on the validation period and evaluated on the held-out future test period.
5. **Imbalance-aware metrics:** report F1 and PR-AUC prominently; raw accuracy should be secondary.

## Plain-language interpretation

On the future test period, the selected model has precision 0.2958 and recall 0.5470. This means the system catches about half of true abnormal demand increases, while roughly three out of ten alerts correspond to true abnormal increases. This is more informative than saying only that the model has high accuracy, because the abnormal class is relatively rare.

## Strongest categories by test F1

- noise: F1=0.4502, precision=0.3521, recall=0.6243
- housing: F1=0.4115, precision=0.3222, recall=0.5691
- public_safety: F1=0.4068, precision=0.3165, recall=0.5693

## Weakest categories by test F1

- sanitation: F1=0.3216, precision=0.2600, recall=0.4215
- infrastructure: F1=0.3465, precision=0.2635, recall=0.5056
- environment: F1=0.3631, precision=0.2759, recall=0.5311

## What not to overclaim

- Do not claim that 311 requests represent all real urban problems; they represent reported service demand.
- Do not claim SHAP/feature importance proves causality.
- Do not claim per-category thresholds are universally optimal; they are validation-selected for this forecasting setup.
- Do not use random split results as the main claim if chronological split results are available.

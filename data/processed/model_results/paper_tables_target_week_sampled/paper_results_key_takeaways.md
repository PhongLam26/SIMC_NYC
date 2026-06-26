# Key takeaways for paper writing

## Most important claims to use

1. **Final model:** Ensemble LGBM(0.50) + XGB(0.50) with per_category thresholding is the selected final method because it ranks first by validation F1.
2. **Future-test performance:** test F1 = 0.3800, precision = 0.3012, recall = 0.5146, PR-AUC = 0.3294.
3. **Semantics-aware decision layer:** category-specific thresholds improve the validation-ranked decision strategy and fit the paper's semantic service-demand framing.
4. **Modeling protocol:** all thresholds are selected on the validation period and evaluated on the held-out future test period.
5. **Imbalance-aware metrics:** report F1 and PR-AUC prominently; raw accuracy should be secondary.

## Plain-language interpretation

On the future test period, the selected model has precision 0.3012 and recall 0.5146. This means the system catches about half of true abnormal demand increases, while roughly three out of ten alerts correspond to true abnormal increases. This is more informative than saying only that the model has high accuracy, because the abnormal class is relatively rare.

## Strongest categories by test F1

- noise: F1=0.4423, precision=0.3446, recall=0.6173
- housing: F1=0.4049, precision=0.3476, recall=0.4849
- public_safety: F1=0.3968, precision=0.3157, recall=0.5339

## Weakest categories by test F1

- sanitation: F1=0.3194, precision=0.2625, recall=0.4076
- infrastructure: F1=0.3495, precision=0.2764, recall=0.4753
- environment: F1=0.3585, precision=0.2665, recall=0.5477

## What not to overclaim

- Do not claim that 311 requests represent all real urban problems; they represent reported service demand.
- Do not claim SHAP/feature importance proves causality.
- Do not claim per-category thresholds are universally optimal; they are validation-selected for this forecasting setup.
- Do not use random split results as the main claim if chronological split results are available.

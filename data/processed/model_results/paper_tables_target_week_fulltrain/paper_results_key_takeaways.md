# Key takeaways for paper writing

## Most important claims to use

1. **Final model:** Ensemble LGBM(0.50) + XGB(0.50) with per_category thresholding is the selected final method because it ranks first by validation F1.
2. **Future-test performance:** test F1 = 0.3839, precision = 0.3071, recall = 0.5117, PR-AUC = 0.3365.
3. **Semantics-aware decision layer:** category-specific thresholds improve the validation-ranked decision strategy and fit the paper's semantic service-demand framing.
4. **Modeling protocol:** all thresholds are selected on the validation period and evaluated on the held-out future test period.
5. **Imbalance-aware metrics:** report F1 and PR-AUC prominently; raw accuracy should be secondary.

## Plain-language interpretation

On the future test period, the selected model has precision 0.3071 and recall 0.5117. This means the system catches about half of true abnormal demand increases, while roughly three out of ten alerts correspond to true abnormal increases. This is more informative than saying only that the model has high accuracy, because the abnormal class is relatively rare.

## Strongest categories by test F1

- noise: F1=0.4578, precision=0.3880, recall=0.5583
- public_safety: F1=0.4097, precision=0.3166, recall=0.5803
- housing: F1=0.4025, precision=0.3396, recall=0.4941

## Weakest categories by test F1

- sanitation: F1=0.3165, precision=0.2685, recall=0.3852
- infrastructure: F1=0.3484, precision=0.2803, recall=0.4600
- environment: F1=0.3633, precision=0.2704, recall=0.5532

## What not to overclaim

- Do not claim that 311 requests represent all real urban problems; they represent reported service demand.
- Do not claim SHAP/feature importance proves causality.
- Do not claim per-category thresholds are universally optimal; they are validation-selected for this forecasting setup.
- Do not use random split results as the main claim if chronological split results are available.

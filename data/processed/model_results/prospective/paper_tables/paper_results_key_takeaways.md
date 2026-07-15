# Key takeaways for paper writing

## Most important claims to use

1. **Final strategy:** Ensemble LGBM(0.50) + XGB(0.50) with per_category thresholding is the validation-selected category-aware operating strategy.
2. **Future-test performance:** test F1 = 0.3802, precision = 0.2933, recall = 0.5404, PR-AUC = 0.3310.
3. **Category-aware decision layer:** category-specific thresholds fit the paper's service-domain framing and are selected only on validation.
4. **Modeling protocol:** all thresholds are selected on the validation period and evaluated on the held-out future test period.
5. **Imbalance-aware metrics:** report F1 and PR-AUC prominently; raw accuracy should be secondary.

## Plain-language interpretation

On the future test period, the selected model has precision 0.2933 and recall 0.5404. This means the system catches about half of true abnormal demand increases, while roughly three out of ten alerts correspond to true abnormal increases. This is more informative than saying only that the model has high accuracy, because the abnormal class is relatively rare.

## Strongest categories by test F1

- noise: F1=0.4555, precision=0.3739, recall=0.5826
- public_safety: F1=0.4058, precision=0.3000, recall=0.6270
- housing: F1=0.4010, precision=0.3421, recall=0.4844

## Weakest categories by test F1

- sanitation: F1=0.3173, precision=0.2572, recall=0.4139
- infrastructure: F1=0.3540, precision=0.2598, recall=0.5556
- environment: F1=0.3556, precision=0.2704, recall=0.5193

## What not to overclaim

- Do not claim that 311 requests represent all real urban problems; they represent reported service demand.
- Do not claim SHAP/feature importance establishes causality.
- Do not claim per-category thresholds are universally optimal; they are validation-selected for this forecasting setup.
- Do not use random split results as the main claim if chronological split results are available.

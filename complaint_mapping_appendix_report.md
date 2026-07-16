# Complaint Mapping Appendix Report

This report generates reviewer-facing artifacts for P2-10: the mapping from original NYC 311 complaint types to the nine analysis categories and the composition of the `other` category.

## Overall Category Summary

| complaint_category | complaint_types | complaint_count | share_of_all_requests |
| --- | --- | --- | --- |
| housing | 57 | 8061962 | 0.2606 |
| noise | 9 | 6713922 | 0.2170 |
| parking_traffic | 39 | 6521679 | 0.2108 |
| sanitation | 36 | 3090901 | 0.0999 |
| infrastructure | 28 | 2673125 | 0.0864 |
| public_safety | 38 | 1597023 | 0.0516 |
| water_sewer | 10 | 1212939 | 0.0392 |
| environment | 17 | 626930 | 0.0203 |
| other | 86 | 441648 | 0.0143 |

## Other Category Composition

- Total mapped requests: 30,940,129.
- `other` contains 86 complaint types and 441,648 requests (1.4274% of all mapped requests).
- The rows below are the largest `other` complaint types by request count.

| complaint_type | complaint_count | share_of_other_requests |
| --- | --- | --- |
| Encampment | 191408 | 0.4334 |
| Drug Activity | 93873 | 0.2126 |
| Animal in a Park | 29914 | 0.0677 |
| Vending | 27575 | 0.0624 |
| Electronics Waste | 26487 | 0.0600 |
| Miscellaneous Categories | 13752 | 0.0311 |
| Unleashed Dog | 11718 | 0.0265 |
| DOF Property - Payment Issue | 11414 | 0.0258 |
| DRIE | 4023 | 0.0091 |
| Storm | 3003 | 0.0068 |
| Posting Advertisement | 2745 | 0.0062 |
| DPR Internal | 2520 | 0.0057 |
| Uprooted Stump | 2343 | 0.0053 |
| Adopt-A-Basket | 2313 | 0.0052 |
| Private or Charter School Reopening | 2099 | 0.0048 |
| Face Covering Violation | 1798 | 0.0041 |
| Facades | 1769 | 0.0040 |
| LinkNYC | 1406 | 0.0032 |
| Quality of Life | 1324 | 0.0030 |
| Executive Inspections | 1289 | 0.0029 |
| Pet Shop | 1281 | 0.0029 |
| Cannabis Retailer | 1111 | 0.0025 |
| Illegal Animal Sold | 817 | 0.0018 |
| AHV Inspection Unit | 749 | 0.0017 |
| Special Natural Area District (SNAD) | 743 | 0.0017 |
| Special Operations | 440 | 0.0010 |
| FATF | 435 | 0.0010 |
| Incorrect Data | 356 | 0.0008 |
| Lifeguard | 259 | 0.0006 |
| Squeegee | 244 | 0.0006 |

## Appendix Files

- `complaint_type_category_mapping_appendix.csv`: all complaint types, assigned category, counts, and shares.
- `complaint_category_summary.csv`: category-level type counts and request shares.
- `other_category_composition.csv`: all complaint types mapped to `other`.
- `other_category_top200.csv`: existing top-200 `other` composition table with shares.

## Guardrails

- This pass documents the current deterministic mapping; it does not complete a sensitivity analysis with alternate groupings.
- The manuscript should avoid implying that `other` is semantically homogeneous.

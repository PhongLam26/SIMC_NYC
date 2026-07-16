# Error Severity Analysis

This report analyzes final-style 2025 errors using `platt` scores from the no-shortcut LightGBM bootstrap prediction rows.

Realized exceedance magnitude is computed as `z = (count_t+1 - mu_8w) / (sigma_8w + epsilon)`. Because near-zero sigma can inflate z, the report also includes absolute increase and target-count bands.

## Overall Error Counts

| target_definition | rows | positive_rows | alert_rows | true_positive_rows | false_positive_rows | false_negative_rows | precision | recall | median_positive_score | median_fn_score | median_tp_score |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| T0_current_reference | 122616 | 15712 | 34611 | 8917 | 25694 | 6795 | 0.2576 | 0.5675 | 0.1890 | 0.1035 | 0.2750 |
| T2_min_count_3 | 122616 | 13562 | 29566 | 7790 | 21776 | 5772 | 0.2635 | 0.5744 | 0.1947 | 0.0979 | 0.2894 |

## Recall by Realized z Severity

| target_definition | bin | positive_rows | share_of_target_positives | true_positive_rows | false_negative_rows | recall | median_score | median_target_count | median_realized_z |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| T0_current_reference | z_1.5_to_2 | 4785 | 0.3045 | 2462 | 2323 | 0.5145 | 0.1707 | 21.0000 | 1.7152 |
| T0_current_reference | z_2_to_3 | 5202 | 0.3311 | 2845 | 2357 | 0.5469 | 0.1828 | 23.0000 | 2.4749 |
| T0_current_reference | z_ge_3 | 5721 | 0.3641 | 3609 | 2112 | 0.6308 | 0.2161 | 22.0000 | 4.5380 |
| T0_current_reference | z_lt_1.5 | 4 | 0.0003 | 1 | 3 | 0.2500 | 0.1430 | 11.0000 | 1.5000 |
| T2_min_count_3 | z_1.5_to_2 | 4158 | 0.3066 | 2095 | 2063 | 0.5038 | 0.1681 | 26.0000 | 1.7287 |
| T2_min_count_3 | z_2_to_3 | 4576 | 0.3374 | 2585 | 1991 | 0.5649 | 0.1906 | 28.0000 | 2.4078 |
| T2_min_count_3 | z_ge_3 | 4824 | 0.3557 | 3108 | 1716 | 0.6443 | 0.2290 | 30.0000 | 4.2499 |
| T2_min_count_3 | z_lt_1.5 | 4 | 0.0003 | 2 | 2 | 0.5000 | 0.1483 | 11.0000 | 1.5000 |

## Recall by Absolute Increase

| target_definition | bin | positive_rows | true_positive_rows | false_negative_rows | recall | median_score | median_absolute_increase | median_ratio_increase |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| T0_current_reference | lt_1 | 817 | 499 | 318 | 0.6108 | 0.1907 | 0.8750 | 7.9999 |
| T0_current_reference | plus_1 | 1444 | 703 | 741 | 0.4868 | 0.1623 | 1.3750 | 8.0000 |
| T0_current_reference | plus_10_plus | 7947 | 4538 | 3409 | 0.5710 | 0.1946 | 22.7500 | 1.7455 |
| T0_current_reference | plus_2_to_3 | 1755 | 1068 | 687 | 0.6085 | 0.1945 | 2.8750 | 3.0000 |
| T0_current_reference | plus_4_to_9 | 3749 | 2109 | 1640 | 0.5626 | 0.1866 | 6.5000 | 2.0000 |
| T2_min_count_3 | plus_1 | 200 | 145 | 55 | 0.7250 | 0.2142 | 1.7500 | 2.4000 |
| T2_min_count_3 | plus_10_plus | 7947 | 4637 | 3310 | 0.5835 | 0.2001 | 22.7500 | 1.7455 |
| T2_min_count_3 | plus_2_to_3 | 1666 | 850 | 816 | 0.5102 | 0.1704 | 2.8750 | 2.8571 |
| T2_min_count_3 | plus_4_to_9 | 3749 | 2158 | 1591 | 0.5756 | 0.1932 | 6.5000 | 2.0000 |

## Recall by Target Count

| target_definition | bin | positive_rows | true_positive_rows | false_negative_rows | recall | median_score | median_target_count |
| --- | --- | --- | --- | --- | --- | --- | --- |
| T0_current_reference | count_0_or_1 | 1345 | 601 | 744 | 0.4468 | 0.1470 | 1.0000 |
| T0_current_reference | count_10_plus | 10832 | 6129 | 4703 | 0.5658 | 0.1903 | 39.0000 |
| T0_current_reference | count_2 | 805 | 488 | 317 | 0.6062 | 0.1969 | 2.0000 |
| T0_current_reference | count_3 | 577 | 374 | 203 | 0.6482 | 0.1993 | 3.0000 |
| T0_current_reference | count_4_to_9 | 2153 | 1325 | 828 | 0.6154 | 0.2016 | 6.0000 |
| T2_min_count_3 | count_10_plus | 10832 | 6313 | 4519 | 0.5828 | 0.1981 | 39.0000 |
| T2_min_count_3 | count_3 | 577 | 229 | 348 | 0.3969 | 0.1290 | 3.0000 |
| T2_min_count_3 | count_4_to_9 | 2153 | 1248 | 905 | 0.5797 | 0.1958 | 6.0000 |

## Guardrails

- Severity bins are computed only for positive rows; precision is reported in the overall table, not within positive-only bins.
- T0 and T2 use different positive labels, so compare their severity profiles descriptively rather than as a paired model comparison.
- Near-zero sigma can make z very large; use the absolute-increase and target-count bands alongside z bins.

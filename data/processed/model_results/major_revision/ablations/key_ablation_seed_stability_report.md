# Key Ablation Seed-Stability Report

This report reruns the key full-training ablation rows for `T2_min_count_3` across seeds 42, 123, 2026, 3407, 7777.

Scope: the rows that support final feature-scope claims are rerun across five seeds. Auxiliary negative-control rows such as calendar-only/weather-only are not used for final claims and are left in the single-seed full-ablation artifact.

## Held-Out 2025 Summary Across Seeds

| feature_config | seeds | pr_auc_mean | pr_auc_std | precision_at_5pct_mean | precision_at_5pct_std | f1_mean | f1_std | precision_mean | recall_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 03_history_calendar | 5 | 0.2882 | 0.0029 | 0.3852 | 0.0043 | 0.3457 | 0.0032 | 0.2543 | 0.5426 |
| 04_history_calendar_weather | 5 | 0.2979 | 0.0026 | 0.3903 | 0.0042 | 0.3527 | 0.0015 | 0.2595 | 0.5507 |
| 05_history_calendar_weather_borough | 5 | 0.2978 | 0.0010 | 0.3902 | 0.0023 | 0.3525 | 0.0003 | 0.2609 | 0.5447 |
| 06_history_calendar_weather_nta_fe | 5 | 0.3002 | 0.0019 | 0.3937 | 0.0031 | 0.3528 | 0.0010 | 0.2613 | 0.5435 |
| 07_final_no_shortcut_borough | 5 | 0.3185 | 0.0023 | 0.4194 | 0.0034 | 0.3638 | 0.0015 | 0.2721 | 0.5494 |
| 08_final_no_shortcut_nta_fe | 5 | 0.3188 | 0.0011 | 0.4215 | 0.0007 | 0.3642 | 0.0005 | 0.2713 | 0.5541 |

## Per-Seed Validation Rows

| feature_config | seed | pr_auc | precision_at_5pct | f1 | precision | recall | threshold |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 03_history_calendar | 42 | 0.2754 | 0.3633 | 0.3436 | 0.2472 | 0.5633 | 0.5466 |
| 03_history_calendar | 123 | 0.2744 | 0.3639 | 0.3415 | 0.2428 | 0.5752 | 0.5339 |
| 03_history_calendar | 2026 | 0.2810 | 0.3729 | 0.3455 | 0.2526 | 0.5466 | 0.5600 |
| 03_history_calendar | 3407 | 0.2770 | 0.3674 | 0.3455 | 0.2589 | 0.5191 | 0.5649 |
| 03_history_calendar | 7777 | 0.2823 | 0.3727 | 0.3464 | 0.2662 | 0.4960 | 0.5800 |
| 04_history_calendar_weather | 42 | 0.2738 | 0.3639 | 0.3466 | 0.2649 | 0.5011 | 0.6000 |
| 04_history_calendar_weather | 123 | 0.2749 | 0.3647 | 0.3483 | 0.2610 | 0.5232 | 0.5880 |
| 04_history_calendar_weather | 2026 | 0.2710 | 0.3605 | 0.3451 | 0.2586 | 0.5185 | 0.5948 |
| 04_history_calendar_weather | 3407 | 0.2734 | 0.3650 | 0.3463 | 0.2563 | 0.5336 | 0.5900 |
| 04_history_calendar_weather | 7777 | 0.2773 | 0.3716 | 0.3488 | 0.2614 | 0.5241 | 0.5831 |
| 05_history_calendar_weather_borough | 42 | 0.2722 | 0.3650 | 0.3448 | 0.2639 | 0.4973 | 0.5900 |
| 05_history_calendar_weather_borough | 123 | 0.2751 | 0.3705 | 0.3455 | 0.2589 | 0.5190 | 0.5937 |
| 05_history_calendar_weather_borough | 2026 | 0.2746 | 0.3676 | 0.3455 | 0.2551 | 0.5347 | 0.5745 |
| 05_history_calendar_weather_borough | 3407 | 0.2729 | 0.3631 | 0.3467 | 0.2569 | 0.5331 | 0.5800 |
| 05_history_calendar_weather_borough | 7777 | 0.2759 | 0.3722 | 0.3479 | 0.2694 | 0.4909 | 0.5946 |
| 06_history_calendar_weather_nta_fe | 42 | 0.2759 | 0.3644 | 0.3480 | 0.2650 | 0.5070 | 0.5950 |
| 06_history_calendar_weather_nta_fe | 123 | 0.2788 | 0.3730 | 0.3487 | 0.2572 | 0.5412 | 0.5900 |
| 06_history_calendar_weather_nta_fe | 2026 | 0.2791 | 0.3689 | 0.3481 | 0.2650 | 0.5072 | 0.5997 |
| 06_history_calendar_weather_nta_fe | 3407 | 0.2769 | 0.3685 | 0.3482 | 0.2572 | 0.5390 | 0.5819 |
| 06_history_calendar_weather_nta_fe | 7777 | 0.2778 | 0.3687 | 0.3475 | 0.2639 | 0.5086 | 0.5900 |
| 07_final_no_shortcut_borough | 42 | 0.2939 | 0.3906 | 0.3611 | 0.2667 | 0.5590 | 0.5452 |
| 07_final_no_shortcut_borough | 123 | 0.2961 | 0.3914 | 0.3617 | 0.2801 | 0.5105 | 0.5880 |
| 07_final_no_shortcut_borough | 2026 | 0.2984 | 0.3948 | 0.3631 | 0.2721 | 0.5455 | 0.5817 |
| 07_final_no_shortcut_borough | 3407 | 0.2947 | 0.3911 | 0.3624 | 0.2807 | 0.5115 | 0.5960 |
| 07_final_no_shortcut_borough | 7777 | 0.2966 | 0.3929 | 0.3601 | 0.2741 | 0.5246 | 0.5698 |
| 08_final_no_shortcut_nta_fe | 42 | 0.2951 | 0.3924 | 0.3618 | 0.2711 | 0.5436 | 0.5832 |
| 08_final_no_shortcut_nta_fe | 123 | 0.2960 | 0.3941 | 0.3623 | 0.2758 | 0.5277 | 0.5817 |
| 08_final_no_shortcut_nta_fe | 2026 | 0.2962 | 0.3977 | 0.3632 | 0.2722 | 0.5456 | 0.5792 |
| 08_final_no_shortcut_nta_fe | 3407 | 0.2977 | 0.3948 | 0.3633 | 0.2766 | 0.5289 | 0.5900 |
| 08_final_no_shortcut_nta_fe | 7777 | 0.2980 | 0.3962 | 0.3636 | 0.2748 | 0.5371 | 0.5800 |

## Per-Seed Test Rows

| feature_config | seed | pr_auc | precision_at_5pct | f1 | precision | recall | threshold |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 03_history_calendar | 42 | 0.2861 | 0.3804 | 0.3446 | 0.2478 | 0.5652 | 0.5466 |
| 03_history_calendar | 123 | 0.2841 | 0.3807 | 0.3414 | 0.2429 | 0.5740 | 0.5339 |
| 03_history_calendar | 2026 | 0.2909 | 0.3879 | 0.3452 | 0.2520 | 0.5476 | 0.5600 |
| 03_history_calendar | 3407 | 0.2898 | 0.3879 | 0.3474 | 0.2601 | 0.5230 | 0.5649 |
| 03_history_calendar | 7777 | 0.2899 | 0.3890 | 0.3500 | 0.2684 | 0.5029 | 0.5800 |
| 04_history_calendar_weather | 42 | 0.2955 | 0.3887 | 0.3537 | 0.2633 | 0.5386 | 0.6000 |
| 04_history_calendar_weather | 123 | 0.3010 | 0.3960 | 0.3547 | 0.2610 | 0.5536 | 0.5880 |
| 04_history_calendar_weather | 2026 | 0.2950 | 0.3851 | 0.3509 | 0.2575 | 0.5507 | 0.5948 |
| 04_history_calendar_weather | 3407 | 0.2983 | 0.3890 | 0.3523 | 0.2568 | 0.5607 | 0.5900 |
| 04_history_calendar_weather | 7777 | 0.2996 | 0.3929 | 0.3519 | 0.2587 | 0.5500 | 0.5831 |
| 05_history_calendar_weather_borough | 42 | 0.2987 | 0.3918 | 0.3527 | 0.2643 | 0.5297 | 0.5900 |
| 05_history_calendar_weather_borough | 123 | 0.2980 | 0.3898 | 0.3529 | 0.2610 | 0.5445 | 0.5937 |
| 05_history_calendar_weather_borough | 2026 | 0.2982 | 0.3913 | 0.3522 | 0.2559 | 0.5644 | 0.5745 |
| 05_history_calendar_weather_borough | 3407 | 0.2978 | 0.3864 | 0.3528 | 0.2559 | 0.5680 | 0.5800 |
| 05_history_calendar_weather_borough | 7777 | 0.2962 | 0.3916 | 0.3522 | 0.2671 | 0.5167 | 0.5946 |
| 06_history_calendar_weather_nta_fe | 42 | 0.2991 | 0.3926 | 0.3530 | 0.2657 | 0.5257 | 0.5950 |
| 06_history_calendar_weather_nta_fe | 123 | 0.3002 | 0.3905 | 0.3532 | 0.2569 | 0.5645 | 0.5900 |
| 06_history_calendar_weather_nta_fe | 2026 | 0.3032 | 0.3983 | 0.3532 | 0.2643 | 0.5321 | 0.5997 |
| 06_history_calendar_weather_nta_fe | 3407 | 0.3003 | 0.3954 | 0.3535 | 0.2575 | 0.5637 | 0.5819 |
| 06_history_calendar_weather_nta_fe | 7777 | 0.2983 | 0.3918 | 0.3510 | 0.2621 | 0.5313 | 0.5900 |
| 07_final_no_shortcut_borough | 42 | 0.3165 | 0.4180 | 0.3613 | 0.2635 | 0.5744 | 0.5452 |
| 07_final_no_shortcut_borough | 123 | 0.3197 | 0.4239 | 0.3645 | 0.2777 | 0.5302 | 0.5880 |
| 07_final_no_shortcut_borough | 2026 | 0.3213 | 0.4200 | 0.3651 | 0.2696 | 0.5654 | 0.5817 |
| 07_final_no_shortcut_borough | 3407 | 0.3193 | 0.4206 | 0.3645 | 0.2766 | 0.5341 | 0.5960 |
| 07_final_no_shortcut_borough | 7777 | 0.3158 | 0.4146 | 0.3635 | 0.2732 | 0.5431 | 0.5698 |
| 08_final_no_shortcut_nta_fe | 42 | 0.3187 | 0.4215 | 0.3634 | 0.2682 | 0.5632 | 0.5832 |
| 08_final_no_shortcut_nta_fe | 123 | 0.3197 | 0.4215 | 0.3645 | 0.2735 | 0.5460 | 0.5817 |
| 08_final_no_shortcut_nta_fe | 2026 | 0.3173 | 0.4206 | 0.3645 | 0.2688 | 0.5659 | 0.5792 |
| 08_final_no_shortcut_nta_fe | 3407 | 0.3201 | 0.4226 | 0.3647 | 0.2743 | 0.5438 | 0.5900 |
| 08_final_no_shortcut_nta_fe | 7777 | 0.3181 | 0.4211 | 0.3639 | 0.2716 | 0.5514 | 0.5800 |

## Guardrails

- These seeds quantify stability for key ablation evidence; final selection still follows the pre-registered validation/backtest rule.
- Differences on the order of a few thousandths should be interpreted with the bootstrap and paired-CI evidence, not as standalone decisive improvements.

# Evidence for the research on frab-binance-spot (built 2026-10-29)

## frab-binance-spot -- trial 80585 (2026-10-25)
route: needs-forward -- forward test too short to tell either way: 14 days; it decides at 108 days, 33% of fitting plus forward (216 days of fitting); so far it would route needs-more-data (metric(s) could not be computed: atomic_execution) [venue_supported: venue binance has no order-placing adapter of ours (venues/binance.yaml execution_adapter: false); this is a fact about our infrastructure, not about access (docs/VENUES.md) and not about the strategy; whether the instruments exist on Hyperliquid, where orders could be placed, is not checked here; the book held 22 instrument(s): AAVE, AAVE-SPOT, ARK, ARK-SPOT, ASTER, ASTER-SPOT, BERA, BERA-SPOT and 14 more]
- accrual_applied: 1
- ann_return_net: -0.00959
- ann_return_net_ex_best_1pct: -0.00959
- data_forward_available: 1
- effective_profitable_days: 110.6
- effective_profitable_days_share: 0.3292
- fit_accrual_applied: 1
- fit_ann_return_net: 0.0737
- fit_ann_return_net_ex_best_1pct: 0.06386
- fit_data_forward_available: 1
- fit_effective_profitable_days: 1420
- fit_effective_profitable_days_share: 0.2741
- fit_fragility_days: 20
- fit_fragility_share: 0.00386
- fit_history_truncated_instruments: 200
- fit_hold_ann_return: 1.188
- fit_hold_ann_return_ex_best_1pct: 0.5315
- fit_max_dd: -0.007767
- fit_min_capital_usd: 97.2
- fit_n_periods: 5181
- fit_point_in_time_universe: 1
- fit_sharpe_net: 1.499
- fit_skew: -0.1336
- fit_turnover: 11.52
- fit_venue_supported: 0
- fit_window_days: 215.9
- fit_worst_week: -0.002147
- fix_date_known: 1
- forward_days: 14.04
- forward_days_needed: 108
- fragility_days: 0
- fragility_share: 0
- history_truncated_instruments: 200
- hold_ann_return: -0.9929
- hold_ann_return_ex_best_1pct: -0.9901
- judged_on_forward: 1
- max_dd: -0.002769
- min_capital_usd: 80
- n_periods: 336
- noise_power: 0.01
- noise_return_percentile: 0.43
- noise_sharpe_percentile: 0.42
- point_in_time_universe: 1
- regime_bear_ann_return: -0.00959
- regime_bear_below: -0.0377
- regime_bear_btc_return: -0.006743
- regime_bear_days: 15
- regime_bear_max_dd: -0.002769
- regime_bear_noise_percentile: 0.43
- regime_bear_return: -0.0003693
- regime_bear_share: 1
- regime_bear_sharpe: -0.287
- regime_bull_above: 0.09335
- regime_bull_days: 0
- regime_bull_share: 0
- regime_flat_days: 0
- regime_flat_share: 0
- selection_days: 215.9
- selection_rules_failed: 1
- sharpe_net: -0.287
- skew: -0.6881
- turnover: 61.83
- venue_supported: 0
- worst_week: -0.0009887
selection-period regime breakdown: regime_bear_ann_return=0.04168, regime_bear_below=-0.0377, regime_bear_btc_return=-0.2288, regime_bear_days=37, regime_bear_max_dd=-0.003602, regime_bear_return=0.004145, regime_bear_share=0.1714, regime_bear_sharpe=0.9039, regime_bull_above=0.09335, regime_bull_ann_return=0.07351, regime_bull_btc_return=0.4263, regime_bull_days=59, regime_bull_max_dd=-0.007767, regime_bull_return=0.01152, regime_bull_share=0.2733, regime_bull_sharpe=1.337, regime_flat_ann_return=0.08387, regime_flat_btc_return=0.09113, regime_flat_days=120, regime_flat_max_dd=-0.006429, regime_flat_return=0.02678, regime_flat_share=0.5553, regime_flat_sharpe=1.789

## frab-binance-spot-300 -- trial 117447 (2026-10-28)
route: needs-forward -- forward test too short to tell either way: 17 days; it decides at 108 days, 33% of fitting plus forward (216 days of fitting); so far it would route needs-more-data (metric(s) could not be computed: atomic_execution) [venue_supported: venue binance has no order-placing adapter of ours (venues/binance.yaml execution_adapter: false); this is a fact about our infrastructure, not about access (docs/VENUES.md) and not about the strategy; whether the instruments exist on Hyperliquid, where orders could be placed, is not checked here; the book held 22 instrument(s): AAVE, AAVE-SPOT, ARK, ARK-SPOT, ASTER, ASTER-SPOT, BERA, BERA-SPOT and 14 more]
- accrual_applied: 1
- ann_return_net: -0.05439
- ann_return_net_ex_best_1pct: -0.05439
- cap_ann_return: -0.05527
- cap_ann_return_ex_best_1pct: -0.05527
- cap_max_dd: -0.005068
- cap_reinvested: 0
- cap_sharpe: -1.523
- cap_total_return: -0.002573
- data_forward_available: 1
- effective_profitable_days: 133.3
- effective_profitable_days_share: 0.3267
- fit_accrual_applied: 1
- fit_ann_return_net: 0.03834
- fit_ann_return_net_ex_best_1pct: 0.02853
- fit_cap_ann_return: 0.03888
- fit_cap_ann_return_ex_best_1pct: 0.02935
- fit_cap_max_dd: -0.01941
- fit_cap_reinvested: 0
- fit_cap_sharpe: 0.777
- fit_cap_total_return: 0.02298
- fit_data_forward_available: 1
- fit_effective_profitable_days: 1421
- fit_effective_profitable_days_share: 0.2742
- fit_fragility_days: 9
- fit_fragility_share: 0.001737
- fit_history_truncated_instruments: 200
- fit_hold_ann_return: 1.188
- fit_hold_ann_return_ex_best_1pct: 0.5315
- fit_max_dd: -0.01942
- fit_min_capital_usd: 97.2
- fit_n_periods: 5181
- fit_point_in_time_universe: 1
- fit_sharpe_net: 0.777
- fit_skew: -0.799
- fit_total_return_net: 0.02249
- fit_turnover: 11.24
- fit_venue_supported: 0
- fit_window_days: 215.9
- fit_worst_week: -0.007653
- fix_date_known: 1
- forward_days: 17.04
- forward_days_needed: 108
- fragility_days: 0
- fragility_share: 0
- history_truncated_instruments: 200
- hold_ann_return: -0.9804
- hold_ann_return_ex_best_1pct: -0.9757
- judged_on_forward: 1
- max_dd: -0.005071
- min_capital_usd: 80
- n_periods: 408
- noise_cap_return_percentile: 0.395
- noise_power: 0.01
- noise_return_percentile: 0.395
- noise_sharpe_percentile: 0.35
- point_in_time_universe: 1
- regime_bear_ann_return: -0.05439
- regime_bear_below: -0.03788
- regime_bear_btc_return: -0.01952
- regime_bear_days: 18
- regime_bear_max_dd: -0.005071
- regime_bear_noise_percentile: 0.395
- regime_bear_return: -0.0026
- regime_bear_share: 1
- regime_bear_sharpe: -1.523
- regime_bull_above: 0.09306
- regime_bull_days: 0
- regime_bull_share: 0
- regime_flat_days: 0
- regime_flat_share: 0
- selection_days: 215.9
- selection_rules_failed: 1
- sharpe_net: -1.523
- skew: -2.872
- total_return_net: -0.0026
- turnover: 48.61
- venue_supported: 0
- worst_week: -0.003134
selection-period regime breakdown: regime_bear_ann_return=0.03115, regime_bear_below=-0.03788, regime_bear_btc_return=-0.2288, regime_bear_days=37, regime_bear_max_dd=-0.004533, regime_bear_return=0.003113, regime_bear_share=0.1714, regime_bear_sharpe=0.6794, regime_bull_above=0.09306, regime_bull_ann_return=0.06732, regime_bull_btc_return=0.4263, regime_bull_days=59, regime_bull_max_dd=-0.007767, regime_bull_return=0.01058, regime_bull_share=0.2733, regime_bull_sharpe=1.227, regime_flat_ann_return=0.02657, regime_flat_btc_return=0.09113, regime_flat_days=120, regime_flat_max_dd=-0.01848, regime_flat_return=0.008643, regime_flat_share=0.5553, regime_flat_sharpe=0.5616

## frab-binance-spot-400 -- trial 104180 (2026-10-28)
route: needs-forward -- forward test too short to tell either way: 17 days; it decides at 108 days, 33% of fitting plus forward (216 days of fitting); so far it would route needs-more-data (metric(s) could not be computed: atomic_execution) [venue_supported: venue binance has no order-placing adapter of ours (venues/binance.yaml execution_adapter: false); this is a fact about our infrastructure, not about access (docs/VENUES.md) and not about the strategy; whether the instruments exist on Hyperliquid, where orders could be placed, is not checked here; the book held 22 instrument(s): AAVE, AAVE-SPOT, ARK, ARK-SPOT, ASTER, ASTER-SPOT, BERA, BERA-SPOT and 14 more]
- accrual_applied: 1
- ann_return_net: -0.03716
- ann_return_net_ex_best_1pct: -0.03716
- data_forward_available: 1
- effective_profitable_days: 133.2
- effective_profitable_days_share: 0.3266
- fit_accrual_applied: 1
- fit_ann_return_net: 0.0502
- fit_ann_return_net_ex_best_1pct: 0.04039
- fit_data_forward_available: 1
- fit_effective_profitable_days: 1421
- fit_effective_profitable_days_share: 0.2743
- fit_fragility_days: 13
- fit_fragility_share: 0.002509
- fit_history_truncated_instruments: 200
- fit_hold_ann_return: 1.188
- fit_hold_ann_return_ex_best_1pct: 0.5315
- fit_max_dd: -0.01415
- fit_min_capital_usd: 97.2
- fit_n_periods: 5181
- fit_point_in_time_universe: 1
- fit_sharpe_net: 1.02
- fit_skew: -0.4467
- fit_total_return_net: 0.02937
- fit_turnover: 11.26
- fit_venue_supported: 0
- fit_window_days: 215.9
- fit_worst_week: -0.005658
- fix_date_known: 1
- forward_days: 17.04
- forward_days_needed: 108
- fragility_days: 0
- fragility_share: 0
- history_truncated_instruments: 200
- hold_ann_return: -0.9804
- hold_ann_return_ex_best_1pct: -0.9757
- judged_on_forward: 1
- max_dd: -0.004246
- min_capital_usd: 80
- n_periods: 408
- noise_power: 0.01
- noise_return_percentile: 0.4
- noise_sharpe_percentile: 0.365
- point_in_time_universe: 1
- regime_bear_ann_return: -0.03716
- regime_bear_below: -0.03788
- regime_bear_btc_return: -0.01952
- regime_bear_days: 18
- regime_bear_max_dd: -0.004246
- regime_bear_noise_percentile: 0.4
- regime_bear_return: -0.001761
- regime_bear_share: 1
- regime_bear_sharpe: -1.077
- regime_bull_above: 0.09306
- regime_bull_days: 0
- regime_bull_share: 0
- regime_flat_days: 0
- regime_flat_share: 0
- selection_days: 215.9
- selection_rules_failed: 1
- sharpe_net: -1.077
- skew: -1.729
- total_return_net: -0.001761
- turnover: 48.87
- venue_supported: 0
- worst_week: -0.002298
selection-period regime breakdown: regime_bear_ann_return=0.03454, regime_bear_below=-0.03788, regime_bear_btc_return=-0.2288, regime_bear_days=37, regime_bear_max_dd=-0.004201, regime_bear_return=0.003446, regime_bear_share=0.1714, regime_bear_sharpe=0.7518, regime_bull_above=0.09306, regime_bull_ann_return=0.06953, regime_bull_btc_return=0.4263, regime_bull_days=59, regime_bull_max_dd=-0.007767, regime_bull_return=0.01092, regime_bull_share=0.2733, regime_bull_sharpe=1.266, regime_flat_ann_return=0.04566, regime_flat_btc_return=0.09113, regime_flat_days=120, regime_flat_max_dd=-0.01354, regime_flat_return=0.01476, regime_flat_share=0.5553, regime_flat_sharpe=0.966

## frab-binance-spot-allbinance -- trial 81391 (2026-10-25)
route: reject -- fails on the selection period, the data its parameters were chosen on and which flatters it: net_edge_positive [venue_supported: venue binance has no order-placing adapter of ours (venues/binance.yaml execution_adapter: false); this is a fact about our infrastructure, not about access (docs/VENUES.md) and not about the strategy; whether the instruments exist on Hyperliquid, where orders could be placed, is not checked here; the book held 92 instrument(s): 0G, 0G-SPOT, 1000FLOKI, 1000FLOKI-SPOT, 1000PEPE, 1000PEPE-SPOT, 1000XEC, 1000XEC-SPOT and 84 more]
- accrual_applied: 1
- ann_return_net: 0.06172
- ann_return_net_ex_best_1pct: 0.06172
- data_forward_available: 1
- effective_profitable_days: 101.1
- effective_profitable_days_share: 0.301
- fit_accrual_applied: 1
- fit_ann_return_net: -0.1333
- fit_ann_return_net_ex_best_1pct: -0.1847
- fit_data_forward_available: 1
- fit_effective_profitable_days: 1026
- fit_effective_profitable_days_share: 0.1979
- fit_fragility_days: 0
- fit_fragility_share: 0
- fit_hold_ann_return: -0.1595
- fit_hold_ann_return_ex_best_1pct: -0.3355
- fit_max_dd: -0.1377
- fit_min_capital_usd: 877.6
- fit_n_periods: 5183
- fit_point_in_time_universe: 1
- fit_sharpe_net: -0.6013
- fit_skew: -8.741
- fit_turnover: 155
- fit_venue_supported: 0
- fit_window_days: 216
- fit_worst_week: -0.07136
- fix_date_known: 1
- forward_days: 14.04
- forward_days_needed: 108
- fragility_days: 2
- fragility_share: 0.005952
- hold_ann_return: -0.7881
- hold_ann_return_ex_best_1pct: -0.765
- judged_on_forward: 1
- max_dd: -0.002697
- min_capital_usd: 290.3
- n_periods: 336
- noise_power: 0.01
- noise_return_percentile: 0.65
- noise_sharpe_percentile: 0.67
- point_in_time_universe: 1
- regime_bear_ann_return: 0.06172
- regime_bear_below: -0.0377
- regime_bear_btc_return: -0.006743
- regime_bear_days: 15
- regime_bear_max_dd: -0.002697
- regime_bear_noise_percentile: 0.65
- regime_bear_return: 0.002298
- regime_bear_share: 1
- regime_bear_sharpe: 1.085
- regime_bull_above: 0.09335
- regime_bull_days: 0
- regime_bull_share: 0
- regime_flat_days: 0
- regime_flat_share: 0
- selection_days: 216
- selection_rules_failed: 2
- sharpe_net: 1.085
- skew: -0.03744
- turnover: 85.42
- venue_supported: 0
- worst_week: -0.001617
selection-period regime breakdown: regime_bear_ann_return=-0.01745, regime_bear_below=-0.0377, regime_bear_btc_return=-0.2288, regime_bear_days=37, regime_bear_max_dd=-0.01879, regime_bear_return=-0.001781, regime_bear_share=0.1713, regime_bear_sharpe=-0.1057, regime_bull_above=0.09335, regime_bull_ann_return=-0.2243, regime_bull_btc_return=0.4263, regime_bull_days=59, regime_bull_max_dd=-0.08815, regime_bull_return=-0.0402, regime_bull_share=0.2732, regime_bull_sharpe=-1.242, regime_flat_ann_return=-0.1194, regime_flat_btc_return=0.09113, regime_flat_days=120, regime_flat_max_dd=-0.08222, regime_flat_return=-0.04091, regime_flat_share=0.5555, regime_flat_sharpe=-0.4361

## frab-binance-spot-allbinance-300 -- trial 81793 (2026-10-25)
route: reject -- fails on the selection period, the data its parameters were chosen on and which flatters it: net_edge_positive [venue_supported: venue binance has no order-placing adapter of ours (venues/binance.yaml execution_adapter: false); this is a fact about our infrastructure, not about access (docs/VENUES.md) and not about the strategy; whether the instruments exist on Hyperliquid, where orders could be placed, is not checked here; the book held 92 instrument(s): 0G, 0G-SPOT, 1000FLOKI, 1000FLOKI-SPOT, 1000PEPE, 1000PEPE-SPOT, 1000XEC, 1000XEC-SPOT and 84 more]
- accrual_applied: 1
- ann_return_net: 0.06172
- ann_return_net_ex_best_1pct: 0.06172
- data_forward_available: 1
- effective_profitable_days: 101.1
- effective_profitable_days_share: 0.301
- fit_accrual_applied: 1
- fit_ann_return_net: -0.1333
- fit_ann_return_net_ex_best_1pct: -0.1847
- fit_data_forward_available: 1
- fit_effective_profitable_days: 1026
- fit_effective_profitable_days_share: 0.1979
- fit_fragility_days: 0
- fit_fragility_share: 0
- fit_hold_ann_return: -0.1595
- fit_hold_ann_return_ex_best_1pct: -0.3355
- fit_max_dd: -0.1377
- fit_min_capital_usd: 877.6
- fit_n_periods: 5183
- fit_point_in_time_universe: 1
- fit_sharpe_net: -0.6013
- fit_skew: -8.741
- fit_turnover: 155
- fit_venue_supported: 0
- fit_window_days: 216
- fit_worst_week: -0.07136
- fix_date_known: 1
- forward_days: 14.04
- forward_days_needed: 108
- fragility_days: 2
- fragility_share: 0.005952
- hold_ann_return: -0.7881
- hold_ann_return_ex_best_1pct: -0.765
- judged_on_forward: 1
- max_dd: -0.002697
- min_capital_usd: 290.3
- n_periods: 336
- noise_power: 0.01
- noise_return_percentile: 0.65
- noise_sharpe_percentile: 0.67
- point_in_time_universe: 1
- regime_bear_ann_return: 0.06172
- regime_bear_below: -0.0377
- regime_bear_btc_return: -0.006743
- regime_bear_days: 15
- regime_bear_max_dd: -0.002697
- regime_bear_noise_percentile: 0.65
- regime_bear_return: 0.002298
- regime_bear_share: 1
- regime_bear_sharpe: 1.085
- regime_bull_above: 0.09335
- regime_bull_days: 0
- regime_bull_share: 0
- regime_flat_days: 0
- regime_flat_share: 0
- selection_days: 216
- selection_rules_failed: 2
- sharpe_net: 1.085
- skew: -0.03744
- turnover: 85.42
- venue_supported: 0
- worst_week: -0.001617
selection-period regime breakdown: regime_bear_ann_return=-0.01745, regime_bear_below=-0.0377, regime_bear_btc_return=-0.2288, regime_bear_days=37, regime_bear_max_dd=-0.01879, regime_bear_return=-0.001781, regime_bear_share=0.1713, regime_bear_sharpe=-0.1057, regime_bull_above=0.09335, regime_bull_ann_return=-0.2243, regime_bull_btc_return=0.4263, regime_bull_days=59, regime_bull_max_dd=-0.08815, regime_bull_return=-0.0402, regime_bull_share=0.2732, regime_bull_sharpe=-1.242, regime_flat_ann_return=-0.1194, regime_flat_btc_return=0.09113, regime_flat_days=120, regime_flat_max_dd=-0.08222, regime_flat_return=-0.04091, regime_flat_share=0.5555, regime_flat_sharpe=-0.4361

## frab-binance-spot-allbinance-400 -- trial 81994 (2026-10-25)
route: reject -- fails on the selection period, the data its parameters were chosen on and which flatters it: net_edge_positive [venue_supported: venue binance has no order-placing adapter of ours (venues/binance.yaml execution_adapter: false); this is a fact about our infrastructure, not about access (docs/VENUES.md) and not about the strategy; whether the instruments exist on Hyperliquid, where orders could be placed, is not checked here; the book held 92 instrument(s): 0G, 0G-SPOT, 1000FLOKI, 1000FLOKI-SPOT, 1000PEPE, 1000PEPE-SPOT, 1000XEC, 1000XEC-SPOT and 84 more]
- accrual_applied: 1
- ann_return_net: 0.06172
- ann_return_net_ex_best_1pct: 0.06172
- data_forward_available: 1
- effective_profitable_days: 101.1
- effective_profitable_days_share: 0.301
- fit_accrual_applied: 1
- fit_ann_return_net: -0.1333
- fit_ann_return_net_ex_best_1pct: -0.1847
- fit_data_forward_available: 1
- fit_effective_profitable_days: 1026
- fit_effective_profitable_days_share: 0.1979
- fit_fragility_days: 0
- fit_fragility_share: 0
- fit_hold_ann_return: -0.1595
- fit_hold_ann_return_ex_best_1pct: -0.3355
- fit_max_dd: -0.1377
- fit_min_capital_usd: 877.6
- fit_n_periods: 5183
- fit_point_in_time_universe: 1
- fit_sharpe_net: -0.6013
- fit_skew: -8.741
- fit_turnover: 155
- fit_venue_supported: 0
- fit_window_days: 216
- fit_worst_week: -0.07136
- fix_date_known: 1
- forward_days: 14.04
- forward_days_needed: 108
- fragility_days: 2
- fragility_share: 0.005952
- hold_ann_return: -0.7881
- hold_ann_return_ex_best_1pct: -0.765
- judged_on_forward: 1
- max_dd: -0.002697
- min_capital_usd: 290.3
- n_periods: 336
- noise_power: 0.01
- noise_return_percentile: 0.65
- noise_sharpe_percentile: 0.67
- point_in_time_universe: 1
- regime_bear_ann_return: 0.06172
- regime_bear_below: -0.0377
- regime_bear_btc_return: -0.006743
- regime_bear_days: 15
- regime_bear_max_dd: -0.002697
- regime_bear_noise_percentile: 0.65
- regime_bear_return: 0.002298
- regime_bear_share: 1
- regime_bear_sharpe: 1.085
- regime_bull_above: 0.09335
- regime_bull_days: 0
- regime_bull_share: 0
- regime_flat_days: 0
- regime_flat_share: 0
- selection_days: 216
- selection_rules_failed: 2
- sharpe_net: 1.085
- skew: -0.03744
- turnover: 85.42
- venue_supported: 0
- worst_week: -0.001617
selection-period regime breakdown: regime_bear_ann_return=-0.01745, regime_bear_below=-0.0377, regime_bear_btc_return=-0.2288, regime_bear_days=37, regime_bear_max_dd=-0.01879, regime_bear_return=-0.001781, regime_bear_share=0.1713, regime_bear_sharpe=-0.1057, regime_bull_above=0.09335, regime_bull_ann_return=-0.2243, regime_bull_btc_return=0.4263, regime_bull_days=59, regime_bull_max_dd=-0.08815, regime_bull_return=-0.0402, regime_bull_share=0.2732, regime_bull_sharpe=-1.242, regime_flat_ann_return=-0.1194, regime_flat_btc_return=0.09113, regime_flat_days=120, regime_flat_max_dd=-0.08222, regime_flat_return=-0.04091, regime_flat_share=0.5555, regime_flat_sharpe=-0.4361

## frab-binance-spot-five-coins -- trial 81188 (2026-10-25)
route: needs-forward -- forward test too short to tell either way: 14 days; it decides at 108 days, 33% of fitting plus forward (216 days of fitting); so far it would route needs-more-data (metric(s) could not be computed: atomic_execution) [venue_supported: venue binance has no order-placing adapter of ours (venues/binance.yaml execution_adapter: false); this is a fact about our infrastructure, not about access (docs/VENUES.md) and not about the strategy; whether the instruments exist on Hyperliquid, where orders could be placed, is not checked here; the book held 8 instrument(s): AVAX, AVAX-SPOT, BTC, BTC-SPOT, ETH, ETH-SPOT, SOL, SOL-SPOT]
- accrual_applied: 1
- ann_return_net: 0.008008
- ann_return_net_ex_best_1pct: 0.008008
- data_forward_available: 1
- effective_profitable_days: 116
- effective_profitable_days_share: 0.3451
- fit_accrual_applied: 1
- fit_ann_return_net: 0.002563
- fit_ann_return_net_ex_best_1pct: -0.0004128
- fit_data_forward_available: 1
- fit_effective_profitable_days: 1495
- fit_effective_profitable_days_share: 0.2885
- fit_fragility_days: 2
- fit_fragility_share: 0.000386
- fit_history_truncated_instruments: 200
- fit_hold_ann_return: 0.5045
- fit_hold_ann_return_ex_best_1pct: 0.1172
- fit_max_dd: -0.01321
- fit_min_capital_usd: 105.8
- fit_n_periods: 5181
- fit_point_in_time_universe: 1
- fit_sharpe_net: 0.2024
- fit_skew: -1.289
- fit_turnover: 27.47
- fit_venue_supported: 0
- fit_window_days: 215.9
- fit_worst_week: -0.001924
- fix_date_known: 1
- forward_days: 14.04
- forward_days_needed: 108
- fragility_days: 1
- fragility_share: 0.002976
- history_truncated_instruments: 200
- hold_ann_return: -0.8311
- hold_ann_return_ex_best_1pct: -0.8098
- judged_on_forward: 1
- max_dd: -0.0007073
- min_capital_usd: 54.01
- n_periods: 336
- noise_power: 0.01
- noise_return_percentile: 0.645
- noise_sharpe_percentile: 0.61
- point_in_time_universe: 1
- regime_bear_ann_return: 0.008008
- regime_bear_below: -0.0377
- regime_bear_btc_return: -0.006743
- regime_bear_days: 15
- regime_bear_max_dd: -0.0007073
- regime_bear_noise_percentile: 0.645
- regime_bear_return: 0.0003058
- regime_bear_share: 1
- regime_bear_sharpe: 0.4829
- regime_bull_above: 0.09335
- regime_bull_days: 0
- regime_bull_share: 0
- regime_flat_days: 0
- regime_flat_share: 0
- selection_days: 215.9
- selection_rules_failed: 1
- sharpe_net: 0.4829
- skew: -2.09
- turnover: 49.15
- venue_supported: 0
- worst_week: -0.001608
selection-period regime breakdown: regime_bear_ann_return=-0.02556, regime_bear_below=-0.0377, regime_bear_btc_return=-0.2288, regime_bear_days=37, regime_bear_max_dd=-0.002881, regime_bear_return=-0.00262, regime_bear_share=0.1714, regime_bear_sharpe=-2.084, regime_bull_above=0.09335, regime_bull_ann_return=0.0284, regime_bull_btc_return=0.4263, regime_bull_days=59, regime_bull_max_dd=-0.002054, regime_bull_return=0.004534, regime_bull_share=0.2733, regime_bull_sharpe=2.098, regime_flat_ann_return=-0.001179, regime_flat_btc_return=0.09113, regime_flat_days=120, regime_flat_max_dd=-0.009817, regime_flat_return=-0.0003872, regime_flat_share=0.5553, regime_flat_sharpe=-0.08345

## frab-live-own-spot -- trial 116844 (2026-10-28)
route: needs-forward -- forward test too short to tell either way: 17 days; it decides at 108 days, 33% of fitting plus forward (216 days of fitting); so far it would route reject (rule(s) failed: shape_aware_edge; fatal rule: shape_aware_edge); and on this window the noise bar would pass a strategy of this shape truly earning 7% a year only 2% of the time (it decides from 80%)
- accrual_applied: 1
- ann_return_net: 0.03871
- ann_return_net_ex_best_1pct: 0.03871
- atomic_execution: 1
- cap_ann_return: 0.03846
- cap_ann_return_ex_best_1pct: 0.03846
- cap_max_dd: -0.001594
- cap_reinvested: 0
- cap_sharpe: 1.248
- cap_total_return: 0.00179
- data_forward_available: 1
- effective_profitable_days: 121.8
- effective_profitable_days_share: 0.2984
- fit_accrual_applied: 1
- fit_ann_return_net: 0.1338
- fit_ann_return_net_ex_best_1pct: 0.04584
- fit_atomic_execution: 1
- fit_cap_ann_return: 0.139
- fit_cap_ann_return_ex_best_1pct: 0.05688
- fit_cap_max_dd: -0.05704
- fit_cap_reinvested: 0
- fit_cap_sharpe: 0.8443
- fit_cap_total_return: 0.08217
- fit_data_forward_available: 1
- fit_effective_profitable_days: 313.8
- fit_effective_profitable_days_share: 0.06056
- fit_fragility_days: 2
- fit_fragility_share: 0.000386
- fit_history_truncated_instruments: 200
- fit_hold_ann_return: 1.715
- fit_hold_ann_return_ex_best_1pct: 0.855
- fit_max_dd: -0.05627
- fit_min_capital_usd: 92.4
- fit_n_periods: 5181
- fit_point_in_time_universe: 1
- fit_sharpe_net: 0.8443
- fit_skew: 5.224
- fit_total_return_net: 0.07702
- fit_turnover: 25.53
- fit_venue_supported: 1
- fit_window_days: 215.9
- fit_worst_week: -0.00168
- fix_date_known: 1
- forward_days: 17.04
- forward_days_needed: 108
- fragility_days: 2
- fragility_share: 0.004902
- history_truncated_instruments: 200
- hold_ann_return: -0.7742
- hold_ann_return_ex_best_1pct: -0.7548
- judged_on_forward: 1
- max_dd: -0.001594
- min_capital_usd: 92.4
- n_periods: 408
- noise_cap_return_percentile: 0.555
- noise_power: 0.02
- noise_return_percentile: 0.58
- noise_sharpe_percentile: 0.825
- point_in_time_universe: 1
- regime_bear_ann_return: 0.03871
- regime_bear_below: -0.03788
- regime_bear_btc_return: -0.01952
- regime_bear_days: 18
- regime_bear_max_dd: -0.001594
- regime_bear_noise_percentile: 0.58
- regime_bear_return: 0.001769
- regime_bear_share: 1
- regime_bear_sharpe: 1.248
- regime_bull_above: 0.09306
- regime_bull_days: 0
- regime_bull_share: 0
- regime_flat_days: 0
- regime_flat_share: 0
- selection_days: 215.9
- selection_rules_failed: 0
- sharpe_net: 1.248
- skew: 0.07293
- total_return_net: 0.001769
- turnover: 35.97
- venue_supported: 1
- worst_week: -0.001242
selection-period regime breakdown: regime_bear_ann_return=0.5554, regime_bear_below=-0.03788, regime_bear_btc_return=-0.2288, regime_bear_days=37, regime_bear_max_dd=-0.05627, regime_bear_return=0.04576, regime_bear_share=0.1714, regime_bear_sharpe=1.359, regime_bull_above=0.09306, regime_bull_ann_return=0.05743, regime_bull_btc_return=0.4263, regime_bull_days=59, regime_bull_max_dd=-0.00429, regime_bull_return=0.009062, regime_bull_share=0.2733, regime_bull_sharpe=0.9864, regime_flat_ann_return=0.06424, regime_flat_btc_return=0.09113, regime_flat_days=120, regime_flat_max_dd=-0.004752, regime_flat_return=0.02064, regime_flat_share=0.5553, regime_flat_sharpe=1.111

## Paper reconciliation (2026-10-29)
**Bv2 (book 3 = b2, sized as created with carry on; book 4 = b2_cold)**
    book 3 (headroom 1.0, sized with carry: True)
      matched hours 4360, price mismatches 0
      paper  +4.15%   replay +4.15%
      max |hourly equity diff| 0.0000
      daily return correlation 1.0000 (n=46)
    book 4 (headroom 1.5, sized with carry: False)
      matched hours 4356, price mismatches 0
      paper  +5.58%   replay +5.58%
      max |hourly equity diff| 0.0000
      daily return correlation 1.0000 (n=46)
**trend**
    matched hours 981 (2026-09-14 14:00:00+00:00 .. 2026-10-25 13:00:00+00:00)
    paper  2999.29 -> 2947.01 (-1.74%)
    replay 2999.29 -> 2946.52 (-1.76%)
    max |hourly equity diff| 15.4801
    daily return correlation 0.9998 (n=41)
    legs paper/replay at end 21/21; fees 7.88/7.95; funding -8.23/-8.37
    fills replay 285
    fills matched 255, paper only 26, replay only 30
    max |qty rel diff| 7.15e-01, max |price rel diff| 0.00e+00
**Bv2 re-tuned monthly (book 8 = b2_retune)**
    book 8 (b2_retune: headroom 1.0 or 1.5 chosen monthly; live from 2026-10-25 14:00 UTC)
      replay stepped through 2026-10-28 23:00 UTC (paper's last bar 2026-10-29 01:00)
      matched hours 83, price mismatches 0
      paper  +0.14%   replay +0.14%
      max |hourly equity diff| 0.0000
      daily return correlation not computed: n=3 complete UTC days, at least 28 needed
      history before launch: 3518 hours both sides, paper +7.96% replay +7.96%, max |hourly equity diff| 0.0000 (0.00% of equity)
      history re-tunings: frab 7, replay 7, same choice 7, scores off by more than 1e-06: 0 (largest gap 0.00e+00)
      history last choice (refit 7): b2
**trend_retune_3k**
    replay of strategy 6: history 662 daily bars to 2026-10-25, replay through 2026-10-28 23:00 UTC (paper's last bar 2026-10-29 01:00)
      matched hours 84
      paper  2999.64 -> 3004.20 (+0.15%)
      replay 2999.64 -> 3004.20 (+0.15%)
      max |hourly equity diff| 0.0000
      daily return correlation not computed: n=3 complete UTC days, at least 28 needed
      legs paper/replay at end 18.0/18.0; fees 0.75/0.75; funding -0.33/-0.33
      history before launch: frab 94 weekly re-tunings, replay 94, same choice 94, scores off by more than 1e-06: 0 (largest gap 0.00e+00); last choice cap/ens
      fills matched 35, paper only 0, replay only 0; max |qty rel diff| 4.13e-13, max |price rel diff| 0.00e+00
      live bars after launch: 3; volume universes equal to frab's log 3 of 3
**Бумага и стенд за один отрезок**
    Bv2, запас 1.0: бумага за 13.09–27.10 +4.1%, повтор бумаги на стенде +4.1%; стенд за 06.05–28.10 +10.4% (от денег стратегии) — отрезки разные, числа сравнивать нельзя
    Bv2, запас 1.5: за 03.10–28.10 бумага −1.8%, стенд −2.6%, повтор бумаги на стенде −1.8% (от денег стратегии)
    Тренд: за 14.09–25.10 бумага −1.7%, стенд +3.3%, повтор бумаги на стенде −1.8%
    Bv2, запас выбирает код раз в месяц: бумага за 25.10–28.10 +0.1%, повтор бумаги на стенде +0.1%; стенд за 01.06–28.10 +8.2% (от денег стратегии) — отрезки разные, числа сравнивать нельзя
    Тренд, монеты выбирает код ($3000): бумага за 25.10–28.10 +0.2%, повтор бумаги на стенде +0.2%; стенд за 09.06.2025–28.10.2026 +7.5% — отрезки разные, числа сравнивать нельзя

# Where the strategy loses (drawdown evidence, before the cut only)

## Drawdown evidence: specs/frab-binance-spot.yaml (idea frab-binance-spot, data source hyperliquid, window 2025-01-01..2026-10-24)
До даты отреза 2026-10-10; данные после неё — для проверки исправлений, на них не смотреть.
Cut date: 2026-10-10 (params_fixed_at 2026-10-10). Everything below is computed on the data BEFORE the cut: 2025-01-01 .. 2026-10-09, 15527 periods.
The data after the cut (14 days) is held out.
- note: drawdowns and months are on the strategy's own money: the sum of its returns, depth in points of the capital it started with, not a share of the peak (docs/CAPITAL_BASE.md)

Deepest drawdowns (peak -> trough -> recovery, depth):
- 2025-10-10 -> 2026-05-17 -> not recovered by the cut, -3.6%
- 2025-04-04 -> 2025-04-23 -> 2025-06-12, -1.4%
- 2025-06-29 -> 2025-07-20 -> 2025-07-27, -1.0%

Returns by BTC market regime (the strategy's compounded return on the regime's days only; BTC's own return over those same days in brackets):
- bull: +0.6% over 128 days, worst fall -1.3% (BTC +135.3%)
- flat: -0.9% over 314 days, worst fall -4.0% (BTC -10.7%)
- bear: +2.9% over 205 days, worst fall -2.3% (BTC -58.1%)

Worst calendar months:
- 2026-03: -1.0% (744 periods)
- 2025-04: -0.8% (720 periods)
- 2026-05: -0.5% (744 periods)
- 2025-08: -0.5% (744 periods)
- 2025-12: -0.3% (744 periods)

## Drawdown evidence: specs/research/frab-binance-spot/frab-binance-spot-300.yaml (idea frab-binance-spot-300, data source hyperliquid, window 2025-01-01..2026-10-24)
До даты отреза 2026-10-10; данные после неё — для проверки исправлений, на них не смотреть.
Cut date: 2026-10-10 (params_fixed_at 2026-10-10). Everything below is computed on the data BEFORE the cut: 2025-01-01 .. 2026-10-09, 15527 periods.
The data after the cut (14 days) is held out.
- note: drawdowns and months are on the strategy's own money: the sum of its returns, depth in points of the capital it started with, not a share of the peak (docs/CAPITAL_BASE.md)

Deepest drawdowns (peak -> trough -> recovery, depth):
- 2025-04-04 -> 2026-05-17 -> not recovered by the cut, -14.8%
- 2025-01-01 -> 2025-01-01 -> 2025-01-09, -0.7%
- 2025-01-12 -> 2025-01-13 -> 2025-01-20, -0.5%

Returns by BTC market regime (the strategy's compounded return on the regime's days only; BTC's own return over those same days in brackets):
- bull: -2.5% over 128 days, worst fall -4.0% (BTC +135.3%)
- flat: -7.9% over 314 days, worst fall -9.9% (BTC -10.7%)
- bear: -0.1% over 205 days, worst fall -2.3% (BTC -58.1%)

Worst calendar months:
- 2026-03: -3.5% (744 periods)
- 2025-04: -2.8% (720 periods)
- 2025-08: -1.7% (744 periods)
- 2026-05: -1.3% (744 periods)
- 2025-12: -1.2% (744 periods)

## Drawdown evidence: specs/research/frab-binance-spot/frab-binance-spot-400.yaml (idea frab-binance-spot-400, data source hyperliquid, window 2025-01-01..2026-10-24)
До даты отреза 2026-10-10; данные после неё — для проверки исправлений, на них не смотреть.
Cut date: 2026-10-10 (params_fixed_at 2026-10-10). Everything below is computed on the data BEFORE the cut: 2025-01-01 .. 2026-10-09, 15527 periods.
The data after the cut (14 days) is held out.
- note: drawdowns and months are on the strategy's own money: the sum of its returns, depth in points of the capital it started with, not a share of the peak (docs/CAPITAL_BASE.md)

Deepest drawdowns (peak -> trough -> recovery, depth):
- 2025-04-04 -> 2026-05-17 -> not recovered by the cut, -10.4%
- 2025-01-12 -> 2025-01-13 -> 2025-01-25, -0.7%
- 2025-01-01 -> 2025-01-01 -> 2025-01-07, -0.6%

Returns by BTC market regime (the strategy's compounded return on the regime's days only; BTC's own return over those same days in brackets):
- bull: -1.4% over 128 days, worst fall -3.0% (BTC +135.3%)
- flat: -5.5% over 314 days, worst fall -7.5% (BTC -10.7%)
- bear: +0.9% over 205 days, worst fall -2.3% (BTC -58.1%)

Worst calendar months:
- 2026-03: -2.6% (744 periods)
- 2025-04: -2.0% (720 periods)
- 2025-08: -1.3% (744 periods)
- 2026-05: -1.0% (744 periods)
- 2025-11: -0.7% (720 periods)

## Drawdown evidence: specs/research/frab-binance-spot/frab-binance-spot-allbinance.yaml (idea frab-binance-spot-allbinance, data source binance, window 2026-03-08..2026-10-24)
До даты отреза 2026-10-10; данные после неё — для проверки исправлений, на них не смотреть.
Cut date: 2026-10-10 (params_fixed_at 2026-10-10). Everything below is computed on the data BEFORE the cut: 2026-03-08 .. 2026-10-09, 5183 periods.
The data after the cut (14 days) is held out.
- note: drawdowns and months are on the strategy's own money: the sum of its returns, depth in points of the capital it started with, not a share of the peak (docs/CAPITAL_BASE.md)

Deepest drawdowns (peak -> trough -> recovery, depth):
- 2026-03-24 -> 2026-08-10 -> not recovered by the cut, -13.8%
- 2026-03-24 -> 2026-03-24 -> 2026-03-24, -0.9%
- 2026-03-18 -> 2026-03-19 -> 2026-03-21, -0.7%

Returns by BTC market regime (the strategy's compounded return on the regime's days only; BTC's own return over those same days in brackets):
- bull: -4.1% over 60 days, worst fall -8.8% (BTC +43.0%)
- flat: -4.0% over 119 days, worst fall -8.2% (BTC +8.8%)
- bear: -0.2% over 37 days, worst fall -1.9% (BTC -22.9%)

Worst calendar months:
- 2026-04: -3.8% (720 periods)
- 2026-08: -3.5% (744 periods)
- 2026-03: -2.7% (575 periods)
- 2026-05: -1.9% (744 periods)
- 2026-10: +0.2% (216 periods)

## Drawdown evidence: specs/research/frab-binance-spot/frab-binance-spot-allbinance-300.yaml (idea frab-binance-spot-allbinance-300, data source binance, window 2026-03-08..2026-10-24)
До даты отреза 2026-10-10; данные после неё — для проверки исправлений, на них не смотреть.
Cut date: 2026-10-10 (params_fixed_at 2026-10-10). Everything below is computed on the data BEFORE the cut: 2026-03-08 .. 2026-10-09, 5183 periods.
The data after the cut (14 days) is held out.
- note: drawdowns and months are on the strategy's own money: the sum of its returns, depth in points of the capital it started with, not a share of the peak (docs/CAPITAL_BASE.md)

Deepest drawdowns (peak -> trough -> recovery, depth):
- 2026-03-24 -> 2026-08-10 -> not recovered by the cut, -13.8%
- 2026-03-24 -> 2026-03-24 -> 2026-03-24, -0.9%
- 2026-03-18 -> 2026-03-19 -> 2026-03-21, -0.7%

Returns by BTC market regime (the strategy's compounded return on the regime's days only; BTC's own return over those same days in brackets):
- bull: -4.1% over 60 days, worst fall -8.8% (BTC +43.0%)
- flat: -4.0% over 119 days, worst fall -8.2% (BTC +8.8%)
- bear: -0.2% over 37 days, worst fall -1.9% (BTC -22.9%)

Worst calendar months:
- 2026-04: -3.8% (720 periods)
- 2026-08: -3.5% (744 periods)
- 2026-03: -2.7% (575 periods)
- 2026-05: -1.9% (744 periods)
- 2026-10: +0.2% (216 periods)

## Drawdown evidence: specs/research/frab-binance-spot/frab-binance-spot-allbinance-400.yaml (idea frab-binance-spot-allbinance-400, data source binance, window 2026-03-08..2026-10-24)
До даты отреза 2026-10-10; данные после неё — для проверки исправлений, на них не смотреть.
Cut date: 2026-10-10 (params_fixed_at 2026-10-10). Everything below is computed on the data BEFORE the cut: 2026-03-08 .. 2026-10-09, 5183 periods.
The data after the cut (14 days) is held out.
- note: drawdowns and months are on the strategy's own money: the sum of its returns, depth in points of the capital it started with, not a share of the peak (docs/CAPITAL_BASE.md)

Deepest drawdowns (peak -> trough -> recovery, depth):
- 2026-03-24 -> 2026-08-10 -> not recovered by the cut, -13.8%
- 2026-03-24 -> 2026-03-24 -> 2026-03-24, -0.9%
- 2026-03-18 -> 2026-03-19 -> 2026-03-21, -0.7%

Returns by BTC market regime (the strategy's compounded return on the regime's days only; BTC's own return over those same days in brackets):
- bull: -4.1% over 60 days, worst fall -8.8% (BTC +43.0%)
- flat: -4.0% over 119 days, worst fall -8.2% (BTC +8.8%)
- bear: -0.2% over 37 days, worst fall -1.9% (BTC -22.9%)

Worst calendar months:
- 2026-04: -3.8% (720 periods)
- 2026-08: -3.5% (744 periods)
- 2026-03: -2.7% (575 periods)
- 2026-05: -1.9% (744 periods)
- 2026-10: +0.2% (216 periods)

## Drawdown evidence: specs/research/frab-binance-spot/frab-binance-spot-five-coins.yaml (idea frab-binance-spot-five-coins, data source hyperliquid, window 2025-01-01..2026-10-24)
До даты отреза 2026-10-10; данные после неё — для проверки исправлений, на них не смотреть.
Cut date: 2026-10-10 (params_fixed_at 2026-10-10). Everything below is computed on the data BEFORE the cut: 2025-01-01 .. 2026-10-09, 15527 periods.
The data after the cut (14 days) is held out.
- note: drawdowns and months are on the strategy's own money: the sum of its returns, depth in points of the capital it started with, not a share of the peak (docs/CAPITAL_BASE.md)

Deepest drawdowns (peak -> trough -> recovery, depth):
- 2026-03-08 -> 2026-06-30 -> 2026-10-01, -1.3%
- 2026-10-05 -> 2026-10-06 -> 2026-10-07, -0.1%
- 2026-10-07 -> 2026-10-07 -> 2026-10-07, -0.1%

Returns by BTC market regime (the strategy's compounded return on the regime's days only; BTC's own return over those same days in brackets):
- bull: +0.4% over 128 days, worst fall -0.2% (BTC +135.3%)
- flat: -0.0% over 314 days, worst fall -1.0% (BTC -10.7%)
- bear: -0.3% over 205 days, worst fall -0.3% (BTC -58.1%)

Worst calendar months:
- 2026-04: -0.4% (720 periods)
- 2026-06: -0.4% (720 periods)
- 2026-03: -0.4% (744 periods)
- 2026-05: -0.0% (744 periods)
- 2025-01: +0.0% (743 periods)

## Drawdown evidence: specs/research/frab/frab-live-own-spot.yaml (idea frab-live-own-spot, data source hyperliquid, window 2025-01-01..2026-10-24)
До даты отреза 2026-10-10; данные после неё — для проверки исправлений, на них не смотреть.
Cut date: 2026-10-10 (params_fixed_at 2026-10-10). Everything below is computed on the data BEFORE the cut: 2025-01-01 .. 2026-10-09, 15527 periods.
The data after the cut (14 days) is held out.
- note: drawdowns and months are on the strategy's own money: the sum of its returns, depth in points of the capital it started with, not a share of the peak (docs/CAPITAL_BASE.md)

Deepest drawdowns (peak -> trough -> recovery, depth):
- 2026-06-06 -> 2026-06-07 -> 2026-06-08, -5.7%
- 2026-06-02 -> 2026-06-06 -> 2026-06-06, -3.7%
- 2026-05-30 -> 2026-05-31 -> 2026-06-02, -3.2%

Returns by BTC market regime (the strategy's compounded return on the regime's days only; BTC's own return over those same days in brackets):
- bull: +1.0% over 128 days, worst fall -0.4% (BTC +135.3%)
- flat: +2.0% over 314 days, worst fall -0.5% (BTC -10.7%)
- bear: +4.6% over 205 days, worst fall -5.6% (BTC -58.1%)

Worst calendar months:
- 2025-01: +0.0% (743 periods)
- 2025-02: +0.0% (672 periods)
- 2025-03: +0.0% (744 periods)
- 2025-04: +0.0% (720 periods)
- 2025-05: +0.0% (744 periods)

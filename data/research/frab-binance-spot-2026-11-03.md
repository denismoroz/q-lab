# Evidence for the research on frab-binance-spot (built 2026-11-03)

## frab-binance-spot -- trial 193720 (2026-11-01)
route: needs-forward -- forward test too short to tell either way: 21 days; it decides at 108 days, 33% of fitting plus forward (216 days of fitting); so far it would route reject (rule(s) failed: shape_aware_edge; fatal rule: shape_aware_edge; also missing infrastructure: venue_supported, atomic_execution); and on this window the noise bar would pass a strategy of this shape truly earning 7% a year only 1% of the time (it decides from 80%) [venue_supported: venue binance has no order-placing adapter of ours (venues/binance.yaml execution_adapter: false); this is a fact about our infrastructure, not about access (docs/VENUES.md) and not about the strategy; whether the instruments exist on Hyperliquid, where orders could be placed, is not checked here; the book held 22 instrument(s): AAVE, AAVE-SPOT, ARK, ARK-SPOT, ASTER, ASTER-SPOT, BERA, BERA-SPOT and 14 more]
numbers on the strategy's own money (от денег стратегии, прибыль не вкладывается обратно, считается суммой): в год +0.1%, Шарп 0.02, просадка −0.3%, обгоняет шум 46%
(keys cap_* and noise_cap_* are on the strategy's own money -- the rules decide on them; the other money keys, ann_return_net, sharpe_net, max_dd, total_return_net, noise_return_percentile, regime_*, are the stand's share of the book's balance and are not comparable across books)
- accrual_applied: 1
- ann_return_net: 0.0002709
- ann_return_net_ex_best_1pct: 0.0002709
- atomic_execution: 0
- cap_ann_return: 0.0007681
- cap_ann_return_ex_best_1pct: 0.0007681
- cap_max_dd: -0.002764
- cap_regime_bear_ann_return: 0.005987
- cap_regime_bear_btc_return: -0.06877
- cap_regime_bear_days: 21
- cap_regime_bear_max_dd: -0.002582
- cap_regime_bear_noise_percentile: 0.46
- cap_regime_bear_return: 0.0003278
- cap_regime_bear_share: 0.9524
- cap_regime_bear_sharpe: 0.1922
- cap_regime_bull_days: 0
- cap_regime_bull_share: 0
- cap_regime_flat_ann_return: -0.1036
- cap_regime_flat_btc_return: 0.01604
- cap_regime_flat_days: 1
- cap_regime_flat_max_dd: -0.001009
- cap_regime_flat_noise_percentile: 0.39
- cap_regime_flat_return: -0.0002837
- cap_regime_flat_share: 0.04762
- cap_regime_flat_sharpe: -2.611
- cap_reinvested: 0
- cap_sharpe: 0.02433
- cap_total_return: 4.416e-05
- cap_worst_week: -0.0009776
- data_forward_available: 1
- effective_profitable_days: 165.1
- effective_profitable_days_share: 0.3275
- fit_accrual_applied: 1
- fit_ann_return_net: 0.0737
- fit_ann_return_net_ex_best_1pct: 0.06386
- fit_atomic_execution: 0
- fit_cap_ann_return: 0.07227
- fit_cap_ann_return_ex_best_1pct: 0.06303
- fit_cap_max_dd: -0.007757
- fit_cap_reinvested: 0
- fit_cap_sharpe: 1.499
- fit_cap_total_return: 0.04271
- fit_cap_worst_week: -0.002142
- fit_data_forward_available: 1
- fit_effective_profitable_days: 1420
- fit_effective_profitable_days_share: 0.2741
- fit_fragility_days: 20
- fit_fragility_share: 0.00386
- fit_history_truncated_instruments: 200
- fit_hold_ann_return: 1.188
- fit_hold_ann_return_ex_best_1pct: 0.5315
- fit_max_dd: -0.007767
- fit_min_capital_from_orders: 1
- fit_min_capital_usd: 80.3
- fit_min_capital_weights_usd: 97.2
- fit_n_periods: 5181
- fit_orders_opening: 22
- fit_orders_skipped: 0
- fit_point_in_time_universe: 1
- fit_sharpe_net: 1.499
- fit_skew: -0.1336
- fit_total_return_net: 0.04292
- fit_turnover: 11.52
- fit_venue_supported: 0
- fit_window_days: 215.9
- fit_worst_week: -0.002147
- fix_date_known: 1
- forward_days: 21.04
- forward_days_needed: 108
- fragility_days: 1
- fragility_share: 0.001984
- history_truncated_instruments: 200
- hold_ann_return: -0.9764
- hold_ann_return_ex_best_1pct: -0.972
- judged_on_forward: 1
- max_dd: -0.002769
- min_capital_from_orders: 1
- min_capital_usd: 80
- min_capital_weights_usd: 80
- n_periods: 504
- noise_cap_power: 0.01
- noise_cap_return_percentile: 0.455
- noise_n_replaced: 0
- noise_n_requested: 200
- noise_n_unusable: 0
- noise_n_usable: 200
- noise_power: 0.01
- noise_return_percentile: 0.46
- noise_sharpe_percentile: 0.46
- orders_opening: 2
- orders_skipped: 0
- point_in_time_universe: 1
- regime_bear_ann_return: 0.005518
- regime_bear_below: -0.03791
- regime_bear_btc_return: -0.06877
- regime_bear_days: 21
- regime_bear_max_dd: -0.002583
- regime_bear_noise_percentile: 0.48
- regime_bear_return: 0.0003013
- regime_bear_share: 0.9524
- regime_bear_sharpe: 0.1922
- regime_bull_above: 0.09281
- regime_bull_days: 0
- regime_bull_share: 0
- regime_flat_ann_return: -0.0991
- regime_flat_btc_return: 0.01604
- regime_flat_days: 1
- regime_flat_max_dd: -0.001009
- regime_flat_noise_percentile: 0.39
- regime_flat_return: -0.0002857
- regime_flat_share: 0.04762
- regime_flat_sharpe: -2.611
- selection_days: 215.9
- selection_rules_failed: 2
- sharpe_net: 0.02433
- skew: -0.4439
- total_return_net: 1.557e-05
- turnover: 41.22
- venue_supported: 0
- worst_week: -0.0009887
selection-period regime breakdown: cap_regime_bear_ann_return=0.04761, cap_regime_bear_btc_return=-0.214, cap_regime_bear_days=36, cap_regime_bear_max_dd=-0.003599, cap_regime_bear_return=0.004692, cap_regime_bear_share=0.1668, cap_regime_bear_sharpe=1.019, cap_regime_bull_ann_return=0.07261, cap_regime_bull_btc_return=0.4303, cap_regime_bull_days=60, cap_regime_bull_max_dd=-0.007757, cap_regime_bull_return=0.01193, cap_regime_bull_share=0.2779, cap_regime_bull_sharpe=1.349, cap_regime_flat_ann_return=0.07951, cap_regime_flat_btc_return=0.06759, cap_regime_flat_days=120, cap_regime_flat_max_dd=-0.006411, cap_regime_flat_return=0.02609, cap_regime_flat_share=0.5553, cap_regime_flat_sharpe=1.743, regime_bear_ann_return=0.04762, regime_bear_below=-0.03791, regime_bear_btc_return=-0.214, regime_bear_days=36, regime_bear_max_dd=-0.003602, regime_bear_return=0.004595, regime_bear_share=0.1668, regime_bear_sharpe=1.019, regime_bull_above=0.09281, regime_bull_ann_return=0.07376, regime_bull_btc_return=0.4303, regime_bull_days=60, regime_bull_max_dd=-0.007767, regime_bull_return=0.01176, regime_bull_share=0.2779, regime_bull_sharpe=1.349, regime_flat_ann_return=0.08162, regime_flat_btc_return=0.06759, regime_flat_days=120, regime_flat_max_dd=-0.006429, regime_flat_return=0.02609, regime_flat_share=0.5553, regime_flat_sharpe=1.743

## frab-binance-spot-300 -- trial 181808 (2026-10-30)
route: needs-forward -- forward test too short to tell either way: 17 days; it decides at 108 days, 33% of fitting plus forward (216 days of fitting); so far it would route reject (rule(s) failed: net_edge_positive, shape_aware_edge; fatal rule: net_edge_positive; also missing infrastructure: venue_supported, atomic_execution); and on this window the noise bar would pass a strategy of this shape truly earning 7% a year only 2% of the time (it decides from 80%) [venue_supported: venue binance has no order-placing adapter of ours (venues/binance.yaml execution_adapter: false); this is a fact about our infrastructure, not about access (docs/VENUES.md) and not about the strategy; whether the instruments exist on Hyperliquid, where orders could be placed, is not checked here; the book held 22 instrument(s): AAVE, AAVE-SPOT, ARK, ARK-SPOT, ASTER, ASTER-SPOT, BERA, BERA-SPOT and 14 more]
numbers on the strategy's own money (от денег стратегии, прибыль не вкладывается обратно, считается суммой): в год −5.5%, Шарп -1.52, просадка −0.5%, обгоняет шум 40%; как считает стенд (от баланса книги): в год −5.4%, Шарп -1.52, просадка −0.5%
(keys cap_* and noise_cap_* are on the strategy's own money -- the rules decide on them; the other money keys, ann_return_net, sharpe_net, max_dd, total_return_net, noise_return_percentile, regime_*, are the stand's share of the book's balance and are not comparable across books)
- accrual_applied: 1
- ann_return_net: -0.05439
- ann_return_net_ex_best_1pct: -0.05439
- atomic_execution: 0
- cap_ann_return: -0.05527
- cap_ann_return_ex_best_1pct: -0.05527
- cap_max_dd: -0.005068
- cap_regime_bear_ann_return: -0.05213
- cap_regime_bear_btc_return: -0.035
- cap_regime_bear_days: 17
- cap_regime_bear_max_dd: -0.004917
- cap_regime_bear_noise_percentile: 0.39
- cap_regime_bear_return: -0.002283
- cap_regime_bear_share: 0.9412
- cap_regime_bear_sharpe: -1.447
- cap_regime_bull_days: 0
- cap_regime_bull_share: 0
- cap_regime_flat_ann_return: -0.1056
- cap_regime_flat_btc_return: 0.01604
- cap_regime_flat_days: 1
- cap_regime_flat_max_dd: -0.001011
- cap_regime_flat_noise_percentile: 0.35
- cap_regime_flat_return: -0.0002891
- cap_regime_flat_share: 0.05882
- cap_regime_flat_sharpe: -2.572
- cap_reinvested: 0
- cap_sharpe: -1.523
- cap_total_return: -0.002573
- cap_worst_week: -0.00312
- data_forward_available: 1
- effective_profitable_days: 133.3
- effective_profitable_days_share: 0.3267
- fit_accrual_applied: 1
- fit_ann_return_net: 0.03834
- fit_ann_return_net_ex_best_1pct: 0.02853
- fit_atomic_execution: 0
- fit_cap_ann_return: 0.03888
- fit_cap_ann_return_ex_best_1pct: 0.02935
- fit_cap_max_dd: -0.01941
- fit_cap_reinvested: 0
- fit_cap_sharpe: 0.777
- fit_cap_total_return: 0.02298
- fit_cap_worst_week: -0.00763
- fit_data_forward_available: 1
- fit_effective_profitable_days: 1421
- fit_effective_profitable_days_share: 0.2742
- fit_fragility_days: 9
- fit_fragility_share: 0.001737
- fit_history_truncated_instruments: 200
- fit_hold_ann_return: 1.188
- fit_hold_ann_return_ex_best_1pct: 0.5315
- fit_max_dd: -0.01942
- fit_min_capital_from_orders: 1
- fit_min_capital_usd: 80.3
- fit_min_capital_weights_usd: 97.2
- fit_n_periods: 5181
- fit_orders_opening: 22
- fit_orders_skipped: 0
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
- min_capital_from_orders: 1
- min_capital_usd: 80
- min_capital_weights_usd: 80
- n_periods: 408
- noise_cap_power: 0.015
- noise_cap_return_percentile: 0.395
- noise_n_replaced: 0
- noise_n_requested: 200
- noise_n_unusable: 0
- noise_n_usable: 200
- noise_power: 0.01
- noise_return_percentile: 0.395
- noise_sharpe_percentile: 0.35
- orders_opening: 2
- orders_skipped: 0
- point_in_time_universe: 1
- regime_bear_ann_return: -0.05141
- regime_bear_below: -0.03788
- regime_bear_btc_return: -0.035
- regime_bear_days: 17
- regime_bear_max_dd: -0.004916
- regime_bear_noise_percentile: 0.39
- regime_bear_return: -0.002309
- regime_bear_share: 0.9412
- regime_bear_sharpe: -1.447
- regime_bull_above: 0.09288
- regime_bull_days: 0
- regime_bull_share: 0
- regime_flat_ann_return: -0.1009
- regime_flat_btc_return: 0.01604
- regime_flat_days: 1
- regime_flat_max_dd: -0.001012
- regime_flat_noise_percentile: 0.35
- regime_flat_return: -0.0002913
- regime_flat_share: 0.05882
- regime_flat_sharpe: -2.572
- selection_days: 215.9
- selection_rules_failed: 2
- sharpe_net: -1.523
- skew: -2.872
- total_return_net: -0.0026
- turnover: 48.61
- venue_supported: 0
- worst_week: -0.003134
selection-period regime breakdown: cap_regime_bear_ann_return=0.03177, cap_regime_bear_btc_return=-0.2288, cap_regime_bear_days=37, cap_regime_bear_max_dd=-0.004532, cap_regime_bear_return=0.003219, cap_regime_bear_share=0.1714, cap_regime_bear_sharpe=0.6794, cap_regime_bull_ann_return=0.06693, cap_regime_bull_btc_return=0.4303, cap_regime_bull_days=60, cap_regime_bull_max_dd=-0.007757, cap_regime_bull_return=0.01099, cap_regime_bull_share=0.2779, cap_regime_bull_sharpe=1.24, cap_regime_flat_ann_return=0.02693, cap_regime_flat_btc_return=0.08812, cap_regime_flat_days=119, cap_regime_flat_max_dd=-0.01849, cap_regime_flat_return=0.008764, cap_regime_flat_share=0.5507, cap_regime_flat_sharpe=0.55, regime_bear_ann_return=0.03115, regime_bear_below=-0.03788, regime_bear_btc_return=-0.2288, regime_bear_days=37, regime_bear_max_dd=-0.004533, regime_bear_return=0.003113, regime_bear_share=0.1714, regime_bear_sharpe=0.6794, regime_bull_above=0.09288, regime_bull_ann_return=0.06766, regime_bull_btc_return=0.4303, regime_bull_days=60, regime_bull_max_dd=-0.007767, regime_bull_return=0.01081, regime_bull_share=0.2779, regime_bull_sharpe=1.24, regime_flat_ann_return=0.02606, regime_flat_btc_return=0.08812, regime_flat_days=119, regime_flat_max_dd=-0.01848, regime_flat_return=0.008409, regime_flat_share=0.5507, regime_flat_sharpe=0.55

## frab-binance-spot-400 -- trial 104180 (2026-10-28)
route: needs-forward -- forward test too short to tell either way: 17 days; it decides at 108 days, 33% of fitting plus forward (216 days of fitting); so far it would route needs-more-data (metric(s) could not be computed: atomic_execution) [venue_supported: venue binance has no order-placing adapter of ours (venues/binance.yaml execution_adapter: false); this is a fact about our infrastructure, not about access (docs/VENUES.md) and not about the strategy; whether the instruments exist on Hyperliquid, where orders could be placed, is not checked here; the book held 22 instrument(s): AAVE, AAVE-SPOT, ARK, ARK-SPOT, ASTER, ASTER-SPOT, BERA, BERA-SPOT and 14 more]
numbers on the strategy's own money (по счёту стенда (у прогона нет чисел на деньгах стратегии)): в год −3.7%, Шарп -1.08, просадка −0.4%, обгоняет шум 40%
(keys cap_* and noise_cap_* are on the strategy's own money -- the rules decide on them; the other money keys, ann_return_net, sharpe_net, max_dd, total_return_net, noise_return_percentile, regime_*, are the stand's share of the book's balance and are not comparable across books)
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

## frab-binance-spot-allbinance -- trial 196275 (2026-11-01)
route: reject -- fails on the selection period, the data its parameters were chosen on and which flatters it: net_edge_positive [venue_supported: venue binance has no order-placing adapter of ours (venues/binance.yaml execution_adapter: false); this is a fact about our infrastructure, not about access (docs/VENUES.md) and not about the strategy; whether the instruments exist on Hyperliquid, where orders could be placed, is not checked here; the book held 92 instrument(s): 0G, 0G-SPOT, 1000FLOKI, 1000FLOKI-SPOT, 1000PEPE, 1000PEPE-SPOT, 1000XEC, 1000XEC-SPOT and 84 more]
numbers on the strategy's own money (от денег стратегии, прибыль не вкладывается обратно, считается суммой): в год +5.0%, Шарп 0.86, просадка −0.3%, обгоняет шум 68%
(keys cap_* and noise_cap_* are on the strategy's own money -- the rules decide on them; the other money keys, ann_return_net, sharpe_net, max_dd, total_return_net, noise_return_percentile, regime_*, are the stand's share of the book's balance and are not comparable across books)
- accrual_applied: 1
- ann_return_net: 0.04905
- ann_return_net_ex_best_1pct: 0.04905
- atomic_execution: 0
- cap_ann_return: 0.04953
- cap_ann_return_ex_best_1pct: 0.04953
- cap_max_dd: -0.002696
- cap_regime_bear_ann_return: 0.03524
- cap_regime_bear_btc_return: -0.06877
- cap_regime_bear_days: 21
- cap_regime_bear_max_dd: -0.002696
- cap_regime_bear_noise_percentile: 0.68
- cap_regime_bear_return: 0.00193
- cap_regime_bear_share: 0.9524
- cap_regime_bear_sharpe: 0.6116
- cap_regime_bull_days: 0
- cap_regime_bull_share: 0
- cap_regime_flat_ann_return: 0.3352
- cap_regime_flat_btc_return: 0.01604
- cap_regime_flat_days: 1
- cap_regime_flat_max_dd: -0.001552
- cap_regime_flat_noise_percentile: 0.635
- cap_regime_flat_return: 0.0009178
- cap_regime_flat_share: 0.04762
- cap_regime_flat_sharpe: 6.592
- cap_reinvested: 0
- cap_sharpe: 0.8647
- cap_total_return: 0.002847
- cap_worst_week: -0.001608
- data_forward_available: 1
- effective_profitable_days: 155.6
- effective_profitable_days_share: 0.3088
- fit_accrual_applied: 1
- fit_ann_return_net: -0.1333
- fit_ann_return_net_ex_best_1pct: -0.1847
- fit_atomic_execution: 0
- fit_cap_ann_return: -0.1221
- fit_cap_ann_return_ex_best_1pct: -0.1846
- fit_cap_max_dd: -0.1377
- fit_cap_reinvested: 0
- fit_cap_sharpe: -0.6013
- fit_cap_total_return: -0.07222
- fit_cap_worst_week: -0.07003
- fit_data_forward_available: 1
- fit_effective_profitable_days: 1026
- fit_effective_profitable_days_share: 0.1979
- fit_fragility_days: 0
- fit_fragility_share: 0
- fit_hold_ann_return: -0.1595
- fit_hold_ann_return_ex_best_1pct: -0.3355
- fit_max_dd: -0.1377
- fit_min_capital_from_orders: 1
- fit_min_capital_usd: 23.25
- fit_min_capital_weights_usd: 87.76
- fit_n_periods: 5183
- fit_orders_opening: 212
- fit_orders_skipped: 0
- fit_point_in_time_universe: 1
- fit_sharpe_net: -0.6013
- fit_skew: -8.741
- fit_total_return_net: -0.0811
- fit_turnover: 155
- fit_venue_supported: 0
- fit_window_days: 216
- fit_worst_week: -0.07136
- fix_date_known: 1
- forward_days: 21.04
- forward_days_needed: 108
- fragility_days: 2
- fragility_share: 0.003968
- hold_ann_return: -0.8022
- hold_ann_return_ex_best_1pct: -0.7871
- judged_on_forward: 1
- max_dd: -0.002697
- min_capital_from_orders: 1
- min_capital_usd: 23.02
- min_capital_weights_usd: 29.03
- n_periods: 504
- noise_cap_power: 0.01
- noise_cap_return_percentile: 0.685
- noise_n_replaced: 0
- noise_n_requested: 200
- noise_n_unusable: 0
- noise_n_usable: 200
- noise_power: 0.01
- noise_return_percentile: 0.695
- noise_sharpe_percentile: 0.715
- orders_opening: 2
- orders_skipped: 0
- point_in_time_universe: 1
- regime_bear_ann_return: 0.03415
- regime_bear_below: -0.03791
- regime_bear_btc_return: -0.06877
- regime_bear_days: 21
- regime_bear_max_dd: -0.002697
- regime_bear_noise_percentile: 0.68
- regime_bear_return: 0.001841
- regime_bear_share: 0.9524
- regime_bear_sharpe: 0.6116
- regime_bull_above: 0.09281
- regime_bull_days: 0
- regime_bull_share: 0
- regime_flat_ann_return: 0.3965
- regime_flat_btc_return: 0.01604
- regime_flat_days: 1
- regime_flat_max_dd: -0.001551
- regime_flat_noise_percentile: 0.635
- regime_flat_return: 0.0009149
- regime_flat_share: 0.04762
- regime_flat_sharpe: 6.592
- selection_days: 216
- selection_rules_failed: 3
- sharpe_net: 0.8647
- skew: -0.1406
- total_return_net: 0.002757
- turnover: 56.95
- venue_supported: 0
- worst_week: -0.001617
selection-period regime breakdown: cap_regime_bear_ann_return=-0.01351, cap_regime_bear_btc_return=-0.214, cap_regime_bear_days=36, cap_regime_bear_max_dd=-0.01886, cap_regime_bear_return=-0.001331, cap_regime_bear_share=0.1667, cap_regime_bear_sharpe=-0.122, cap_regime_bull_ann_return=-0.2376, cap_regime_bull_btc_return=0.4303, cap_regime_bull_days=60, cap_regime_bull_max_dd=-0.09112, cap_regime_bull_return=-0.03904, cap_regime_bull_share=0.2778, cap_regime_bull_sharpe=-1.26, cap_regime_flat_ann_return=-0.09697, cap_regime_flat_btc_return=0.06759, cap_regime_flat_days=120, cap_regime_flat_max_dd=-0.08248, cap_regime_flat_return=-0.03185, cap_regime_flat_share=0.5555, cap_regime_flat_sharpe=-0.4219, regime_bear_ann_return=-0.01944, regime_bear_below=-0.03791, regime_bear_btc_return=-0.214, regime_bear_days=36, regime_bear_max_dd=-0.01879, regime_bear_return=-0.001933, regime_bear_share=0.1667, regime_bear_sharpe=-0.122, regime_bull_above=0.09281, regime_bull_ann_return=-0.2254, regime_bull_btc_return=0.4303, regime_bull_days=60, regime_bull_max_dd=-0.08815, regime_bull_return=-0.04109, regime_bull_share=0.2778, regime_bull_sharpe=-1.26, regime_flat_ann_return=-0.1165, regime_flat_btc_return=0.06759, regime_flat_days=120, regime_flat_max_dd=-0.08222, regime_flat_return=-0.03987, regime_flat_share=0.5555, regime_flat_sharpe=-0.4219

## frab-binance-spot-allbinance-300 -- trial 195873 (2026-11-01)
route: reject -- fails on the selection period, the data its parameters were chosen on and which flatters it: net_edge_positive [venue_supported: venue binance has no order-placing adapter of ours (venues/binance.yaml execution_adapter: false); this is a fact about our infrastructure, not about access (docs/VENUES.md) and not about the strategy; whether the instruments exist on Hyperliquid, where orders could be placed, is not checked here; the book held 92 instrument(s): 0G, 0G-SPOT, 1000FLOKI, 1000FLOKI-SPOT, 1000PEPE, 1000PEPE-SPOT, 1000XEC, 1000XEC-SPOT and 84 more]
numbers on the strategy's own money (от денег стратегии, прибыль не вкладывается обратно, считается суммой): в год +5.0%, Шарп 0.86, просадка −0.3%, обгоняет шум 68%
(keys cap_* and noise_cap_* are on the strategy's own money -- the rules decide on them; the other money keys, ann_return_net, sharpe_net, max_dd, total_return_net, noise_return_percentile, regime_*, are the stand's share of the book's balance and are not comparable across books)
- accrual_applied: 1
- ann_return_net: 0.04905
- ann_return_net_ex_best_1pct: 0.04905
- atomic_execution: 0
- cap_ann_return: 0.04953
- cap_ann_return_ex_best_1pct: 0.04953
- cap_max_dd: -0.002696
- cap_regime_bear_ann_return: 0.03524
- cap_regime_bear_btc_return: -0.06877
- cap_regime_bear_days: 21
- cap_regime_bear_max_dd: -0.002696
- cap_regime_bear_noise_percentile: 0.68
- cap_regime_bear_return: 0.00193
- cap_regime_bear_share: 0.9524
- cap_regime_bear_sharpe: 0.6116
- cap_regime_bull_days: 0
- cap_regime_bull_share: 0
- cap_regime_flat_ann_return: 0.3352
- cap_regime_flat_btc_return: 0.01604
- cap_regime_flat_days: 1
- cap_regime_flat_max_dd: -0.001552
- cap_regime_flat_noise_percentile: 0.635
- cap_regime_flat_return: 0.0009178
- cap_regime_flat_share: 0.04762
- cap_regime_flat_sharpe: 6.592
- cap_reinvested: 0
- cap_sharpe: 0.8647
- cap_total_return: 0.002847
- cap_worst_week: -0.001608
- data_forward_available: 1
- effective_profitable_days: 155.6
- effective_profitable_days_share: 0.3088
- fit_accrual_applied: 1
- fit_ann_return_net: -0.1333
- fit_ann_return_net_ex_best_1pct: -0.1847
- fit_atomic_execution: 0
- fit_cap_ann_return: -0.1221
- fit_cap_ann_return_ex_best_1pct: -0.1846
- fit_cap_max_dd: -0.1377
- fit_cap_reinvested: 0
- fit_cap_sharpe: -0.6013
- fit_cap_total_return: -0.07222
- fit_cap_worst_week: -0.07003
- fit_data_forward_available: 1
- fit_effective_profitable_days: 1026
- fit_effective_profitable_days_share: 0.1979
- fit_fragility_days: 0
- fit_fragility_share: 0
- fit_hold_ann_return: -0.1595
- fit_hold_ann_return_ex_best_1pct: -0.3355
- fit_max_dd: -0.1377
- fit_min_capital_from_orders: 1
- fit_min_capital_usd: 23.25
- fit_min_capital_weights_usd: 87.76
- fit_n_periods: 5183
- fit_orders_opening: 212
- fit_orders_skipped: 0
- fit_point_in_time_universe: 1
- fit_sharpe_net: -0.6013
- fit_skew: -8.741
- fit_total_return_net: -0.0811
- fit_turnover: 155
- fit_venue_supported: 0
- fit_window_days: 216
- fit_worst_week: -0.07136
- fix_date_known: 1
- forward_days: 21.04
- forward_days_needed: 108
- fragility_days: 2
- fragility_share: 0.003968
- hold_ann_return: -0.8022
- hold_ann_return_ex_best_1pct: -0.7871
- judged_on_forward: 1
- max_dd: -0.002697
- min_capital_from_orders: 1
- min_capital_usd: 23.02
- min_capital_weights_usd: 29.03
- n_periods: 504
- noise_cap_power: 0.01
- noise_cap_return_percentile: 0.685
- noise_n_replaced: 0
- noise_n_requested: 200
- noise_n_unusable: 0
- noise_n_usable: 200
- noise_power: 0.01
- noise_return_percentile: 0.695
- noise_sharpe_percentile: 0.715
- orders_opening: 2
- orders_skipped: 0
- point_in_time_universe: 1
- regime_bear_ann_return: 0.03415
- regime_bear_below: -0.03791
- regime_bear_btc_return: -0.06877
- regime_bear_days: 21
- regime_bear_max_dd: -0.002697
- regime_bear_noise_percentile: 0.68
- regime_bear_return: 0.001841
- regime_bear_share: 0.9524
- regime_bear_sharpe: 0.6116
- regime_bull_above: 0.09281
- regime_bull_days: 0
- regime_bull_share: 0
- regime_flat_ann_return: 0.3965
- regime_flat_btc_return: 0.01604
- regime_flat_days: 1
- regime_flat_max_dd: -0.001551
- regime_flat_noise_percentile: 0.635
- regime_flat_return: 0.0009149
- regime_flat_share: 0.04762
- regime_flat_sharpe: 6.592
- selection_days: 216
- selection_rules_failed: 3
- sharpe_net: 0.8647
- skew: -0.1406
- total_return_net: 0.002757
- turnover: 56.95
- venue_supported: 0
- worst_week: -0.001617
selection-period regime breakdown: cap_regime_bear_ann_return=-0.01351, cap_regime_bear_btc_return=-0.214, cap_regime_bear_days=36, cap_regime_bear_max_dd=-0.01886, cap_regime_bear_return=-0.001331, cap_regime_bear_share=0.1667, cap_regime_bear_sharpe=-0.122, cap_regime_bull_ann_return=-0.2376, cap_regime_bull_btc_return=0.4303, cap_regime_bull_days=60, cap_regime_bull_max_dd=-0.09112, cap_regime_bull_return=-0.03904, cap_regime_bull_share=0.2778, cap_regime_bull_sharpe=-1.26, cap_regime_flat_ann_return=-0.09697, cap_regime_flat_btc_return=0.06759, cap_regime_flat_days=120, cap_regime_flat_max_dd=-0.08248, cap_regime_flat_return=-0.03185, cap_regime_flat_share=0.5555, cap_regime_flat_sharpe=-0.4219, regime_bear_ann_return=-0.01944, regime_bear_below=-0.03791, regime_bear_btc_return=-0.214, regime_bear_days=36, regime_bear_max_dd=-0.01879, regime_bear_return=-0.001933, regime_bear_share=0.1667, regime_bear_sharpe=-0.122, regime_bull_above=0.09281, regime_bull_ann_return=-0.2254, regime_bull_btc_return=0.4303, regime_bull_days=60, regime_bull_max_dd=-0.08815, regime_bull_return=-0.04109, regime_bull_share=0.2778, regime_bull_sharpe=-1.26, regime_flat_ann_return=-0.1165, regime_flat_btc_return=0.06759, regime_flat_days=120, regime_flat_max_dd=-0.08222, regime_flat_return=-0.03987, regime_flat_share=0.5555, regime_flat_sharpe=-0.4219

## frab-binance-spot-allbinance-400 -- trial 196074 (2026-11-01)
route: reject -- fails on the selection period, the data its parameters were chosen on and which flatters it: net_edge_positive [venue_supported: venue binance has no order-placing adapter of ours (venues/binance.yaml execution_adapter: false); this is a fact about our infrastructure, not about access (docs/VENUES.md) and not about the strategy; whether the instruments exist on Hyperliquid, where orders could be placed, is not checked here; the book held 92 instrument(s): 0G, 0G-SPOT, 1000FLOKI, 1000FLOKI-SPOT, 1000PEPE, 1000PEPE-SPOT, 1000XEC, 1000XEC-SPOT and 84 more]
numbers on the strategy's own money (от денег стратегии, прибыль не вкладывается обратно, считается суммой): в год +5.0%, Шарп 0.86, просадка −0.3%, обгоняет шум 68%
(keys cap_* and noise_cap_* are on the strategy's own money -- the rules decide on them; the other money keys, ann_return_net, sharpe_net, max_dd, total_return_net, noise_return_percentile, regime_*, are the stand's share of the book's balance and are not comparable across books)
- accrual_applied: 1
- ann_return_net: 0.04905
- ann_return_net_ex_best_1pct: 0.04905
- atomic_execution: 0
- cap_ann_return: 0.04953
- cap_ann_return_ex_best_1pct: 0.04953
- cap_max_dd: -0.002696
- cap_regime_bear_ann_return: 0.03524
- cap_regime_bear_btc_return: -0.06877
- cap_regime_bear_days: 21
- cap_regime_bear_max_dd: -0.002696
- cap_regime_bear_noise_percentile: 0.68
- cap_regime_bear_return: 0.00193
- cap_regime_bear_share: 0.9524
- cap_regime_bear_sharpe: 0.6116
- cap_regime_bull_days: 0
- cap_regime_bull_share: 0
- cap_regime_flat_ann_return: 0.3352
- cap_regime_flat_btc_return: 0.01604
- cap_regime_flat_days: 1
- cap_regime_flat_max_dd: -0.001552
- cap_regime_flat_noise_percentile: 0.635
- cap_regime_flat_return: 0.0009178
- cap_regime_flat_share: 0.04762
- cap_regime_flat_sharpe: 6.592
- cap_reinvested: 0
- cap_sharpe: 0.8647
- cap_total_return: 0.002847
- cap_worst_week: -0.001608
- data_forward_available: 1
- effective_profitable_days: 155.6
- effective_profitable_days_share: 0.3088
- fit_accrual_applied: 1
- fit_ann_return_net: -0.1333
- fit_ann_return_net_ex_best_1pct: -0.1847
- fit_atomic_execution: 0
- fit_cap_ann_return: -0.1221
- fit_cap_ann_return_ex_best_1pct: -0.1846
- fit_cap_max_dd: -0.1377
- fit_cap_reinvested: 0
- fit_cap_sharpe: -0.6013
- fit_cap_total_return: -0.07222
- fit_cap_worst_week: -0.07003
- fit_data_forward_available: 1
- fit_effective_profitable_days: 1026
- fit_effective_profitable_days_share: 0.1979
- fit_fragility_days: 0
- fit_fragility_share: 0
- fit_hold_ann_return: -0.1595
- fit_hold_ann_return_ex_best_1pct: -0.3355
- fit_max_dd: -0.1377
- fit_min_capital_from_orders: 1
- fit_min_capital_usd: 23.25
- fit_min_capital_weights_usd: 87.76
- fit_n_periods: 5183
- fit_orders_opening: 212
- fit_orders_skipped: 0
- fit_point_in_time_universe: 1
- fit_sharpe_net: -0.6013
- fit_skew: -8.741
- fit_total_return_net: -0.0811
- fit_turnover: 155
- fit_venue_supported: 0
- fit_window_days: 216
- fit_worst_week: -0.07136
- fix_date_known: 1
- forward_days: 21.04
- forward_days_needed: 108
- fragility_days: 2
- fragility_share: 0.003968
- hold_ann_return: -0.8022
- hold_ann_return_ex_best_1pct: -0.7871
- judged_on_forward: 1
- max_dd: -0.002697
- min_capital_from_orders: 1
- min_capital_usd: 23.02
- min_capital_weights_usd: 29.03
- n_periods: 504
- noise_cap_power: 0.01
- noise_cap_return_percentile: 0.685
- noise_n_replaced: 0
- noise_n_requested: 200
- noise_n_unusable: 0
- noise_n_usable: 200
- noise_power: 0.01
- noise_return_percentile: 0.695
- noise_sharpe_percentile: 0.715
- orders_opening: 2
- orders_skipped: 0
- point_in_time_universe: 1
- regime_bear_ann_return: 0.03415
- regime_bear_below: -0.03791
- regime_bear_btc_return: -0.06877
- regime_bear_days: 21
- regime_bear_max_dd: -0.002697
- regime_bear_noise_percentile: 0.68
- regime_bear_return: 0.001841
- regime_bear_share: 0.9524
- regime_bear_sharpe: 0.6116
- regime_bull_above: 0.09281
- regime_bull_days: 0
- regime_bull_share: 0
- regime_flat_ann_return: 0.3965
- regime_flat_btc_return: 0.01604
- regime_flat_days: 1
- regime_flat_max_dd: -0.001551
- regime_flat_noise_percentile: 0.635
- regime_flat_return: 0.0009149
- regime_flat_share: 0.04762
- regime_flat_sharpe: 6.592
- selection_days: 216
- selection_rules_failed: 3
- sharpe_net: 0.8647
- skew: -0.1406
- total_return_net: 0.002757
- turnover: 56.95
- venue_supported: 0
- worst_week: -0.001617
selection-period regime breakdown: cap_regime_bear_ann_return=-0.01351, cap_regime_bear_btc_return=-0.214, cap_regime_bear_days=36, cap_regime_bear_max_dd=-0.01886, cap_regime_bear_return=-0.001331, cap_regime_bear_share=0.1667, cap_regime_bear_sharpe=-0.122, cap_regime_bull_ann_return=-0.2376, cap_regime_bull_btc_return=0.4303, cap_regime_bull_days=60, cap_regime_bull_max_dd=-0.09112, cap_regime_bull_return=-0.03904, cap_regime_bull_share=0.2778, cap_regime_bull_sharpe=-1.26, cap_regime_flat_ann_return=-0.09697, cap_regime_flat_btc_return=0.06759, cap_regime_flat_days=120, cap_regime_flat_max_dd=-0.08248, cap_regime_flat_return=-0.03185, cap_regime_flat_share=0.5555, cap_regime_flat_sharpe=-0.4219, regime_bear_ann_return=-0.01944, regime_bear_below=-0.03791, regime_bear_btc_return=-0.214, regime_bear_days=36, regime_bear_max_dd=-0.01879, regime_bear_return=-0.001933, regime_bear_share=0.1667, regime_bear_sharpe=-0.122, regime_bull_above=0.09281, regime_bull_ann_return=-0.2254, regime_bull_btc_return=0.4303, regime_bull_days=60, regime_bull_max_dd=-0.08815, regime_bull_return=-0.04109, regime_bull_share=0.2778, regime_bull_sharpe=-1.26, regime_flat_ann_return=-0.1165, regime_flat_btc_return=0.06759, regime_flat_days=120, regime_flat_max_dd=-0.08222, regime_flat_return=-0.03987, regime_flat_share=0.5555, regime_flat_sharpe=-0.4219

## frab-binance-spot-five-coins -- trial 196476 (2026-11-01)
route: needs-forward -- forward test too short to tell either way: 21 days; it decides at 108 days, 33% of fitting plus forward (216 days of fitting); so far it would route reject (rule(s) failed: shape_aware_edge; fatal rule: shape_aware_edge; also missing infrastructure: venue_supported, atomic_execution); and on this window the noise bar would pass a strategy of this shape truly earning 7% a year only 4% of the time (it decides from 80%) [venue_supported: venue binance has no order-placing adapter of ours (venues/binance.yaml execution_adapter: false); this is a fact about our infrastructure, not about access (docs/VENUES.md) and not about the strategy; whether the instruments exist on Hyperliquid, where orders could be placed, is not checked here; the book held 8 instrument(s): AVAX, AVAX-SPOT, BTC, BTC-SPOT, ETH, ETH-SPOT, SOL, SOL-SPOT]
numbers on the strategy's own money (от денег стратегии, прибыль не вкладывается обратно, считается суммой): в год +3.3%, Шарп 2.09, просадка −0.2%, обгоняет шум 70%; как считает стенд (от баланса книги): в год +3.4%, Шарп 2.09, просадка −0.1%
(keys cap_* and noise_cap_* are on the strategy's own money -- the rules decide on them; the other money keys, ann_return_net, sharpe_net, max_dd, total_return_net, noise_return_percentile, regime_*, are the stand's share of the book's balance and are not comparable across books)
- accrual_applied: 1
- ann_return_net: 0.03375
- ann_return_net_ex_best_1pct: 0.03375
- atomic_execution: 0
- cap_ann_return: 0.03332
- cap_ann_return_ex_best_1pct: 0.03332
- cap_max_dd: -0.001859
- cap_regime_bear_ann_return: 0.03669
- cap_regime_bear_btc_return: -0.06877
- cap_regime_bear_days: 21
- cap_regime_bear_max_dd: -0.001859
- cap_regime_bear_noise_percentile: 0.7
- cap_regime_bear_return: 0.002009
- cap_regime_bear_share: 0.9524
- cap_regime_bear_sharpe: 2.324
- cap_regime_bull_days: 0
- cap_regime_bull_share: 0
- cap_regime_flat_ann_return: -0.03399
- cap_regime_flat_btc_return: 0.01604
- cap_regime_flat_days: 1
- cap_regime_flat_max_dd: -0.0006594
- cap_regime_flat_noise_percentile: 0.42
- cap_regime_flat_return: -9.307e-05
- cap_regime_flat_share: 0.04762
- cap_regime_flat_sharpe: -1.796
- cap_reinvested: 0
- cap_sharpe: 2.092
- cap_total_return: 0.001916
- cap_worst_week: -0.001608
- data_forward_available: 1
- effective_profitable_days: 165.2
- effective_profitable_days_share: 0.3278
- fit_accrual_applied: 1
- fit_ann_return_net: 0.002563
- fit_ann_return_net_ex_best_1pct: -0.0004128
- fit_atomic_execution: 0
- fit_cap_ann_return: 0.002645
- fit_cap_ann_return_ex_best_1pct: -0.0003282
- fit_cap_max_dd: -0.01327
- fit_cap_reinvested: 0
- fit_cap_sharpe: 0.2024
- fit_cap_total_return: 0.001563
- fit_cap_worst_week: -0.001925
- fit_data_forward_available: 1
- fit_effective_profitable_days: 1495
- fit_effective_profitable_days_share: 0.2885
- fit_fragility_days: 2
- fit_fragility_share: 0.000386
- fit_history_truncated_instruments: 200
- fit_hold_ann_return: 0.5045
- fit_hold_ann_return_ex_best_1pct: 0.1172
- fit_max_dd: -0.01321
- fit_min_capital_from_orders: 1
- fit_min_capital_usd: 80.05
- fit_min_capital_weights_usd: 105.8
- fit_n_periods: 5181
- fit_orders_opening: 40
- fit_orders_skipped: 0
- fit_point_in_time_universe: 1
- fit_sharpe_net: 0.2024
- fit_skew: -1.289
- fit_total_return_net: 0.001514
- fit_turnover: 27.47
- fit_venue_supported: 0
- fit_window_days: 215.9
- fit_worst_week: -0.001924
- fix_date_known: 1
- forward_days: 21.04
- forward_days_needed: 108
- fragility_days: 5
- fragility_share: 0.009921
- history_truncated_instruments: 200
- hold_ann_return: -0.8665
- hold_ann_return_ex_best_1pct: -0.8537
- judged_on_forward: 1
- max_dd: -0.0009268
- min_capital_from_orders: 1
- min_capital_usd: 80.05
- min_capital_weights_usd: 54.01
- n_periods: 504
- noise_cap_power: 0.035
- noise_cap_return_percentile: 0.705
- noise_n_replaced: 0
- noise_n_requested: 200
- noise_n_unusable: 0
- noise_n_usable: 200
- noise_power: 0.015
- noise_return_percentile: 0.705
- noise_sharpe_percentile: 0.725
- orders_opening: 0
- orders_skipped: 0
- point_in_time_universe: 1
- regime_bear_ann_return: 0.03724
- regime_bear_below: -0.03791
- regime_bear_btc_return: -0.06877
- regime_bear_days: 21
- regime_bear_max_dd: -0.0009268
- regime_bear_noise_percentile: 0.7
- regime_bear_return: 0.002004
- regime_bear_share: 0.9524
- regime_bear_sharpe: 2.324
- regime_bull_above: 0.09281
- regime_bull_days: 0
- regime_bull_share: 0
- regime_flat_ann_return: -0.03359
- regime_flat_btc_return: 0.01604
- regime_flat_days: 1
- regime_flat_max_dd: -0.0006593
- regime_flat_noise_percentile: 0.42
- regime_flat_return: -9.353e-05
- regime_flat_share: 0.04762
- regime_flat_sharpe: -1.796
- selection_days: 215.9
- selection_rules_failed: 2
- sharpe_net: 2.092
- skew: -1.641
- total_return_net: 0.00191
- turnover: 32.76
- venue_supported: 0
- worst_week: -0.001608
selection-period regime breakdown: cap_regime_bear_ann_return=-0.02331, cap_regime_bear_btc_return=-0.214, cap_regime_bear_days=36, cap_regime_bear_max_dd=-0.002878, cap_regime_bear_return=-0.002298, cap_regime_bear_share=0.1668, cap_regime_bear_sharpe=-1.89, cap_regime_bull_ann_return=0.02602, cap_regime_bull_btc_return=0.4303, cap_regime_bull_days=60, cap_regime_bull_max_dd=-0.002055, cap_regime_bull_return=0.004274, cap_regime_bull_share=0.2779, cap_regime_bull_sharpe=1.946, cap_regime_flat_ann_return=-0.001258, cap_regime_flat_btc_return=0.06759, cap_regime_flat_days=120, cap_regime_flat_max_dd=-0.009852, cap_regime_flat_return=-0.0004127, cap_regime_flat_share=0.5553, cap_regime_flat_sharpe=-0.09582, regime_bear_ann_return=-0.02312, regime_bear_below=-0.03791, regime_bear_btc_return=-0.214, regime_bear_days=36, regime_bear_max_dd=-0.002881, regime_bear_return=-0.002303, regime_bear_share=0.1668, regime_bear_sharpe=-1.89, regime_bull_above=0.09281, regime_bull_ann_return=0.02627, regime_bull_btc_return=0.4303, regime_bull_days=60, regime_bull_max_dd=-0.002054, regime_bull_return=0.004268, regime_bull_share=0.2779, regime_bull_sharpe=1.946, regime_flat_ann_return=-0.001343, regime_flat_btc_return=0.06759, regime_flat_days=120, regime_flat_max_dd=-0.009817, regime_flat_return=-0.0004409, regime_flat_share=0.5553, regime_flat_sharpe=-0.09582

## frab-live-own-spot -- trial 181205 (2026-10-30)
route: needs-forward -- forward test too short to tell either way: 17 days; it decides at 108 days, 33% of fitting plus forward (216 days of fitting); so far it would route reject (rule(s) failed: shape_aware_edge; fatal rule: shape_aware_edge); and on this window the noise bar would pass a strategy of this shape truly earning 7% a year only 4% of the time (it decides from 80%)
numbers on the strategy's own money (от денег стратегии, прибыль не вкладывается обратно, считается суммой): в год +3.8%, Шарп 1.25, просадка −0.2%, обгоняет шум 56%
(keys cap_* and noise_cap_* are on the strategy's own money -- the rules decide on them; the other money keys, ann_return_net, sharpe_net, max_dd, total_return_net, noise_return_percentile, regime_*, are the stand's share of the book's balance and are not comparable across books)
- accrual_applied: 1
- ann_return_net: 0.03871
- ann_return_net_ex_best_1pct: 0.03871
- atomic_execution: 1
- cap_ann_return: 0.03846
- cap_ann_return_ex_best_1pct: 0.03846
- cap_max_dd: -0.001594
- cap_regime_bear_ann_return: 0.03274
- cap_regime_bear_btc_return: -0.035
- cap_regime_bear_days: 17
- cap_regime_bear_max_dd: -0.001594
- cap_regime_bear_noise_percentile: 0.615
- cap_regime_bear_return: 0.001434
- cap_regime_bear_share: 0.9412
- cap_regime_bear_sharpe: 1.056
- cap_regime_bull_days: 0
- cap_regime_bull_share: 0
- cap_regime_flat_ann_return: 0.1299
- cap_regime_flat_btc_return: 0.01604
- cap_regime_flat_days: 1
- cap_regime_flat_max_dd: -0.000876
- cap_regime_flat_noise_percentile: 0.465
- cap_regime_flat_return: 0.0003556
- cap_regime_flat_share: 0.05882
- cap_regime_flat_sharpe: 4.668
- cap_reinvested: 0
- cap_sharpe: 1.248
- cap_total_return: 0.00179
- cap_worst_week: -0.00124
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
- fit_cap_worst_week: -0.001678
- fit_data_forward_available: 1
- fit_effective_profitable_days: 313.8
- fit_effective_profitable_days_share: 0.06056
- fit_fragility_days: 2
- fit_fragility_share: 0.000386
- fit_history_truncated_instruments: 200
- fit_hold_ann_return: 1.715
- fit_hold_ann_return_ex_best_1pct: 0.855
- fit_max_dd: -0.05627
- fit_min_capital_from_orders: 1
- fit_min_capital_usd: 80.09
- fit_min_capital_weights_usd: 92.4
- fit_n_periods: 5181
- fit_orders_opening: 36
- fit_orders_skipped: 0
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
- min_capital_from_orders: 1
- min_capital_usd: 80.09
- min_capital_weights_usd: 92.4
- n_periods: 408
- noise_cap_power: 0.035
- noise_cap_return_percentile: 0.555
- noise_n_replaced: 0
- noise_n_requested: 200
- noise_n_unusable: 0
- noise_n_usable: 200
- noise_power: 0.02
- noise_return_percentile: 0.58
- noise_sharpe_percentile: 0.825
- orders_opening: 0
- orders_skipped: 0
- point_in_time_universe: 1
- regime_bear_ann_return: 0.03279
- regime_bear_below: -0.03788
- regime_bear_btc_return: -0.035
- regime_bear_days: 17
- regime_bear_max_dd: -0.001594
- regime_bear_noise_percentile: 0.645
- regime_bear_return: 0.001414
- regime_bear_share: 0.9412
- regime_bear_sharpe: 1.056
- regime_bull_above: 0.09288
- regime_bull_days: 0
- regime_bull_share: 0
- regime_flat_ann_return: 0.1383
- regime_flat_btc_return: 0.01604
- regime_flat_days: 1
- regime_flat_max_dd: -0.0008761
- regime_flat_noise_percentile: 0.47
- regime_flat_return: 0.0003546
- regime_flat_share: 0.05882
- regime_flat_sharpe: 4.668
- selection_days: 215.9
- selection_rules_failed: 0
- sharpe_net: 1.248
- skew: 0.07293
- total_return_net: 0.001769
- turnover: 35.97
- venue_supported: 1
- worst_week: -0.001242
selection-period regime breakdown: cap_regime_bear_ann_return=0.5123, cap_regime_bear_btc_return=-0.2288, cap_regime_bear_days=37, cap_regime_bear_max_dd=-0.05704, cap_regime_bear_return=0.05189, cap_regime_bear_share=0.1714, cap_regime_bear_sharpe=1.359, cap_regime_bull_ann_return=0.06002, cap_regime_bull_btc_return=0.4303, cap_regime_bull_days=60, cap_regime_bull_max_dd=-0.004293, cap_regime_bull_return=0.00986, cap_regime_bull_share=0.2779, cap_regime_bull_sharpe=1.028, cap_regime_flat_ann_return=0.06272, cap_regime_flat_btc_return=0.08812, cap_regime_flat_days=119, cap_regime_flat_max_dd=-0.004733, cap_regime_flat_return=0.02041, cap_regime_flat_share=0.5507, cap_regime_flat_sharpe=1.091, regime_bear_ann_return=0.5554, regime_bear_below=-0.03788, regime_bear_btc_return=-0.2288, regime_bear_days=37, regime_bear_max_dd=-0.05627, regime_bear_return=0.04576, regime_bear_share=0.1714, regime_bear_sharpe=1.359, regime_bull_above=0.09288, regime_bull_ann_return=0.06005, regime_bull_btc_return=0.4303, regime_bull_days=60, regime_bull_max_dd=-0.00429, regime_bull_return=0.009626, regime_bull_share=0.2779, regime_bull_sharpe=1.028, regime_flat_ann_return=0.06297, regime_flat_btc_return=0.08812, regime_flat_days=119, regime_flat_max_dd=-0.004752, regime_flat_return=0.02007, regime_flat_share=0.5507, regime_flat_sharpe=1.091

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
- bull: +0.7% over 129 days, worst fall -1.3% (BTC +123.2%)
- flat: -1.0% over 315 days, worst fall -4.0% (BTC -9.9%)
- bear: +2.9% over 203 days, worst fall -2.3% (BTC -56.2%)

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
- bull: -2.4% over 129 days, worst fall -4.0% (BTC +123.2%)
- flat: -8.1% over 315 days, worst fall -9.9% (BTC -9.9%)
- bear: -0.0% over 203 days, worst fall -2.3% (BTC -56.2%)

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
- bull: -1.4% over 129 days, worst fall -3.0% (BTC +123.2%)
- flat: -5.6% over 315 days, worst fall -7.5% (BTC -9.9%)
- bear: +1.0% over 203 days, worst fall -2.3% (BTC -56.2%)

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
- flat: -4.0% over 120 days, worst fall -8.2% (BTC +6.8%)
- bear: -0.2% over 36 days, worst fall -1.9% (BTC -21.4%)

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
- flat: -4.0% over 120 days, worst fall -8.2% (BTC +6.8%)
- bear: -0.2% over 36 days, worst fall -1.9% (BTC -21.4%)

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
- flat: -4.0% over 120 days, worst fall -8.2% (BTC +6.8%)
- bear: -0.2% over 36 days, worst fall -1.9% (BTC -21.4%)

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
- bull: +0.4% over 129 days, worst fall -0.2% (BTC +123.2%)
- flat: -0.0% over 315 days, worst fall -1.0% (BTC -9.9%)
- bear: -0.2% over 203 days, worst fall -0.3% (BTC -56.2%)

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
- bull: +1.0% over 129 days, worst fall -0.4% (BTC +123.2%)
- flat: +2.0% over 315 days, worst fall -0.5% (BTC -9.9%)
- bear: +4.6% over 203 days, worst fall -5.6% (BTC -56.2%)

Worst calendar months:
- 2025-01: +0.0% (743 periods)
- 2025-02: +0.0% (672 periods)
- 2025-03: +0.0% (744 periods)
- 2025-04: +0.0% (720 periods)
- 2025-05: +0.0% (744 periods)

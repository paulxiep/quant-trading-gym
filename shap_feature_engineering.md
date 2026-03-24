# V6.2 SHAP Feature Analysis & Trimming

## Dataset

- 6 training runs × ~40K rows = 240K total training rows
- 5000 ticks per run, 1000 warmup
- 55 V6.1 features across 8 groups
- 5 tree-based models analyzed

## Model Baseline Accuracy (55 features)

| Model | Type | Accuracy | SHAP Method |
|-------|------|----------|-------------|
| medium_decision_tree | DT (depth 12) | ~51-63% | SHAP TreeExplainer |
| deep_decision_tree | DT (depth 16) | ~51-63% | SHAP TreeExplainer |
| small_random_forest | RF (24 trees, depth 12) | ~64.5% | SHAP TreeExplainer |
| fast_gradient_boosted | GB (24 trees, depth 8, lr 0.4) | ~71-85% | sklearn feature_importances |
| slow_gradient_boosted | GB (36 trees, depth 10, lr 0.25) | ~71-85% | sklearn feature_importances |

Note: GB models use sklearn Gini importance because SHAP TreeExplainer doesn't support multi-class GradientBoostingClassifier. Importance is normalized before cross-model averaging, making Gini and SHAP comparable.

## Group-Level Importance

| Group | #Feat | DT deep | GB fast | DT med | GB slow | RF small |
|-------|-------|---------|---------|--------|---------|----------|
| Price | 25 | 33.7% | 37.4% | 29.2% | 40.1% | 29.9% |
| Technical | 13 | 32.1% | 31.6% | 31.7% | 30.7% | 28.2% |
| Volatility | 3 | 12.1% | 14.0% | 12.1% | 13.7% | 12.7% |
| Fundamental | 2 | 16.0% | 10.2% | 20.8% | 7.9% | 24.2% |
| MomentumQuality | 2 | 3.5% | 3.9% | 3.6% | 4.4% | 2.6% |
| VolumeCross | 3 | 1.5% | 1.1% | 1.6% | 1.4% | 1.0% |
| Microstructure | 3 | 1.0% | 1.0% | 0.7% | 1.2% | 1.1% |
| News | 4 | 0.2% | 0.8% | 0.3% | 0.6% | 0.3% |

**Key observations:**
- Price + Technical dominate (~60-70% combined), consistent across all model types
- Fundamental is disproportionately important per-feature (2 features, 8-24%)
- Volatility (3 features, 12-14%) is highly efficient — all 3 features appear in every model's top-20
- News (<1%), Microstructure (~1%), VolumeCross (~1.3%) contribute almost nothing
- Price has 25 features but most importance concentrates in ~8 of them

## Cross-Model Feature Consensus (All 55 Features)

Importance normalized per model (divide by sum), then averaged across 5 models. Top20 = how many models had the feature in their top-20.

| Rank | Feature | Group | AvgNorm | Top20 |
|------|---------|-------|---------|-------|
| 1 | f_fair_value_dev | Fundamental | 0.1000 | 5/5 |
| 2 | f_price_to_fair | Fundamental | 0.0583 | 5/5 |
| 3 | f_realized_vol_32 | Volatility | 0.0539 | 5/5 |
| 4 | f_realized_vol_8 | Volatility | 0.0517 | 5/5 |
| 5 | f_bb_lower | Technical | 0.0436 | 5/5 |
| 6 | f_mid_price | Price | 0.0402 | 5/5 |
| 7 | f_atr_8 | Technical | 0.0331 | 5/5 |
| 8 | f_log_return_64 | Price | 0.0328 | 5/5 |
| 9 | f_price_change_1 | Price | 0.0305 | 3/5 |
| 10 | f_macd_histogram | Technical | 0.0293 | 5/5 |
| 11 | f_sma_16 | Technical | 0.0270 | 5/5 |
| 12 | f_ema_16 | Technical | 0.0266 | 4/5 |
| 13 | f_bb_upper | Technical | 0.0253 | 5/5 |
| 14 | f_price_change_64 | Price | 0.0246 | 5/5 |
| 15 | f_vol_ratio | Volatility | 0.0234 | 4/5 |
| 16 | f_macd_line | Technical | 0.0232 | 3/5 |
| 17 | f_log_return_48 | Price | 0.0209 | 4/5 |
| 18 | f_bb_percent_b | Technical | 0.0207 | 4/5 |
| 19 | f_trend_strength | MomentumQuality | 0.0202 | 4/5 |
| 20 | f_price_change_48 | Price | 0.0193 | 3/5 |
| 21 | f_rsi_8 | Technical | 0.0189 | 3/5 |
| **--- trim line (keep above, drop below) ---** | | | | |
| 22 | f_macd_signal | Technical | 0.0158 | 2/5 |
| 23 | f_ema_8 | Technical | 0.0157 | 0/5 |
| 24 | f_rsi_divergence | MomentumQuality | 0.0156 | 0/5 |
| 25 | f_price_change_32 | Price | 0.0156 | 1/5 |
| 26 | f_log_return_32 | Price | 0.0150 | 2/5 |
| 27 | f_sma_8 | Technical | 0.0150 | 1/5 |
| 28 | f_bb_middle | Technical | 0.0145 | 1/5 |
| **--- below here: all 0/5 or 1/5, excluded ---** | | | | |
| 29 | f_price_change_16 | Price | 0.0127 | 0/5 |
| 30 | f_log_return_1 | Price | 0.0122 | 1/5 |
| 31 | f_log_return_24 | Price | 0.0120 | 0/5 |
| 32 | f_price_change_24 | Price | 0.0117 | 0/5 |
| 33 | f_volume_surge | VolumeCross | 0.0116 | 0/5 |
| 34 | f_log_return_12 | Price | 0.0116 | 0/5 |
| 35 | f_log_return_16 | Price | 0.0109 | 0/5 |
| 36 | f_price_change_12 | Price | 0.0105 | 0/5 |
| 37 | f_net_order_flow | Microstructure | 0.0099 | 0/5 |
| 38 | f_price_change_8 | Price | 0.0093 | 0/5 |
| 39 | f_log_return_6 | Price | 0.0086 | 0/5 |
| 40 | f_price_change_6 | Price | 0.0085 | 0/5 |
| 41 | f_log_return_8 | Price | 0.0080 | 0/5 |
| 42 | f_price_change_4 | Price | 0.0055 | 0/5 |
| 43 | f_log_return_4 | Price | 0.0045 | 0/5 |
| 44 | f_log_return_3 | Price | 0.0040 | 0/5 |
| 45 | f_log_return_2 | Price | 0.0040 | 0/5 |
| 46 | f_price_change_3 | Price | 0.0040 | 0/5 |
| 47 | f_price_change_2 | Price | 0.0037 | 0/5 |
| 48 | f_news_sentiment | News | 0.0022 | 0/5 |
| 49 | f_news_magnitude | News | 0.0016 | 0/5 |
| 50 | f_sentiment_price_gap | VolumeCross | 0.0015 | 0/5 |
| 51 | f_news_ticks_remaining | News | 0.0008 | 0/5 |
| 52 | f_has_active_news | News | 0.0001 | 0/5 |
| 53 | f_spread_bps | Microstructure | 0.0000 | 0/5 |
| 54 | f_book_imbalance | Microstructure | 0.0000 | 0/5 |
| 55 | f_trade_intensity | VolumeCross | 0.0000 | 0/5 |

## Trimming Decision: 55 → 28 Features

### What we drop (27 features)

**Whole groups (10 features):**
- News (4): `f_has_active_news`, `f_news_sentiment`, `f_news_magnitude`, `f_news_ticks_remaining` — <1% across all models
- Microstructure (3): `f_spread_bps`, `f_book_imbalance`, `f_net_order_flow` — ~1% across all models
- VolumeCross (3): `f_volume_surge`, `f_trade_intensity`, `f_sentiment_price_gap` — ~1.3% across all models

**Individual Price features (17 features):**
- Short/mid horizon price changes that never reach any model's top-20: `f_price_change_{2,3,4,6,8,12,16,24}` (8 features)
- Short/mid horizon log returns: `f_log_return_{1,2,3,4,6,8,12,16,24}` (9 features)
- Note: `f_price_change_1` (rank 9, 3/5) is kept while `f_log_return_1` (rank 30, 1/5) is dropped — they're near-identical (`log(1+r) ≈ r` for small r)

### What we keep (28 features)

| Group | #Kept | Features |
|-------|-------|----------|
| Fundamental | 2 | f_fair_value_dev, f_price_to_fair |
| Volatility | 3 | f_realized_vol_32, f_realized_vol_8, f_vol_ratio |
| Technical | 13 | f_bb_lower, f_atr_8, f_macd_histogram, f_sma_16, f_ema_16, f_bb_upper, f_macd_line, f_bb_percent_b, f_rsi_8, f_macd_signal, f_ema_8, f_sma_8, f_bb_middle |
| Price | 8 | f_mid_price, f_log_return_64, f_price_change_1, f_price_change_64, f_log_return_48, f_price_change_48, f_price_change_32, f_log_return_32 |
| MomentumQuality | 2 | f_trend_strength, f_rsi_divergence |

### Rationale

- **Conservative cut**: all 27 dropped features are 0/5 top-20 across models (one exception: `f_log_return_1` at 1/5)
- **Price redundancy**: `price_change_X` and `log_return_X` are mathematically near-identical at same horizon. Short horizons (2-24) have neither variant in any model's top-20. Long horizons (32, 48, 64) keep both variants since trees can exploit the small nonlinearity
- **Groups removed entirely**: News/Microstructure/VolumeCross contribute <1% each. News features being near-zero may reflect simulation characteristics (rare events, already captured by price impact)
- **All Technical features kept**: even the lower-ranked ones (ema_8, sma_8, bb_middle) rank above the cut line

### Notes on News features

News contributing almost nothing may be a simulation artifact — news events are rare and their price impact is already captured by price/volatility features. In a real market data setting, news features would likely be more important. Keeping them in the Rust feature extractor for future re-evaluation.

---

## V6.2 Retrained Results (28 Features)

### Accuracy Comparison

Retrained all 5 models on 28 SHAP-validated features. No signal loss — some models slightly improve due to reduced noise.

| Model | Type | 55-feat | 28-feat | Delta |
|-------|------|---------|---------|-------|
| medium_decision_tree | DT (depth 12) | ~51% | 51.12% | ~0% |
| deep_decision_tree | DT (depth 16) | ~63% | 63.13% | ~0% |
| small_random_forest | RF (24 trees, depth 12) | ~64.5% | 65.12% | +0.6% |
| fast_gradient_boosted | GB (24 trees, depth 8, lr 0.4) | ~71% | 72.24% | +1.2% |
| slow_gradient_boosted | GB (36 trees, depth 10, lr 0.25) | ~85% | 84.48% | -0.5% |

### Group Importance (28 features, 5 groups)

| Group | #Feat | DT deep | GB fast | DT med | GB slow | RF small |
|-------|-------|---------|---------|--------|---------|----------|
| Technical | 13 | 37.0% | 37.7% | 37.5% | 37.9% | 34.2% |
| Price | 8 | 24.0% | 26.2% | 20.8% | 27.2% | 20.0% |
| Fundamental | 2 | 18.7% | 12.2% | 24.6% | 9.8% | 29.4% |
| Volatility | 3 | 15.9% | 19.2% | 13.0% | 20.1% | 13.4% |
| MomentumQuality | 2 | 4.4% | 4.7% | 4.1% | 5.0% | 3.0% |

### Cross-Model Feature Consensus (28 Features)

| Rank | Feature | Group | AvgNorm | Top20 |
|------|---------|-------|---------|-------|
| 1 | f_fair_value_dev | Fundamental | 0.1168 | 5/5 |
| 2 | f_price_to_fair | Fundamental | 0.0680 | 5/5 |
| 3 | f_realized_vol_32 | Volatility | 0.0623 | 5/5 |
| 4 | f_realized_vol_8 | Volatility | 0.0598 | 5/5 |
| 5 | f_bb_lower | Technical | 0.0513 | 5/5 |
| 6 | f_mid_price | Price | 0.0470 | 5/5 |
| 7 | f_atr_8 | Technical | 0.0390 | 5/5 |
| 8 | f_log_return_64 | Price | 0.0382 | 5/5 |
| 9 | f_price_change_1 | Price | 0.0355 | 5/5 |
| 10 | f_macd_histogram | Technical | 0.0343 | 5/5 |
| 11 | f_sma_16 | Technical | 0.0316 | 5/5 |
| 12 | f_ema_16 | Technical | 0.0311 | 5/5 |
| 13 | f_bb_upper | Technical | 0.0295 | 5/5 |
| 14 | f_price_change_64 | Price | 0.0287 | 5/5 |
| 15 | f_vol_ratio | Volatility | 0.0273 | 5/5 |
| 16 | f_macd_line | Technical | 0.0271 | 5/5 |
| 17 | f_log_return_48 | Price | 0.0244 | 5/5 |
| 18 | f_bb_percent_b | Technical | 0.0243 | 5/5 |
| 19 | f_trend_strength | MomentumQuality | 0.0237 | 5/5 |
| 20 | f_price_change_48 | Price | 0.0225 | 5/5 |
| 21 | f_rsi_8 | Technical | 0.0221 | 5/5 |
| 22 | f_macd_signal | Technical | 0.0185 | 3/5 |
| 23 | f_ema_8 | Technical | 0.0183 | 2/5 |
| 24 | f_rsi_divergence | MomentumQuality | 0.0183 | 2/5 |
| 25 | f_price_change_32 | Price | 0.0182 | 2/5 |
| 26 | f_log_return_32 | Price | 0.0176 | 2/5 |
| 27 | f_sma_8 | Technical | 0.0175 | 1/5 |
| 28 | f_bb_middle | Technical | 0.0169 | 1/5 |

**Key observations (28 features):**
- All 28 features contribute meaningfully — no zero-importance features remain
- Top 21 features are 5/5 consensus (up from top 14 at 55 features)
- Per-feature importance increases ~17% (1/28 vs 1/55 baseline) as noise features removed
- Fundamental remains disproportionately important: 2 features, ~19% importance
- Volatility efficiency confirmed: 3 features, ~16% importance

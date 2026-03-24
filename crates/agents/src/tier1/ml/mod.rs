//! ML model inference and prediction-reading agents (V5.5, V6.2).
//!
//! This module provides:
//! - Rust inference for trained sklearn models (trees, linear, NB, ensemble)
//! - [`MlAgent`] — reads cached predictions by model name, generates orders
//! - [`ModelRegistry`] — centralized prediction computation
//! - [`FeatureExtractor`] trait — swappable feature extraction (V5/V6)
//!
//! All models implement the [`MlModel`] trait and produce class probabilities
//! for 3-class classification: sell (-1), hold (0), buy (1).
//!
//! # Architecture (V5.6+)
//!
//! ```text
//! Runner -> extract features -> cache -> ModelRegistry.predict_from_cache()
//!   -> cache.insert_prediction(model_name, symbol, probs)
//!   -> MlAgent reads from cache -> threshold logic -> orders
//! ```
//!
//! # Usage
//!
//! ```ignore
//! use agents::tier1::ml::{MlAgent, MlAgentConfig};
//!
//! let agent = MlAgent::new(
//!     AgentId(100),
//!     "RandomForest_small".to_string(),
//!     MlAgentConfig {
//!         symbols: vec!["ACME".into()],
//!         buy_threshold: 0.55,
//!         sell_threshold: 0.55,
//!         ..Default::default()
//!     },
//! );
//! ```

mod canonical_features;
mod decision_tree;
mod ensemble_model;
mod feature_extractor;
mod full_features;
mod gaussian_nb;
mod gradient_boosted;
pub mod group_extractors;
mod linear_predictor;
mod ml_agent;
mod model_registry;
mod random_forest;

pub use canonical_features::CanonicalFeatures;
pub use decision_tree::DecisionTree;
pub use ensemble_model::EnsembleModel;
#[allow(deprecated)]
pub use feature_extractor::{MinimalFeatures, extract_features, extract_features_raw};
pub use full_features::FullFeatures;
pub use gaussian_nb::GaussianNBPredictor;
pub use gradient_boosted::GradientBoosted;
pub use linear_predictor::LinearPredictor;
pub use ml_agent::{MlAgent, MlAgentConfig};
pub use model_registry::ModelRegistry;
pub use random_forest::RandomForest;

/// Class probabilities: [p_sell, p_hold, p_buy] for classes [-1, 0, 1].
pub type ClassProbabilities = [f64; 3];

/// Trait for ML models that produce class probabilities.
///
/// Implementors must be `Send + Sync` for parallel agent execution.
pub trait MlModel: Send + Sync {
    /// Predict class probabilities from market features.
    ///
    /// # Arguments
    /// * `features` - Slice of features in extraction order (28 for V6.3 canonical, 42 for V5, 55 for V6.1)
    ///
    /// # Returns
    /// `[p_sell, p_hold, p_buy]` probabilities that sum to 1.0.
    /// Returns `[0.0, 1.0, 0.0]` (hold) if prediction fails.
    fn predict(&self, features: &[f64]) -> ClassProbabilities;

    /// Model name for logging, cache keys, and agent lookup.
    fn name(&self) -> &str;

    /// Number of features expected by this model.
    fn n_features(&self) -> usize {
        42
    }
}

/// Trait for extracting market-level features from simulation state.
///
/// Implementors produce a feature vector from `StrategyContext` for a given symbol.
/// Market features are shared across agents (cacheable). Per-agent features
/// (portfolio state) are NOT part of this trait — they are computed locally.
///
/// # Feature Prefix Invariant
///
/// All `FeatureExtractor` implementations MUST preserve the V5 feature ordering
/// as a prefix. That is, `features[0..42]` must correspond to `MARKET_FEATURE_NAMES`
/// in order. New features are appended after index 41. This invariant enables
/// mixed V5/V6 models in the same ensemble, where each sub-model receives
/// `features[..model.n_features()]`. Enforced by tests in the types crate.
///
/// # Pipeline
///
/// Extraction is **pure** — `extract_market()` returns raw features with NaN
/// for missing values. Imputation is a separate step using `neutral_values()`.
/// The runner applies imputation before caching:
///
/// ```ignore
/// let raw = extractor.extract_market(symbol, ctx);
/// let neutrals = extractor.neutral_values();
/// let imputed: FeatureVec = raw.iter().zip(neutrals).map(|(f, n)| {
///     if f.is_nan() { *n } else { *f }
/// }).collect();
/// cache.insert_features(symbol, imputed);
/// ```
pub trait FeatureExtractor: Send + Sync {
    /// Number of features this extractor produces.
    fn n_features(&self) -> usize;

    /// Extract raw market features for a symbol. NaN values preserved.
    fn extract_market(
        &self,
        symbol: &types::Symbol,
        ctx: &crate::StrategyContext<'_>,
    ) -> crate::ml_cache::FeatureVec;

    /// Feature names in extraction order (for Parquet schema, logging).
    fn feature_names(&self) -> &[&str];

    /// Per-feature neutral values for NaN imputation.
    ///
    /// Length must equal `n_features()`. Each value is the "no signal" default
    /// for that feature when data is missing (e.g. RSI → 50, vol_ratio → 1.0).
    fn neutral_values(&self) -> &[f64];

    /// Feature registry providing metadata (groups, valid ranges, descriptors).
    ///
    /// Used by downstream consumers: V6.2 SHAP analysis (group names),
    /// V6.3 gym (observation space bounds), V7.2 deep RL (normalization ranges).
    fn registry(&self) -> &'static types::FeatureRegistry;
}

/// Apply per-feature NaN imputation using neutral values from an extractor.
///
/// Replaces NaN values in `features` with the corresponding neutral value.
/// This is the single imputation point in the pipeline — called by the runner
/// after extraction and before cache insertion.
#[inline]
pub fn impute_features(features: &mut crate::ml_cache::FeatureVec, neutrals: &[f64]) {
    features.iter_mut().zip(neutrals).for_each(|(f, n)| {
        if f.is_nan() {
            *f = *n;
        }
    });
}

/// Compute mapping from model feature positions to full-vector positions.
///
/// Returns `None` if no remapping is needed (model uses 28 canonical, 42 market,
/// or 55 full features in standard order). Returns `Some(indices)` when the model
/// was trained on a feature subset, where `indices[model_pos] = full_vector_pos`.
pub(crate) fn compute_feature_indices(
    model_feature_names: &[String],
    n_features: usize,
) -> Option<Vec<usize>> {
    // V6.3 canonical: 28 features = canonical order, no remap
    if n_features == types::N_CANONICAL_FEATURES {
        return None;
    }
    // V5 backward compat: 42 features = market prefix, no remap
    if n_features == types::N_MARKET_FEATURES {
        return None;
    }
    // V6.1 full: 55 features = full order, no remap
    if n_features == types::N_FULL_FEATURES {
        return None;
    }

    // Build name→index lookup for FULL_FEATURE_NAMES
    let full_idx: std::collections::HashMap<&str, usize> = types::FULL_FEATURE_NAMES
        .iter()
        .enumerate()
        .map(|(i, &name)| (name, i))
        .collect();

    let indices: Vec<usize> = model_feature_names
        .iter()
        .map(|name| {
            *full_idx
                .get(name.as_str())
                .unwrap_or_else(|| panic!("Unknown feature '{}' in model", name))
        })
        .collect();

    Some(indices)
}

/// Remap a full feature vector to model feature order using precomputed indices.
///
/// If `indices` is `None`, returns `features` unchanged. Otherwise, extracts
/// the relevant features into `buffer` and returns a reference to it.
#[inline]
pub(crate) fn remap_features<'a>(
    features: &'a [f64],
    indices: &Option<Vec<usize>>,
    buffer: &'a mut Vec<f64>,
) -> &'a [f64] {
    match indices {
        None => features,
        Some(idx) => {
            buffer.clear();
            buffer.extend(
                idx.iter()
                    .map(|&i| features.get(i).copied().unwrap_or(f64::NAN)),
            );
            buffer
        }
    }
}

/// Compute numerically stable softmax (zero-alloc, 3-class).
///
/// Subtracts max value before exponentiation to prevent overflow.
/// Uses inline `[f64; 3]` instead of Vec for zero heap allocation.
#[inline]
pub(crate) fn softmax(scores: &[f64]) -> ClassProbabilities {
    let max_score = scores[0].max(scores[1]).max(scores[2]);
    let e = [
        (scores[0] - max_score).exp(),
        (scores[1] - max_score).exp(),
        (scores[2] - max_score).exp(),
    ];
    let sum = e[0] + e[1] + e[2];
    if sum > 0.0 && sum.is_finite() {
        [e[0] / sum, e[1] / sum, e[2] / sum]
    } else {
        // Fallback to hold if softmax fails
        [0.0, 1.0, 0.0]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_softmax_basic() {
        let scores = [1.0, 2.0, 3.0];
        let probs = softmax(&scores);

        // Should sum to 1
        let sum: f64 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10);

        // Higher score should have higher probability
        assert!(probs[2] > probs[1]);
        assert!(probs[1] > probs[0]);
    }

    #[test]
    fn test_softmax_numerical_stability() {
        // Large scores that would overflow naive exp()
        let scores = [1000.0, 1001.0, 1002.0];
        let probs = softmax(&scores);

        // Should still sum to 1 and not be NaN/Inf
        let sum: f64 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10);
        assert!(probs.iter().all(|&p| p.is_finite()));
    }

    #[test]
    fn test_softmax_equal_scores() {
        let scores = [0.0, 0.0, 0.0];
        let probs = softmax(&scores);

        // Equal scores should give ~equal probabilities
        for p in &probs {
            assert!((*p - 1.0 / 3.0).abs() < 1e-10);
        }
    }
}

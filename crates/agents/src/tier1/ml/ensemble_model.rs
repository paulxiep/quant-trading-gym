//! Ensemble model for classification (V6.2).
//!
//! Composes any mix of `MlModel` implementations via weighted probability
//! averaging. Registered in `ModelRegistry` like any other model — predictions
//! land in `MlPredictionCache` under the ensemble's name.
//!
//! # Architecture
//!
//! The ensemble is a MODEL, not an AGENT. `MlAgent` reads its cached predictions
//! identically to how it reads any other model. This avoids duplicating
//! threshold/order logic and keeps predictions accessible to other consumers
//! (recording hooks, gym observations, SHAP analysis).
//!
//! # Mixed Feature Counts
//!
//! Sub-models can have different `n_features()` values (e.g., V5 models with 42
//! features, V6 models with 55). Each sub-model receives `features[..n_features()]`.
//! This works because V5 features are a prefix of V6 features (Feature Prefix Invariant).
//! `EnsembleModel::n_features()` returns the max, ensuring the cache provides enough.
//!
//! # V7.1 Weight Learning
//!
//! The `weights()` accessor exposes current weights for inspection. V7 can
//! reconstruct the ensemble with optimized weights and re-register via
//! `ModelRegistry.register()` (which overwrites by name).

use std::sync::Arc;

use super::{ClassProbabilities, MlModel};

/// Weighted ensemble of ML models.
///
/// Produces class probabilities by weighted averaging of sub-model predictions.
/// All sub-models must implement `MlModel`.
pub struct EnsembleModel {
    /// Ensemble name (e.g., "Ensemble_v6").
    name: String,
    /// Sub-models (shared via Arc for zero-copy).
    models: Vec<Arc<dyn MlModel>>,
    /// Per-model weights (must be > 0).
    weights: Vec<f64>,
    /// Max n_features across all sub-models.
    n_features: usize,
}

impl std::fmt::Debug for EnsembleModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EnsembleModel")
            .field("name", &self.name)
            .field("n_models", &self.models.len())
            .field("weights", &self.weights)
            .field("n_features", &self.n_features)
            .finish()
    }
}

impl EnsembleModel {
    /// Create a new ensemble from sub-models and weights.
    ///
    /// # Errors
    /// - Fewer than 2 models
    /// - models.len() != weights.len()
    /// - Any weight <= 0
    pub fn new(
        name: String,
        models: Vec<Arc<dyn MlModel>>,
        weights: Vec<f64>,
    ) -> Result<Self, String> {
        if models.len() < 2 {
            return Err(format!(
                "Ensemble must have at least 2 models, got {}",
                models.len()
            ));
        }
        if models.len() != weights.len() {
            return Err(format!(
                "models count ({}) and weights count ({}) must match",
                models.len(),
                weights.len()
            ));
        }
        if let Some((i, &w)) = weights.iter().enumerate().find(|&(_, &w)| w <= 0.0) {
            return Err(format!(
                "All ensemble weights must be > 0, but weight[{}] = {}",
                i, w
            ));
        }

        let n_features = models.iter().map(|m| m.n_features()).max().unwrap();

        Ok(Self {
            name,
            models,
            weights,
            n_features,
        })
    }

    /// Get the ensemble weights (V7.1 uses this for weight optimization).
    pub fn weights(&self) -> &[f64] {
        &self.weights
    }

    /// Get the number of sub-models.
    pub fn n_models(&self) -> usize {
        self.models.len()
    }

    /// Get sub-model names for diagnostics.
    pub fn member_names(&self) -> Vec<&str> {
        self.models.iter().map(|m| m.name()).collect()
    }
}

impl MlModel for EnsembleModel {
    fn predict(&self, features: &[f64]) -> ClassProbabilities {
        let mut weighted = [0.0f64; 3];
        let mut total_weight = 0.0;

        for (model, &weight) in self.models.iter().zip(&self.weights) {
            let n = model.n_features().min(features.len());
            debug_assert!(
                features.len() >= model.n_features(),
                "Ensemble received {} features, sub-model '{}' needs {}",
                features.len(),
                model.name(),
                model.n_features()
            );
            let probs = model.predict(&features[..n]);
            weighted[0] += probs[0] * weight;
            weighted[1] += probs[1] * weight;
            weighted[2] += probs[2] * weight;
            total_weight += weight;
        }

        if total_weight > 0.0 {
            weighted[0] /= total_weight;
            weighted[1] /= total_weight;
            weighted[2] /= total_weight;
        }

        weighted
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn n_features(&self) -> usize {
        self.n_features
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Mock model that always returns fixed probabilities.
    struct FixedModel {
        name: String,
        probs: ClassProbabilities,
        n_features: usize,
    }

    impl MlModel for FixedModel {
        fn predict(&self, _features: &[f64]) -> ClassProbabilities {
            self.probs
        }
        fn name(&self) -> &str {
            &self.name
        }
        fn n_features(&self) -> usize {
            self.n_features
        }
    }

    fn make_model(name: &str, probs: [f64; 3], n_features: usize) -> Arc<dyn MlModel> {
        Arc::new(FixedModel {
            name: name.to_string(),
            probs,
            n_features,
        })
    }

    #[test]
    fn test_weighted_average() {
        let models = vec![
            make_model("A", [0.8, 0.1, 0.1], 4),
            make_model("B", [0.2, 0.6, 0.2], 4),
        ];
        let weights = vec![1.0, 1.0];
        let ensemble = EnsembleModel::new("test".into(), models, weights).unwrap();

        let probs = ensemble.predict(&[0.0; 4]);
        // Equal weights: average of [0.8, 0.1, 0.1] and [0.2, 0.6, 0.2]
        assert!((probs[0] - 0.5).abs() < 1e-10);
        assert!((probs[1] - 0.35).abs() < 1e-10);
        assert!((probs[2] - 0.15).abs() < 1e-10);
    }

    #[test]
    fn test_unequal_weights() {
        let models = vec![
            make_model("A", [1.0, 0.0, 0.0], 4),
            make_model("B", [0.0, 0.0, 1.0], 4),
        ];
        let weights = vec![3.0, 1.0];
        let ensemble = EnsembleModel::new("test".into(), models, weights).unwrap();

        let probs = ensemble.predict(&[0.0; 4]);
        // Weighted: (3*1.0 + 1*0.0)/4, (3*0.0 + 1*0.0)/4, (3*0.0 + 1*1.0)/4
        assert!((probs[0] - 0.75).abs() < 1e-10);
        assert!((probs[1] - 0.0).abs() < 1e-10);
        assert!((probs[2] - 0.25).abs() < 1e-10);
    }

    #[test]
    fn test_mixed_feature_counts() {
        // V5 model (42 features) and V6 model (55 features)
        let models = vec![
            make_model("V5", [0.6, 0.2, 0.2], 42),
            make_model("V6", [0.2, 0.2, 0.6], 55),
        ];
        let weights = vec![1.0, 1.0];
        let ensemble = EnsembleModel::new("test".into(), models, weights).unwrap();

        assert_eq!(ensemble.n_features(), 55); // max of sub-models

        // Provide 55 features — both sub-models should work
        let probs = ensemble.predict(&[0.0; 55]);
        let sum: f64 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_member_names() {
        let models = vec![
            make_model("RandomForest_small", [0.5, 0.3, 0.2], 42),
            make_model("LinearModel_v6", [0.2, 0.5, 0.3], 55),
        ];
        let ensemble = EnsembleModel::new("test".into(), models, vec![1.0, 1.0]).unwrap();
        assert_eq!(
            ensemble.member_names(),
            vec!["RandomForest_small", "LinearModel_v6"]
        );
    }

    #[test]
    fn test_validation_too_few_models() {
        let result = EnsembleModel::new(
            "test".into(),
            vec![make_model("A", [0.5, 0.3, 0.2], 4)],
            vec![1.0],
        );
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("at least 2"));
    }

    #[test]
    fn test_validation_mismatched_lengths() {
        let result = EnsembleModel::new(
            "test".into(),
            vec![
                make_model("A", [0.5, 0.3, 0.2], 4),
                make_model("B", [0.5, 0.3, 0.2], 4),
            ],
            vec![1.0], // Only 1 weight for 2 models
        );
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("must match"));
    }

    #[test]
    fn test_validation_zero_weight() {
        let result = EnsembleModel::new(
            "test".into(),
            vec![
                make_model("A", [0.5, 0.3, 0.2], 4),
                make_model("B", [0.5, 0.3, 0.2], 4),
            ],
            vec![1.0, 0.0],
        );
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("weight[1]"));
    }

    #[test]
    fn test_validation_negative_weight() {
        let result = EnsembleModel::new(
            "test".into(),
            vec![
                make_model("A", [0.5, 0.3, 0.2], 4),
                make_model("B", [0.5, 0.3, 0.2], 4),
            ],
            vec![1.0, -0.5],
        );
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("weight[1]"));
    }

    #[test]
    fn test_probabilities_sum_to_one() {
        let models = vec![
            make_model("A", [0.7, 0.2, 0.1], 4),
            make_model("B", [0.1, 0.3, 0.6], 4),
            make_model("C", [0.3, 0.5, 0.2], 4),
        ];
        let weights = vec![0.8, 1.2, 0.5];
        let ensemble = EnsembleModel::new("test".into(), models, weights).unwrap();

        let probs = ensemble.predict(&[0.0; 4]);
        let sum: f64 = probs.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-10,
            "Ensemble probabilities should sum to 1.0, got {}",
            sum
        );
    }
}

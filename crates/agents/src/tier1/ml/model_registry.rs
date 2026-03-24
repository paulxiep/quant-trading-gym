//! Model registry for centralized ML prediction (V5.6, SoC refactor).
//!
//! The [`ModelRegistry`] holds registered ML models and computes predictions
//! from pre-extracted features in the [`MlPredictionCache`]. This enables
//! O(M × S) predictions instead of O(N) per-agent computations.
//!
//! # SoC
//!
//! ModelRegistry is **prediction only**. Feature extraction is the runner's
//! responsibility — the runner extracts features once and populates the cache.
//! ModelRegistry reads cached features and computes predictions.
//!
//! # Usage
//!
//! ```ignore
//! use agents::tier1::ml::{ModelRegistry, DecisionTree};
//!
//! let mut registry = ModelRegistry::new();
//! registry.register(DecisionTree::from_json("model1.json")?);
//! registry.register(DecisionTree::from_json("model2.json")?);
//!
//! // Runner populates cache with features, then:
//! registry.predict_from_cache(&mut cache);
//! ```

use std::collections::HashMap;
use std::sync::Arc;

use crate::ml_cache::MlPredictionCache;
use crate::tier1::ml::MlModel;

/// Registry of ML models for centralized prediction computation.
///
/// Stores models by name. Provides bulk prediction computation for all
/// (model, symbol) pairs from pre-extracted features.
///
/// # SoC (pre-V6 refactor section F)
///
/// ModelRegistry is responsible for **prediction only**. Feature extraction
/// is the runner's responsibility — the runner extracts features once and
/// feeds them to both the ML cache (for prediction) and recording hooks
/// (for Parquet). This separation ensures training-serving parity and
/// avoids recalculation.
pub struct ModelRegistry {
    /// Models indexed by name.
    models: HashMap<String, Arc<dyn MlModel>>,
}

impl Default for ModelRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl ModelRegistry {
    /// Create a new empty model registry.
    pub fn new() -> Self {
        Self {
            models: HashMap::new(),
        }
    }

    /// Register a model with the registry.
    ///
    /// If a model with the same name already exists, it will be replaced.
    pub fn register<M: MlModel + 'static>(&mut self, model: M) {
        let name = model.name().to_string();
        self.models.insert(name, Arc::new(model));
    }

    /// Register a model wrapped in Arc.
    ///
    /// Useful when the model is already Arc-wrapped from elsewhere.
    pub fn register_arc(&mut self, model: Arc<dyn MlModel>) {
        let name = model.name().to_string();
        self.models.insert(name, model);
    }

    /// Get a model by name.
    pub fn get(&self, name: &str) -> Option<&Arc<dyn MlModel>> {
        self.models.get(name)
    }

    /// Check if a model is registered.
    pub fn contains(&self, name: &str) -> bool {
        self.models.contains_key(name)
    }

    /// Get the number of registered models.
    pub fn len(&self) -> usize {
        self.models.len()
    }

    /// Check if the registry is empty.
    pub fn is_empty(&self) -> bool {
        self.models.is_empty()
    }

    /// Get all model names.
    pub fn model_names(&self) -> Vec<&str> {
        self.models.keys().map(|s| s.as_str()).collect()
    }

    /// Compute predictions for all (model, symbol) pairs from pre-extracted features.
    ///
    /// Features must already be populated in the cache (by the runner).
    /// This method only handles the prediction step: O(M × S), parallelized.
    ///
    /// # SoC
    ///
    /// Feature extraction is NOT this method's responsibility. The runner extracts
    /// features once and inserts them into the cache. This method reads those features
    /// and computes predictions. The same extracted features are also passed to
    /// recording hooks for Parquet output — guaranteeing training-serving parity.
    pub fn predict_from_cache(&self, cache: &mut MlPredictionCache) {
        let symbols: Vec<_> = cache.feature_symbols().cloned().collect();

        // Create work items: (model_name, model, symbol)
        let work_items: Vec<_> = self
            .models
            .iter()
            .flat_map(|(name, model)| {
                symbols
                    .iter()
                    .map(move |symbol| (name.clone(), model.clone(), symbol.clone()))
            })
            .collect();

        // Parallel prediction computation
        let predictions: Vec<_> = parallel::filter_map_slice(
            &work_items,
            |(model_name, model, symbol)| {
                cache.get_features(symbol).map(|features| {
                    let probs = model.predict(features);
                    (model_name.clone(), symbol.clone(), probs)
                })
            },
            false, // Use parallel execution
        );

        // Insert predictions into cache
        for (model_name, symbol, probs) in predictions {
            cache.insert_prediction(&model_name, &symbol, probs);
        }
    }
}

impl std::fmt::Debug for ModelRegistry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ModelRegistry")
            .field("model_count", &self.models.len())
            .field("models", &self.model_names())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Mock model for testing
    struct MockModel {
        name: String,
        prediction: [f64; 3],
    }

    impl MockModel {
        fn new(name: &str, prediction: [f64; 3]) -> Self {
            Self {
                name: name.to_string(),
                prediction,
            }
        }
    }

    impl MlModel for MockModel {
        fn predict(&self, _features: &[f64]) -> [f64; 3] {
            self.prediction
        }

        fn name(&self) -> &str {
            &self.name
        }
    }

    #[test]
    fn test_registry_creation() {
        let registry = ModelRegistry::new();
        assert!(registry.is_empty());
        assert_eq!(registry.len(), 0);
    }

    #[test]
    fn test_register_and_get() {
        let mut registry = ModelRegistry::new();
        let model = MockModel::new("test_model", [0.1, 0.2, 0.7]);

        registry.register(model);

        assert!(!registry.is_empty());
        assert_eq!(registry.len(), 1);
        assert!(registry.contains("test_model"));
        assert!(registry.get("test_model").is_some());
    }

    #[test]
    fn test_multiple_models() {
        let mut registry = ModelRegistry::new();

        registry.register(MockModel::new("model_a", [0.3, 0.3, 0.4]));
        registry.register(MockModel::new("model_b", [0.1, 0.1, 0.8]));
        registry.register(MockModel::new("model_c", [0.5, 0.3, 0.2]));

        assert_eq!(registry.len(), 3);

        let names = registry.model_names();
        assert!(names.contains(&"model_a"));
        assert!(names.contains(&"model_b"));
        assert!(names.contains(&"model_c"));
    }

    #[test]
    fn test_model_replacement() {
        let mut registry = ModelRegistry::new();

        registry.register(MockModel::new("model", [0.1, 0.1, 0.8]));
        registry.register(MockModel::new("model", [0.8, 0.1, 0.1])); // Same name

        assert_eq!(registry.len(), 1);

        // Should have the new prediction
        let model = registry.get("model").unwrap();
        let probs = model.predict(&[]);
        assert_eq!(probs, [0.8, 0.1, 0.1]);
    }

    #[test]
    fn test_register_arc() {
        let mut registry = ModelRegistry::new();
        let model: Arc<dyn MlModel> = Arc::new(MockModel::new("arc_model", [0.2, 0.6, 0.2]));

        registry.register_arc(model);

        assert!(registry.contains("arc_model"));
    }

    #[test]
    fn test_debug_format() {
        let mut registry = ModelRegistry::new();
        registry.register(MockModel::new("debug_test", [0.3, 0.4, 0.3]));

        let debug_str = format!("{:?}", registry);
        assert!(debug_str.contains("ModelRegistry"));
        assert!(debug_str.contains("model_count"));
    }
}

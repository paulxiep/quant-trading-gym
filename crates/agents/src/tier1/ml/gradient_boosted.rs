//! Gradient Boosted model for classification (V5.5).
//!
//! Loads sklearn GradientBoostingClassifier from JSON and performs inference
//! using staged regression trees with a learning rate.
//!
//! # JSON Format
//!
//! ```json
//! {
//!   "model_type": "gradient_boosted",
//!   "model_name": "fast",
//!   "n_features": 42,
//!   "n_classes": 3,
//!   "classes": [-1, 0, 1],
//!   "n_estimators": 50,
//!   "learning_rate": 0.2,
//!   "init_value": [0.31, 0.41, 0.27],
//!   "stages": [
//!     [
//!       { "n_nodes": 113, "nodes": [...] },  // Tree for class -1
//!       { "n_nodes": 115, "nodes": [...] },  // Tree for class 0
//!       { "n_nodes": 109, "nodes": [...] }   // Tree for class 1
//!     ],
//!     ...
//!   ]
//! }
//! ```
//!
//! # Prediction
//!
//! 1. Initialize raw scores from `init_value`
//! 2. For each stage, traverse each class's tree and accumulate:
//!    `scores[class] += learning_rate * leaf_value`
//! 3. Apply softmax to convert scores to probabilities

use std::path::Path;

use serde::Deserialize;

use super::{ClassProbabilities, MlModel, compute_feature_indices, remap_features, softmax};

/// A single node in a regression tree (gradient boosting uses regression trees).
#[derive(Debug, Clone, Deserialize)]
pub struct RegressionNode {
    /// Feature index to split on (-1 for leaf nodes).
    pub feature: i32,
    /// Threshold value for the split.
    pub threshold: f64,
    /// Index of left child (-1 for leaf nodes).
    pub left: i32,
    /// Index of right child (-1 for leaf nodes).
    pub right: i32,
    /// Scalar prediction value for leaf nodes (None for internal nodes).
    /// This is a raw score, not a probability.
    pub value: Option<f64>,
}

/// Tree structure from JSON.
#[derive(Debug, Deserialize)]
struct TreeJson {
    #[allow(dead_code)]
    n_nodes: usize,
    nodes: Vec<RegressionNode>,
}

/// Complete gradient boosted model from JSON.
#[derive(Debug, Deserialize)]
struct GradientBoostedJson {
    model_type: String,
    model_name: String,
    #[allow(dead_code)]
    feature_names: Vec<String>,
    n_features: usize,
    n_classes: usize,
    #[allow(dead_code)]
    classes: Vec<i32>,
    n_estimators: usize,
    learning_rate: f64,
    /// Initial class probabilities (prior).
    init_value: Vec<f64>,
    /// Stages[stage_idx][class_idx] = tree for that class at that stage.
    stages: Vec<Vec<TreeJson>>,
}

/// Gradient Boosted classifier loaded from sklearn JSON export.
#[derive(Debug, Clone)]
pub struct GradientBoosted {
    /// Model name for identification.
    name: String,
    /// Number of features the model was trained on.
    n_features: usize,
    /// Number of classes (should be 3).
    #[allow(dead_code)]
    n_classes: usize,
    /// Learning rate for boosting.
    learning_rate: f64,
    /// Initial log-odds for each class.
    init_value: Vec<f64>,
    /// Stages of regression trees: stages[stage][class] = nodes.
    stages: Vec<Vec<Vec<RegressionNode>>>,
    /// Feature index remapping for subset-trained models (V6.2+).
    feature_indices: Option<Vec<usize>>,
}

impl GradientBoosted {
    /// Load a gradient boosted model from a JSON file.
    ///
    /// The model name is derived from the filename by stripping the `_gradient_boosted` suffix,
    /// formatted as `GradientBoosted_{name}`.
    ///
    /// # Arguments
    /// * `path` - Path to the JSON file exported from sklearn
    ///
    /// # Errors
    /// Returns error if file cannot be read or JSON is malformed.
    pub fn from_json<P: AsRef<Path>>(path: P) -> Result<Self, String> {
        let path = path.as_ref();
        let content = std::fs::read_to_string(path)
            .map_err(|e| format!("Failed to read {}: {}", path.display(), e))?;

        // Derive model name from filename (e.g., "fast_gradient_boosted.json" -> "GradientBoosted_fast")
        let file_stem = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("unknown");
        let model_name = file_stem
            .strip_suffix("_gradient_boosted")
            .unwrap_or(file_stem);
        let name = format!("GradientBoosted_{}", model_name);

        Self::from_json_str_with_name(&content, name)
    }

    /// Load a gradient boosted model from a JSON string with a custom name.
    fn from_json_str_with_name(json: &str, name: String) -> Result<Self, String> {
        let model: GradientBoostedJson =
            serde_json::from_str(json).map_err(|e| format!("JSON parse error: {}", e))?;

        // Validate model type
        if model.model_type != "gradient_boosted" {
            return Err(format!(
                "Expected model_type 'gradient_boosted', got '{}'",
                model.model_type
            ));
        }

        // Validate feature count
        if model.n_features == 0 || model.n_features > types::N_FULL_FEATURES {
            return Err(format!(
                "Expected 1-{} features, got {}",
                types::N_FULL_FEATURES,
                model.n_features
            ));
        }

        // Validate class count
        if model.n_classes != 3 {
            return Err(format!("Expected 3 classes, got {}", model.n_classes));
        }

        // Validate init_value
        if model.init_value.len() != model.n_classes {
            return Err(format!(
                "init_value has {} elements, expected {}",
                model.init_value.len(),
                model.n_classes
            ));
        }

        // Validate stages count
        if model.stages.len() != model.n_estimators {
            return Err(format!(
                "n_estimators ({}) doesn't match stages count ({})",
                model.n_estimators,
                model.stages.len()
            ));
        }

        // Validate each stage has n_classes trees
        for (i, stage) in model.stages.iter().enumerate() {
            if stage.len() != model.n_classes {
                return Err(format!(
                    "Stage {} has {} trees, expected {}",
                    i,
                    stage.len(),
                    model.n_classes
                ));
            }
        }

        // Validate learning rate
        if model.learning_rate <= 0.0 || model.learning_rate > 1.0 {
            return Err(format!(
                "Invalid learning_rate: {} (should be 0 < lr <= 1)",
                model.learning_rate
            ));
        }

        // Extract stages
        let stages: Vec<Vec<Vec<RegressionNode>>> = model
            .stages
            .into_iter()
            .map(|stage| stage.into_iter().map(|t| t.nodes).collect())
            .collect();

        let feature_indices = compute_feature_indices(&model.feature_names, model.n_features);

        Ok(Self {
            name,
            n_features: model.n_features,
            n_classes: model.n_classes,
            learning_rate: model.learning_rate,
            init_value: model.init_value,
            stages,
            feature_indices,
        })
    }

    /// Load a gradient boosted model from a JSON string (uses model_name from JSON).
    pub fn from_json_str(json: &str) -> Result<Self, String> {
        // Parse just to get the model_name for backwards compatibility
        let model: GradientBoostedJson =
            serde_json::from_str(json).map_err(|e| format!("JSON parse error: {}", e))?;
        let name = format!("GradientBoosted_{}", model.model_name);
        Self::from_json_str_with_name(json, name)
    }

    /// Traverse a regression tree and return the leaf value.
    #[inline]
    fn traverse_tree(nodes: &[RegressionNode], features: &[f64]) -> f64 {
        let mut node_idx = 0usize;

        loop {
            let node = &nodes[node_idx];

            // Leaf node - return value
            if node.feature == -1 {
                return node.value.unwrap_or(0.0);
            }

            // Get feature value (NaN-safe)
            let feature_val = features
                .get(node.feature as usize)
                .copied()
                .unwrap_or(f64::NAN);

            // Decision: NaN or <= threshold goes left (conservative)
            if feature_val.is_nan() || feature_val <= node.threshold {
                node_idx = node.left as usize;
            } else {
                node_idx = node.right as usize;
            }
        }
    }

    /// Number of boosting stages.
    pub fn n_estimators(&self) -> usize {
        self.stages.len()
    }
}

impl MlModel for GradientBoosted {
    fn predict(&self, features: &[f64]) -> ClassProbabilities {
        let mut buf = Vec::new();
        let feats = remap_features(features, &self.feature_indices, &mut buf);

        // Initialize raw scores from init_value (log-odds prior)
        let mut scores = self.init_value.clone();

        // Accumulate predictions from each stage
        for stage in &self.stages {
            for (class_idx, tree) in stage.iter().enumerate() {
                let leaf_value = Self::traverse_tree(tree, feats);
                scores[class_idx] += self.learning_rate * leaf_value;
            }
        }

        // Apply softmax for numerical stability
        softmax(&scores)
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn n_features(&self) -> usize {
        if self.feature_indices.is_some() {
            types::N_FULL_FEATURES
        } else {
            self.n_features
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_gbt_json() -> &'static str {
        r#"{
            "model_type": "gradient_boosted",
            "model_name": "test",
            "feature_names": ["f0","f1","f2","f3","f4","f5","f6","f7","f8","f9","f10","f11","f12","f13","f14","f15","f16","f17","f18","f19","f20","f21","f22","f23","f24","f25","f26","f27","f28","f29","f30","f31","f32","f33","f34","f35","f36","f37","f38","f39","f40","f41"],
            "n_features": 42,
            "n_classes": 3,
            "classes": [-1, 0, 1],
            "n_estimators": 1,
            "learning_rate": 0.1,
            "init_value": [0.0, 0.0, 0.0],
            "stages": [
                [
                    {
                        "n_nodes": 3,
                        "nodes": [
                            {"feature": 0, "threshold": 50.0, "left": 1, "right": 2, "value": null},
                            {"feature": -1, "threshold": 0.0, "left": -1, "right": -1, "value": 2.0},
                            {"feature": -1, "threshold": 0.0, "left": -1, "right": -1, "value": -1.0}
                        ]
                    },
                    {
                        "n_nodes": 3,
                        "nodes": [
                            {"feature": 0, "threshold": 50.0, "left": 1, "right": 2, "value": null},
                            {"feature": -1, "threshold": 0.0, "left": -1, "right": -1, "value": 0.0},
                            {"feature": -1, "threshold": 0.0, "left": -1, "right": -1, "value": 0.0}
                        ]
                    },
                    {
                        "n_nodes": 3,
                        "nodes": [
                            {"feature": 0, "threshold": 50.0, "left": 1, "right": 2, "value": null},
                            {"feature": -1, "threshold": 0.0, "left": -1, "right": -1, "value": -1.0},
                            {"feature": -1, "threshold": 0.0, "left": -1, "right": -1, "value": 2.0}
                        ]
                    }
                ]
            ]
        }"#
    }

    #[test]
    fn test_load_from_json() {
        let gbt = GradientBoosted::from_json_str(sample_gbt_json()).unwrap();
        assert_eq!(gbt.name, "GradientBoosted_test");
        assert_eq!(gbt.n_features, 42);
        assert_eq!(gbt.n_classes, 3);
        assert_eq!(gbt.n_estimators(), 1);
        assert!((gbt.learning_rate - 0.1).abs() < 1e-10);
    }

    #[test]
    fn test_predict_low_feature() {
        let gbt = GradientBoosted::from_json_str(sample_gbt_json()).unwrap();

        // Feature[0] = 30 <= 50 → left branches
        // Class 0 (sell): 0.0 + 0.1 * 2.0 = 0.2
        // Class 1 (hold): 0.0 + 0.1 * 0.0 = 0.0
        // Class 2 (buy):  0.0 + 0.1 * -1.0 = -0.1
        // After softmax: sell should be highest
        let mut features = [0.0; 42];
        features[0] = 30.0;

        let probs = gbt.predict(&features);
        assert!(probs[0] > probs[1]); // sell > hold
        assert!(probs[0] > probs[2]); // sell > buy
    }

    #[test]
    fn test_predict_high_feature() {
        let gbt = GradientBoosted::from_json_str(sample_gbt_json()).unwrap();

        // Feature[0] = 70 > 50 → right branches
        // Class 0 (sell): 0.0 + 0.1 * -1.0 = -0.1
        // Class 1 (hold): 0.0 + 0.1 * 0.0 = 0.0
        // Class 2 (buy):  0.0 + 0.1 * 2.0 = 0.2
        // After softmax: buy should be highest
        let mut features = [0.0; 42];
        features[0] = 70.0;

        let probs = gbt.predict(&features);
        assert!(probs[2] > probs[1]); // buy > hold
        assert!(probs[2] > probs[0]); // buy > sell
    }

    #[test]
    fn test_probabilities_sum_to_one() {
        let gbt = GradientBoosted::from_json_str(sample_gbt_json()).unwrap();

        let features = [0.0; 42];
        let probs = gbt.predict(&features);

        let sum: f64 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_invalid_model_type() {
        let json = r#"{"model_type": "random_forest", "model_name": "x", "feature_names": [], "n_features": 42, "n_classes": 3, "classes": [], "n_estimators": 0, "learning_rate": 0.1, "init_value": [0,0,0], "stages": []}"#;
        let result = GradientBoosted::from_json_str(json);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("gradient_boosted"));
    }
}

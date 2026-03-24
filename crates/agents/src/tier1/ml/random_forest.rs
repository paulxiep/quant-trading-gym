//! Random Forest model for classification (V5.5).
//!
//! Loads sklearn RandomForestClassifier from JSON and performs inference
//! by averaging predictions from all trees in the ensemble.
//!
//! # JSON Format
//!
//! ```json
//! {
//!   "model_type": "random_forest",
//!   "model_name": "small",
//!   "n_features": 42,
//!   "n_classes": 3,
//!   "classes": [-1, 0, 1],
//!   "n_estimators": 50,
//!   "trees": [
//!     { "n_nodes": 393, "nodes": [...] },
//!     ...
//!   ]
//! }
//! ```
//!
//! # Prediction
//!
//! 1. Traverse each tree to get leaf probabilities
//! 2. Average probabilities across all trees
//! 3. Return averaged [p_sell, p_hold, p_buy]

use std::path::Path;

use serde::Deserialize;

use super::decision_tree::TreeNode;
use super::{ClassProbabilities, MlModel, compute_feature_indices, remap_features};

/// Tree structure from JSON.
#[derive(Debug, Deserialize)]
struct TreeJson {
    #[allow(dead_code)]
    n_nodes: usize,
    nodes: Vec<TreeNode>,
}

/// Complete random forest model from JSON.
#[derive(Debug, Deserialize)]
struct RandomForestJson {
    model_type: String,
    model_name: String,
    #[allow(dead_code)]
    feature_names: Vec<String>,
    n_features: usize,
    n_classes: usize,
    #[allow(dead_code)]
    classes: Vec<i32>,
    n_estimators: usize,
    trees: Vec<TreeJson>,
}

/// Random Forest classifier loaded from sklearn JSON export.
#[derive(Debug, Clone)]
pub struct RandomForest {
    /// Model name for identification.
    name: String,
    /// Number of features the model was trained on.
    n_features: usize,
    /// Number of classes (should be 3).
    #[allow(dead_code)]
    n_classes: usize,
    /// All trees in the ensemble.
    trees: Vec<Vec<TreeNode>>,
    /// Feature index remapping for subset-trained models (V6.2+).
    feature_indices: Option<Vec<usize>>,
}

impl RandomForest {
    /// Load a random forest from a JSON file.
    ///
    /// The model name is derived from the filename by stripping the `_random_forest` suffix,
    /// formatted as `RandomForest_{name}`.
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

        // Derive model name from filename (e.g., "small_random_forest.json" -> "RandomForest_small")
        let file_stem = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("unknown");
        let model_name = file_stem
            .strip_suffix("_random_forest")
            .unwrap_or(file_stem);
        let name = format!("RandomForest_{}", model_name);

        Self::from_json_str_with_name(&content, name)
    }

    /// Load a random forest from a JSON string with a custom name.
    fn from_json_str_with_name(json: &str, name: String) -> Result<Self, String> {
        let model: RandomForestJson =
            serde_json::from_str(json).map_err(|e| format!("JSON parse error: {}", e))?;

        // Validate model type
        if model.model_type != "random_forest" {
            return Err(format!(
                "Expected model_type 'random_forest', got '{}'",
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

        // Validate tree count
        if model.trees.len() != model.n_estimators {
            return Err(format!(
                "n_estimators ({}) doesn't match trees count ({})",
                model.n_estimators,
                model.trees.len()
            ));
        }

        if model.trees.is_empty() {
            return Err("Random forest has no trees".to_string());
        }

        // Extract trees
        let trees: Vec<Vec<TreeNode>> = model.trees.into_iter().map(|t| t.nodes).collect();

        let feature_indices = compute_feature_indices(&model.feature_names, model.n_features);

        Ok(Self {
            name,
            n_features: model.n_features,
            n_classes: model.n_classes,
            trees,
            feature_indices,
        })
    }

    /// Load a random forest from a JSON string (uses model_name from JSON).
    pub fn from_json_str(json: &str) -> Result<Self, String> {
        // Parse just to get the model_name for backwards compatibility
        let model: RandomForestJson =
            serde_json::from_str(json).map_err(|e| format!("JSON parse error: {}", e))?;
        let name = format!("RandomForest_{}", model.model_name);
        Self::from_json_str_with_name(json, name)
    }

    /// Traverse a single tree and return leaf probabilities.
    #[inline]
    fn traverse_tree(nodes: &[TreeNode], features: &[f64]) -> ClassProbabilities {
        let mut node_idx = 0usize;

        loop {
            let node = &nodes[node_idx];

            // Leaf node - return probabilities
            if node.feature == -1 {
                return match &node.value {
                    Some(probs) if probs.len() >= 3 => [probs[0], probs[1], probs[2]],
                    _ => [0.0, 1.0, 0.0], // Fallback to hold
                };
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

    /// Number of trees in the ensemble.
    pub fn n_estimators(&self) -> usize {
        self.trees.len()
    }
}

impl MlModel for RandomForest {
    fn predict(&self, features: &[f64]) -> ClassProbabilities {
        if self.trees.is_empty() {
            return [0.0, 1.0, 0.0]; // No trees = hold
        }

        let mut buf = Vec::new();
        let feats = remap_features(features, &self.feature_indices, &mut buf);

        // Sum probabilities from all trees
        let sum = self.trees.iter().fold([0.0, 0.0, 0.0], |mut acc, tree| {
            let probs = Self::traverse_tree(tree, feats);
            acc[0] += probs[0];
            acc[1] += probs[1];
            acc[2] += probs[2];
            acc
        });

        // Average
        let n = self.trees.len() as f64;
        [sum[0] / n, sum[1] / n, sum[2] / n]
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

    fn sample_forest_json() -> &'static str {
        r#"{
            "model_type": "random_forest",
            "model_name": "test",
            "feature_names": ["f0","f1","f2","f3","f4","f5","f6","f7","f8","f9","f10","f11","f12","f13","f14","f15","f16","f17","f18","f19","f20","f21","f22","f23","f24","f25","f26","f27","f28","f29","f30","f31","f32","f33","f34","f35","f36","f37","f38","f39","f40","f41"],
            "n_features": 42,
            "n_classes": 3,
            "classes": [-1, 0, 1],
            "n_estimators": 2,
            "trees": [
                {
                    "n_nodes": 3,
                    "nodes": [
                        {"feature": 0, "threshold": 50.0, "left": 1, "right": 2, "value": null},
                        {"feature": -1, "threshold": 0.0, "left": -1, "right": -1, "value": [0.6, 0.2, 0.2]},
                        {"feature": -1, "threshold": 0.0, "left": -1, "right": -1, "value": [0.2, 0.2, 0.6]}
                    ]
                },
                {
                    "n_nodes": 3,
                    "nodes": [
                        {"feature": 0, "threshold": 60.0, "left": 1, "right": 2, "value": null},
                        {"feature": -1, "threshold": 0.0, "left": -1, "right": -1, "value": [0.4, 0.4, 0.2]},
                        {"feature": -1, "threshold": 0.0, "left": -1, "right": -1, "value": [0.1, 0.1, 0.8]}
                    ]
                }
            ]
        }"#
    }

    #[test]
    fn test_load_from_json() {
        let forest = RandomForest::from_json_str(sample_forest_json()).unwrap();
        assert_eq!(forest.name, "RandomForest_test");
        assert_eq!(forest.n_features, 42);
        assert_eq!(forest.n_classes, 3);
        assert_eq!(forest.n_estimators(), 2);
    }

    #[test]
    fn test_predict_averaging() {
        let forest = RandomForest::from_json_str(sample_forest_json()).unwrap();

        // Feature[0] = 30 → both trees go left
        // Tree 1: [0.6, 0.2, 0.2]
        // Tree 2: [0.4, 0.4, 0.2]
        // Average: [0.5, 0.3, 0.2]
        let mut features = [0.0; 42];
        features[0] = 30.0;

        let probs = forest.predict(&features);
        assert!((probs[0] - 0.5).abs() < 0.01);
        assert!((probs[1] - 0.3).abs() < 0.01);
        assert!((probs[2] - 0.2).abs() < 0.01);
    }

    #[test]
    fn test_predict_mixed_paths() {
        let forest = RandomForest::from_json_str(sample_forest_json()).unwrap();

        // Feature[0] = 55 → Tree 1 goes right (>50), Tree 2 goes left (<=60)
        // Tree 1: [0.2, 0.2, 0.6]
        // Tree 2: [0.4, 0.4, 0.2]
        // Average: [0.3, 0.3, 0.4]
        let mut features = [0.0; 42];
        features[0] = 55.0;

        let probs = forest.predict(&features);
        assert!((probs[0] - 0.3).abs() < 0.01);
        assert!((probs[1] - 0.3).abs() < 0.01);
        assert!((probs[2] - 0.4).abs() < 0.01);
    }

    #[test]
    fn test_invalid_model_type() {
        let json = r#"{"model_type": "decision_tree", "model_name": "x", "feature_names": [], "n_features": 42, "n_classes": 3, "classes": [], "n_estimators": 0, "trees": []}"#;
        let result = RandomForest::from_json_str(json);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("random_forest"));
    }
}

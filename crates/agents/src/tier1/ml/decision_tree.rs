//! Decision Tree model for classification (V5.5).
//!
//! Loads sklearn DecisionTreeClassifier from JSON and performs inference.
//!
//! # JSON Format
//!
//! ```json
//! {
//!   "model_type": "decision_tree",
//!   "model_name": "shallow",
//!   "n_features": 42,
//!   "n_classes": 3,
//!   "classes": [-1, 0, 1],
//!   "tree": {
//!     "n_nodes": 245,
//!     "nodes": [
//!       { "feature": 37, "threshold": 0.46, "left": 1, "right": 92, "value": null },
//!       { "feature": -1, "threshold": 0.0, "left": -1, "right": -1, "value": [1.0, 0.0, 0.0] }
//!     ]
//!   }
//! }
//! ```
//!
//! # Tree Traversal
//!
//! - Start at node 0 (root)
//! - If `feature == -1`, this is a leaf node; return `value` as probabilities
//! - Else: compare `features[node.feature]` to `node.threshold`
//!   - If `<= threshold` or `NaN`, go to `left` child
//!   - Else go to `right` child
//! - Repeat until reaching a leaf

use std::path::Path;

use serde::Deserialize;

use super::{ClassProbabilities, MlModel, compute_feature_indices, remap_features};

/// A single node in the decision tree.
#[derive(Debug, Clone, Deserialize)]
pub struct TreeNode {
    /// Feature index to split on (-1 for leaf nodes).
    pub feature: i32,
    /// Threshold value for the split.
    pub threshold: f64,
    /// Index of left child (-1 for leaf nodes).
    pub left: i32,
    /// Index of right child (-1 for leaf nodes).
    pub right: i32,
    /// Class probabilities for leaf nodes (None for internal nodes).
    /// Format: [p_sell, p_hold, p_buy] for classes [-1, 0, 1]
    pub value: Option<Vec<f64>>,
}

/// Tree structure from JSON.
#[derive(Debug, Deserialize)]
struct TreeJson {
    n_nodes: usize,
    nodes: Vec<TreeNode>,
}

/// Complete decision tree model from JSON.
#[derive(Debug, Deserialize)]
struct DecisionTreeJson {
    model_type: String,
    model_name: String,
    #[allow(dead_code)]
    feature_names: Vec<String>,
    n_features: usize,
    n_classes: usize,
    #[allow(dead_code)]
    classes: Vec<i32>,
    tree: TreeJson,
}

/// Decision Tree classifier loaded from sklearn JSON export.
#[derive(Debug, Clone)]
pub struct DecisionTree {
    /// Model name for identification.
    name: String,
    /// Number of features the model was trained on.
    n_features: usize,
    /// Number of classes (should be 3).
    #[allow(dead_code)]
    n_classes: usize,
    /// Tree nodes in pre-order traversal.
    nodes: Vec<TreeNode>,
    /// Feature index remapping for subset-trained models (V6.2+).
    /// Maps model feature positions to full-vector positions.
    /// None when model uses canonical 42 or 55 features.
    feature_indices: Option<Vec<usize>>,
}

impl DecisionTree {
    /// Load a decision tree from a JSON file.
    ///
    /// The model name is derived from the filename by stripping the `_decision_tree` suffix,
    /// formatted as `DecisionTree_{name}`.
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

        // Derive model name from filename (e.g., "deep_decision_tree.json" -> "DecisionTree_deep")
        let file_stem = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("unknown");
        let model_name = file_stem
            .strip_suffix("_decision_tree")
            .unwrap_or(file_stem);
        let name = format!("DecisionTree_{}", model_name);

        Self::from_json_str_with_name(&content, name)
    }

    /// Load a decision tree from a JSON string with a custom name.
    fn from_json_str_with_name(json: &str, name: String) -> Result<Self, String> {
        let model: DecisionTreeJson =
            serde_json::from_str(json).map_err(|e| format!("JSON parse error: {}", e))?;

        // Validate model type
        if model.model_type != "decision_tree" {
            return Err(format!(
                "Expected model_type 'decision_tree', got '{}'",
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

        // Validate node indices
        for (i, node) in model.tree.nodes.iter().enumerate() {
            if node.feature != -1 {
                // Internal node - validate children
                if node.left < 0 || node.left as usize >= model.tree.n_nodes {
                    return Err(format!("Node {} has invalid left child {}", i, node.left));
                }
                if node.right < 0 || node.right as usize >= model.tree.n_nodes {
                    return Err(format!("Node {} has invalid right child {}", i, node.right));
                }
                // Validate feature index
                if node.feature < 0 || node.feature as usize >= model.n_features {
                    return Err(format!(
                        "Node {} has invalid feature index {}",
                        i, node.feature
                    ));
                }
            } else {
                // Leaf node - validate value
                match &node.value {
                    Some(v) if v.len() == 3 => {}
                    Some(v) => {
                        return Err(format!(
                            "Leaf node {} has {} probabilities, expected 3",
                            i,
                            v.len()
                        ));
                    }
                    None => {
                        return Err(format!("Leaf node {} missing value array", i));
                    }
                }
            }
        }

        let feature_indices = compute_feature_indices(&model.feature_names, model.n_features);

        Ok(Self {
            name,
            n_features: model.n_features,
            n_classes: model.n_classes,
            nodes: model.tree.nodes,
            feature_indices,
        })
    }

    /// Load a decision tree from a JSON string (uses model_name from JSON).
    pub fn from_json_str(json: &str) -> Result<Self, String> {
        // Parse just to get the model_name for backwards compatibility
        let model: DecisionTreeJson =
            serde_json::from_str(json).map_err(|e| format!("JSON parse error: {}", e))?;
        let name = format!("DecisionTree_{}", model.model_name);
        Self::from_json_str_with_name(json, name)
    }

    /// Traverse the tree for given features and return leaf node index.
    #[inline]
    fn traverse(&self, features: &[f64]) -> usize {
        let mut node_idx = 0usize;

        loop {
            let node = &self.nodes[node_idx];

            // Leaf node - return index
            if node.feature == -1 {
                return node_idx;
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
}

impl MlModel for DecisionTree {
    fn predict(&self, features: &[f64]) -> ClassProbabilities {
        let mut buf = Vec::new();
        let feats = remap_features(features, &self.feature_indices, &mut buf);
        let leaf_idx = self.traverse(feats);
        let node = &self.nodes[leaf_idx];

        match &node.value {
            Some(probs) if probs.len() >= 3 => [probs[0], probs[1], probs[2]],
            _ => [0.0, 1.0, 0.0], // Fallback to hold
        }
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn n_features(&self) -> usize {
        // When remapping is active, report full vector size so callers pass the full vector
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

    fn sample_tree_json() -> &'static str {
        r#"{
            "model_type": "decision_tree",
            "model_name": "test",
            "feature_names": ["f0","f1","f2","f3","f4","f5","f6","f7","f8","f9","f10","f11","f12","f13","f14","f15","f16","f17","f18","f19","f20","f21","f22","f23","f24","f25","f26","f27","f28","f29","f30","f31","f32","f33","f34","f35","f36","f37","f38","f39","f40","f41"],
            "n_features": 42,
            "n_classes": 3,
            "classes": [-1, 0, 1],
            "tree": {
                "n_nodes": 5,
                "nodes": [
                    {"feature": 0, "threshold": 50.0, "left": 1, "right": 2, "value": null},
                    {"feature": -1, "threshold": 0.0, "left": -1, "right": -1, "value": [0.8, 0.1, 0.1]},
                    {"feature": 1, "threshold": 0.5, "left": 3, "right": 4, "value": null},
                    {"feature": -1, "threshold": 0.0, "left": -1, "right": -1, "value": [0.1, 0.8, 0.1]},
                    {"feature": -1, "threshold": 0.0, "left": -1, "right": -1, "value": [0.1, 0.1, 0.8]}
                ]
            }
        }"#
    }

    #[test]
    fn test_load_from_json() {
        let tree = DecisionTree::from_json_str(sample_tree_json()).unwrap();
        assert_eq!(tree.name, "DecisionTree_test");
        assert_eq!(tree.n_features, 42);
        assert_eq!(tree.n_classes, 3);
        assert_eq!(tree.nodes.len(), 5);
    }

    #[test]
    fn test_traverse_left() {
        let tree = DecisionTree::from_json_str(sample_tree_json()).unwrap();

        // Feature[0] = 30 <= 50 → go left → leaf 1 (sell)
        let mut features = [0.0; 42];
        features[0] = 30.0;

        let probs = tree.predict(&features);
        assert!((probs[0] - 0.8).abs() < 0.01); // p_sell
    }

    #[test]
    fn test_traverse_right_then_left() {
        let tree = DecisionTree::from_json_str(sample_tree_json()).unwrap();

        // Feature[0] = 60 > 50 → go right
        // Feature[1] = 0.3 <= 0.5 → go left → leaf 3 (hold)
        let mut features = [0.0; 42];
        features[0] = 60.0;
        features[1] = 0.3;

        let probs = tree.predict(&features);
        assert!((probs[1] - 0.8).abs() < 0.01); // p_hold
    }

    #[test]
    fn test_traverse_right_then_right() {
        let tree = DecisionTree::from_json_str(sample_tree_json()).unwrap();

        // Feature[0] = 60 > 50 → go right
        // Feature[1] = 0.8 > 0.5 → go right → leaf 4 (buy)
        let mut features = [0.0; 42];
        features[0] = 60.0;
        features[1] = 0.8;

        let probs = tree.predict(&features);
        assert!((probs[2] - 0.8).abs() < 0.01); // p_buy
    }

    #[test]
    fn test_nan_goes_left() {
        let tree = DecisionTree::from_json_str(sample_tree_json()).unwrap();

        // Feature[0] = NaN → go left (conservative) → leaf 1 (sell)
        let mut features = [0.0; 42];
        features[0] = f64::NAN;

        let probs = tree.predict(&features);
        assert!((probs[0] - 0.8).abs() < 0.01); // p_sell (left child)
    }

    #[test]
    fn test_invalid_model_type() {
        let json = r#"{"model_type": "random_forest", "model_name": "x", "feature_names": [], "n_features": 42, "n_classes": 3, "classes": [], "tree": {"n_nodes": 0, "nodes": []}}"#;
        let result = DecisionTree::from_json_str(json);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("decision_tree"));
    }
}

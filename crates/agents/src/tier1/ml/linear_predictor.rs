//! Linear model for classification (V6.2).
//!
//! Unified inference for sklearn LogisticRegression and LinearSVC.
//! Both compute `softmax(W*x + b)` — the difference is training loss, not inference.
//!
//! # JSON Format (LogisticRegression)
//!
//! ```json
//! {
//!   "model_type": "linear_model",
//!   "model_name": "logistic_v6",
//!   "n_features": 55,
//!   "n_classes": 3,
//!   "classes": [-1, 0, 1],
//!   "coefficients": [[...55 weights...], [...], [...]],
//!   "intercepts": [0.1, -0.3, 0.2]
//! }
//! ```
//!
//! # JSON Format (LinearSVC)
//!
//! ```json
//! {
//!   "model_type": "svm_linear",
//!   "model_name": "linear_svc_v6",
//!   "n_features": 55,
//!   "n_classes": 3,
//!   "classes": [-1, 0, 1],
//!   "weights": [[...55 weights...], [...], [...]],
//!   "biases": [0.1, -0.3, 0.2]
//! }
//! ```
//!
//! Both formats produce the same `LinearPredictor` struct via serde aliasing.

use std::path::Path;

use serde::Deserialize;

use super::{ClassProbabilities, MlModel, softmax};

/// JSON deserialization: accept both sklearn naming conventions via serde alias.
#[derive(Debug, Deserialize)]
struct LinearPredictorJson {
    model_type: String,
    model_name: String,
    n_features: usize,
    n_classes: usize,
    #[allow(dead_code)]
    classes: Vec<i32>,
    /// LogisticRegression uses "coefficients", LinearSVC uses "weights".
    #[serde(alias = "coefficients")]
    weights: Vec<Vec<f64>>,
    /// LogisticRegression uses "intercepts", LinearSVC uses "biases".
    #[serde(alias = "intercepts")]
    biases: Vec<f64>,
}

/// Linear classifier loaded from sklearn JSON export.
///
/// Handles both LogisticRegression and LinearSVC inference via `softmax(W*x + b)`.
/// The `model_type` field in JSON determines the name prefix for logging.
#[derive(Debug)]
pub struct LinearPredictor {
    /// Model name (e.g., "LinearModel_logistic_v6" or "SvmLinear_linear_svc_v6").
    name: String,
    /// Weight matrix: [3][n_features].
    weights: Vec<Vec<f64>>,
    /// Bias vector: [3].
    biases: Vec<f64>,
    /// Number of features expected.
    n_features: usize,
}

impl LinearPredictor {
    /// Load a linear predictor from a JSON file.
    pub fn from_json<P: AsRef<Path>>(path: P) -> Result<Self, String> {
        let path = path.as_ref();
        let content = std::fs::read_to_string(path)
            .map_err(|e| format!("Failed to read {}: {}", path.display(), e))?;
        Self::from_json_str(&content)
    }

    /// Load a linear predictor from a JSON string.
    pub fn from_json_str(json: &str) -> Result<Self, String> {
        let parsed: LinearPredictorJson =
            serde_json::from_str(json).map_err(|e| format!("JSON parse error: {}", e))?;

        // Validate model type
        if parsed.model_type != "linear_model" && parsed.model_type != "svm_linear" {
            return Err(format!(
                "Expected model_type 'linear_model' or 'svm_linear', got '{}'",
                parsed.model_type
            ));
        }

        // Validate class count
        if parsed.n_classes != 3 {
            return Err(format!("Expected 3 classes, got {}", parsed.n_classes));
        }

        // Validate weight matrix dimensions
        if parsed.weights.len() != 3 {
            return Err(format!(
                "Expected 3 weight rows, got {}",
                parsed.weights.len()
            ));
        }
        for (i, row) in parsed.weights.iter().enumerate() {
            if row.len() != parsed.n_features {
                return Err(format!(
                    "Weight row {} has {} elements, expected {}",
                    i,
                    row.len(),
                    parsed.n_features
                ));
            }
        }

        // Validate bias vector
        if parsed.biases.len() != 3 {
            return Err(format!("Expected 3 biases, got {}", parsed.biases.len()));
        }

        // Name prefix from model_type for log clarity
        let prefix = match parsed.model_type.as_str() {
            "svm_linear" => "SvmLinear",
            _ => "LinearModel",
        };
        let name = format!("{}_{}", prefix, parsed.model_name);

        Ok(Self {
            name,
            weights: parsed.weights,
            biases: parsed.biases,
            n_features: parsed.n_features,
        })
    }
}

impl MlModel for LinearPredictor {
    fn predict(&self, features: &[f64]) -> ClassProbabilities {
        let n = features.len().min(self.n_features);
        let mut scores = [0.0f64; 3];
        for (class, score) in scores.iter_mut().enumerate() {
            *score = self.weights[class][..n]
                .iter()
                .zip(features[..n].iter())
                .map(|(w, f)| w * f)
                .sum::<f64>()
                + self.biases[class];
        }
        softmax(&scores)
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

    fn sample_logistic_json() -> &'static str {
        r#"{
            "model_type": "linear_model",
            "model_name": "logistic_v6",
            "n_features": 4,
            "n_classes": 3,
            "classes": [-1, 0, 1],
            "coefficients": [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0]
            ],
            "intercepts": [0.0, 0.0, 0.0]
        }"#
    }

    fn sample_svm_json() -> &'static str {
        r#"{
            "model_type": "svm_linear",
            "model_name": "linear_svc_v6",
            "n_features": 4,
            "n_classes": 3,
            "classes": [-1, 0, 1],
            "weights": [
                [2.0, 0.0, 0.0, 0.0],
                [0.0, 2.0, 0.0, 0.0],
                [0.0, 0.0, 2.0, 0.0]
            ],
            "biases": [0.0, 0.0, 0.0]
        }"#
    }

    #[test]
    fn test_load_logistic_regression() {
        let model = LinearPredictor::from_json_str(sample_logistic_json()).unwrap();
        assert_eq!(model.name(), "LinearModel_logistic_v6");
        assert_eq!(model.n_features(), 4);
    }

    #[test]
    fn test_load_svm_linear() {
        let model = LinearPredictor::from_json_str(sample_svm_json()).unwrap();
        assert_eq!(model.name(), "SvmLinear_linear_svc_v6");
        assert_eq!(model.n_features(), 4);
    }

    #[test]
    fn test_predictions_sum_to_one() {
        let model = LinearPredictor::from_json_str(sample_logistic_json()).unwrap();
        let features = [1.0, 0.5, 0.3, 0.1];
        let probs = model.predict(&features);

        let sum: f64 = probs.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-10,
            "Probabilities should sum to 1.0, got {}",
            sum
        );
        assert!(probs.iter().all(|&p| p >= 0.0 && p <= 1.0));
    }

    #[test]
    fn test_strongest_class_wins() {
        let model = LinearPredictor::from_json_str(sample_logistic_json()).unwrap();

        // Feature[0] = 5.0 dominates → class 0 (sell) should have highest probability
        let features = [5.0, 0.0, 0.0, 0.0];
        let probs = model.predict(&features);
        assert!(probs[0] > probs[1], "sell should > hold");
        assert!(probs[0] > probs[2], "sell should > buy");

        // Feature[2] = 5.0 dominates → class 2 (buy) should have highest probability
        let features = [0.0, 0.0, 5.0, 0.0];
        let probs = model.predict(&features);
        assert!(probs[2] > probs[0], "buy should > sell");
        assert!(probs[2] > probs[1], "buy should > hold");
    }

    #[test]
    fn test_equal_scores_uniform_probs() {
        let model = LinearPredictor::from_json_str(sample_logistic_json()).unwrap();
        let features = [0.0, 0.0, 0.0, 0.0];
        let probs = model.predict(&features);

        for &p in &probs {
            assert!(
                (p - 1.0 / 3.0).abs() < 1e-10,
                "Equal scores should give ~uniform probabilities"
            );
        }
    }

    #[test]
    fn test_svm_alias_works() {
        // Verify serde alias: "weights"/"biases" in SVM JSON maps to same fields
        let model = LinearPredictor::from_json_str(sample_svm_json()).unwrap();
        let features = [5.0, 0.0, 0.0, 0.0];
        let probs = model.predict(&features);
        assert!(probs[0] > probs[1], "sell should dominate with svm weights");
    }

    #[test]
    fn test_invalid_model_type() {
        let json = r#"{"model_type": "decision_tree", "model_name": "x", "n_features": 4, "n_classes": 3, "classes": [-1,0,1], "weights": [[0.0],[0.0],[0.0]], "biases": [0.0,0.0,0.0]}"#;
        let result = LinearPredictor::from_json_str(json);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("linear_model"));
    }

    #[test]
    fn test_wrong_class_count() {
        let json = r#"{"model_type": "linear_model", "model_name": "x", "n_features": 4, "n_classes": 2, "classes": [-1,1], "coefficients": [[0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0]], "intercepts": [0.0,0.0]}"#;
        let result = LinearPredictor::from_json_str(json);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("3 classes"));
    }

    #[test]
    fn test_mismatched_weight_dimensions() {
        let json = r#"{"model_type": "linear_model", "model_name": "x", "n_features": 4, "n_classes": 3, "classes": [-1,0,1], "coefficients": [[0.0,0.0],[0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0]], "intercepts": [0.0,0.0,0.0]}"#;
        let result = LinearPredictor::from_json_str(json);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Weight row 0"));
    }
}
